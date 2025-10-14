#!/usr/bin/env python3
"""Training script for fine-tuning BERT on patent classification dataset."""

import argparse
import os
import json
import logging
import torch
from datetime import datetime
from typing import Dict, Optional

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np

from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatentClassificationTrainer:
    """Trainer for fine-tuning BERT models on patent classification."""
    
    def __init__(self, model_name: str = "bert-base-uncased", config: Config = Config()):
        self.model_name = model_name
        self.config = config
        self.tokenizer = None
        self.model = None
        self.dataset = None
        
        # Create output directories
        os.makedirs("./models", exist_ok=True)
        os.makedirs("./training_logs", exist_ok=True)
        
    def load_dataset(self, max_samples: Optional[int] = None):
        """Load and prepare the patent classification dataset."""
        logger.info("Loading patent classification dataset...")
        
        # Load the dataset
        dataset = load_dataset("ccdv/patent-classification", "abstract")
        
        # Limit samples if specified (for testing)
        if max_samples:
            for split in dataset:
                dataset[split] = dataset[split].select(range(min(max_samples, len(dataset[split]))))
                logger.info(f"{split}: {len(dataset[split])} samples")
        
        # Map string labels to integers if needed
        def map_labels(examples):
            # The dataset should already have integer labels, but let's make sure
            return examples
        
        dataset = dataset.map(map_labels, batched=True)
        self.dataset = dataset
        
        logger.info("Dataset loaded successfully:")
        for split in dataset:
            logger.info(f"  {split}: {len(dataset[split])} examples")
            
        return dataset
    
    def load_model_and_tokenizer(self):
        """Load the model and tokenizer."""
        logger.info(f"Loading model and tokenizer: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model for sequence classification
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=9,  # 9 patent classes
            problem_type="single_label_classification"
        )
        
        logger.info("Model and tokenizer loaded successfully")
        
    def tokenize_dataset(self):
        """Tokenize the dataset."""
        logger.info("Tokenizing dataset...")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],  # The dataset uses 'text' not 'abstract'
                truncation=True,
                padding=False,  # We'll pad dynamically during training
                max_length=512,  # Standard BERT max length
                return_tensors=None
            )
        
        # Check what columns exist
        print(f"Dataset columns: {self.dataset['train'].column_names}")
        
        # Tokenize all splits
        tokenized_dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']  # Remove text column, keep labels
        )
        
        self.dataset = tokenized_dataset
        logger.info("Dataset tokenized successfully")
        
        return tokenized_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, 
              output_dir: str = "./models/patent-bert",
              num_epochs: int = 3,
              learning_rate: float = 2e-5,
              batch_size: int = 16,
              warmup_steps: int = 500,
              weight_decay: float = 0.01,
              save_strategy: str = "epoch",
              evaluation_strategy: str = "epoch",
              early_stopping_patience: int = 3,
              push_to_hub: bool = False,
              hub_model_id: Optional[str] = None):
        """Fine-tune the model."""
        
        logger.info("Starting training...")
        
        # Create timestamp for this training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = f"{output_dir}_{timestamp}"
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=run_output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            logging_dir=f'./training_logs/patent_bert_{timestamp}',
            logging_steps=100,
            eval_strategy=evaluation_strategy,  # Changed from evaluation_strategy
            eval_steps=500 if evaluation_strategy == "steps" else None,
            save_strategy=save_strategy,
            save_steps=500 if save_strategy == "steps" else None,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to=None,  # Disable wandb/tensorboard for now
            seed=42,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
        )
        
        # Data collator for dynamic padding
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=512
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"] if "validation" in self.dataset else None,
            compute_metrics=self.compute_metrics,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
        )
        
        # Train the model
        logger.info(f"Training configuration:")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Output directory: {run_output_dir}")
        
        train_result = trainer.train()
        
        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(run_output_dir)
        
        # Save training metrics
        with open(f"{run_output_dir}/training_results.json", "w") as f:
            json.dump({
                "train_runtime": train_result.metrics["train_runtime"],
                "train_samples_per_second": train_result.metrics["train_samples_per_second"],
                "train_loss": train_result.metrics["train_loss"],
                "model_name": self.model_name,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "timestamp": timestamp
            }, f, indent=2)
        
        logger.info(f"Training completed! Model saved to: {run_output_dir}")
        
        # Evaluate on test set if available
        if "test" in self.dataset:
            logger.info("Evaluating on test set...")
            test_results = trainer.evaluate(eval_dataset=self.dataset["test"])
            
            # Save test results
            with open(f"{run_output_dir}/test_results.json", "w") as f:
                json.dump(test_results, f, indent=2)
            
            logger.info(f"Test Results:")
            logger.info(f"  Accuracy: {test_results['eval_accuracy']:.4f}")
            logger.info(f"  F1-Score: {test_results['eval_f1']:.4f}")
            logger.info(f"  Precision: {test_results['eval_precision']:.4f}")
            logger.info(f"  Recall: {test_results['eval_recall']:.4f}")
            
            # Generate detailed classification report
            test_predictions = trainer.predict(self.dataset["test"])
            y_true = test_predictions.label_ids
            y_pred = np.argmax(test_predictions.predictions, axis=1)
            
            # Create classification report
            class_report = classification_report(
                y_true, y_pred, 
                target_names=[self.config.CLASS_LABELS[i] for i in range(9)],
                output_dict=True
            )
            
            # Save detailed report
            with open(f"{run_output_dir}/classification_report.json", "w") as f:
                json.dump(class_report, f, indent=2)
            
            # Print summary
            print(f"\n{'='*80}")
            print(f"TRAINING SUMMARY")
            print(f"{'='*80}")
            print(f"Model: {self.model_name}")
            print(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds")
            print(f"Final training loss: {train_result.metrics['train_loss']:.4f}")
            print(f"Test accuracy: {test_results['eval_accuracy']:.4f}")
            print(f"Test F1-score: {test_results['eval_f1']:.4f}")
            print(f"Model saved to: {run_output_dir}")
            print(f"{'='*80}")
        
        return run_output_dir, train_result

def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for patent classification")
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                       help='Model name or path (default: bert-base-uncased)')
    parser.add_argument('--output_dir', type=str, default='./models/patent-bert',
                       help='Output directory for saved model')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs (default: 3)')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate (default: 2e-5)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size (default: 16)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples per split for testing (default: None)')
    parser.add_argument('--warmup_steps', type=int, default=500,
                       help='Number of warmup steps (default: 500)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay (default: 0.01)')
    
    # Early stopping and evaluation
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                       help='Early stopping patience (default: 3)')
    parser.add_argument('--evaluation_strategy', type=str, default='epoch',
                       choices=['no', 'steps', 'epoch'],
                       help='Evaluation strategy (default: epoch)')
    parser.add_argument('--save_strategy', type=str, default='epoch',
                       choices=['no', 'steps', 'epoch'],
                       help='Save strategy (default: epoch)')
    
    # Hub integration
    parser.add_argument('--push_to_hub', action='store_true',
                       help='Push model to HuggingFace Hub after training')
    parser.add_argument('--hub_model_id', type=str, default=None,
                       help='Model ID for HuggingFace Hub (required if push_to_hub)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.push_to_hub and not args.hub_model_id:
        parser.error("--hub_model_id is required when --push_to_hub is specified")
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize trainer
    trainer = PatentClassificationTrainer(model_name=args.model_name)
    
    # Load and prepare dataset
    trainer.load_dataset(max_samples=args.max_samples)
    trainer.load_model_and_tokenizer()
    trainer.tokenize_dataset()
    
    # Train the model
    output_dir, train_result = trainer.train(
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        save_strategy=args.save_strategy,
        evaluation_strategy=args.evaluation_strategy,
        early_stopping_patience=args.early_stopping_patience,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id
    )
    
    print(f"\n🎉 Training completed successfully!")
    print(f"📁 Model saved to: {output_dir}")
    print(f"\nTo use your trained model:")
    print(f"python main.py --mode classify --model {output_dir} --model_type classification")

if __name__ == "__main__":
    main()