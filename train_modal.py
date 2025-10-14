#!/usr/bin/env python3
"""Modal-based training script for fine-tuning BERT on patent classification with GPU acceleration."""

import modal
from typing import Dict, Optional
import json

# Create Modal app for training
app = modal.App("patent-bert-training")

# Define training image with all required dependencies
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "huggingface_hub>=0.25.0",
        "tensorboard>=2.14.0",
        "tiktoken>=0.5.0",  # Required for DeBERTa-v3-large
        "sentencepiece>=0.1.99",  # Required for DeBERTa tokenizer
    ])
)

@app.function(
    image=training_image,
    gpu="A10G",  # Use A10G for cost-effective training
    memory=32768,  # 32GB memory for training
    timeout=7200,  # 2 hours timeout
    secrets=[modal.Secret.from_name("huggingface-secret")],  # HF token for dataset access
)
def train_patent_bert(
    model_name: str = "bert-base-uncased",
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    max_samples: Optional[int] = None,
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
    run_name: Optional[str] = None
) -> Dict:
    """Train BERT model on patent classification dataset using Modal GPU."""
    
    import os
    import torch
    from datetime import datetime
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        DebertaV2Tokenizer
    )
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
    import numpy as np
    from huggingface_hub import login
    
    # Setup - disable wandb
    os.environ["WANDB_DISABLED"] = "true"
    
    print(f"🚀 Starting Modal training on GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"📋 Training Configuration:")
    print(f"   Model: {model_name}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Batch size: {batch_size}")
    print(f"   Max samples: {max_samples or 'all'}")
    
    # Authenticate with HuggingFace if token available
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("✅ Authenticated with HuggingFace")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not run_name:
        run_name = f"patent-bert-{timestamp}"
    
    # Load dataset
    print("📥 Loading patent classification dataset...")
    dataset = load_dataset("ccdv/patent-classification", "abstract")
    
    if max_samples:
        for split in dataset:
            dataset[split] = dataset[split].select(range(min(max_samples, len(dataset[split]))))
            print(f"   {split}: {len(dataset[split])} samples")
    
    # Load model and tokenizer
    print(f"🔧 Loading model and tokenizer: {model_name}")
    # Use specific tokenizer for DeBERTa models to avoid conversion issues
    if "deberta" in model_name.lower():
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=9,  # 9 patent classes
        problem_type="single_label_classification"
    )
    
    # Tokenize dataset
    print("⚙️ Tokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],  # Use 'text' instead of 'abstract'
            truncation=True,
            padding=False,
            max_length=512,
            return_tensors=None
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']  # Remove 'text' column only
    )
    
    # Compute metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"/tmp/{run_name}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4 if batch_size <= 4 else 1,  # Use gradient accumulation for small batch sizes
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=learning_rate,
        logging_steps=50,
        eval_strategy="epoch",  # Fixed parameter name
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=[],  # Explicitly disable all reporting
        seed=42,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        hub_strategy="end" if push_to_hub else None,  # Reference the flag correctly
        dataloader_num_workers=2,  # Reduce workers for large models
        fp16=True,  # Use mixed precision for faster training
        dataloader_pin_memory=False,  # Disable pin memory to save GPU memory
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        max_length=512
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"] if "validation" in tokenized_dataset else None,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train the model
    print("🏋️ Starting training...")
    train_result = trainer.train()
    
    print("✅ Training completed!")
    print(f"   Training loss: {train_result.metrics['train_loss']:.4f}")
    print(f"   Training time: {train_result.metrics['train_runtime']:.2f} seconds")
    print(f"   Samples/second: {train_result.metrics['train_samples_per_second']:.2f}")
    
    # Evaluate on test set
    test_results = {}
    if "test" in tokenized_dataset:
        print("📊 Evaluating on test set...")
        test_results = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
        
        print(f"📈 Test Results:")
        print(f"   Accuracy: {test_results['eval_accuracy']:.4f}")
        print(f"   F1-Score: {test_results['eval_f1']:.4f}")
        print(f"   Precision: {test_results['eval_precision']:.4f}")
        print(f"   Recall: {test_results['eval_recall']:.4f}")
        
        # Generate predictions for detailed analysis
        test_predictions = trainer.predict(tokenized_dataset["test"])
        y_true = test_predictions.label_ids
        y_pred = np.argmax(test_predictions.predictions, axis=1)
        
        # Class labels
        class_labels = {
            0: "Human Necessities",
            1: "Performing Operations; Transporting", 
            2: "Chemistry; Metallurgy",
            3: "Textiles; Paper",
            4: "Fixed Constructions",
            5: "Mechanical Engineering; Lightning; Heating; Weapons; Blasting",
            6: "Physics",
            7: "Electricity",
            8: "General tagging of new or cross-sectional technology"
        }
        
        class_report = classification_report(
            y_true, y_pred, 
            target_names=[class_labels[i] for i in range(9)],
            output_dict=True
        )
        
        test_results['classification_report'] = class_report
    
    # Save model to Hub if requested
    if push_to_hub and hub_model_id:
        print(f"📤 Pushing model to HuggingFace Hub: {hub_model_id}")
        trainer.push_to_hub(commit_message="Fine-tuned BERT for patent classification")
        tokenizer.push_to_hub(hub_model_id)
    
    # Return results
    results = {
        'training_results': train_result.metrics,
        'test_results': test_results,
        'model_name': model_name,
        'run_name': run_name,
        'timestamp': timestamp,
        'hub_model_id': hub_model_id if push_to_hub else None
    }
    
    print(f"\n🎉 Modal training completed successfully!")
    if push_to_hub:
        print(f"📁 Model published to: https://huggingface.co/{hub_model_id}")
    
    return results

@app.local_entrypoint()
def main(
    model_name: str = "bert-base-uncased",
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    max_samples: int = None,
    push_to_hub: bool = False,
    hub_model_id: str = None,
    run_name: str = None
):
    """Local entrypoint to start Modal training."""
    
    print("🚀 Starting Modal-based BERT training for patent classification...")
    print(f"📋 Configuration:")
    print(f"   Model: {model_name}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Batch size: {batch_size}")
    print(f"   Max samples: {max_samples or 'all'}")
    print(f"   Push to Hub: {push_to_hub}")
    if push_to_hub:
        print(f"   Hub Model ID: {hub_model_id}")
    
    # Run training on Modal
    results = train_patent_bert.remote(
        model_name=model_name,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_samples=max_samples,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        run_name=run_name
    )
    
    # Print final results
    print(f"\n{'='*80}")
    print(f"MODAL TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"Model: {results['model_name']}")
    print(f"Run name: {results['run_name']}")
    print(f"Training loss: {results['training_results']['train_loss']:.4f}")
    print(f"Training time: {results['training_results']['train_runtime']:.2f} seconds")
    
    if results['test_results']:
        print(f"Test accuracy: {results['test_results']['eval_accuracy']:.4f}")
        print(f"Test F1-score: {results['test_results']['eval_f1']:.4f}")
    
    if results['hub_model_id']:
        print(f"🔗 Model URL: https://huggingface.co/{results['hub_model_id']}")
        print(f"\nTo use your trained model:")
        print(f"python main.py --mode classify --model {results['hub_model_id']} --model_type classification")
    
    print(f"{'='*80}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune BERT on Modal with GPU acceleration")
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                       help='Model name or path (default: bert-base-uncased)')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs (default: 3)')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate (default: 2e-5)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size (default: 16)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples per split for testing (default: None)')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Custom run name (default: auto-generated)')
    
    # HuggingFace Hub arguments
    parser.add_argument('--push_to_hub', action='store_true',
                       help='Push trained model to HuggingFace Hub')
    parser.add_argument('--hub_model_id', type=str, default=None,
                       help='Model ID for HuggingFace Hub (required if --push_to_hub)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.push_to_hub and not args.hub_model_id:
        parser.error("--hub_model_id is required when --push_to_hub is specified")
    
    # Convert args to dict
    kwargs = {
        'model_name': args.model_name,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'max_samples': args.max_samples,
        'push_to_hub': args.push_to_hub,
        'hub_model_id': args.hub_model_id,
        'run_name': args.run_name
    }
    
    main(**kwargs)