"""Flexible classification pipeline for patent classification supporting different model types."""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime
import logging
from tqdm import tqdm

from config import Config
from data_loader import PatentDataLoader
from modal_client_flexible import FlexibleModalPatentClassifier
from cost_tracker import CostTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlexiblePatentClassificationPipeline:
    """Flexible pipeline for patent classification supporting different model types."""
    
    def __init__(self, max_sequence_length: int, config: Config = Config(), 
                 enable_cost_tracking: bool = True, model_type: str = "auto"):
        self.config = config
        self.model_type = model_type
        self.max_sequence_length = max_sequence_length
        self.data_loader = PatentDataLoader(config)
        
        # Initialize cost tracking
        self.cost_tracker = CostTracker(config.MODEL_NAME) if enable_cost_tracking else None
        
        # Use flexible Modal client
        try:
            self.classifier = FlexibleModalPatentClassifier(
                self.max_sequence_length,
                config, 
                cost_tracker=self.cost_tracker, 
                model_type=model_type
            )
            logger.info(f"Initialized flexible Modal classifier successfully (type: {model_type})")
        except Exception as e:
            logger.error(f"Failed to initialize flexible Modal classifier: {e}")
            raise RuntimeError(f"Flexible Modal is required but failed to initialize: {e}. "
                             f"Please ensure Modal is properly configured and accessible.")
        
        self.results = {}
        
        # Create output directories
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.config.LOGS_DIR, exist_ok=True)
        
    def run_inference(self, split: str = 'test', max_samples: Optional[int] = None, 
                     save_results: bool = True, note: Optional[str] = None) -> Dict:
        """Run inference on specified dataset split."""
        logger.info(f"Starting flexible inference on {split} set with model type: {self.model_type}")
        
        # Start cost tracking
        if self.cost_tracker:
            self.cost_tracker.start_run()
            
        try:
            # Load data
            data = self.data_loader.prepare_data_for_model(split, max_samples)
            logger.info(f"Processing {len(data)} samples")
            
            # Run batch classification
            results = []
            correct_predictions = 0
            
            # Extract texts and labels for batch processing
            texts = [sample['text'] for sample in data]
            labels = [sample['label'] for sample in data]
            
            try:
                # Use batch processing to keep Modal container alive
                predictions = self.classifier.classify_batch(texts, true_labels=labels)
                
                # Process results
                for i, (sample, prediction) in enumerate(tqdm(zip(data, predictions), 
                                                            desc=f"Processing {split} results", 
                                                            total=len(data))):
                    result = {
                        'sample_id': i,
                        'true_label': sample['label'],
                        'true_label_name': sample['label_name'],
                        'predicted_label': prediction['predicted_class'],
                        'predicted_label_name': prediction['class_name'],
                        'raw_response': prediction['raw_response'],
                        'text_preview': sample['text'][:200] + "...",
                        'is_correct': prediction['predicted_class'] == sample['label'] if prediction['predicted_class'] is not None else False,
                        'input_tokens': prediction.get('input_tokens', 0),
                        'output_tokens': prediction.get('output_tokens', 0),
                        'method': prediction.get('method', 'unknown'),
                        'confidence': prediction.get('confidence', None)
                    }
                    
                    if result['is_correct']:
                        correct_predictions += 1
                        
                    results.append(result)
                    
                    # Log progress every 10 samples
                    if (i + 1) % 10 == 0:
                        current_accuracy = correct_predictions / (i + 1)
                        cost_info = ""
                        if self.cost_tracker:
                            current_metrics = self.cost_tracker.get_metrics()
                            cost_info = f", Cost so far: ${current_metrics.total_estimated_cost_usd:.4f}"
                        logger.info(f"Processed {i+1}/{len(data)}, Current accuracy: {current_accuracy:.3f}{cost_info}")
                
                # Print final results after batch processing
                print(f"\n🎯 CLASSIFICATION RESULTS:")
                print(f"   Correct Classifications: {correct_predictions}/{len(data)} ({correct_predictions/len(data)*100:.1f}%)")
                print(f"   Model Type: {self.model_type}")
                print(f"   Method Used: {predictions[0].get('method', 'unknown') if predictions else 'none'}")
                
                if self.cost_tracker:
                    current_metrics = self.cost_tracker.get_metrics()
                    print(f"💰 TOKEN COSTS:")
                    print(f"   Input Tokens: {current_metrics.total_input_tokens:,}")
                    print(f"   Output Tokens: {current_metrics.total_output_tokens:,}")
                    print(f"   Total Cost: ${current_metrics.total_estimated_cost_usd:.6f}")
                        
            except Exception as e:
                logger.error(f"Error in flexible batch classification: {e}")
                # Fallback to individual error results
                for i, sample in enumerate(data):
                    results.append({
                        'sample_id': i,
                        'true_label': sample['label'],
                        'true_label_name': sample['label_name'],
                        'predicted_label': None,
                        'predicted_label_name': "Error",
                        'raw_response': f"Flexible Batch Error: {str(e)}",
                        'text_preview': sample['text'][:200] + "...",
                        'is_correct': False,
                        'input_tokens': 0,
                        'output_tokens': 0,
                        'method': 'error',
                        'confidence': None
                    })
        
            # Calculate final metrics
            final_accuracy = correct_predictions / len(results)
            valid_predictions = sum(1 for r in results if r['predicted_label'] is not None)
            
            # End cost tracking
            if self.cost_tracker:
                self.cost_tracker.end_run()
                cost_metrics = self.cost_tracker.get_metrics()
            else:
                cost_metrics = None
            
            summary = {
                'split': split,
                'total_samples': len(results),
                'valid_predictions': valid_predictions,
                'correct_predictions': correct_predictions,
                'accuracy': final_accuracy,
                'prediction_rate': valid_predictions / len(results),
                'timestamp': datetime.now().isoformat(),
                'note': note,
                'model_type': self.model_type,
                'method': results[0].get('method', 'unknown') if results else 'none',
                'config': {
                    'model_name': self.config.MODEL_NAME,
                    'max_sequence_length': self.max_sequence_length,
                    'temperature': self.config.TEMPERATURE,
                    'top_p': self.config.TOP_P
                },
                'cost_metrics': cost_metrics.__dict__ if cost_metrics else None
            }
            
            self.results[split] = {
                'summary': summary,
                'detailed_results': results
            }
            
            logger.info(f"Flexible inference completed!")
            logger.info(f"Accuracy: {final_accuracy:.3f}")
            logger.info(f"Valid predictions: {valid_predictions}/{len(results)} ({valid_predictions/len(results):.3f})")
            logger.info(f"Method used: {summary['method']}")
            
            # Print cost summary if tracking is enabled
            if self.cost_tracker:
                self.cost_tracker.print_cost_summary()
            
            # Save results
            if save_results:
                self.save_results(split)
                if self.cost_tracker:
                    self.save_cost_metrics(split)
                
            return self.results[split]
            
        except Exception as e:
            # Ensure cost tracking is stopped even on error
            if self.cost_tracker:
                self.cost_tracker.end_run()
            raise e
    
    def save_results(self, split: str):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"patent_classification_results_{split}_{timestamp}.json"
        filepath = os.path.join(self.config.OUTPUT_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results[split], f, indent=2, ensure_ascii=False)
            
        logger.info(f"Results saved to: {filepath}")
    
    def save_cost_metrics(self, split: str):
        """Save cost metrics to separate JSON file."""
        if not self.cost_tracker:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"patent_classification_costs_{split}_{timestamp}.json"
        filepath = os.path.join(self.config.OUTPUT_DIR, filename)
        
        self.cost_tracker.save_metrics(filepath)
        logger.info(f"Cost metrics saved to: {filepath}")
        
    def analyze_results(self, split: str) -> Dict:
        """Analyze classification results by class."""
        if split not in self.results:
            raise ValueError(f"No results found for split '{split}'. Run inference first.")
            
        results = self.results[split]['detailed_results']
        
        # Per-class analysis
        class_stats = {}
        for class_id, class_name in self.config.CLASS_LABELS.items():
            true_positives = sum(1 for r in results if r['true_label'] == class_id and r['is_correct'])
            false_positives = sum(1 for r in results if r['predicted_label'] == class_id and r['true_label'] != class_id)
            false_negatives = sum(1 for r in results if r['true_label'] == class_id and not r['is_correct'])
            total_true = sum(1 for r in results if r['true_label'] == class_id)
            total_predicted = sum(1 for r in results if r['predicted_label'] == class_id)
            
            precision = true_positives / total_predicted if total_predicted > 0 else 0
            recall = true_positives / total_true if total_true > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_stats[class_id] = {
                'class_name': class_name,
                'true_positives': true_positives,
                'false_positives': false_positives, 
                'false_negatives': false_negatives,
                'total_true': total_true,
                'total_predicted': total_predicted,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
        return class_stats
    
    def print_analysis(self, split: str):
        """Print detailed analysis of results."""
        class_stats = self.analyze_results(split)
        
        print(f"\n{'='*60}")
        print(f"FLEXIBLE CLASSIFICATION ANALYSIS - {split.upper()} SET")
        print(f"{'='*60}")
        
        print(f"\nOverall Accuracy: {self.results[split]['summary']['accuracy']:.3f}")
        print(f"Valid Predictions: {self.results[split]['summary']['prediction_rate']:.3f}")
        print(f"Model Type: {self.results[split]['summary']['model_type']}")
        print(f"Method Used: {self.results[split]['summary']['method']}")
        
        # Print note if provided
        note = self.results[split]['summary'].get('note')
        if note:
            print(f"\n📝 Note: {note}")
        
        print(f"\n{'Class':<5} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8} Class Name")
        print("-" * 80)
        
        for class_id in sorted(class_stats.keys()):
            stats = class_stats[class_id]
            class_name = stats['class_name'][:35] + "..." if len(stats['class_name']) > 35 else stats['class_name']
            print(f"{class_id:<5} {stats['precision']:<10.3f} {stats['recall']:<10.3f} {stats['f1_score']:<10.3f} {stats['total_true']:<8} {class_name}")

if __name__ == "__main__":
    # Demo usage
    pipeline = FlexiblePatentClassificationPipeline(model_type="auto")
    
    # Run on a small sample first
    results = pipeline.run_inference('test', max_samples=20)
    pipeline.print_analysis('test')