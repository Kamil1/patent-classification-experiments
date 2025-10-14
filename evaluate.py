"""Evaluation utilities and metrics for patent classification."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
from typing import Dict, List, Tuple
import json
import os
from config import Config

class PatentClassificationEvaluator:
    """Evaluator for patent classification results."""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        
    def load_results(self, results_path: str) -> Dict:
        """Load results from JSON file."""
        with open(results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_predictions_and_labels(self, results: Dict) -> Tuple[List[int], List[int]]:
        """Extract true labels and predictions from results."""
        detailed_results = results['detailed_results']
        
        y_true = []
        y_pred = []
        
        for result in detailed_results:
            if result['predicted_label'] is not None:
                y_true.append(result['true_label'])
                y_pred.append(result['predicted_label'])
                
        return y_true, y_pred
    
    def calculate_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict:
        """Calculate comprehensive classification metrics."""
        
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0, labels=list(range(9))
        )
        
        metrics = {
            'overall': {
                'accuracy': accuracy,
                'weighted_precision': precision,
                'weighted_recall': recall,
                'weighted_f1': f1,
                'total_samples': len(y_true)
            },
            'per_class': {}
        }
        
        for i in range(9):
            metrics['per_class'][i] = {
                'class_name': self.config.CLASS_LABELS[i],
                'precision': precision_per_class[i] if i < len(precision_per_class) else 0,
                'recall': recall_per_class[i] if i < len(recall_per_class) else 0,
                'f1_score': f1_per_class[i] if i < len(f1_per_class) else 0,
                'support': int(support_per_class[i]) if i < len(support_per_class) else 0
            }
            
        return metrics
    
    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], save_path: str = None):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred, labels=list(range(9)))
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=[f"{i}: {self.config.CLASS_LABELS[i][:20]}..." for i in range(9)],
            yticklabels=[f"{i}: {self.config.CLASS_LABELS[i][:20]}..." for i in range(9)]
        )
        
        plt.title('Patent Classification Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        else:
            plt.show()
            
        plt.close()
    
    def plot_class_performance(self, metrics: Dict, save_path: str = None):
        """Plot per-class performance metrics."""
        classes = list(range(9))
        class_names = [self.config.CLASS_LABELS[i][:25] + "..." if len(self.config.CLASS_LABELS[i]) > 25 
                      else self.config.CLASS_LABELS[i] for i in classes]
        
        precision_scores = [metrics['per_class'][i]['precision'] for i in classes]
        recall_scores = [metrics['per_class'][i]['recall'] for i in classes]
        f1_scores = [metrics['per_class'][i]['f1_score'] for i in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        bars1 = ax.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recall_scores, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Patent Classes')
        ax.set_ylabel('Scores')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        autolabel(bars1)
        autolabel(bars2)
        autolabel(bars3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance plot saved to: {save_path}")
        else:
            plt.show()
            
        plt.close()
    
    def generate_report(self, results_path: str, output_dir: str = None):
        """Generate comprehensive evaluation report."""
        if output_dir is None:
            output_dir = self.config.OUTPUT_DIR
            
        # Load results
        results = self.load_results(results_path)
        y_true, y_pred = self.extract_predictions_and_labels(results)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # Create plots
        base_name = os.path.splitext(os.path.basename(results_path))[0]
        cm_path = os.path.join(output_dir, f"{base_name}_confusion_matrix.png")
        perf_path = os.path.join(output_dir, f"{base_name}_performance.png")
        
        self.plot_confusion_matrix(y_true, y_pred, cm_path)
        self.plot_class_performance(metrics, perf_path)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, f"{base_name}_metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "="*60)
        print("PATENT CLASSIFICATION EVALUATION REPORT")
        print("="*60)
        
        print(f"\nOverall Performance:")
        print(f"  Accuracy: {metrics['overall']['accuracy']:.3f}")
        print(f"  Weighted Precision: {metrics['overall']['weighted_precision']:.3f}")
        print(f"  Weighted Recall: {metrics['overall']['weighted_recall']:.3f}")
        print(f"  Weighted F1-Score: {metrics['overall']['weighted_f1']:.3f}")
        print(f"  Total Samples: {metrics['overall']['total_samples']}")
        
        print(f"\nPer-Class Performance:")
        print(f"{'Class':<5} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8} Class Name")
        print("-" * 80)
        
        for i in range(9):
            class_metrics = metrics['per_class'][i]
            class_name = class_metrics['class_name'][:35] + "..." if len(class_metrics['class_name']) > 35 else class_metrics['class_name']
            print(f"{i:<5} {class_metrics['precision']:<10.3f} {class_metrics['recall']:<10.3f} "
                  f"{class_metrics['f1_score']:<10.3f} {class_metrics['support']:<8} {class_name}")
        
        print(f"\nFiles generated:")
        print(f"  - Metrics: {metrics_path}")
        print(f"  - Confusion Matrix: {cm_path}")
        print(f"  - Performance Plot: {perf_path}")
        
        return metrics

if __name__ == "__main__":
    # Demo usage
    evaluator = PatentClassificationEvaluator()
    
    # You would typically run this after having results from pipeline.py
    # evaluator.generate_report("results/patent_classification_results_test_20240101_120000.json")