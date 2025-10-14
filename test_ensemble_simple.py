#!/usr/bin/env python3
"""
Simple Ensemble Test with Confidence-Based Reasoning Fallback

This script demonstrates the ensemble concept using existing infrastructure.
"""

import subprocess
import json
import sys
from typing import Dict, List

def run_classification(model: str, model_type: str, samples: int = 10) -> Dict:
    """Run classification and return results."""
    
    cmd = [
        "python", "main.py",
        "--mode", "classify",
        "--model", model,
        "--model_type", model_type,
        "--max_samples", str(samples)
    ]
    
    print(f"🔍 Running: {model}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error running {model}: {result.stderr}")
        return None
    
    # Find and load the results file
    lines = result.stdout.split('\n')
    results_line = None
    for line in lines:
        if "Results saved to:" in line:
            results_line = line
            break
    
    if not results_line:
        print(f"⚠️ Could not find results file for {model}")
        return None
    
    # Extract file path
    results_file = results_line.split("Results saved to: ")[1].split("patent_classification_results")[1]
    results_path = f"./results/patent_classification_results{results_file}"
    
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading results for {model}: {e}")
        return None

def analyze_confidence_ensemble(deberta_results: Dict, bert_results: Dict, confidence_threshold: float = 0.75):
    """Analyze ensemble performance with confidence-based fallback."""
    
    if not deberta_results or not bert_results:
        print("❌ Missing results for ensemble analysis")
        return
    
    deberta_predictions = deberta_results['detailed_results']
    bert_predictions = bert_results['detailed_results'] 
    
    print(f"\n🎯 ENSEMBLE ANALYSIS")
    print(f"{'='*60}")
    print(f"Confidence Threshold: {confidence_threshold}")
    print(f"Samples: {len(deberta_predictions)}")
    
    # Analyze predictions
    high_confidence_correct = 0
    low_confidence_total = 0
    low_confidence_correct = 0
    ensemble_correct = 0
    
    confidence_buckets = {'high': [], 'medium': [], 'low': []}
    
    for i, (deberta_pred, bert_pred) in enumerate(zip(deberta_predictions, bert_predictions)):
        
        deberta_conf = deberta_pred.get('confidence', 0.5)
        bert_conf = bert_pred.get('confidence', 0.5)
        deberta_class = deberta_pred.get('predicted_label')
        bert_class = bert_pred.get('predicted_label')
        true_class = deberta_pred.get('true_label')
        
        # Ensemble logic: weighted by confidence
        total_conf = deberta_conf + bert_conf
        if total_conf > 0:
            if deberta_class == bert_class:
                # Both models agree
                ensemble_class = deberta_class
                ensemble_conf = (deberta_conf + bert_conf) / 2
            else:
                # Models disagree - use higher confidence
                if deberta_conf > bert_conf:
                    ensemble_class = deberta_class
                    ensemble_conf = deberta_conf
                else:
                    ensemble_class = bert_class
                    ensemble_conf = bert_conf
        else:
            ensemble_class = deberta_class
            ensemble_conf = 0.5
        
        # Track accuracy
        is_correct = (ensemble_class == true_class)
        if is_correct:
            ensemble_correct += 1
            
        # Confidence analysis
        if ensemble_conf >= confidence_threshold:
            confidence_buckets['high'].append({
                'confidence': ensemble_conf,
                'correct': is_correct,
                'predicted': ensemble_class,
                'true': true_class,
                'agreement': deberta_class == bert_class
            })
            if is_correct:
                high_confidence_correct += 1
        elif ensemble_conf >= 0.5:
            confidence_buckets['medium'].append({
                'confidence': ensemble_conf,
                'correct': is_correct,
                'predicted': ensemble_class,
                'true': true_class,
                'agreement': deberta_class == bert_class
            })
        else:
            confidence_buckets['low'].append({
                'confidence': ensemble_conf,
                'correct': is_correct,
                'predicted': ensemble_class,
                'true': true_class,
                'agreement': deberta_class == bert_class
            })
            low_confidence_total += 1
            if is_correct:
                low_confidence_correct += 1
    
    # Results summary
    total_samples = len(deberta_predictions)
    ensemble_accuracy = ensemble_correct / total_samples
    high_conf_accuracy = high_confidence_correct / len(confidence_buckets['high']) if confidence_buckets['high'] else 0
    low_conf_accuracy = low_confidence_correct / low_confidence_total if low_confidence_total > 0 else 0
    
    print(f"\n📊 ENSEMBLE RESULTS:")
    print(f"   Overall Ensemble Accuracy: {ensemble_accuracy:.3f} ({ensemble_correct}/{total_samples})")
    print(f"   High Confidence Accuracy: {high_conf_accuracy:.3f} ({len(confidence_buckets['high'])} samples)")
    print(f"   Low Confidence Accuracy: {low_conf_accuracy:.3f} ({low_confidence_total} samples)")
    
    print(f"\n🔍 CONFIDENCE DISTRIBUTION:")
    print(f"   High (≥{confidence_threshold}): {len(confidence_buckets['high'])} samples")
    print(f"   Medium (0.5-{confidence_threshold}): {len(confidence_buckets['medium'])} samples") 
    print(f"   Low (<0.5): {len(confidence_buckets['low'])} samples")
    
    # Model agreement analysis
    agreements = sum(1 for bucket in confidence_buckets.values() for item in bucket if item['agreement'])
    agreement_rate = agreements / total_samples
    print(f"\n🤝 MODEL AGREEMENT: {agreement_rate:.3f} ({agreements}/{total_samples})")
    
    # Identify cases needing reasoning model
    reasoning_candidates = confidence_buckets['low'] + [
        item for item in confidence_buckets['medium'] 
        if not item['agreement'] or item['confidence'] < 0.6
    ]
    
    print(f"\n🧠 REASONING MODEL CANDIDATES:")
    print(f"   {len(reasoning_candidates)} samples would benefit from reasoning model")
    print(f"   Current accuracy on these samples: {sum(c['correct'] for c in reasoning_candidates) / len(reasoning_candidates):.3f}")
    
    return {
        'ensemble_accuracy': ensemble_accuracy,
        'high_confidence_accuracy': high_conf_accuracy,
        'low_confidence_accuracy': low_conf_accuracy,
        'reasoning_candidates': len(reasoning_candidates),
        'agreement_rate': agreement_rate
    }

def main():
    """Run ensemble analysis."""
    
    print("🚀 SMART ENSEMBLE WITH CONFIDENCE ANALYSIS")
    print("="*80)
    
    # Test with small sample first
    sample_size = 50
    
    # Run DeBERTa
    deberta_results = run_classification(
        "KamilHugsFaces/patent-deberta-v3-large", 
        "classification", 
        sample_size
    )
    
    # Run BERT  
    bert_results = run_classification(
        "KamilHugsFaces/patent-bert-classifier",
        "classification", 
        sample_size
    )
    
    # Analyze ensemble
    if deberta_results and bert_results:
        analysis = analyze_confidence_ensemble(deberta_results, bert_results)
        
        print(f"\n✅ ENSEMBLE ANALYSIS COMPLETE!")
        print(f"\nKey Insights:")
        print(f"• Ensemble accuracy: {analysis['ensemble_accuracy']:.1%}")
        print(f"• High-confidence samples are {analysis['high_confidence_accuracy']:.1%} accurate")
        print(f"• {analysis['reasoning_candidates']} samples need reasoning model")
        print(f"• Models agree {analysis['agreement_rate']:.1%} of the time")
        
        # Estimate potential improvement
        potential_improvement = analysis['reasoning_candidates'] * 0.15  # Assume 15% boost from reasoning
        estimated_new_accuracy = analysis['ensemble_accuracy'] + (potential_improvement / sample_size)
        
        print(f"\n🎯 PROJECTED IMPROVEMENT WITH REASONING:")
        print(f"• Current: {analysis['ensemble_accuracy']:.1%}")
        print(f"• With reasoning model: {estimated_new_accuracy:.1%}")
        print(f"• Potential gain: +{(estimated_new_accuracy - analysis['ensemble_accuracy']):.1%}")
    
if __name__ == "__main__":
    main()