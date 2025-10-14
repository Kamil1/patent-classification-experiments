#!/usr/bin/env python3
"""
Analyze existing classification results to demonstrate ensemble approach.
"""

import json
import os
from typing import Dict, List

def load_results(file_pattern: str) -> Dict:
    """Load the most recent results file matching pattern."""
    results_dir = "./results"
    files = [f for f in os.listdir(results_dir) if file_pattern in f and f.endswith('.json')]
    
    if not files:
        return None
        
    # Get most recent file
    latest_file = max(files, key=lambda f: os.path.getctime(os.path.join(results_dir, f)))
    
    with open(os.path.join(results_dir, latest_file), 'r') as f:
        return json.load(f)

def analyze_confidence_patterns(results: Dict, model_name: str):
    """Analyze confidence patterns in results."""
    
    if not results or 'detailed_results' not in results:
        print(f"❌ No detailed results found for {model_name}")
        return None
    
    predictions = results['detailed_results']
    
    # Confidence buckets
    high_conf = []  # >= 0.8
    med_conf = []   # 0.6-0.8  
    low_conf = []   # < 0.6
    
    total_correct = 0
    
    for pred in predictions:
        conf = pred.get('confidence', 0.5)
        correct = pred.get('is_correct', False)
        
        if correct:
            total_correct += 1
            
        if conf >= 0.8:
            high_conf.append({'confidence': conf, 'correct': correct, 'class': pred.get('predicted_label')})
        elif conf >= 0.6:
            med_conf.append({'confidence': conf, 'correct': correct, 'class': pred.get('predicted_label')})
        else:
            low_conf.append({'confidence': conf, 'correct': correct, 'class': pred.get('predicted_label')})
    
    total_samples = len(predictions)
    overall_accuracy = total_correct / total_samples
    
    # Calculate accuracy by confidence bucket
    high_acc = sum(p['correct'] for p in high_conf) / len(high_conf) if high_conf else 0
    med_acc = sum(p['correct'] for p in med_conf) / len(med_conf) if med_conf else 0
    low_acc = sum(p['correct'] for p in low_conf) / len(low_conf) if low_conf else 0
    
    return {
        'model': model_name,
        'total_samples': total_samples,
        'overall_accuracy': overall_accuracy,
        'high_confidence': {'count': len(high_conf), 'accuracy': high_acc},
        'medium_confidence': {'count': len(med_conf), 'accuracy': med_acc},
        'low_confidence': {'count': len(low_conf), 'accuracy': low_acc},
        'predictions': predictions
    }

def simulate_ensemble(deberta_analysis: Dict, bert_analysis: Dict):
    """Simulate ensemble voting between DeBERTa and BERT."""
    
    if not deberta_analysis or not bert_analysis:
        print("❌ Missing analysis data for ensemble simulation")
        return
    
    # Assume same samples (they should be from same test set)
    deberta_preds = deberta_analysis['predictions']
    bert_preds = bert_analysis['predictions']
    
    min_samples = min(len(deberta_preds), len(bert_preds))
    
    print(f"\n🎯 ENSEMBLE SIMULATION")
    print(f"{'='*50}")
    print(f"Comparing {min_samples} predictions")
    
    ensemble_correct = 0
    high_conf_cases = 0
    low_conf_cases = 0
    reasoning_candidates = []
    
    for i in range(min_samples):
        deberta = deberta_preds[i]
        bert = bert_preds[i]
        
        # Get confidence scores
        deberta_conf = deberta.get('confidence', 0.5)
        bert_conf = bert.get('confidence', 0.5)
        deberta_class = deberta.get('predicted_label')
        bert_class = bert.get('predicted_label')
        true_class = deberta.get('true_label')
        
        # Ensemble decision logic
        if deberta_class == bert_class:
            # Models agree - use average confidence
            ensemble_class = deberta_class
            ensemble_conf = (deberta_conf + bert_conf) / 2
        else:
            # Models disagree - use higher confidence prediction
            if deberta_conf > bert_conf:
                ensemble_class = deberta_class
                ensemble_conf = deberta_conf
            else:
                ensemble_class = bert_class
                ensemble_conf = bert_conf
        
        # Check correctness
        is_correct = (ensemble_class == true_class)
        if is_correct:
            ensemble_correct += 1
        
        # Categorize by confidence
        if ensemble_conf >= 0.75:
            high_conf_cases += 1
        elif ensemble_conf < 0.6 or deberta_class != bert_class:
            low_conf_cases += 1
            reasoning_candidates.append({
                'sample_id': i,
                'ensemble_confidence': ensemble_conf,
                'deberta_pred': deberta_class,
                'bert_pred': bert_class,
                'true_class': true_class,
                'models_agree': deberta_class == bert_class,
                'text_preview': deberta.get('text_preview', '')[:100]
            })
    
    ensemble_accuracy = ensemble_correct / min_samples
    
    print(f"\n📊 RESULTS:")
    print(f"   Individual Models:")
    print(f"   • DeBERTa: {deberta_analysis['overall_accuracy']:.3f}")
    print(f"   • BERT: {bert_analysis['overall_accuracy']:.3f}")
    print(f"   Ensemble: {ensemble_accuracy:.3f}")
    
    improvement = ensemble_accuracy - max(deberta_analysis['overall_accuracy'], bert_analysis['overall_accuracy'])
    print(f"   Improvement: +{improvement:.3f} ({improvement*100:.1f}%)")
    
    print(f"\n🎯 CONFIDENCE ANALYSIS:")
    print(f"   High Confidence (≥0.75): {high_conf_cases} cases")
    print(f"   Low Confidence/Disagreement: {low_conf_cases} cases")
    print(f"   Reasoning Model Candidates: {len(reasoning_candidates)}")
    
    print(f"\n🧠 TOP REASONING CANDIDATES:")
    for i, candidate in enumerate(reasoning_candidates[:5]):
        agree_text = "✅ Agree" if candidate['models_agree'] else "❌ Disagree" 
        print(f"   {i+1}. Conf: {candidate['ensemble_confidence']:.3f} | {agree_text}")
        print(f"      DeBERTa: {candidate['deberta_pred']} | BERT: {candidate['bert_pred']} | True: {candidate['true_class']}")
        print(f"      Text: {candidate['text_preview']}...")
        print()
    
    # Estimate reasoning model impact
    current_wrong_in_candidates = len([c for c in reasoning_candidates if c['deberta_pred'] != c['true_class'] and c['bert_pred'] != c['true_class']])
    potential_fixes = min(len(reasoning_candidates), current_wrong_in_candidates * 0.7)  # 70% improvement assumption
    
    estimated_new_accuracy = ensemble_accuracy + (potential_fixes / min_samples)
    
    print(f"🚀 PROJECTED WITH REASONING MODEL:")
    print(f"   Current Ensemble: {ensemble_accuracy:.3f}")
    print(f"   With Reasoning: {estimated_new_accuracy:.3f}")
    print(f"   Total Potential Gain: +{estimated_new_accuracy - ensemble_accuracy:.3f}")

def main():
    """Analyze existing results for ensemble patterns."""
    
    print("🔍 ANALYZING EXISTING RESULTS FOR ENSEMBLE PATTERNS")
    print("="*60)
    
    # Load most recent results for each model
    deberta_results = load_results("patent_classification_results_test_20251012_105849")
    bert_results = load_results("patent_classification_results_test_20251011_131106")  
    
    if not deberta_results:
        print("❌ Could not find DeBERTa results")
        return
        
    if not bert_results:
        print("❌ Could not find BERT results") 
        return
    
    print("✅ Found results files")
    print(f"DeBERTa samples: {deberta_results['summary']['total_samples']}")
    print(f"BERT samples: {bert_results['summary']['total_samples']}")
    
    # Analyze confidence patterns
    deberta_analysis = analyze_confidence_patterns(deberta_results, "DeBERTa-v3-large")
    bert_analysis = analyze_confidence_patterns(bert_results, "BERT-classifier")
    
    if deberta_analysis and bert_analysis:
        print(f"\n📊 INDIVIDUAL MODEL ANALYSIS:")
        print(f"DeBERTa-v3-large:")
        print(f"  Overall: {deberta_analysis['overall_accuracy']:.3f}")
        print(f"  High conf (≥0.8): {deberta_analysis['high_confidence']['count']} samples, {deberta_analysis['high_confidence']['accuracy']:.3f} acc")
        print(f"  Low conf (<0.6): {deberta_analysis['low_confidence']['count']} samples, {deberta_analysis['low_confidence']['accuracy']:.3f} acc")
        
        print(f"\nBERT-classifier:")
        print(f"  Overall: {bert_analysis['overall_accuracy']:.3f}")
        print(f"  High conf (≥0.8): {bert_analysis['high_confidence']['count']} samples, {bert_analysis['high_confidence']['accuracy']:.3f} acc")
        print(f"  Low conf (<0.6): {bert_analysis['low_confidence']['count']} samples, {bert_analysis['low_confidence']['accuracy']:.3f} acc")
        
        # Simulate ensemble
        simulate_ensemble(deberta_analysis, bert_analysis)
    
    print(f"\n✅ ANALYSIS COMPLETE!")

if __name__ == "__main__":
    main()