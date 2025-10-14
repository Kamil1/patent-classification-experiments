#!/usr/bin/env python3
"""
Ensemble Classification with Reasoning Model Fallback

This script implements a smart ensemble approach:
1. Run multiple BERT/DeBERTa models for fast classification
2. For low-confidence predictions, use a reasoning model for detailed analysis
3. Combine results using confidence-weighted voting
"""

import asyncio
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from modal_client_flexible import FlexibleModalPatentClassifier
import time

@dataclass
class EnsemblePrediction:
    """Result from ensemble classification."""
    predicted_class: int
    confidence: float
    method: str  # 'ensemble', 'reasoning_fallback'
    component_predictions: List[Dict]
    reasoning_explanation: Optional[str] = None

class ConfidenceEnsemble:
    """Ensemble classifier with reasoning model fallback for low confidence cases."""
    
    def __init__(self, confidence_threshold: float = 0.75):
        self.confidence_threshold = confidence_threshold
        
        # Initialize model clients
        self.models = {
            'deberta': FlexibleModalPatentClassifier(model_type="classification"),
            'bert': FlexibleModalPatentClassifier(model_type="classification"),
            'reasoning': FlexibleModalPatentClassifier(model_type="generative")
        }
        
        self.class_labels = {
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
    
    async def classify_single(self, patent_text: str) -> EnsemblePrediction:
        """Classify a single patent with ensemble + reasoning fallback."""
        
        # Step 1: Get predictions from fast classification models
        print(f"🔍 Getting ensemble predictions...")
        
        deberta_pred = await self.models['deberta'].classify_single(patent_text)
        bert_pred = await self.models['bert'].classify_single(patent_text)
        
        component_predictions = [deberta_pred, bert_pred]
        
        # Step 2: Calculate ensemble prediction using confidence weighting
        ensemble_result = self._weighted_ensemble_vote(component_predictions)
        
        print(f"📊 Ensemble result: Class {ensemble_result['predicted_class']} (confidence: {ensemble_result['confidence']:.3f})")
        
        # Step 3: Check if confidence is high enough
        if ensemble_result['confidence'] >= self.confidence_threshold:
            print("✅ High confidence - using ensemble prediction")
            return EnsemblePrediction(
                predicted_class=ensemble_result['predicted_class'],
                confidence=ensemble_result['confidence'],
                method='ensemble',
                component_predictions=component_predictions
            )
        
        # Step 4: Low confidence - use reasoning model
        print(f"🤔 Low confidence ({ensemble_result['confidence']:.3f}) - consulting reasoning model...")
        reasoning_result = await self._reasoning_classification(
            patent_text, 
            ensemble_result,
            component_predictions
        )
        
        return reasoning_result
    
    def _weighted_ensemble_vote(self, predictions: List[Dict]) -> Dict:
        """Combine predictions using confidence-weighted voting."""
        
        # Calculate weighted average of class probabilities
        total_weight = sum(pred.get('confidence', 0.5) for pred in predictions)
        
        if total_weight == 0:
            # Fallback to simple averaging
            total_weight = len(predictions)
            weights = [1.0] * len(predictions)
        else:
            weights = [pred.get('confidence', 0.5) for pred in predictions]
        
        # Weighted vote
        class_votes = {}
        confidence_sum = 0
        
        for pred, weight in zip(predictions, weights):
            predicted_class = pred.get('predicted_class')
            if predicted_class is not None:
                if predicted_class not in class_votes:
                    class_votes[predicted_class] = 0
                class_votes[predicted_class] += weight
                confidence_sum += weight * pred.get('confidence', 0.5)
        
        if not class_votes:
            # Fallback
            return {'predicted_class': 0, 'confidence': 0.1}
        
        # Get class with highest weighted vote
        winning_class = max(class_votes.keys(), key=lambda k: class_votes[k])
        winning_confidence = confidence_sum / total_weight if total_weight > 0 else 0.5
        
        return {
            'predicted_class': winning_class,
            'confidence': winning_confidence,
            'vote_distribution': class_votes
        }
    
    async def _reasoning_classification(
        self, 
        patent_text: str, 
        ensemble_result: Dict,
        component_predictions: List[Dict]
    ) -> EnsemblePrediction:
        """Use reasoning model for detailed analysis of uncertain cases."""
        
        # Create detailed reasoning prompt
        reasoning_prompt = self._create_reasoning_prompt(
            patent_text, 
            ensemble_result, 
            component_predictions
        )
        
        # Get reasoning model prediction
        reasoning_pred = await self.models['reasoning'].classify_single(reasoning_prompt)
        
        # Parse reasoning model response
        reasoning_class = self._parse_reasoning_response(reasoning_pred['raw_response'])
        
        if reasoning_class is not None:
            print(f"🧠 Reasoning model suggests: Class {reasoning_class}")
            return EnsemblePrediction(
                predicted_class=reasoning_class,
                confidence=0.85,  # High confidence in reasoning
                method='reasoning_fallback',
                component_predictions=component_predictions,
                reasoning_explanation=reasoning_pred['raw_response']
            )
        else:
            # Fallback to ensemble if reasoning parsing failed
            print("⚠️ Could not parse reasoning response, using ensemble")
            return EnsemblePrediction(
                predicted_class=ensemble_result['predicted_class'],
                confidence=ensemble_result['confidence'] * 0.8,  # Reduce confidence
                method='ensemble',
                component_predictions=component_predictions
            )
    
    def _create_reasoning_prompt(self, patent_text: str, ensemble_result: Dict, component_predictions: List[Dict]) -> str:
        """Create a detailed reasoning prompt for the reasoning model."""
        
        # Get component prediction details
        pred_details = []
        for i, pred in enumerate(component_predictions):
            model_name = ['DeBERTa', 'BERT'][i]
            pred_class = pred.get('predicted_class', 'Unknown')
            pred_conf = pred.get('confidence', 0.0)
            pred_details.append(f"{model_name}: Class {pred_class} ({pred_conf:.3f})")
        
        prompt = f"""You are a patent classification expert. Two AI models have made predictions but with low confidence. Please analyze this patent abstract step by step and determine the correct classification.

Patent Abstract:
{patent_text[:1000]}...

Current AI Predictions:
{chr(10).join(pred_details)}
Ensemble: Class {ensemble_result['predicted_class']} (confidence: {ensemble_result['confidence']:.3f})

Patent Classification Categories:
0: Human Necessities (food, agriculture, medicine, etc.)
1: Performing Operations; Transporting (manufacturing processes, transport)
2: Chemistry; Metallurgy (chemical processes, materials)
3: Textiles; Paper (textile production, paper making)
4: Fixed Constructions (buildings, roads, bridges)
5: Mechanical Engineering; Lightning; Heating; Weapons; Blasting
6: Physics (measurement, optics, nuclear, etc.)
7: Electricity (electrical circuits, power generation, etc.)
8: General tagging of new or cross-sectional technology

Please analyze this step by step:
1. Identify the main technical domain
2. Look for key technical terms and concepts
3. Consider the primary application or purpose
4. Determine which category best fits

Respond with: "CLASSIFICATION: [number]" followed by your reasoning."""
        
        return prompt
    
    def _parse_reasoning_response(self, response: str) -> Optional[int]:
        """Parse the reasoning model's response to extract the classification."""
        
        # Look for "CLASSIFICATION: X" pattern
        import re
        
        # Try different patterns
        patterns = [
            r"CLASSIFICATION:\s*(\d+)",
            r"CLASS:\s*(\d+)", 
            r"CATEGORY:\s*(\d+)",
            r"Answer:\s*(\d+)",
            r"Result:\s*(\d+)",
            r"\b([0-8])\b"  # Any digit 0-8
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    class_num = int(match.group(1))
                    if 0 <= class_num <= 8:
                        return class_num
                except ValueError:
                    continue
        
        return None

async def main():
    """Test the ensemble reasoning approach."""
    
    # Initialize ensemble
    ensemble = ConfidenceEnsemble(confidence_threshold=0.75)
    
    # Test with some sample patents
    test_patents = [
        "A method for producing solar cells using silicon wafers with improved efficiency through surface texturing and anti-reflective coatings.",
        "A pharmaceutical composition comprising a novel antibody for treating autoimmune diseases with reduced side effects.",
        "A distributed computing system for managing blockchain transactions with enhanced security protocols."
    ]
    
    results = []
    
    for i, patent in enumerate(test_patents):
        print(f"\n{'='*80}")
        print(f"TEST PATENT {i+1}")
        print(f"{'='*80}")
        print(f"Text: {patent}")
        
        result = await ensemble.classify_single(patent)
        
        print(f"\n🎯 FINAL RESULT:")
        print(f"   Class: {result.predicted_class} ({ensemble.class_labels[result.predicted_class]})")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Method: {result.method}")
        
        if result.reasoning_explanation:
            print(f"   Reasoning: {result.reasoning_explanation[:200]}...")
        
        results.append(result)
    
    # Save results
    with open('./results/ensemble_reasoning_test.json', 'w') as f:
        json.dump({
            'test_results': [
                {
                    'patent_text': patent,
                    'predicted_class': result.predicted_class,
                    'confidence': result.confidence,
                    'method': result.method,
                    'reasoning_explanation': result.reasoning_explanation
                }
                for patent, result in zip(test_patents, results)
            ]
        }, f, indent=2)
    
    print(f"\n✅ Ensemble reasoning test complete! Results saved.")

if __name__ == "__main__":
    asyncio.run(main())