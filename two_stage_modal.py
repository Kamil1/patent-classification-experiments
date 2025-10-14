#!/usr/bin/env python3
"""
Modal-based Two-Stage Patent Classification

This runs both DeBERTa and Qwen on Modal for optimal performance:
1. DeBERTa for fast classification with confidence scores
2. Qwen reasoning for low-confidence cases
3. All running efficiently on Modal GPU infrastructure
"""

import modal
from typing import Dict, List, Optional, Any
import time
import re

app = modal.App("two-stage-patent-classification")

# Enhanced Modal image with both models' dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0",
        "datasets>=2.14.0",
        "numpy<2.0.0",
        "tqdm>=4.65.0",
        "huggingface_hub>=0.25.0",
        "hf_transfer>=0.1.0",
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.3.0",
        "sentencepiece>=0.1.99",  # For DeBERTa
        "tiktoken>=0.5.0",  # For Qwen
    ])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

CLASS_LABELS = {
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

@app.cls(
    image=image,
    gpu="A10G", 
    memory=32768,  # 32GB for both models
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class TwoStagePatentClassifier:
    """Modal class for two-stage patent classification."""
    
    @modal.enter()
    def load_models(self):
        """Load both DeBERTa and Qwen models on Modal startup."""
        import os
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            AutoModelForCausalLM, DebertaV2Tokenizer
        )
        from huggingface_hub import login
        
        # Authenticate with HuggingFace
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
            print("✅ Authenticated with HuggingFace")
        
        print("📥 Loading DeBERTa-v3-large for classification...")
        
        # Load DeBERTa model and tokenizer
        deberta_model_name = "KamilHugsFaces/patent-deberta-v3-large"
        self.deberta_tokenizer = DebertaV2Tokenizer.from_pretrained(deberta_model_name)
        self.deberta_model = AutoModelForSequenceClassification.from_pretrained(
            deberta_model_name,
            num_labels=9,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("✅ DeBERTa loaded successfully")
        
        print("📥 Loading Qwen2.5-Coder for reasoning...")
        
        # Load Qwen model and tokenizer  
        qwen_model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(
            qwen_model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.qwen_tokenizer.pad_token is None:
            self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token
            
        # Load with 4-bit quantization for memory efficiency
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        print("✅ Qwen loaded successfully")
        
        print("🚀 Two-stage classifier ready!")
    
    def _classify_with_deberta(self, patent_text: str) -> Dict[str, Any]:
        """Classify using DeBERTa and return prediction with confidence."""
        import torch
        import torch.nn.functional as F
        
        # Tokenize input
        inputs = self.deberta_tokenizer(
            patent_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.deberta_model.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.deberta_model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy().tolist(),
            'method': 'deberta'
        }
    
    def _classify_with_qwen_reasoning(self, patent_text: str, deberta_pred: Dict) -> Dict[str, Any]:
        """Use Qwen for detailed reasoning on low-confidence cases."""
        import torch
        
        reasoning_prompt = self._create_reasoning_prompt(
            patent_text, 
            deberta_pred['predicted_class'], 
            deberta_pred['confidence']
        )
        
        # Tokenize reasoning prompt
        inputs = self.qwen_tokenizer(
            reasoning_prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        ).to(self.qwen_model.device)
        
        # Generate reasoning response
        with torch.no_grad():
            outputs = self.qwen_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,  # Lower temperature for more focused reasoning
                do_sample=True,
                pad_token_id=self.qwen_tokenizer.eos_token_id,
                eos_token_id=self.qwen_tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.qwen_tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], 
            skip_special_tokens=True
        )
        
        # Parse classification from response
        parsed_class = self._parse_qwen_response(response)
        
        if parsed_class is not None:
            return {
                'predicted_class': parsed_class,
                'confidence': 0.85,  # High confidence in reasoning
                'method': 'qwen_reasoning',
                'reasoning_explanation': response,
                'original_deberta': deberta_pred
            }
        else:
            # Fallback to DeBERTa if parsing failed
            return {
                'predicted_class': deberta_pred['predicted_class'],
                'confidence': deberta_pred['confidence'] * 0.8,  # Reduced confidence
                'method': 'deberta_fallback',
                'reasoning_explanation': response,
                'original_deberta': deberta_pred
            }
    
    def _create_reasoning_prompt(self, patent_text: str, deberta_pred: int, deberta_conf: float) -> str:
        """Create detailed reasoning prompt for Qwen."""
        
        class_descriptions = {
            0: "Human Necessities (food, agriculture, medicine, healthcare)",
            1: "Performing Operations; Transporting (manufacturing, industrial processes)",
            2: "Chemistry; Metallurgy (chemical processes, materials science)",
            3: "Textiles; Paper (textile production, paper making)",
            4: "Fixed Constructions (buildings, construction)",
            5: "Mechanical Engineering; Lightning; Heating; Weapons; Blasting",
            6: "Physics (measurement, optics, scientific instruments)",
            7: "Electricity (electrical circuits, electronics)",
            8: "General tagging (interdisciplinary, emerging technology)"
        }
        
        prompt = f"""You are a patent classification expert. Analyze this patent abstract and determine the correct classification category.

PATENT ABSTRACT:
{patent_text[:600]}

CURRENT AI PREDICTION:
- Class {deberta_pred}: {class_descriptions.get(deberta_pred, 'Unknown')}
- Confidence: {deberta_conf:.3f} (UNCERTAIN - needs expert analysis)

CLASSIFICATION CATEGORIES:
0: Human Necessities - food, agriculture, medicine, healthcare, personal care
1: Performing Operations; Transporting - manufacturing, industrial processes
2: Chemistry; Metallurgy - chemical processes, materials science, polymers
3: Textiles; Paper - textile production, paper making, fiber processing
4: Fixed Constructions - buildings, roads, bridges, construction
5: Mechanical Engineering - mechanical systems, engines, machinery
6: Physics - measurement, optics, nuclear physics, scientific instruments
7: Electricity - electrical circuits, power generation, electronics
8: General tagging - AI, nanotechnology, interdisciplinary technology

Analyze the patent step by step:
1. What is the main technical domain?
2. What problem does it solve?
3. What is the primary application?
4. Which category fits best?

Provide your reasoning and conclude with: FINAL_CLASSIFICATION: [0-8]"""
        
        return prompt
    
    def _parse_qwen_response(self, response: str) -> Optional[int]:
        """Parse Qwen's response to extract classification."""
        
        patterns = [
            r"FINAL_CLASSIFICATION:\s*(\d+)",
            r"CLASSIFICATION:\s*(\d+)",
            r"CLASS:\s*(\d+)",
            r"CATEGORY:\s*(\d+)",
            r"Answer:\s*(\d+)",
            r"Result:\s*(\d+)"
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
    
    @modal.method()
    def classify_two_stage(self, max_samples: int = 50, confidence_threshold: float = 0.75) -> List[Dict[str, Any]]:
        """Run two-stage classification on patent dataset."""
        
        print(f"🚀 Loading patent dataset (max_samples: {max_samples})")
        
        # Load dataset within Modal to avoid serialization issues
        from datasets import load_dataset
        dataset = load_dataset("ccdv/patent-classification", "abstract")
        test_data = dataset["test"]
        
        if max_samples and max_samples < len(test_data):
            test_data = test_data.select(range(max_samples))
        
        patent_texts = test_data["text"]
        true_labels = test_data["label"]
        
        print(f"✅ Loaded {len(patent_texts)} patent samples")
        print(f"   Confidence threshold: {confidence_threshold}")
        
        results = []
        high_confidence_count = 0
        reasoning_count = 0
        correct_predictions = 0
        
        for i, (text, true_label) in enumerate(zip(patent_texts, true_labels)):
            print(f"📄 Processing patent {i+1}/{len(patent_texts)}")
            
            # Stage 1: DeBERTa classification
            deberta_result = self._classify_with_deberta(text)
            
            if deberta_result['confidence'] >= confidence_threshold:
                # High confidence - use DeBERTa result
                high_confidence_count += 1
                predicted_class = deberta_result['predicted_class']
                result = {
                    **deberta_result,
                    'patent_text': text[:200] + "...",
                    'stage': 'deberta_only',
                    'true_label': true_label,
                    'is_correct': predicted_class == true_label
                }
                print(f"   ✅ High confidence: {deberta_result['confidence']:.3f}")
                
            else:
                # Low confidence - use reasoning
                reasoning_count += 1
                print(f"   🤔 Low confidence ({deberta_result['confidence']:.3f}) - applying reasoning...")
                
                reasoning_result = self._classify_with_qwen_reasoning(text, deberta_result)
                predicted_class = reasoning_result['predicted_class']
                result = {
                    **reasoning_result,
                    'patent_text': text[:200] + "...",
                    'stage': 'two_stage',
                    'true_label': true_label,
                    'is_correct': predicted_class == true_label
                }
                print(f"   🧠 Reasoning result: Class {reasoning_result['predicted_class']}")
            
            if result['is_correct']:
                correct_predictions += 1
                
            results.append(result)
        
        accuracy = correct_predictions / len(results)
        print(f"\n📊 Two-stage processing complete:")
        print(f"   Total accuracy: {accuracy:.3f} ({correct_predictions}/{len(results)})")
        print(f"   High confidence (DeBERTa only): {high_confidence_count}")
        print(f"   Reasoning applied: {reasoning_count}")
        print(f"   Reasoning rate: {reasoning_count/len(patent_texts):.1%}")
        
        return results

@app.local_entrypoint()
def main(max_samples: int = 50, confidence_threshold: float = 0.75):
    """Run two-stage classification on Modal using the patent dataset."""
    
    print("🚀 MODAL TWO-STAGE PATENT CLASSIFICATION")
    print("="*60)
    
    # Initialize classifier
    classifier = TwoStagePatentClassifier()
    
    # Run classification (dataset loading happens inside Modal)
    results = classifier.classify_two_stage.remote(max_samples, confidence_threshold)
    
    print(f"\n🎯 CLASSIFICATION RESULTS:")
    print(f"{'='*60}")
    
    # Calculate breakdown by method
    deberta_only_results = [r for r in results if r['stage'] == 'deberta_only']
    reasoning_results = [r for r in results if r['stage'] == 'two_stage']
    
    deberta_only_correct = sum(1 for r in deberta_only_results if r['is_correct'])
    reasoning_correct = sum(1 for r in reasoning_results if r['is_correct'])
    total_correct = sum(1 for r in results if r['is_correct'])
    
    # Show first 5 examples
    for i, result in enumerate(results[:5]):
        status = "✅" if result['is_correct'] else "❌"
        method_icon = "🤖" if result['stage'] == 'deberta_only' else "🧠"
        
        print(f"\n{status} {method_icon} Patent {i+1}:")
        print(f"   Text: {result['patent_text']}")
        print(f"   Predicted: {result['predicted_class']} ({CLASS_LABELS[result['predicted_class']]})")
        print(f"   True: {result['true_label']} ({CLASS_LABELS[result['true_label']]})")
        print(f"   Confidence: {result['confidence']:.3f}")
        
        if 'reasoning_explanation' in result:
            print(f"   Reasoning: {result['reasoning_explanation'][:100]}...")
    
    # Calculate accuracies
    overall_accuracy = total_correct / len(results)
    deberta_accuracy = deberta_only_correct / len(deberta_only_results) if deberta_only_results else 0
    reasoning_accuracy = reasoning_correct / len(reasoning_results) if reasoning_results else 0
    
    print(f"\n📊 ACCURACY ANALYSIS:")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {overall_accuracy:.3f} ({total_correct}/{len(results)})")
    print(f"DeBERTa Only: {deberta_accuracy:.3f} ({deberta_only_correct}/{len(deberta_only_results)})")
    print(f"With Reasoning: {reasoning_accuracy:.3f} ({reasoning_correct}/{len(reasoning_results)})")
    print(f"")
    print(f"Method Distribution:")
    print(f"• High confidence (DeBERTa only): {len(deberta_only_results)}")
    print(f"• Low confidence (+ Reasoning): {len(reasoning_results)}")
    print(f"• Reasoning utilization: {len(reasoning_results)/len(results):.1%}")
    
    # Compare with expected DeBERTa baseline
    expected_deberta_accuracy = 0.675  # From our previous results
    improvement = overall_accuracy - expected_deberta_accuracy
    
    print(f"\n🚀 IMPROVEMENT ANALYSIS:")
    print(f"• Expected DeBERTa baseline: {expected_deberta_accuracy:.3f}")
    print(f"• Two-stage system: {overall_accuracy:.3f}")
    print(f"• Improvement: {improvement:+.3f} ({improvement*100:+.1f}%)")
    
    return results

if __name__ == "__main__":
    import sys
    
    # Default parameters
    confidence_threshold = 0.75
    max_samples = 10  # Start with smaller number for testing
    
    if len(sys.argv) > 1:
        try:
            max_samples = int(sys.argv[1])
            if len(sys.argv) > 2:
                confidence_threshold = float(sys.argv[2])
        except ValueError:
            print("Usage: python two_stage_modal.py [max_samples] [confidence_threshold]")
            sys.exit(1)
    
    print(f"Running with max_samples={max_samples}, confidence_threshold={confidence_threshold}")
    
    # Use modal run with the local_entrypoint
    # The parameters will be passed to the main function directly
    import modal
    modal.run.main