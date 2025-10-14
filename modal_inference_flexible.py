"""Flexible Modal inference for different model types."""

import modal
from typing import List, Dict, Any, Optional
import json
import time
from datetime import datetime

# Create Modal app
app = modal.App("patent-classification-flexible")

# Define the Modal image with all required dependencies
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
        "sentence-transformers>=2.2.0",  # For embedding models
        "scikit-learn>=1.3.0",  # For classification tasks
        "sentencepiece>=0.1.99",  # Required for DeBERTa tokenizer
        "tiktoken>=0.5.0",  # Required for DeBERTa-v3-large
    ])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Class labels for patent classification
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
    memory=16384,
    timeout=3600,
    min_containers=0,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.concurrent(max_inputs=4)
class FlexiblePatentClassifier:
    """Modal class for flexible patent classification supporting different model types."""
    
    model_name: str = modal.parameter(default="meta-llama/Llama-3.1-8B-Instruct")
    max_length: int = modal.parameter(default=2048)
    model_type: str = modal.parameter(default="generative")  # "generative" or "classification"
    
    # Float parameters as class attributes
    temperature: float = 0.1
    top_p: float = 0.9
        
    @modal.enter()
    def load_model(self):
        """Load the appropriate model based on model_type."""
        import torch
        import os
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
            AutoConfig, BitsAndBytesConfig
        )
        from huggingface_hub import login
        
        print(f"🔧 Modal Container Starting - Loading {self.model_name}...")
        print(f"📋 Model Configuration:")
        print(f"   Model Type: {self.model_type}")
        print(f"   Max Length: {self.max_length}")
        print(f"   Temperature: {self.temperature}")
        print(f"   Top-p: {self.top_p}")
        
        # Authenticate with HuggingFace
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
            print("✅ Authenticated with HuggingFace")
        
        print(f"📥 Loading model: {self.model_name}")
        
        # Load tokenizer (common for both types)
        # Use specific tokenizer for DeBERTa models to avoid conversion issues
        if "deberta" in self.model_name.lower():
            from transformers import DebertaV2Tokenizer
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        if self.model_type == "generative":
            # Load causal LM for generative models (Llama, GPT, etc.)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            print("✅ Generative model loaded successfully!")
            
        elif self.model_type == "classification":
            # Try direct classification first for fine-tuned models
            try:
                print("🔍 Attempting to load as classification model...")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    num_labels=9  # Patent classification has 9 classes
                )
                print("✅ Classification model loaded successfully!")
            except Exception as e:
                print(f"⚠️ Failed to load as classification model: {e}")
                try:
                    from sentence_transformers import SentenceTransformer
                    print("🔍 Attempting to load as SentenceTransformer...")
                    self.model = SentenceTransformer(self.model_name)
                    self.model_type = "sentence_transformer"
                    print("✅ SentenceTransformer model loaded successfully!")
                except Exception as e2:
                    print(f"⚠️ Failed to load as SentenceTransformer: {e2}")
                    print("🔄 Falling back to generative approach...")
                    self.model_type = "generative"
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        dtype=torch.float16,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    print("✅ Fallback to generative model successful!")
        
        print("✅ Model loaded successfully on GPU - Ready for inference!")
        
    def _classify_generative(self, patent_text: str) -> Dict[str, Any]:
        """Classification using generative model (like Llama)."""
        import torch
        
        start_time = time.time()
        
        # Create classification prompt
        classes_text = """0: Human Necessities (food, medicine, personal items)
1: Operations/Transport (manufacturing, transport)
2: Chemistry/Metallurgy (chemicals, materials)
3: Textiles/Paper (textile, paper production)
4: Fixed Constructions (buildings, construction)
5: Mechanical Engineering (engines, machines, heating)
6: Physics (instruments, optics, nuclear)
7: Electricity (electronics, communications)
8: General Technology (cross-sectional tech)"""
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Classify this patent abstract into one of 9 categories. Respond with only the number (0-8).

Categories:
{classes_text}

<|eot_id|><|start_header_id|>user<|end_header_id|>

Patent: {patent_text.strip()[:500]}

Class:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # Tokenize and generate
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=self.max_length,
            truncation=True,
            padding=True
        ).to(self.model.device)
        
        input_token_count = inputs['input_ids'].shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        output_token_count = outputs.shape[1] - input_token_count
        
        # Decode and extract class
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        predicted_class = self._extract_class_number(response)
        inference_time = time.time() - start_time
        
        return {
            'predicted_class': predicted_class,
            'raw_response': response,
            'class_name': CLASS_LABELS.get(predicted_class, "Unknown") if predicted_class is not None else "Unknown",
            'input_tokens': input_token_count,
            'output_tokens': output_token_count,
            'inference_time': inference_time,
            'method': 'generative'
        }
    
    def _classify_direct(self, patent_text: str) -> Dict[str, Any]:
        """Direct classification using a fine-tuned classification model."""
        import torch
        import torch.nn.functional as F
        
        start_time = time.time()
        
        # Tokenize input
        inputs = self.tokenizer(
            patent_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        ).to(self.model.device)
        
        input_token_count = inputs['input_ids'].shape[1]
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
            # Get all class probabilities for detailed analysis
            all_probabilities = probabilities[0].cpu().numpy().tolist()
        
        inference_time = time.time() - start_time
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': all_probabilities,  # Full probability distribution
            'top_3_predictions': sorted(enumerate(all_probabilities), key=lambda x: x[1], reverse=True)[:3],
            'raw_response': f"Class {predicted_class} (confidence: {confidence:.3f})",
            'class_name': CLASS_LABELS.get(predicted_class, "Unknown"),
            'input_tokens': input_token_count,
            'output_tokens': 0,  # No generation tokens for classification
            'inference_time': inference_time,
            'method': 'classification'
        }
    
    def _classify_sentence_transformer(self, patent_text: str) -> Dict[str, Any]:
        """Classification using sentence transformer embeddings with a simple classifier."""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        start_time = time.time()
        
        # Get embedding for input text
        embedding = self.model.encode([patent_text])
        
        # Simple prototype-based classification using class descriptions
        class_descriptions = [
            "Human necessities including food, medicine, personal items, healthcare",
            "Operations, transporting, manufacturing, industrial processes",
            "Chemistry, metallurgy, chemical processes, materials science",
            "Textiles, paper manufacturing, fabric production",
            "Fixed constructions, buildings, architecture, construction",
            "Mechanical engineering, engines, machines, heating, weapons",
            "Physics, instruments, optics, nuclear physics, measurements",
            "Electricity, electronics, communications, electrical devices",
            "General technology, cross-sectional technologies, emerging tech"
        ]
        
        # Get embeddings for class descriptions
        if not hasattr(self, '_class_embeddings'):
            self._class_embeddings = self.model.encode(class_descriptions)
        
        # Calculate similarities
        similarities = cosine_similarity(embedding, self._class_embeddings)[0]
        predicted_class = int(np.argmax(similarities))
        confidence = float(similarities[predicted_class])
        
        inference_time = time.time() - start_time
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'raw_response': f"Embedding-based classification: Class {predicted_class} (similarity: {confidence:.3f})",
            'class_name': CLASS_LABELS.get(predicted_class, "Unknown"),
            'input_tokens': len(patent_text.split()),  # Approximate token count
            'output_tokens': 0,  # No generation tokens
            'inference_time': inference_time,
            'method': 'sentence_transformer'
        }
    
    def _extract_class_number(self, response: str) -> Optional[int]:
        """Extract class number from model response."""
        import re
        
        # Look for single digits 0-8 at the start of response
        numbers = re.findall(r'^[0-8]', response.strip())
        if numbers:
            return int(numbers[0])
            
        # Fallback: look for any digit 0-8
        numbers = re.findall(r'\b[0-8]\b', response)
        if numbers:
            return int(numbers[0])
            
        # Last resort: look for any digit 0-8 anywhere
        for char in response:
            if char.isdigit() and 0 <= int(char) <= 8:
                return int(char)
                
        return None

    @modal.method()
    def classify_single(self, patent_text: str) -> Dict[str, Any]:
        """Classify a single patent abstract."""
        if self.model_type == "generative":
            return self._classify_generative(patent_text)
        elif self.model_type == "sentence_transformer":
            return self._classify_sentence_transformer(patent_text)
        else:
            return self._classify_direct(patent_text)
    
    @modal.method()
    def classify_batch(self, patent_texts: List[str], true_labels: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Classify multiple patent abstracts efficiently."""
        batch_start_time = time.time()
        print(f"🚀 Starting Modal batch processing for {len(patent_texts)} samples...")
        print(f"📋 Using {self.model_type} approach with {self.model_name}")
        
        results = []
        correct_predictions = 0
        total_input_tokens = 0
        total_output_tokens = 0
        
        for i, text in enumerate(patent_texts):
            if self.model_type == "generative":
                result = self._classify_generative(text)
            elif self.model_type == "sentence_transformer":
                result = self._classify_sentence_transformer(text)
            else:
                result = self._classify_direct(text)
                
            results.append(result)
            
            # Track metrics
            if result.get('predicted_class') is not None and true_labels:
                if i < len(true_labels) and result['predicted_class'] == true_labels[i]:
                    correct_predictions += 1
            
            total_input_tokens += result.get('input_tokens', 0)
            total_output_tokens += result.get('output_tokens', 0)
            
            # Log progress every 5 samples
            if (i + 1) % 5 == 0 or (i + 1) == len(patent_texts):
                accuracy_info = ""
                if true_labels:
                    current_accuracy = correct_predictions / (i + 1) if i + 1 > 0 else 0
                    accuracy_info = f", Accuracy: {current_accuracy:.3f} ({correct_predictions}/{i+1})"
                print(f"📊 Modal Progress: {i+1}/{len(patent_texts)} samples processed{accuracy_info}")
                print(f"   Tokens so far: {total_input_tokens:,} input, {total_output_tokens:,} output")
        
        # Calculate costs
        batch_runtime = time.time() - batch_start_time
        modal_a10g_cost_per_second = 0.000306
        modal_compute_cost = batch_runtime * modal_a10g_cost_per_second
        
        print(f"✅ Modal batch processing completed!")
        print(f"📈 Final Modal Stats:")
        print(f"   Total samples: {len(patent_texts)}")
        if true_labels:
            final_accuracy = correct_predictions / len(patent_texts) if len(patent_texts) > 0 else 0
            print(f"   Classification Accuracy: {final_accuracy:.3f} ({correct_predictions}/{len(patent_texts)})")
        print(f"   Total input tokens: {total_input_tokens:,}")
        print(f"   Total output tokens: {total_output_tokens:,}")
        print(f"   Method: {self.model_type}")
        print(f"💰 Modal Costs:")
        print(f"   Compute time: {batch_runtime:.2f} seconds")
        print(f"   A10G GPU cost: ${modal_compute_cost:.6f}")
        
        return results

# Local interface functions
def get_flexible_modal_client(model_name: str, max_length: int = 2048, model_type: str = "generative"):
    """Get flexible Modal client for local usage."""
    return FlexiblePatentClassifier(model_name=model_name, max_length=max_length, model_type=model_type)

if __name__ == "__main__":
    # Test the flexible model loading
    print("Testing flexible Modal inference...")
    
    with app.run():
        # Test with generative model
        classifier = FlexiblePatentClassifier(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            model_type="generative"
        )
        test_text = "A method for producing biodegradable plastic from agricultural waste materials."
        result = classifier.classify_single.remote(test_text)
        print(f"Test result: {result}")