"""Modal-based inference for patent classification with GPU acceleration."""

import modal
from typing import List, Dict, Any, Optional
import json
import time
from datetime import datetime

# Create Modal app
app = modal.App("patent-classification")

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
        "hf_transfer>=0.1.0"  # Add hf_transfer for faster downloads
    ])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Model configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_LENGTH = 2048
TEMPERATURE = 0.1
TOP_P = 0.9

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
    # nit: why use an A10G over a newer H100 or A100 (that would have higher throughput)
    gpu="A10G",            # Use A10G GPU for cost-effective inference
    memory=16384,          # 16GB memory
    timeout=3600,          # 1 hour timeout
    min_containers=0,      # Don't keep warm initially to save costs
    secrets=[modal.Secret.from_name("huggingface-secret")],  # Add HF token
)
@modal.concurrent(max_inputs=4)  # Allow 4 concurrent requests
class PatentClassifierModel:
    """Modal class for patent classification using configurable models."""
    
    # Use modal parameters (only int, str, bytes, bool supported)
    model_name: str = modal.parameter(default=MODEL_NAME)
    max_length: int = modal.parameter(default=MAX_LENGTH)
    
    # Float parameters as class attributes (not modal parameters since floats aren't supported)
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
        
    @modal.enter()
    def load_model(self):
        """Load the model when the container starts."""
        import torch
        import os
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from huggingface_hub import login
        
        print(f"🔧 Modal Container Starting - Loading {self.model_name}...")
        print(f"📋 Model Configuration:")
        print(f"   Max Length: {self.max_length}")
        print(f"   Temperature: {self.temperature}")
        print(f"   Top-p: {self.top_p}")
        
        # Authenticate with HuggingFace using the secret token
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
            print("✅ Authenticated with HuggingFace")
        else:
            print("⚠️  No HF_TOKEN found, proceeding without authentication")
        
        print(f"📥 Loading model: {self.model_name}")
        
        # Configure 4-bit quantization for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model with 4-bit quantization
        # What's the reasoning for using AutoModelForCausalLM over vLLM or SGLang?
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        print("✅ Model loaded successfully on GPU - Ready for inference!")
        
    def create_classification_prompt(self, patent_text: str) -> str:
        """Create optimized prompt for patent classification."""

        # Why not try adding some k shot examples?

        # nit: could using tokenizer.apply_chat_template be easier than manipulating the chat template text
        # ourselves??
        # Shortened class descriptions to reduce token usage
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
        return prompt
    
    def _classify_text_internal(self, patent_text: str) -> Dict[str, Any]:
        """Internal method to classify a single text (used by both single and batch methods)."""
        import torch
        
        start_time = time.time()
        
        prompt = self.create_classification_prompt(patent_text)
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=self.max_length,
            truncation=True,
            padding=True
        ).to(self.model.device)
        
        input_token_count = inputs['input_ids'].shape[1]
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,  # Only need a single digit
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        output_token_count = outputs.shape[1] - input_token_count
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        # Extract class number
        predicted_class = self._extract_class_number(response)
        inference_time = time.time() - start_time
        
        return {
            'predicted_class': predicted_class,
            'raw_response': response,
            'class_name': CLASS_LABELS.get(predicted_class, "Unknown") if predicted_class is not None else "Unknown",
            'input_tokens': input_token_count,
            'output_tokens': output_token_count,
            'inference_time': inference_time,
            'prompt_length': len(prompt)
        }

    @modal.method()
    def classify_single(self, patent_text: str) -> Dict[str, Any]:
        """Classify a single patent abstract."""
        return self._classify_text_internal(patent_text)
    
    @modal.method()
    def classify_batch(self, patent_texts: List[str], true_labels: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Classify multiple patent abstracts efficiently."""
        print(f"🚀 Starting Modal batch processing for {len(patent_texts)} samples...")
        
        results = []
        correct_predictions = 0
        total_input_tokens = 0
        total_output_tokens = 0
        
        for i, text in enumerate(patent_texts):
            result = self._classify_text_internal(text)
            results.append(result)
            
            # Track metrics for Modal logs
            if result.get('predicted_class') is not None and true_labels:
                # Check correctness if true labels are provided
                if i < len(true_labels) and result['predicted_class'] == true_labels[i]:
                    correct_predictions += 1
            
            total_input_tokens += result.get('input_tokens', 0)
            total_output_tokens += result.get('output_tokens', 0)
            
            # Log progress every 5 samples or at the end
            if (i + 1) % 5 == 0 or (i + 1) == len(patent_texts):
                accuracy_info = ""
                if true_labels:
                    current_accuracy = correct_predictions / (i + 1) if i + 1 > 0 else 0
                    accuracy_info = f", Accuracy: {current_accuracy:.3f} ({correct_predictions}/{i+1})"
                print(f"📊 Modal Progress: {i+1}/{len(patent_texts)} samples processed{accuracy_info}")
                print(f"   Tokens so far: {total_input_tokens:,} input, {total_output_tokens:,} output")
        
        print(f"✅ Modal batch processing completed!")
        print(f"📈 Final Modal Stats:")
        print(f"   Total samples: {len(patent_texts)}")
        if true_labels:
            final_accuracy = correct_predictions / len(patent_texts) if len(patent_texts) > 0 else 0
            print(f"   Classification Accuracy: {final_accuracy:.3f} ({correct_predictions}/{len(patent_texts)})")
        print(f"   Total input tokens: {total_input_tokens:,}")
        print(f"   Total output tokens: {total_output_tokens:,}")
        print(f"   Avg tokens per sample: {total_input_tokens/len(patent_texts):.1f} input, {total_output_tokens/len(patent_texts):.1f} output")
            
        return results
    
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

# Could things like this clutter the submitted code?
# Create a web endpoint for testing (optional)
@app.function(image=image)
def classify_endpoint(item: Dict[str, str]):
    """Web endpoint for patent classification."""
    classifier = PatentClassifierModel()
    patent_text = item.get("text", "")
    
    if not patent_text:
        return {"error": "No text provided"}
    
    result = classifier.classify_single.local(patent_text)
    return result

# Local interface functions
def get_modal_client(model_name: str = MODEL_NAME, max_length: int = MAX_LENGTH):
    """Get Modal client for local usage."""
    return PatentClassifierModel(model_name=model_name, max_length=max_length)

@app.function(image=image)
def test_model_loading():
    """Test function to verify model loads correctly."""
    classifier = PatentClassifierModel()
    test_text = "A method for producing biodegradable plastic from agricultural waste materials."
    result = classifier.classify_single.local(test_text)
    return result

if __name__ == "__main__":
    # Test the model loading and inference
    print("Testing Modal inference...")
    
    with app.run():
        result = test_model_loading.remote()
        print(f"Test result: {result}")
        print("Modal inference test completed!")