"""Model loading and inference utilities for Llama 8B patent classification."""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from typing import List, Dict, Optional, Union
import logging
from config import Config
from cost_tracker import CostTracker
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaPatentClassifier:
    """Llama 8B model for patent classification using prompt-based approach."""
    
    def __init__(self, config: Config = Config(), use_4bit: bool = True, cost_tracker: Optional[CostTracker] = None):
        self.config = config
        self.use_4bit = use_4bit
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cost_tracker = cost_tracker
        
    def load_model(self):
        """Load Llama 8B model and tokenizer."""
        logger.info(f"Loading model: {self.config.MODEL_NAME}")
        logger.info(f"Device: {self.device}")
        
        # Track model loading time
        load_context = self.cost_tracker.track_model_loading() if self.cost_tracker else None
        
        with (load_context if load_context else self._dummy_context()):
            # Configure quantization for memory efficiency
            quantization_config = None
            quantization_str = "none"
            if self.use_4bit and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                quantization_str = "4bit"
                
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.MODEL_NAME,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.MODEL_NAME,
                quantization_config=quantization_config,
                device_map="auto" if torch.cuda.is_available() else None,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        
        # Update cost tracker with configuration
        if self.cost_tracker:
            self.cost_tracker.update_config_info(
                batch_size=self.config.BATCH_SIZE,
                max_length=self.config.MAX_LENGTH,
                quantization=quantization_str
            )
        
        logger.info("Model loaded successfully")
    
    def _dummy_context(self):
        """Dummy context manager for when cost_tracker is None."""
        from contextlib import nullcontext
        return nullcontext()
        
    def create_classification_prompt(self, patent_text: str) -> str:
        """Create a structured prompt for patent classification."""
        
        class_descriptions = {
            0: "Human Necessities - Agriculture, food, medicine, personal items",
            1: "Performing Operations; Transporting - Manufacturing, transport, separation", 
            2: "Chemistry; Metallurgy - Chemical processes, materials, metallurgy",
            3: "Textiles; Paper - Textile manufacturing, paper production",
            4: "Fixed Constructions - Buildings, bridges, construction",
            5: "Mechanical Engineering - Engines, machines, weapons, lighting, heating",
            6: "Physics - Instruments, nuclear physics, optics, photography",
            7: "Electricity - Electrical engineering, electronics, communications",
            8: "General Technology - Cross-sectional or emerging technologies"
        }
        
        classes_text = "\n".join([f"{k}: {v}" for k, v in class_descriptions.items()])
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert patent classifier. Your task is to classify patent abstracts into one of 9 categories based on the International Patent Classification (IPC) system.

Classes:
{classes_text}

Instructions:
1. Read the patent abstract carefully
2. Identify the main technical field and application
3. Choose the most appropriate class (0-8)
4. Respond with only the class number (0, 1, 2, 3, 4, 5, 6, 7, or 8)

<|eot_id|><|start_header_id|>user<|end_header_id|>

Patent Abstract:
{patent_text.strip()}

Classification:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt
        
    def classify_single(self, patent_text: str) -> Dict:
        """Classify a single patent abstract."""
        if self.model is None or self.tokenizer is None:
            self.load_model()
            
        prompt = self.create_classification_prompt(patent_text)
        
        # Track inference time and token usage
        inference_context = self.cost_tracker.track_inference() if self.cost_tracker else None
        
        with (inference_context if inference_context else self._dummy_context()):
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=self.config.MAX_LENGTH,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Count input tokens
            input_token_count = inputs['input_ids'].shape[1]
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,  # We only need a single digit
                    temperature=self.config.TEMPERATURE,
                    top_p=self.config.TOP_P,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
            # Count output tokens
            output_token_count = outputs.shape[1] - input_token_count
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
        
        # Track token usage
        if self.cost_tracker:
            self.cost_tracker.add_token_count(input_token_count, output_token_count)
            self.cost_tracker.add_sample_processed()
        
        # Extract classification number
        predicted_class = self.extract_class_number(response)
        
        return {
            'predicted_class': predicted_class,
            'raw_response': response,
            'class_name': self.config.CLASS_LABELS.get(predicted_class, "Unknown") if predicted_class is not None else "Unknown",
            'prompt': prompt,
            'input_tokens': input_token_count,
            'output_tokens': output_token_count
        }
    
    def extract_class_number(self, response: str) -> Optional[int]:
        """Extract class number from model response."""
        # Look for single digits 0-8
        numbers = re.findall(r'\b[0-8]\b', response)
        if numbers:
            return int(numbers[0])
            
        # Fallback: look for any digit 0-8
        for char in response:
            if char.isdigit() and 0 <= int(char) <= 8:
                return int(char)
                
        return None
    
    def classify_batch(self, patent_texts: List[str], batch_size: int = None) -> List[Dict]:
        """Classify multiple patent abstracts."""
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE
            
        results = []
        for i in range(0, len(patent_texts), batch_size):
            batch = patent_texts[i:i + batch_size]
            for text in batch:
                result = self.classify_single(text)
                results.append(result)
                logger.info(f"Processed {len(results)}/{len(patent_texts)} samples")
                
        return results

if __name__ == "__main__":
    # Demo usage
    classifier = LlamaPatentClassifier()
    
    # Example patent abstract
    sample_text = """
    A method for producing biodegradable plastic from agricultural waste materials. 
    The process involves enzymatic breakdown of cellulose fibers followed by 
    polymerization to create environmentally friendly packaging materials.
    """
    
    result = classifier.classify_single(sample_text)
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Class name: {result['class_name']}")
    print(f"Raw response: {result['raw_response']}")