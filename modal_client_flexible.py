"""Flexible client interface for Modal-based patent classification."""

import modal
from typing import List, Dict, Any, Optional
import time
import logging
from config import Config
from cost_tracker import CostTracker

logger = logging.getLogger(__name__)

class FlexibleModalPatentClassifier:
    """Flexible client for Modal-based patent classification supporting different model types."""
    
    def __init__(self, config: Config = Config(), cost_tracker: Optional[CostTracker] = None, 
                 model_type: str = "auto"):
        self.config = config
        self.cost_tracker = cost_tracker
        self.model_type = model_type
        self.classifier_instance = None
        
    def _detect_model_type(self, model_name: str) -> str:
        """Detect model type based on model name."""
        model_name_lower = model_name.lower()
        
        # Generative models
        generative_keywords = ['llama', 'gpt', 'mistral', 'phi', 'qwen', 'gemma', 'vicuna']
        if any(keyword in model_name_lower for keyword in generative_keywords):
            return "generative"
        
        # Classification models (fine-tuned BERT, RoBERTa, etc.)
        classification_keywords = ['bert', 'roberta', 'distilbert', 'electra', 'deberta', 'esm']
        if any(keyword in model_name_lower for keyword in classification_keywords):
            return "classification"
        
        # Default to generative for unknown models
        return "generative"
        
    def load_model(self):
        """Initialize connection to flexible Modal app."""
        try:
            # Import the flexible classifier class
            from modal_inference_flexible import FlexiblePatentClassifier
            
            # Auto-detect model type if not specified
            if self.model_type == "auto":
                detected_type = self._detect_model_type(self.config.MODEL_NAME)
                logger.info(f"Auto-detected model type: {detected_type} for {self.config.MODEL_NAME}")
                self.model_type = detected_type
            
            # Create an instance with config parameters
            self.classifier_instance = FlexiblePatentClassifier(
                model_name=self.config.MODEL_NAME,
                max_length=self.config.MAX_LENGTH,
                model_type=self.model_type
            )
            
            logger.info(f"Flexible Modal classifier initialized successfully (type: {self.model_type})")
            
            # Update cost tracker configuration
            if self.cost_tracker:
                self.cost_tracker.update_config_info(
                    batch_size=self.config.BATCH_SIZE,
                    max_length=self.config.MAX_LENGTH,
                    quantization=f"4bit-modal-{self.model_type}"
                )
                
        except Exception as e:
            logger.error(f"Failed to initialize flexible Modal classifier: {e}")
            raise RuntimeError(f"Flexible Modal classifier initialization failed: {e}. "
                             f"Ensure the Modal app is deployed and accessible.")
    
    def classify_single(self, patent_text: str) -> Dict[str, Any]:
        """Classify a single patent abstract using flexible Modal."""
        if self.classifier_instance is None:
            self.load_model()
        
        # Track inference time
        inference_context = self.cost_tracker.track_inference() if self.cost_tracker else None
        
        try:
            # Import and run within the app context
            from modal_inference_flexible import app
            
            with app.run():
                if inference_context:
                    with inference_context:
                        result = self.classifier_instance.classify_single.remote(patent_text)
                else:
                    result = self.classifier_instance.classify_single.remote(patent_text)
            
            # Track token usage if cost tracker is available
            if self.cost_tracker:
                input_tokens = result.get('input_tokens', 0)
                output_tokens = result.get('output_tokens', 0)
                self.cost_tracker.add_token_count(input_tokens, output_tokens)
                self.cost_tracker.add_sample_processed()
            
            return result
            
        except Exception as e:
            logger.error(f"Flexible Modal inference failed: {e}")
            # Return error result
            return {
                'predicted_class': None,
                'raw_response': f"Flexible Modal Error: {str(e)}",
                'class_name': "Error",
                'input_tokens': 0,
                'output_tokens': 0,
                'inference_time': 0,
                'method': 'error'
            }
    
    def classify_batch(self, patent_texts: List[str], batch_size: int = None, 
                      true_labels: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Classify multiple patent abstracts using flexible Modal's batch processing."""
        if self.classifier_instance is None:
            self.load_model()
        
        try:
            # Import and run within a single app context
            from modal_inference_flexible import app
            
            with app.run():
                # Use Modal's native batch processing for all samples at once
                batch_results = self.classifier_instance.classify_batch.remote(patent_texts, true_labels)
                
                # Track token usage for each result
                if self.cost_tracker:
                    for result in batch_results:
                        input_tokens = result.get('input_tokens', 0)
                        output_tokens = result.get('output_tokens', 0)
                        self.cost_tracker.add_token_count(input_tokens, output_tokens)
                        self.cost_tracker.add_sample_processed()
                
                return batch_results
                
        except Exception as e:
            logger.error(f"Flexible Modal batch inference failed: {e}")
            # Return error results for all samples
            return [{
                'predicted_class': None,
                'raw_response': f"Flexible Modal Batch Error: {str(e)}",
                'class_name': "Error",
                'input_tokens': 0,
                'output_tokens': 0,
                'inference_time': 0,
                'method': 'error'
            } for _ in patent_texts]

class HybridFlexiblePatentClassifier:
    """Hybrid classifier that can switch between local and flexible Modal inference."""
    
    def __init__(self, config: Config = Config(), cost_tracker: Optional[CostTracker] = None, 
                 use_modal: bool = True, model_type: str = "auto"):
        self.config = config
        self.cost_tracker = cost_tracker
        self.use_modal = use_modal
        self.model_type = model_type
        
        if self.use_modal:
            try:
                self.classifier = FlexibleModalPatentClassifier(config, cost_tracker, model_type)
                logger.info(f"Using Flexible Modal for inference (model type: {model_type})")
            except Exception as e:
                logger.warning(f"Flexible Modal initialization failed, falling back to local: {e}")
                self.use_modal = False
        
        if not self.use_modal:
            # Fallback to local classifier
            from model import LlamaPatentClassifier
            self.classifier = LlamaPatentClassifier(config, cost_tracker=cost_tracker)
            logger.info("Using local inference")
    
    def load_model(self):
        """Load the appropriate model."""
        return self.classifier.load_model()
    
    def classify_single(self, patent_text: str) -> Dict[str, Any]:
        """Classify using the active classifier."""
        return self.classifier.classify_single(patent_text)
    
    def classify_batch(self, patent_texts: List[str], batch_size: int = None, 
                      true_labels: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Classify batch using the active classifier."""
        if hasattr(self.classifier, 'classify_batch'):
            return self.classifier.classify_batch(patent_texts, batch_size, true_labels)
        else:
            # Fallback to single classification for local model
            results = []
            for text in patent_texts:
                results.append(self.classify_single(text))
            return results

if __name__ == "__main__":
    # Test flexible Modal client
    config = Config()
    config.MODEL_NAME = "bert-base-uncased"  # Test with a BERT model
    
    classifier = FlexibleModalPatentClassifier(config, model_type="auto")
    
    test_text = "A method for producing biodegradable plastic from agricultural waste materials."
    
    try:
        result = classifier.classify_single(test_text)
        print(f"Classification result: {result}")
    except Exception as e:
        print(f"Error: {e}")