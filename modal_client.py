"""Client interface for Modal-based patent classification."""

import modal
from typing import List, Dict, Any, Optional
import time
import logging
from config import Config
from cost_tracker import CostTracker

logger = logging.getLogger(__name__)

class ModalPatentClassifier:
    """Client for Modal-based patent classification."""
    
    def __init__(self, config: Config = Config(), cost_tracker: Optional[CostTracker] = None):
        self.config = config
        self.cost_tracker = cost_tracker
        self.classifier_cls = None
        self.classifier_instance = None
        
    def load_model(self):
        """Initialize connection to Modal app."""
        try:
            # Import the classifier class directly from modal_inference
            from modal_inference import PatentClassifierModel
            
            # Create an instance with config parameters (this will trigger model loading on Modal)
            self.classifier_instance = PatentClassifierModel(
                model_name=self.config.MODEL_NAME,
                max_length=self.config.MAX_LENGTH
            )
            
            logger.info("Modal classifier initialized successfully")
            
            # Update cost tracker configuration
            if self.cost_tracker:
                self.cost_tracker.update_config_info(
                    batch_size=self.config.BATCH_SIZE,
                    max_length=self.config.MAX_LENGTH,
                    quantization="4bit-modal"
                )
                
        except Exception as e:
            logger.error(f"Failed to initialize Modal classifier: {e}")
            raise RuntimeError(f"Modal classifier initialization failed: {e}. "
                             f"Ensure the Modal app is deployed and accessible.")
    
    def classify_single(self, patent_text: str) -> Dict[str, Any]:
        """Classify a single patent abstract using Modal."""
        if self.classifier_instance is None:
            self.load_model()
        
        # Track inference time
        inference_context = self.cost_tracker.track_inference() if self.cost_tracker else None
        
        try:
            # Import and run within the app context
            from modal_inference import app
            
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
            logger.error(f"Modal inference failed: {e}")
            # Return error result
            return {
                'predicted_class': None,
                'raw_response': f"Modal Error: {str(e)}",
                'class_name': "Error",
                'input_tokens': 0,
                'output_tokens': 0,
                'inference_time': 0,
                'prompt_length': 0
            }
    
    def classify_batch(self, patent_texts: List[str], batch_size: int = None, true_labels: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Classify multiple patent abstracts using Modal's batch processing."""
        if self.classifier_instance is None:
            self.load_model()
        
        try:
            # Import and run within a single app context
            from modal_inference import app
            
            with app.run():
                # Use Modal's native batch processing for all samples at once
                # This keeps the container alive for the entire batch
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
            logger.error(f"Modal batch inference failed: {e}")
            # Return error results for all samples
            return [{
                'predicted_class': None,
                'raw_response': f"Modal Batch Error: {str(e)}",
                'class_name': "Error",
                'input_tokens': 0,
                'output_tokens': 0,
                'inference_time': 0,
                'prompt_length': 0
            } for _ in patent_texts]

class HybridPatentClassifier:
    """Hybrid classifier that can switch between local and Modal inference."""
    
    def __init__(self, config: Config = Config(), cost_tracker: Optional[CostTracker] = None, 
                 use_modal: bool = True):
        self.config = config
        self.cost_tracker = cost_tracker
        self.use_modal = use_modal
        
        if self.use_modal:
            try:
                self.classifier = ModalPatentClassifier(config, cost_tracker)
                logger.info("Using Modal for inference")
            except Exception as e:
                logger.warning(f"Modal initialization failed, falling back to local: {e}")
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
    
    def classify_batch(self, patent_texts: List[str], batch_size: int = None, true_labels: Optional[List[int]] = None) -> List[Dict[str, Any]]:
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
    # Test Modal client
    config = Config()
    classifier = ModalPatentClassifier(config)
    
    test_text = "A method for producing biodegradable plastic from agricultural waste materials."
    
    try:
        result = classifier.classify_single(test_text)
        print(f"Classification result: {result}")
    except Exception as e:
        print(f"Error: {e}")