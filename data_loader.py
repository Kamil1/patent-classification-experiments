"""Data loading and preprocessing utilities for patent classification."""

from datasets import load_dataset
import pandas as pd
from typing import Dict, List, Tuple, Optional
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatentDataLoader:
    """Handles loading and preprocessing of patent classification dataset."""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.dataset = None
        self.class_counts = None
        
    def load_dataset(self) -> Dict:
        """Load the patent classification dataset from HuggingFace."""
        logger.info(f"Loading dataset: {self.config.DATASET_NAME}")
        
        self.dataset = load_dataset(
            self.config.DATASET_NAME,
            self.config.SUBSET
        )
        
        logger.info(f"Dataset loaded successfully:")
        logger.info(f"Train: {len(self.dataset['train'])} examples")
        logger.info(f"Validation: {len(self.dataset['validation'])} examples") 
        logger.info(f"Test: {len(self.dataset['test'])} examples")
        
        return self.dataset
    
    def analyze_class_distribution(self) -> Dict[int, int]:
        """Analyze class distribution across splits."""
        if self.dataset is None:
            self.load_dataset()
            
        self.class_counts = {}
        
        for split in ['train', 'validation', 'test']:
            labels = self.dataset[split]['label']
            split_counts = pd.Series(labels).value_counts().sort_index()
            self.class_counts[split] = split_counts.to_dict()
            
            logger.info(f"\n{split.upper()} set class distribution:")
            for label, count in split_counts.items():
                class_name = self.config.CLASS_LABELS[label]
                percentage = (count / len(labels)) * 100
                logger.info(f"  {label}: {class_name[:30]}... - {count} ({percentage:.1f}%)")
        
        return self.class_counts
    
    def get_sample_data(self, split: str = 'train', n_samples: int = 5) -> List[Dict]:
        """Get sample data for inspection."""
        if self.dataset is None:
            self.load_dataset()
            
        samples = []
        for i in range(min(n_samples, len(self.dataset[split]))):
            sample = self.dataset[split][i]
            samples.append({
                'text': sample['text'][:200] + "...",  # Truncate for display
                'label': sample['label'],
                'class_name': self.config.CLASS_LABELS[sample['label']]
            })
            
        return samples
    
    def prepare_data_for_model(self, split: str = 'train', max_samples: Optional[int] = None) -> List[Dict]:
        """Prepare data in format suitable for model training/inference."""
        if self.dataset is None:
            self.load_dataset()
            
        data = self.dataset[split]
        if max_samples:
            data = data.select(range(min(max_samples, len(data))))
            
        prepared_data = []
        for example in data:
            prepared_data.append({
                'text': example['text'],
                'label': example['label'],
                'label_name': self.config.CLASS_LABELS[example['label']]
            })
            
        return prepared_data

if __name__ == "__main__":
    # Demo usage
    loader = PatentDataLoader()
    loader.load_dataset()
    loader.analyze_class_distribution()
    
    print("\nSample data:")
    samples = loader.get_sample_data(n_samples=2)
    for i, sample in enumerate(samples):
        print(f"\nExample {i+1}:")
        print(f"Label: {sample['label']} - {sample['class_name']}")
        print(f"Text: {sample['text']}")