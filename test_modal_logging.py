#!/usr/bin/env python3
"""Test script to verify Modal accuracy logging works."""

import sys
import os
from modal_client import ModalPatentClassifier
from config import Config

def test_modal_logging():
    """Test Modal logging with sample data."""
    print("🧪 Testing Modal accuracy logging...")
    
    config = Config()
    classifier = ModalPatentClassifier(config)
    
    # Sample texts
    test_texts = [
        "A method for producing biodegradable plastic from agricultural waste materials.",
        "A system for wireless communication using 5G technology.",
        "A pharmaceutical composition for treating diabetes."
    ]
    
    # Sample true labels (these might not be correct, just for testing)
    true_labels = [2, 7, 0]  # Chemistry, Electricity, Human Necessities
    
    print(f"📊 Testing with {len(test_texts)} samples")
    print("Note: Modal logs appear in the Modal dashboard, not local console")
    
    try:
        # Test batch classification with true labels
        results = classifier.classify_batch(test_texts, true_labels=true_labels)
        
        # Calculate local accuracy to verify
        correct = 0
        for i, result in enumerate(results):
            if result['predicted_class'] == true_labels[i]:
                correct += 1
            print(f"Sample {i+1}: True={true_labels[i]}, Predicted={result['predicted_class']}, Correct={result['predicted_class'] == true_labels[i]}")
        
        local_accuracy = correct / len(results)
        print(f"\n✅ Local verification: {correct}/{len(results)} correct ({local_accuracy:.3f} accuracy)")
        print("🔍 Check Modal dashboard to see the accuracy logs from the remote container")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_modal_logging()
    sys.exit(0 if success else 1)