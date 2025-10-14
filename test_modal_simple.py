#!/usr/bin/env python3
"""
Simple Modal Test for Two-Stage Classification
"""

import modal

app = modal.App("test-two-stage")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(["datasets>=2.14.0", "transformers>=4.35.0", "torch>=2.0.0"])
)

@app.function(
    image=image,
    gpu="T4",  # Use smaller GPU for testing
    memory=8192,
    timeout=600,
)
def simple_test():
    """Simple test to see if Modal is working."""
    print("🚀 Testing Modal connection...")
    
    # Load dataset
    from datasets import load_dataset
    print("📥 Loading dataset...")
    dataset = load_dataset("ccdv/patent-classification", "abstract")
    test_data = dataset["test"].select(range(3))
    
    print(f"✅ Loaded {len(test_data)} samples")
    for i, sample in enumerate(test_data):
        print(f"Sample {i+1}: {sample['text'][:100]}...")
        print(f"Label: {sample['label']}")
    
    return {"status": "success", "samples": len(test_data)}

@app.local_entrypoint()
def main():
    """Run simple test."""
    print("🧪 Running simple Modal test...")
    result = simple_test.remote()
    print(f"Result: {result}")
    return result

if __name__ == "__main__":
    app.run(main)