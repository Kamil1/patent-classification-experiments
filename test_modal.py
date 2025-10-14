#!/usr/bin/env python3
"""Test Modal deployment and functionality."""

import modal

def test_modal_connection():
    """Test basic Modal connection and app lookup."""
    try:
        # Test connection
        print("🔌 Testing Modal connection...")
        
        # Import our Modal app directly
        from modal_inference import app, PatentClassifierModel
        
        print("✅ Modal app imported successfully")
        
        # Try to create an instance
        print("🚀 Testing classifier instantiation...")
        classifier = PatentClassifierModel()
        
        print("✅ Classifier instantiated")
        
        # Test a simple classification
        test_text = "A method for producing biodegradable plastic from agricultural waste materials."
        
        print("🧪 Testing classification...")
        result = classifier.classify_single.local(test_text)
        
        print("✅ Classification completed!")
        print(f"Result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Modal Patent Classification")
    print("=" * 40)
    
    success = test_modal_connection()
    
    if success:
        print("\n✅ All tests passed! Modal is working correctly.")
    else:
        print("\n❌ Tests failed. Check the errors above.")