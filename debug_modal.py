#!/usr/bin/env python3
"""Debug Modal app structure."""

from modal_inference import app

print("🔍 Debugging Modal app structure...")
print(f"App: {app}")
print(f"App attributes: {dir(app)}")

# Check if the class is available in different ways
try:
    print(f"App._function_cls: {getattr(app, '_function_cls', 'NOT FOUND')}")
    print(f"App._web_endpoints: {getattr(app, '_web_endpoints', 'NOT FOUND')}")
    
    # Try to find the PatentClassifierModel
    for attr_name in dir(app):
        attr_value = getattr(app, attr_name)
        print(f"  {attr_name}: {type(attr_value)} - {attr_value}")
        
except Exception as e:
    print(f"Error: {e}")

print("🧪 Trying different access patterns...")

# Try direct import
try:
    from modal_inference import PatentClassifierModel
    print(f"✅ Direct import successful: {PatentClassifierModel}")
except Exception as e:
    print(f"❌ Direct import failed: {e}")

# Try app object lookup
try:
    classifier_cls = app.registered_functions.get('PatentClassifierModel')  # Example attempt
    print(f"Via registered_functions: {classifier_cls}")
except Exception as e:
    print(f"Via registered_functions failed: {e}")

print("Done debugging.")