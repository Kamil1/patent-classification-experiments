#!/usr/bin/env python3
"""Setup script for Modal deployment and configuration."""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd: str, description: str):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"   Error: {e.stderr.strip()}")
        return False

def check_modal_auth():
    """Check if Modal is authenticated."""
    try:
        result = subprocess.run("modal token current", shell=True, check=True, capture_output=True, text=True)
        print("✅ Modal authentication verified")
        return True
    except subprocess.CalledProcessError:
        print("❌ Modal not authenticated")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up Modal for Patent Classification")
    print("=" * 50)
    
    # Check if modal is installed
    try:
        import modal
        print("✅ Modal is already installed")
    except ImportError:
        print("📦 Installing Modal...")
        if not run_command("pip install modal", "Installing Modal"):
            print("❌ Failed to install Modal. Please install manually with: pip install modal")
            return False
    
    # Check Modal authentication
    if not check_modal_auth():
        print("\n🔐 Setting up Modal authentication...")
        print("Please follow the authentication flow:")
        if not run_command("modal token new", "Setting up Modal authentication"):
            print("❌ Authentication failed. Please run 'modal token new' manually")
            return False
    
    # Deploy the Modal app
    print("\n🚀 Deploying Modal app...")
    modal_file = Path(__file__).parent / "modal_inference.py"
    
    if not modal_file.exists():
        print(f"❌ Modal inference file not found: {modal_file}")
        return False
    
    if not run_command(f"modal deploy {modal_file}", "Deploying Modal app"):
        print("❌ Failed to deploy Modal app")
        print("\n🔧 Troubleshooting:")
        print("   1. Check that modal_inference.py exists")
        print("   2. Verify Modal authentication: modal token current")
        print("   3. Try deploying manually: modal deploy modal_inference.py")
        return False
    
    # Test the deployment
    print("\n🧪 Testing Modal deployment...")
    test_cmd = f"python -c \"import modal; app = modal.App.lookup('patent-classification'); print('App found successfully')\""
    
    if run_command(test_cmd, "Testing Modal app lookup"):
        print("\n🎉 Modal setup completed successfully!")
        print("\nYou can now run:")
        print("   python main.py --mode classify --max_samples 5")
        return True
    else:
        print("⚠️  Modal app deployed but lookup test failed")
        print("This might be temporary - try running the classification anyway")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)