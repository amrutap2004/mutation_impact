"""
Debug ML model loading issues.
"""

import json
from pathlib import Path

def debug_ml_loading():
    """Debug ML model loading."""
    print("🔍 Debugging ML Model Loading")
    print("="*50)
    
    models_dir = Path("models/")
    print(f"Models directory: {models_dir}")
    print(f"Directory exists: {models_dir.exists()}")
    
    if models_dir.exists():
        files = list(models_dir.glob("*"))
        print(f"Files in models directory: {files}")
        
        # Check each metadata file
        for metadata_file in models_dir.glob("*_metadata.json"):
            print(f"\nChecking {metadata_file}:")
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    print(f"  ✅ JSON loaded successfully")
                    print(f"  Keys: {list(metadata.keys())}")
                    if 'scaler_mean' in metadata:
                        print(f"  Scaler mean length: {len(metadata['scaler_mean'])}")
                    if 'label_encoder_classes' in metadata:
                        print(f"  Label encoder classes: {metadata['label_encoder_classes']}")
            except Exception as e:
                print(f"  ❌ JSON loading failed: {e}")
                print(f"  File content preview:")
                with open(metadata_file, 'r') as f:
                    content = f.read()
                    print(f"  First 200 chars: {content[:200]}")
                    print(f"  Last 200 chars: {content[-200:]}")
    
    # Test model loading
    print(f"\n🔍 Testing model loading:")
    try:
        import joblib
        model_files = list(models_dir.glob("*_model.joblib"))
        for model_file in model_files:
            print(f"  Loading {model_file}...")
            model = joblib.load(model_file)
            print(f"  ✅ Model loaded: {type(model)}")
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")

if __name__ == "__main__":
    debug_ml_loading()
