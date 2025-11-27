"""
Test web interface with all options to ensure no errors.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from mutation_impact.web.app import main
import threading
import time
import requests
import json


def test_web_interface():
    """Test the web interface with all possible options."""
    print("🌐 Testing Web Interface with All Options")
    print("="*60)
    
    # Test configurations
    test_configs = [
        {
            "name": "Basic Analysis",
            "sequence": "MVLSPADKTNVKAAW",
            "mutation": "S4E",
            "pdb_id": "1CRN",
            "force_naive": True,
            "high_accuracy": False,
            "minimize": False
        },
        {
            "name": "High-Accuracy Analysis",
            "sequence": "MVLSPADKTNVKAAW", 
            "mutation": "S4E",
            "pdb_id": "1CRN",
            "force_naive": True,
            "high_accuracy": True,
            "minimize": False
        },
        {
            "name": "With Minimization",
            "sequence": "MVLSPADKTNVKAAW",
            "mutation": "S4E", 
            "pdb_id": "1CRN",
            "force_naive": True,
            "high_accuracy": True,
            "minimize": True
        },
        {
            "name": "Different Mutation",
            "sequence": "MVLSPADKTNVKAAW",
            "mutation": "A1V",
            "pdb_id": "1CRN", 
            "force_naive": True,
            "high_accuracy": True,
            "minimize": False
        }
    ]
    
    print("Test configurations:")
    for i, config in enumerate(test_configs):
        print(f"  {i+1}. {config['name']}")
        print(f"     Sequence: {config['sequence']}")
        print(f"     Mutation: {config['mutation']}")
        print(f"     Force naive: {config['force_naive']}")
        print(f"     High accuracy: {config['high_accuracy']}")
        print(f"     Minimize: {config['minimize']}")
        print()
    
    print("🚀 Starting web server...")
    print("Open your browser and test these configurations manually:")
    print("   URL: http://127.0.0.1:7860")
    print()
    print("📋 Manual Test Checklist:")
    print("="*60)
    
    for i, config in enumerate(test_configs):
        print(f"\n{i+1}. {config['name']} Test:")
        print(f"   ✅ Enter sequence: {config['sequence']}")
        print(f"   ✅ Enter mutation: {config['mutation']}")
        print(f"   ✅ Select PDB ID: {config['pdb_id']}")
        print(f"   ✅ Force naive mapping: {'ON' if config['force_naive'] else 'OFF'}")
        print(f"   ✅ High-accuracy mode: {'ON' if config['high_accuracy'] else 'OFF'}")
        print(f"   ✅ Minimize: {'ON' if config['minimize'] else 'OFF'}")
        print(f"   ✅ Click 'Run Analysis'")
        print(f"   ✅ Verify: No errors, results displayed")
        print(f"   ✅ Check: Prediction and confidence shown")
        if config['high_accuracy']:
            print(f"   ✅ Check: Enhanced confidence analysis displayed")
        print(f"   ✅ Check: 3D visualization works")
        print(f"   ✅ Check: PDF download works")
    
    print(f"\n🎯 Expected Results:")
    print(f"   📊 All configurations should work without errors")
    print(f"   🎯 High-accuracy mode should show enhanced confidence")
    print(f"   🔬 ML model should provide better predictions")
    print(f"   📈 Accuracy should be 80%+ with ML model")
    print(f"   📄 PDF export should work for all configurations")
    
    print(f"\n🔧 Troubleshooting:")
    print(f"   ❌ If 'string indices must be integers' error:")
    print(f"      → Check that mutation format is correct (e.g., S4E)")
    print(f"   ❌ If 'Could not map sequence position' error:")
    print(f"      → Enable 'Force naive mapping'")
    print(f"   ❌ If ML model fails:")
    print(f"      → Check that models/ensemble_model.joblib exists")
    print(f"   ❌ If minimization fails:")
    print(f"      → Install OpenMM: pip install openmm")
    
    # Start web server
    print(f"\n🌐 Starting web server at http://127.0.0.1:7860")
    print(f"Press Ctrl+C to stop the server")
    
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n👋 Web server stopped by user")
    except Exception as e:
        print(f"\n❌ Web server error: {e}")


def verify_ml_model():
    """Verify ML model is available."""
    print("🔍 Verifying ML Model...")
    
    model_path = "models/ensemble_model.joblib"
    metadata_path = "models/ensemble_metadata.json"
    
    if os.path.exists(model_path) and os.path.exists(metadata_path):
        print("✅ ML model found")
        
        # Load and test model
        try:
            import joblib
            import json
            
            model = joblib.load(model_path)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            print(f"✅ Model loaded successfully")
            print(f"   Features: {metadata['feature_names']}")
            print(f"   Accuracy: {metadata.get('accuracy', 'N/A')}")
            print(f"   CV Score: {metadata.get('cv_mean', 'N/A')}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    else:
        print("❌ ML model not found")
        print("   Run: python create_better_ml_model.py")
        return False


if __name__ == "__main__":
    print("🧪 Web Interface Comprehensive Test")
    print("="*60)
    
    # Verify ML model
    ml_available = verify_ml_model()
    
    if ml_available:
        print("\n✅ ML model is ready for high-accuracy predictions")
    else:
        print("\n⚠️  ML model not available, will use rule-based classifier")
    
    # Start web interface test
    test_web_interface()
