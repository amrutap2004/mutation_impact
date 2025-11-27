#!/usr/bin/env python3
"""
Test the web interface specifically for K8E mutation to verify ML model integration.
"""

from mutation_impact.classifier.simple_ml_only import SimpleMLOnlyClassifier
from mutation_impact.input_module.parser import load_sequence, parse_mutation
from mutation_impact.structure.retrieval import fetch_rcsb_pdb
from mutation_impact.structure.modeling import build_mutant_structure_stub
from mutation_impact.features.interfaces import compute_basic_features

def test_web_k8e_integration():
    """Test K8E mutation exactly as the web interface does."""
    
    print("🌐 Testing Web Interface K8E Integration")
    print("="*60)
    
    # Simulate web interface inputs
    sequence = 'MVLSPADKTNVKAAW'
    mut_text = 'K8E'
    
    print(f"Input: {sequence} with mutation {mut_text}")
    
    try:
        # Step 1: Parse and validate (as web interface does)
        mutation = parse_mutation(mut_text)
        print(f"✅ Parsed mutation: {mutation}")
        
        # Step 2: Get structures (as web interface does)
        wt_path = fetch_rcsb_pdb('1CRN')
        mut_path = build_mutant_structure_stub(wt_path, sequence, mutation, force_naive=True)
        print(f"✅ Built structures")
        
        # Step 3: Compute basic features (as web interface does)
        features = compute_basic_features(sequence, mutation, wt_path, mut_path)
        print(f"\n📊 Basic Features (from web interface):")
        for key, value in features.items():
            print(f"   {key}: {value}")
        
        # Step 4: ML prediction (as web interface does)
        ml_classifier = SimpleMLOnlyClassifier("models/")
        pred = ml_classifier.predict(sequence, mut_text, str(wt_path), str(mut_path), "ensemble")
        
        print(f"\n🤖 ML Prediction:")
        print(f"   Label: {pred['label']}")
        print(f"   Confidence: {pred['confidence']:.1%}")
        print(f"   Model: {pred['model_used']}")
        print(f"   Feature Quality: {pred['feature_quality']:.1%}")
        
        # Step 5: Check what features the ML model actually used
        ml_features = ml_classifier._extract_basic_features(sequence, mut_text, str(wt_path), str(mut_path))
        print(f"\n🔍 ML Model Features (internal):")
        for key, value in ml_features.items():
            print(f"   {key}: {value}")
        
        # Compare the two feature sets
        print(f"\n🔍 Feature Comparison:")
        print(f"   Web features count: {len(features)}")
        print(f"   ML features count: {len(ml_features)}")
        
        missing_in_web = set(ml_features.keys()) - set(features.keys())
        if missing_in_web:
            print(f"   Missing in web features: {missing_in_web}")
        
        # The issue: web interface shows basic features, but ML model uses enhanced features
        print(f"\n💡 Issue Analysis:")
        print(f"   - Web interface displays features from compute_basic_features()")
        print(f"   - ML model internally adds charge_change feature")
        print(f"   - Report shows basic features, not ML model features")
        print(f"   - This creates confusion about what the model actually used")
        
        return pred
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_web_k8e_integration()
