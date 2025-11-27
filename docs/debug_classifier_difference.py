#!/usr/bin/env python3
"""
Debug the difference between direct model loading and SimpleMLOnlyClassifier.
"""

import joblib
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from mutation_impact.classifier.simple_ml_only import SimpleMLOnlyClassifier

def debug_classifier_difference():
    """Debug the difference between direct and classifier approaches."""
    
    print("🔍 DEBUGGING CLASSIFIER DIFFERENCE")
    print("="*60)
    
    # Method 1: Direct model loading (what we tested earlier)
    print("📊 Method 1: Direct Model Loading")
    model_direct = joblib.load("models/ensemble_model.joblib")
    
    with open("models/ensemble_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    scaler_direct = StandardScaler()
    scaler_direct.mean_ = np.array(metadata['scaler_mean'])
    scaler_direct.scale_ = np.array(metadata['scaler_scale'])
    
    # K8E features with charge_change
    k8e_features = [0.0, -4.887125836769883, 0.0, 1.0, 0.3999999999999999, 0.5, 2.0]
    features_scaled = scaler_direct.transform([k8e_features])
    
    prediction_direct = model_direct.predict(features_scaled)[0]
    probabilities_direct = model_direct.predict_proba(features_scaled)[0]
    label_direct = metadata['label_encoder_classes'][prediction_direct]
    confidence_direct = max(probabilities_direct)
    
    print(f"  Features: {k8e_features}")
    print(f"  Scaled: {features_scaled[0]}")
    print(f"  Prediction: {label_direct} ({confidence_direct:.1%})")
    
    # Method 2: SimpleMLOnlyClassifier (what web interface uses)
    print(f"\n📊 Method 2: SimpleMLOnlyClassifier")
    classifier = SimpleMLOnlyClassifier("models/")
    
    # Simulate the exact call from web interface
    sequence = 'MVLSPADKTNVKAAW'
    mutation = 'K8E'
    wt_path = 'C:\\Users\\akash\\.mutation_impact\\1CRN.pdb'
    mut_path = 'C:\\Users\\akash\\.mutation_impact\\1CRN_K8E.pdb'
    
    pred_classifier = classifier.predict(sequence, mutation, wt_path, mut_path, "ensemble")
    
    print(f"  Prediction: {pred_classifier['label']} ({pred_classifier['confidence']:.1%})")
    print(f"  Feature Quality: {pred_classifier['feature_quality']:.1%}")
    
    # Method 3: Check what features the classifier actually extracts
    print(f"\n📊 Method 3: Classifier Feature Extraction")
    ml_features = classifier._extract_basic_features(sequence, mutation, wt_path, mut_path)
    
    print(f"  Extracted features: {ml_features}")
    
    # Compare the features
    print(f"\n🔍 Feature Comparison:")
    print(f"  Direct method features: {k8e_features}")
    print(f"  Classifier features: {list(ml_features.values())}")
    
    # Check if they're the same
    classifier_values = [ml_features[name] for name in metadata['feature_names']]
    
    print(f"\n📊 Detailed Comparison:")
    for i, name in enumerate(metadata['feature_names']):
        direct_val = k8e_features[i]
        classifier_val = classifier_values[i]
        match = "✅" if abs(direct_val - classifier_val) < 1e-10 else "❌"
        print(f"  {name}: {direct_val} vs {classifier_val} {match}")
    
    # If features are the same, check if scaling is different
    if classifier_values == k8e_features:
        print(f"\n✅ Features are identical - issue must be elsewhere")
    else:
        print(f"\n❌ Features are different - this explains the discrepancy")
    
    # Check if the classifier is using the same model file
    print(f"\n📁 Model File Check:")
    print(f"  Direct model file: models/ensemble_model.joblib")
    print(f"  Classifier model path: {classifier.models_dir}")
    print(f"  Classifier available models: {list(classifier.models.keys())}")
    
    # Check if the classifier model is the same as direct model
    classifier_model = classifier.models.get("ensemble")
    if classifier_model is model_direct:
        print(f"  ✅ Same model object")
    else:
        print(f"  ❌ Different model objects")
    
    # Test the classifier model directly
    if classifier_model:
        classifier_scaled = classifier.scaler.transform([classifier_values]) if hasattr(classifier, 'scaler') else scaler_direct.transform([classifier_values])
        classifier_pred = classifier_model.predict(classifier_scaled)[0]
        classifier_prob = classifier_model.predict_proba(classifier_scaled)[0]
        classifier_label = metadata['label_encoder_classes'][classifier_pred]
        classifier_conf = max(classifier_prob)
        
        print(f"\n🧪 Direct test of classifier model:")
        print(f"  Prediction: {classifier_label} ({classifier_conf:.1%})")

if __name__ == "__main__":
    debug_classifier_difference()
