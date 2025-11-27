#!/usr/bin/env python3
"""
Debug model loading to check for version mismatches or other issues.
"""

import joblib
import json
import numpy as np
from sklearn.preprocessing import StandardScaler

def debug_model_loading():
    """Debug the model loading process."""
    
    print("🔍 DEBUGGING MODEL LOADING")
    print("="*60)
    
    # Load model and metadata
    print("📁 Loading model and metadata...")
    model = joblib.load("models/ensemble_model.joblib")
    
    with open("models/ensemble_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    print(f"✅ Model type: {type(model).__name__}")
    print(f"✅ Model classes: {getattr(model, 'classes_', 'No classes')}")
    print(f"✅ Metadata features: {metadata['feature_names']}")
    print(f"✅ Metadata classes: {metadata['label_encoder_classes']}")
    
    # Check if model expects the right number of features
    if hasattr(model, 'n_features_in_'):
        print(f"✅ Model expects {model.n_features_in_} features")
    
    # Recreate scaler
    scaler = StandardScaler()
    scaler.mean_ = np.array(metadata['scaler_mean'])
    scaler.scale_ = np.array(metadata['scaler_scale'])
    
    print(f"\n📏 Scaler configuration:")
    print(f"   Mean: {scaler.mean_}")
    print(f"   Scale: {scaler.scale_}")
    
    # Test with K8E features including charge_change
    k8e_features = [0.0, -4.887125836769883, 0.0, 1.0, 0.3999999999999999, 0.5, 2.0]
    feature_names = metadata['feature_names']
    
    print(f"\n🧬 Testing K8E features:")
    for name, value in zip(feature_names, k8e_features):
        print(f"   {name}: {value}")
    
    # Scale and predict
    features_array = np.array([k8e_features])
    features_scaled = scaler.transform(features_array)
    
    print(f"\n📐 Scaled features:")
    for name, value in zip(feature_names, features_scaled[0]):
        print(f"   {name}: {value:.3f}")
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    print(f"\n🎯 Raw model output:")
    print(f"   Prediction: {prediction}")
    print(f"   Probabilities: {probabilities}")
    print(f"   Classes: {model.classes_}")
    
    # Map to labels
    label = metadata['label_encoder_classes'][prediction]
    confidence = max(probabilities)
    
    print(f"\n📋 Final prediction:")
    print(f"   Label: {label}")
    print(f"   Confidence: {confidence:.1%}")
    
    # Test with a known harmful example from training
    print(f"\n🧪 Testing with known harmful training example:")
    harmful_features = [0.0, -5.0, 0, 1, 0.3, 0.6, 2.0]  # Should be harmful
    harmful_scaled = scaler.transform([harmful_features])
    harmful_pred = model.predict(harmful_scaled)[0]
    harmful_prob = model.predict_proba(harmful_scaled)[0]
    harmful_label = metadata['label_encoder_classes'][harmful_pred]
    
    print(f"   Features: {harmful_features}")
    print(f"   Prediction: {harmful_label} ({max(harmful_prob):.1%})")
    
    # Analyze why K8E might be predicted as neutral
    print(f"\n🔍 Analysis:")
    print(f"   K8E has charge_change=2.0 (should indicate harmful)")
    print(f"   But also has positive BLOSUM62=1.0 (model might see as favorable)")
    print(f"   And small SASA change=-4.9 (below harmful threshold)")
    print(f"   Model might be weighing these factors differently")
    
    # Check feature importance
    if hasattr(model, 'feature_importances_'):
        print(f"\n📊 Feature importance:")
        for name, importance in zip(feature_names, model.feature_importances_):
            print(f"   {name}: {importance:.3f}")
    
    # Test what happens if we modify the features to be more clearly harmful
    print(f"\n🧪 Testing modified K8E features (more clearly harmful):")
    modified_k8e = [0.2, -15.0, -1, -2, 0.4, 0.8, 2.0]  # Enhanced harmful signals
    modified_scaled = scaler.transform([modified_k8e])
    modified_pred = model.predict(modified_scaled)[0]
    modified_prob = model.predict_proba(modified_scaled)[0]
    modified_label = metadata['label_encoder_classes'][modified_pred]
    
    print(f"   Modified features: {modified_k8e}")
    print(f"   Prediction: {modified_label} ({max(modified_prob):.1%})")

if __name__ == "__main__":
    debug_model_loading()
