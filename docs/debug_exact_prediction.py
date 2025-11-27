#!/usr/bin/env python3
"""
Debug the exact prediction process to see why K8E is still predicted as Neutral.
"""

import numpy as np
import joblib
import json
from sklearn.preprocessing import StandardScaler

def debug_exact_prediction():
    """Debug the exact prediction process."""
    
    print("🔍 DEBUGGING EXACT K8E PREDICTION")
    print("="*60)
    
    # Load the model and metadata
    model = joblib.load("models/ensemble_model.joblib")
    
    with open("models/ensemble_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    print("📊 Model Info:")
    print(f"   Features: {metadata['feature_names']}")
    print(f"   Classes: {metadata['label_encoder_classes']}")
    print(f"   Accuracy: {metadata['accuracy']}")
    
    # Recreate scaler
    scaler = StandardScaler()
    scaler.mean_ = np.array(metadata['scaler_mean'])
    scaler.scale_ = np.array(metadata['scaler_scale'])
    
    print(f"\n📏 Scaler Info:")
    print(f"   Mean: {scaler.mean_}")
    print(f"   Scale: {scaler.scale_}")
    
    # Test with actual K8E features
    k8e_features = [0.0, -4.887125836769883, 0.0, 1.0, 0.3999999999999999, 0.5, 2.0]
    feature_names = metadata['feature_names']
    
    print(f"\n🧬 K8E Features:")
    for name, value in zip(feature_names, k8e_features):
        print(f"   {name}: {value}")
    
    # Scale features
    features_array = np.array([k8e_features])
    features_scaled = scaler.transform(features_array)
    
    print(f"\n📐 Scaled Features:")
    for name, value in zip(feature_names, features_scaled[0]):
        print(f"   {name}: {value:.3f}")
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    print(f"\n🎯 Prediction Results:")
    print(f"   Raw prediction: {prediction}")
    print(f"   Probabilities: {probabilities}")
    print(f"   Class 0 (Neutral): {probabilities[0]:.3f}")
    print(f"   Class 1 (Harmful): {probabilities[1]:.3f}")
    
    label = metadata['label_encoder_classes'][prediction]
    confidence = max(probabilities)
    
    print(f"\n📋 Final Result:")
    print(f"   Label: {label}")
    print(f"   Confidence: {confidence:.1%}")
    
    # Analyze decision path
    print(f"\n🌳 Decision Analysis:")
    print(f"   Model type: {type(model).__name__}")
    
    if hasattr(model, 'feature_importances_'):
        print(f"   Feature Importances:")
        for name, importance in zip(feature_names, model.feature_importances_):
            print(f"     {name}: {importance:.3f}")
    
    # Test with training examples to see if they work
    print(f"\n🧪 Testing Training Examples:")
    
    # Test case that should be harmful
    harmful_example = [0.0, -5.0, 0, 1, 0.3, 0.6, 2.0]  # From training data
    harmful_scaled = scaler.transform([harmful_example])
    harmful_pred = model.predict(harmful_scaled)[0]
    harmful_prob = model.predict_proba(harmful_scaled)[0]
    harmful_label = metadata['label_encoder_classes'][harmful_pred]
    
    print(f"   Training Harmful Example:")
    print(f"     Features: {harmful_example}")
    print(f"     Prediction: {harmful_label} ({max(harmful_prob):.1%})")
    
    # Test case that should be neutral
    neutral_example = [0.1, -2.0, 0, 0, 0.1, 0.4, 0.0]  # From training data
    neutral_scaled = scaler.transform([neutral_example])
    neutral_pred = model.predict(neutral_scaled)[0]
    neutral_prob = model.predict_proba(neutral_scaled)[0]
    neutral_label = metadata['label_encoder_classes'][neutral_pred]
    
    print(f"   Training Neutral Example:")
    print(f"     Features: {neutral_example}")
    print(f"     Prediction: {neutral_label} ({max(neutral_prob):.1%})")
    
    # Compare K8E to training examples
    print(f"\n🔍 K8E vs Training Examples:")
    print(f"   K8E charge_change: {k8e_features[6]} (should indicate harmful)")
    print(f"   K8E blosum62: {k8e_features[3]} (positive, might confuse model)")
    print(f"   K8E delta_sasa: {k8e_features[1]} (small change)")
    print(f"   K8E rmsd: {k8e_features[0]} (no structural change)")
    
    print(f"\n💡 Analysis:")
    print(f"   The model might be influenced by:")
    print(f"   - Positive BLOSUM62 score (1.0) suggesting favorable substitution")
    print(f"   - Small SASA change (-4.9) below typical harmful threshold")
    print(f"   - Zero RMSD suggesting no structural disruption")
    print(f"   - Even though charge_change=2.0 indicates charge reversal")

if __name__ == "__main__":
    debug_exact_prediction()
