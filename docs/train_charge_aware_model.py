#!/usr/bin/env python3
"""
Train a charge-aware ML model that properly handles charge reversals.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json
import os

def create_charge_aware_training_data():
    """Create training data that properly handles charge changes."""
    
    # Amino acid properties
    aa_properties = {
        'A': {'charge': 0, 'hydrophobicity': 1.8, 'size': 'small'},
        'R': {'charge': 1, 'hydrophobicity': -4.5, 'size': 'large'},
        'N': {'charge': 0, 'hydrophobicity': -3.5, 'size': 'medium'},
        'D': {'charge': -1, 'hydrophobicity': -3.5, 'size': 'medium'},
        'C': {'charge': 0, 'hydrophobicity': 2.5, 'size': 'small'},
        'E': {'charge': -1, 'hydrophobicity': -3.5, 'size': 'medium'},
        'Q': {'charge': 0, 'hydrophobicity': -3.5, 'size': 'medium'},
        'G': {'charge': 0, 'hydrophobicity': -0.4, 'size': 'small'},
        'H': {'charge': 0.5, 'hydrophobicity': -3.2, 'size': 'medium'},
        'I': {'charge': 0, 'hydrophobicity': 4.5, 'size': 'large'},
        'L': {'charge': 0, 'hydrophobicity': 3.8, 'size': 'large'},
        'K': {'charge': 1, 'hydrophobicity': -3.9, 'size': 'large'},
        'M': {'charge': 0, 'hydrophobicity': 1.9, 'size': 'large'},
        'F': {'charge': 0, 'hydrophobicity': 2.8, 'size': 'large'},
        'P': {'charge': 0, 'hydrophobicity': -1.6, 'size': 'medium'},
        'S': {'charge': 0, 'hydrophobicity': -0.8, 'size': 'small'},
        'T': {'charge': 0, 'hydrophobicity': -0.7, 'size': 'medium'},
        'W': {'charge': 0, 'hydrophobicity': -0.9, 'size': 'large'},
        'Y': {'charge': 0, 'hydrophobicity': -1.3, 'size': 'large'},
        'V': {'charge': 0, 'hydrophobicity': 4.2, 'size': 'medium'},
    }
    
    training_examples = []
    
    # HARMFUL mutations (should be predicted as Harmful)
    harmful_mutations = [
        # Charge reversals (most harmful)
        {'from_aa': 'K', 'to_aa': 'E', 'reason': 'Charge reversal +1 to -1'},
        {'from_aa': 'K', 'to_aa': 'D', 'reason': 'Charge reversal +1 to -1'},
        {'from_aa': 'R', 'to_aa': 'E', 'reason': 'Charge reversal +1 to -1'},
        {'from_aa': 'R', 'to_aa': 'D', 'reason': 'Charge reversal +1 to -1'},
        {'from_aa': 'E', 'to_aa': 'K', 'reason': 'Charge reversal -1 to +1'},
        {'from_aa': 'E', 'to_aa': 'R', 'reason': 'Charge reversal -1 to +1'},
        {'from_aa': 'D', 'to_aa': 'K', 'reason': 'Charge reversal -1 to +1'},
        {'from_aa': 'D', 'to_aa': 'R', 'reason': 'Charge reversal -1 to +1'},
        
        # Large size changes
        {'from_aa': 'A', 'to_aa': 'W', 'reason': 'Small to large size change'},
        {'from_aa': 'G', 'to_aa': 'F', 'reason': 'Small to large size change'},
        {'from_aa': 'S', 'to_aa': 'W', 'reason': 'Small to large size change'},
        
        # Proline disruptions
        {'from_aa': 'P', 'to_aa': 'A', 'reason': 'Proline disruption'},
        {'from_aa': 'P', 'to_aa': 'G', 'reason': 'Proline disruption'},
        
        # Hydrophobicity changes
        {'from_aa': 'S', 'to_aa': 'F', 'reason': 'Hydrophobic to hydrophilic'},
        {'from_aa': 'T', 'to_aa': 'W', 'reason': 'Hydrophobic to hydrophilic'},
    ]
    
    # NEUTRAL mutations (should be predicted as Neutral)
    neutral_mutations = [
        # Conservative changes
        {'from_aa': 'A', 'to_aa': 'V', 'reason': 'Similar size and properties'},
        {'from_aa': 'V', 'to_aa': 'I', 'reason': 'Similar hydrophobicity'},
        {'from_aa': 'L', 'to_aa': 'I', 'reason': 'Similar hydrophobicity'},
        {'from_aa': 'S', 'to_aa': 'T', 'reason': 'Similar properties'},
        {'from_aa': 'N', 'to_aa': 'Q', 'reason': 'Similar properties'},
        {'from_aa': 'D', 'to_aa': 'E', 'reason': 'Same charge, similar properties'},
        {'from_aa': 'K', 'to_aa': 'R', 'reason': 'Same charge, similar properties'},
        {'from_aa': 'F', 'to_aa': 'Y', 'reason': 'Similar aromatic properties'},
        
        # Synonymous-like changes
        {'from_aa': 'A', 'to_aa': 'G', 'reason': 'Both small and flexible'},
        {'from_aa': 'V', 'to_aa': 'L', 'reason': 'Both hydrophobic'},
    ]
    
    # Generate training examples
    for mut in harmful_mutations:
        from_aa = mut['from_aa']
        to_aa = mut['to_aa']
        
        # Calculate features
        charge_change = abs(aa_properties[to_aa]['charge'] - aa_properties[from_aa]['charge'])
        hydro_change = abs(aa_properties[to_aa]['hydrophobicity'] - aa_properties[from_aa]['hydrophobicity'])
        
        # Simulate realistic structural features for harmful mutations
        example = {
            'rmsd': np.random.normal(0.5, 0.2),  # Higher RMSD for harmful
            'delta_sasa': np.random.normal(-20, 10),  # Larger SASA changes
            'delta_hbond_count': np.random.choice([-2, -1, 0, 1, 2]),
            'blosum62': np.random.normal(-2, 1),  # More negative BLOSUM for harmful
            'delta_hydrophobicity': hydro_change,
            'conservation_score': np.random.uniform(0.6, 0.9),  # Higher conservation
            'charge_change': charge_change,  # NEW FEATURE: Charge change magnitude
            'label': 'Harmful'
        }
        training_examples.append(example)
    
    for mut in neutral_mutations:
        from_aa = mut['from_aa']
        to_aa = mut['to_aa']
        
        # Calculate features
        charge_change = abs(aa_properties[to_aa]['charge'] - aa_properties[from_aa]['charge'])
        hydro_change = abs(aa_properties[to_aa]['hydrophobicity'] - aa_properties[from_aa]['hydrophobicity'])
        
        # Simulate realistic structural features for neutral mutations
        example = {
            'rmsd': np.random.normal(0.1, 0.05),  # Lower RMSD for neutral
            'delta_sasa': np.random.normal(-5, 5),  # Smaller SASA changes
            'delta_hbond_count': np.random.choice([-1, 0, 1]),
            'blosum62': np.random.normal(1, 0.5),  # More positive BLOSUM for neutral
            'delta_hydrophobicity': hydro_change,
            'conservation_score': np.random.uniform(0.3, 0.7),  # Lower conservation
            'charge_change': charge_change,  # NEW FEATURE: Charge change magnitude
            'label': 'Neutral'
        }
        training_examples.append(example)
    
    return training_examples

def train_charge_aware_model():
    """Train ML model with charge awareness."""
    print("🧬 Training Charge-Aware ML Model")
    print("="*60)
    
    # Create training data
    training_data = create_charge_aware_training_data()
    df = pd.DataFrame(training_data)
    
    print(f"Created {len(training_data)} training examples")
    print(f"  Harmful: {len(df[df['label'] == 'Harmful'])}")
    print(f"  Neutral: {len(df[df['label'] == 'Neutral'])}")
    
    # Prepare features and labels
    feature_cols = ['rmsd', 'delta_sasa', 'delta_hbond_count', 'blosum62', 
                   'delta_hydrophobicity', 'conservation_score', 'charge_change']
    X = df[feature_cols].values
    y = df['label'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    
    # Train Random Forest with charge awareness
    print("\nTraining Charge-Aware Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.3f}")
    
    # Cross-validation
    cv_scores = cross_val_score(rf, X_scaled, y_encoded, cv=5)
    print(f"Cross-validation: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Feature importance
    print("\nFeature Importance:")
    for feature, importance in zip(feature_cols, rf.feature_importances_):
        print(f"  {feature}: {importance:.3f}")
    
    # Save model and metadata
    os.makedirs("models", exist_ok=True)
    
    # Save model
    model_path = "models/ensemble_model.joblib"
    joblib.dump(rf, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save metadata
    metadata = {
        'feature_names': feature_cols,
        'label_encoder_classes': label_encoder.classes_.tolist(),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'feature_importance': dict(zip(feature_cols, rf.feature_importances_))
    }
    
    metadata_path = "models/ensemble_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")
    
    return rf, scaler, label_encoder, feature_cols

def test_charge_aware_model():
    """Test the charge-aware model."""
    print("\n" + "="*60)
    print("🧪 TESTING CHARGE-AWARE MODEL")
    print("="*60)
    
    # Test cases
    test_cases = [
        {'features': [0.0, -4.9, 0, 1, 0.4, 0.5, 2.0], 'expected': 'Harmful', 'description': 'K8E (charge reversal)'},
        {'features': [0.1, -2.0, 0, 0, 0.1, 0.4, 0.0], 'expected': 'Neutral', 'description': 'A13V (conservative)'},
        {'features': [0.3, -15, -1, -2, 2.5, 0.8, 1.0], 'expected': 'Harmful', 'description': 'Large change'},
    ]
    
    # Load model
    model = joblib.load("models/ensemble_model.joblib")
    
    with open("models/ensemble_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    scaler = StandardScaler()
    scaler.mean_ = np.array(metadata['scaler_mean'])
    scaler.scale_ = np.array(metadata['scaler_scale'])
    
    print("Test Results:")
    for i, case in enumerate(test_cases):
        features = np.array([case['features']])
        features_scaled = scaler.transform(features)
        
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        confidence = max(probability)
        
        label = metadata['label_encoder_classes'][prediction]
        status = "✅" if label == case['expected'] else "❌"
        
        print(f"  {i+1}. {case['description']}")
        print(f"     Predicted: {label} ({confidence:.1%}) {status}")
        print(f"     Expected: {case['expected']}")
        print()

if __name__ == "__main__":
    train_charge_aware_model()
    test_charge_aware_model()
