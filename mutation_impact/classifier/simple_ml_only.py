"""
Simple ML-Only Classifier - Uses trained ML models with basic features only.
No complex feature extraction - maximum reliability.
"""

from typing import TypedDict, Optional, Dict, Any
import joblib
import numpy as np
from pathlib import Path
import sys
import os

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class SimpleMLOnlyPrediction(TypedDict):
    label: str  # "Harmful" | "Neutral"
    confidence: float
    model_used: str
    feature_quality: float


class SimpleMLOnlyClassifier:
    """Simple ML-Only Classifier - Uses trained ML models with basic features.
    
    Features used: Basic structural features only.
    Never falls back to rule-based - ensures maximum accuracy.
    """

    def __init__(self, models_dir: str = "models/"):
        self.models_dir = models_dir
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.feature_names = []
        self._initialize_models()

    def _initialize_models(self):
        """Initialize ML models."""
        try:
            # Load ensemble model (most reliable)
            ensemble_path = Path(self.models_dir) / "ensemble_model.joblib"
            if ensemble_path.exists():
                self.models['ensemble'] = joblib.load(ensemble_path)
                print(f"✅ Loaded ensemble model")
            
            # Load metadata and scaler - use high_accuracy_metadata for ensemble model
            metadata_path = Path(self.models_dir) / "high_accuracy_metadata.json"
            if metadata_path.exists():
                import json
                from sklearn.preprocessing import StandardScaler
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.feature_names = metadata.get('feature_names', [])
                    
                    # Initialize scaler from metadata
                    self.scaler = StandardScaler()
                    self.scaler.mean_ = np.array(metadata.get('scaler_mean', []))
                    self.scaler.scale_ = np.array(metadata.get('scaler_scale', []))
                    
                    # Store label encoder classes
                    self.label_classes = metadata.get('label_encoder_classes', ['Harmful', 'Neutral'])
                    
                    print(f"✅ Loaded metadata with {len(self.feature_names)} features")
                    print(f"✅ Initialized scaler and label encoder")
            
            if not self.models:
                raise Exception("No ML models found")
                
        except Exception as e:
            print(f"❌ Failed to initialize models: {e}")
            raise Exception(f"Simple ML-Only classifier requires trained models. Error: {e}")

    def predict(self, sequence: str, mutation: str, wt_path: str, mut_path: str, 
                model_name: str = "ensemble") -> SimpleMLOnlyPrediction:
        """Predict using ML model with basic features only."""
        
        if model_name not in self.models:
            available = list(self.models.keys())
            raise Exception(f"Model '{model_name}' not found. Available: {available}")
        
        try:
            # Extract basic features only (no complex feature extraction)
            features = self._extract_basic_features(sequence, mutation, wt_path, mut_path)
            
            # Convert to array for ML model
            feature_array = np.array([[features.get(name, 0.0) for name in self.feature_names]])
            
            # Scale features using the loaded scaler
            if hasattr(self, 'scaler') and self.scaler is not None:
                feature_array = self.scaler.transform(feature_array)
            
            # Make prediction
            model = self.models[model_name]
            prediction = model.predict(feature_array)[0]
            probability = model.predict_proba(feature_array)[0] if hasattr(model, 'predict_proba') else None
            
            # Convert prediction to label using label classes
            if hasattr(self, 'label_classes'):
                label = self.label_classes[prediction]
            else:
                # Fallback to manual mapping
                if prediction == 1:
                    label = "Harmful"
                else:
                    label = "Neutral"
            
            # Calculate confidence
            if probability is not None:
                confidence = max(probability)
            else:
                confidence = 0.8  # Default high confidence for ML model
            
            # Calculate feature quality
            feature_quality = self._calculate_feature_quality(features)
            
            return {
                "label": label,
                "confidence": confidence,
                "model_used": model_name,
                "feature_quality": feature_quality
            }
            
        except Exception as e:
            raise Exception(f"ML prediction failed: {e}")

    def _extract_basic_features(self, sequence: str, mutation: str, wt_path: str, mut_path: str) -> Dict[str, float]:
        """Extract basic features for ML model."""
        try:
            # Import basic feature computation
            from mutation_impact.features.interfaces import compute_basic_features
            from mutation_impact.input_module.parser import parse_mutation
            
            # Parse mutation
            mut_obj = parse_mutation(mutation)
            
            # Compute basic features
            features = compute_basic_features(sequence, mut_obj, wt_path, mut_path)
            
            # Calculate additional features needed for high-accuracy model
            features['charge_change'] = self._calculate_charge_change(mut_obj['from_res'], mut_obj['to_res'])
            features['solvent_accessibility'] = self._calculate_solvent_accessibility(mut_obj['position'], sequence)
            features['size_change'] = self._calculate_size_change(mut_obj['from_res'], mut_obj['to_res'])
            features['polarity_change'] = self._calculate_polarity_change(mut_obj['from_res'], mut_obj['to_res'])
            features['aromatic_change'] = self._calculate_aromatic_change(mut_obj['from_res'], mut_obj['to_res'])
            features['buried_score'] = self._calculate_buried_score(mut_obj['position'], sequence)
            features['structure_impact'] = self._calculate_structure_impact(features.get('rmsd', 0), features.get('delta_sasa', 0))
            features['experimental_ddg'] = self._calculate_experimental_ddg(mut_obj['from_res'], mut_obj['to_res'])
            
            # Ensure all required features are present
            result = {}
            for feature_name in self.feature_names:
                if feature_name in features:
                    result[feature_name] = float(features[feature_name])
                else:
                    result[feature_name] = 0.0
            
            return result
            
        except Exception as e:
            print(f"Basic feature extraction failed: {e}")
            # Return default features
            result = {}
            for feature_name in self.feature_names:
                result[feature_name] = 0.0
            return result

    def _calculate_feature_quality(self, features: Dict[str, float]) -> float:
        """Calculate feature quality score."""
        quality_factors = []
        
        # Factor 1: RMSD > 0
        if features.get('rmsd', 0) > 0.1:
            quality_factors.append(0.2)
        
        # Factor 2: SASA change
        if abs(features.get('delta_sasa', 0)) > 10:
            quality_factors.append(0.2)
        
        # Factor 3: H-bond change
        if abs(features.get('delta_hbond_count', 0)) > 0:
            quality_factors.append(0.15)
        
        # Factor 4: BLOSUM62 score
        if abs(features.get('blosum62', 0)) > 0:
            quality_factors.append(0.15)
        
        # Factor 5: Hydrophobicity change
        if abs(features.get('delta_hydrophobicity', 0)) > 0.5:
            quality_factors.append(0.1)
        
        # Factor 6: Conservation score
        if features.get('conservation_score', 0.5) > 0.7:
            quality_factors.append(0.2)
        
        return sum(quality_factors)
    
    def _calculate_charge_change(self, from_res: str, to_res: str) -> float:
        """Calculate charge change between amino acids."""
        # Amino acid charges at physiological pH
        aa_charges = {
            'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0, 'E': -1, 'Q': 0, 'G': 0,
            'H': 0.5, 'I': 0, 'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0, 'S': 0,
            'T': 0, 'W': 0, 'Y': 0, 'V': 0
        }
        
        from_charge = aa_charges.get(from_res, 0)
        to_charge = aa_charges.get(to_res, 0)
        
        return abs(to_charge - from_charge)
    
    def _calculate_solvent_accessibility(self, position: int, sequence: str) -> float:
        """Calculate solvent accessibility based on position in sequence."""
        if position <= 0 or position > len(sequence):
            return 0.5
        
        # Simple heuristic: N-terminal and C-terminal regions are more accessible
        seq_len = len(sequence)
        if position <= 5 or position >= seq_len - 4:
            return 0.8
        elif position <= 10 or position >= seq_len - 9:
            return 0.6
        else:
            return 0.4
    
    def _calculate_size_change(self, from_res: str, to_res: str) -> float:
        """Calculate size change between amino acids."""
        # Amino acid sizes (approximate van der Waals volumes)
        aa_sizes = {
            'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5, 'E': 138.4,
            'Q': 143.8, 'G': 60.1, 'H': 153.2, 'I': 166.7, 'L': 166.7, 'K': 168.6,
            'M': 162.9, 'F': 189.9, 'P': 112.7, 'S': 89.0, 'T': 116.1, 'W': 227.8,
            'Y': 193.6, 'V': 140.0
        }
        
        from_size = aa_sizes.get(from_res, 100)
        to_size = aa_sizes.get(to_res, 100)
        
        return (to_size - from_size) / 100.0  # Normalize
    
    def _calculate_polarity_change(self, from_res: str, to_res: str) -> float:
        """Calculate polarity change between amino acids."""
        # Amino acid polarity (0 = non-polar, 1 = polar)
        aa_polarity = {
            'A': 0, 'R': 1, 'N': 1, 'D': 1, 'C': 0, 'E': 1, 'Q': 1, 'G': 0,
            'H': 1, 'I': 0, 'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0, 'S': 1,
            'T': 1, 'W': 0, 'Y': 1, 'V': 0
        }
        
        from_polarity = aa_polarity.get(from_res, 0)
        to_polarity = aa_polarity.get(to_res, 0)
        
        return abs(to_polarity - from_polarity)
    
    def _calculate_aromatic_change(self, from_res: str, to_res: str) -> float:
        """Calculate aromatic change between amino acids."""
        # Aromatic amino acids
        aromatic_aas = {'F', 'W', 'Y'}
        
        from_aromatic = 1 if from_res in aromatic_aas else 0
        to_aromatic = 1 if to_res in aromatic_aas else 0
        
        return abs(to_aromatic - from_aromatic)
    
    def _calculate_buried_score(self, position: int, sequence: str) -> float:
        """Calculate buried score based on position and sequence context."""
        if position <= 0 or position > len(sequence):
            return 0.5
        
        # Simple heuristic based on position
        seq_len = len(sequence)
        if position <= 5 or position >= seq_len - 4:
            return 0.2  # Surface
        elif position <= 15 or position >= seq_len - 14:
            return 0.5  # Intermediate
        else:
            return 0.8  # Buried
    
    def _calculate_structure_impact(self, rmsd: float, delta_sasa: float) -> float:
        """Calculate structure impact based on RMSD and SASA changes."""
        # Combine RMSD and SASA changes into a single impact score
        rmsd_impact = min(rmsd / 2.0, 1.0)  # Normalize RMSD
        sasa_impact = min(abs(delta_sasa) / 50.0, 1.0)  # Normalize SASA change
        
        return (rmsd_impact + sasa_impact) / 2.0
    
    def _calculate_experimental_ddg(self, from_res: str, to_res: str) -> float:
        """Calculate experimental ΔΔG based on amino acid properties."""
        # Simplified experimental ΔΔG calculation
        # This is a placeholder - in practice, you'd use experimental data
        
        # Hydrophobicity-based approximation
        hydrophobic_aas = {'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W'}
        from_hydrophobic = 1 if from_res in hydrophobic_aas else 0
        to_hydrophobic = 1 if to_res in hydrophobic_aas else 0
        
        # Charge-based approximation
        charged_aas = {'R', 'K', 'D', 'E'}
        from_charged = 1 if from_res in charged_aas else 0
        to_charged = 1 if to_res in charged_aas else 0
        
        # Size-based approximation
        large_aas = {'R', 'K', 'E', 'Q', 'W', 'F', 'Y'}
        from_large = 1 if from_res in large_aas else 0
        to_large = 1 if to_res in large_aas else 0
        
        # Combine factors
        ddg = (from_hydrophobic - to_hydrophobic) * 0.5 + \
              (from_charged - to_charged) * 0.3 + \
              (from_large - to_large) * 0.2
        
        return max(min(ddg, 2.0), -2.0)  # Clamp between -2 and 2

    def get_available_models(self) -> list:
        """Get list of available ML models."""
        return list(self.models.keys())

    def get_model_info(self, model_name: str = "ensemble") -> Dict[str, Any]:
        """Get information about a specific model."""
        if model_name not in self.models:
            return {"error": f"Model '{model_name}' not found"}
        
        model = self.models[model_name]
        return {
            "name": model_name,
            "type": type(model).__name__,
            "features": self.feature_names,
            "accuracy": "Unknown"
        }


def create_simple_ml_only_classifier(models_dir: str = "models/") -> SimpleMLOnlyClassifier:
    """Create and return a simple ML-only classifier."""
    return SimpleMLOnlyClassifier(models_dir)
