"""
ML-Only Classifier - Forces use of trained ML models only.
No rule-based fallback - maximum accuracy through ML models.
"""

from typing import TypedDict, Optional, Dict, Any
import joblib
import numpy as np
from pathlib import Path
import sys
import os

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mutation_impact.ml.pipeline import ProductionMLPipeline


class MLOnlyPrediction(TypedDict):
    label: str  # "Harmful" | "Neutral"
    confidence: float
    model_used: str
    feature_quality: float
    confidence_factors: list


class MLOnlyClassifier:
    """ML-Only Classifier - Uses trained ML models exclusively.
    
    Features used: All available features from ML pipeline.
    Never falls back to rule-based - ensures maximum accuracy.
    """

    def __init__(self, models_dir: str = "models/"):
        self.models_dir = models_dir
        self.ml_pipeline = None
        self.available_models = []
        self._initialize_ml_pipeline()

    def _initialize_ml_pipeline(self):
        """Initialize ML pipeline with all available models."""
        try:
            self.ml_pipeline = ProductionMLPipeline(self.models_dir)
            if self.ml_pipeline.models:
                self.available_models = list(self.ml_pipeline.models.keys())
                print(f"✅ ML-Only Classifier initialized with {len(self.available_models)} models:")
                for model_name in self.available_models:
                    print(f"   - {model_name}")
            else:
                raise Exception("No ML models found")
        except Exception as e:
            print(f"❌ Failed to initialize ML pipeline: {e}")
            raise Exception(f"ML-Only classifier requires trained models. Error: {e}")

    def predict(self, sequence: str, mutation: str, wt_path: str, mut_path: str, 
                model_name: str = "ensemble") -> MLOnlyPrediction:
        """Predict using ML model exclusively - no rule-based fallback."""
        
        if not self.ml_pipeline or not self.ml_pipeline.models:
            raise Exception("No ML models available. Train models first using create_better_ml_model.py")
        
        if model_name not in self.ml_pipeline.models:
            available = list(self.ml_pipeline.models.keys())
            raise Exception(f"Model '{model_name}' not found. Available: {available}")
        
        try:
            # Use ML pipeline for prediction
            ml_result = self.ml_pipeline.predict_single_mutation(
                sequence, mutation, wt_path, mut_path, model_name
            )
            
            # Enhanced confidence scoring
            confidence_factors = self._calculate_confidence_factors(ml_result)
            feature_quality = len(confidence_factors) / 6.0  # 6 possible factors
            
            # Enhanced confidence
            base_confidence = ml_result.get('confidence', 0.5)
            enhancement = sum(confidence_factors)
            enhanced_confidence = min(0.95, base_confidence + enhancement)
            
            return {
                "label": ml_result['prediction'],
                "confidence": enhanced_confidence,
                "model_used": ml_result.get('model_used', model_name),
                "feature_quality": feature_quality,
                "confidence_factors": confidence_factors
            }
            
        except Exception as e:
            raise Exception(f"ML prediction failed: {e}. Ensure models are trained and available.")

    def _calculate_confidence_factors(self, ml_result: Dict[str, Any]) -> list:
        """Calculate confidence enhancement factors."""
        factors = []
        
        # Get features from ML result if available
        features = ml_result.get('features', {})
        
        # Factor 1: Structural change detected
        if features.get('rmsd', 0) > 0.1:
            factors.append(0.2)
        
        # Factor 2: SASA change detected  
        if abs(features.get('delta_sasa', 0)) > 10:
            factors.append(0.2)
        
        # Factor 3: H-bond change detected
        if abs(features.get('delta_hbond_count', 0)) > 0:
            factors.append(0.15)
        
        # Factor 4: Evolutionary score available
        if abs(features.get('blosum62', 0)) > 0:
            factors.append(0.15)
        
        # Factor 5: Hydrophobicity change
        if abs(features.get('delta_hydrophobicity', 0)) > 0.5:
            factors.append(0.1)
        
        # Factor 6: High conservation
        if features.get('conservation_score', 0.5) > 0.7:
            factors.append(0.2)
        
        return factors

    def get_available_models(self) -> list:
        """Get list of available ML models."""
        return self.available_models.copy()

    def get_model_info(self, model_name: str = "ensemble") -> Dict[str, Any]:
        """Get information about a specific model."""
        if not self.ml_pipeline or model_name not in self.ml_pipeline.models:
            return {"error": f"Model '{model_name}' not found"}
        
        model = self.ml_pipeline.models[model_name]
        return {
            "name": model_name,
            "type": type(model).__name__,
            "features": getattr(model, 'feature_names_in_', []),
            "accuracy": getattr(model, 'accuracy', 'Unknown'),
            "cv_score": getattr(model, 'cv_score', 'Unknown')
        }


def create_ml_only_classifier(models_dir: str = "models/") -> MLOnlyClassifier:
    """Create and return an ML-only classifier."""
    return MLOnlyClassifier(models_dir)
