"""
Production ML pipeline for high-accuracy mutation impact prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import joblib
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder

from mutation_impact.ml.feature_engineering import AdvancedFeatureExtractor
from mutation_impact.ml.models import MLModelTrainer
from mutation_impact.ml.validation import ModelValidator


class ProductionMLPipeline:
    """Production-ready ML pipeline for mutation impact prediction."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.feature_extractor = AdvancedFeatureExtractor()
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.feature_names = []
        self.model_metadata = {}
        
        # Load trained models
        self._load_models()
    
    def _load_models(self):
        """Load trained models and metadata."""
        print("Loading trained models...")
        
        model_files = list(self.model_dir.glob("*_model.joblib"))
        for model_file in model_files:
            model_name = model_file.stem.replace("_model", "")
            
            # Try to load model with error handling for missing dependencies
            try:
                model = joblib.load(model_file)
                self.models[model_name] = model
            except (ImportError, ModuleNotFoundError) as e:
                # Skip models that require unavailable dependencies
                error_msg = str(e).lower()
                if 'lightgbm' in error_msg or 'lgb' in error_msg:
                    print(f"⚠️  Warning: Skipping {model_name} model (lightgbm not installed). Install with: pip install lightgbm")
                    continue
                elif 'xgboost' in error_msg or 'xgb' in error_msg:
                    print(f"⚠️  Warning: Skipping {model_name} model (xgboost not installed). Install with: pip install xgboost")
                    continue
                else:
                    print(f"⚠️  Warning: Skipping {model_name} model due to missing dependency: {e}")
                    continue
            except Exception as e:
                # Handle other loading errors
                print(f"⚠️  Warning: Failed to load {model_name} model: {e}")
                continue
            
            # Load metadata
            metadata_file = self.model_dir / f"{model_name}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.model_metadata[model_name] = metadata
                    
                    # Set scaler and label encoder from first model
                    if self.scaler is None:
                        self.scaler = StandardScaler()
                        self.scaler.mean_ = np.array(metadata['scaler_mean'])
                        self.scaler.scale_ = np.array(metadata['scaler_scale'])
                        
                        self.label_encoder = LabelEncoder()
                        self.label_encoder.classes_ = np.array(metadata['label_encoder_classes'])
                        
                        self.feature_names = metadata['feature_names']
        
        print(f"Loaded {len(self.models)} models: {list(self.models.keys())}")
    
    def predict_single_mutation(self, sequence: str, mutation: str, wt_path: str, mut_path: str, 
                              model_name: str = "ensemble") -> Dict[str, Any]:
        """Predict impact of a single mutation."""
        print(f"Predicting impact for mutation {mutation}...")
        
        # Extract features
        features = self.feature_extractor.extract_all_features(sequence, mutation, wt_path, mut_path)
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([features])
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in feature_df.columns:
                feature_df[feature] = 0.0
        
        # Reorder columns to match training data
        feature_df = feature_df[self.feature_names]
        
        # Scale features
        X = self.scaler.transform(feature_df.values)
        
        # Get model
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None
        
        # Decode prediction
        prediction_label = self.label_encoder.inverse_transform([prediction])[0]
        
        # Calculate confidence
        confidence = max(probability) if probability is not None else 0.5
        
        # Get feature importance if available
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_names, model.feature_importances_))
        
        return {
            'mutation': mutation,
            'prediction': prediction_label,
            'confidence': confidence,
            'probabilities': dict(zip(self.label_encoder.classes_, probability)) if probability is not None else {},
            'features': features,
            'feature_importance': feature_importance,
            'model_used': model_name
        }
    
    def predict_batch(self, mutations: List[Dict[str, Any]], model_name: str = "ensemble") -> List[Dict[str, Any]]:
        """Predict impact for multiple mutations."""
        print(f"Predicting impact for {len(mutations)} mutations...")
        
        results = []
        for mutation_data in mutations:
            try:
                result = self.predict_single_mutation(
                    mutation_data['sequence'],
                    mutation_data['mutation'],
                    mutation_data['wt_path'],
                    mutation_data['mut_path'],
                    model_name
                )
                results.append(result)
            except Exception as e:
                print(f"Error predicting {mutation_data['mutation']}: {e}")
                results.append({
                    'mutation': mutation_data['mutation'],
                    'error': str(e),
                    'prediction': 'Unknown',
                    'confidence': 0.0
                })
        
        return results
    
    def ensemble_prediction(self, sequence: str, mutation: str, wt_path: str, mut_path: str) -> Dict[str, Any]:
        """Get ensemble prediction from all available models."""
        print("Getting ensemble prediction...")
        
        predictions = {}
        confidences = {}
        
        for model_name, model in self.models.items():
            try:
                result = self.predict_single_mutation(sequence, mutation, wt_path, mut_path, model_name)
                predictions[model_name] = result['prediction']
                confidences[model_name] = result['confidence']
            except Exception as e:
                print(f"Error with model {model_name}: {e}")
                predictions[model_name] = 'Unknown'
                confidences[model_name] = 0.0
        
        # Calculate ensemble statistics
        harmful_count = sum(1 for pred in predictions.values() if pred == 'Harmful')
        total_models = len(predictions)
        ensemble_confidence = np.mean(list(confidences.values()))
        
        # Majority vote
        ensemble_prediction = 'Harmful' if harmful_count > total_models / 2 else 'Neutral'
        
        return {
            'mutation': mutation,
            'ensemble_prediction': ensemble_prediction,
            'ensemble_confidence': ensemble_confidence,
            'individual_predictions': predictions,
            'individual_confidences': confidences,
            'harmful_votes': harmful_count,
            'total_models': total_models
        }
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all loaded models."""
        performance = {}
        
        for model_name, metadata in self.model_metadata.items():
            performance[model_name] = {
                'cv_score': metadata.get('cv_score', 0.0),
                'feature_importance': metadata.get('feature_importance', {}),
                'model_type': type(self.models[model_name]).__name__
            }
        
        return performance
    
    def retrain_models(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """Retrain models with new data."""
        print("Retraining models with new data...")
        
        # Train new models
        trainer = MLModelTrainer(self.model_dir)
        results = trainer.train_all_models(new_data)
        
        # Reload models
        self._load_models()
        
        return results
    
    def validate_models(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate models against test data."""
        print("Validating models...")
        
        # Prepare test data
        X_test = test_data.drop(['label', 'mutation'], axis=1).values
        y_test = test_data['label'].values
        
        # Validate each model
        validator = ModelValidator(self.model_dir)
        results = {}
        
        for model_name, model in self.models.items():
            try:
                result = validator.validate_against_experimental_data(model, X_test, y_test)
                results[model_name] = result
            except Exception as e:
                print(f"Error validating {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results


def main():
    """Example usage of production ML pipeline."""
    # Initialize pipeline
    pipeline = ProductionMLPipeline()
    
    # Example prediction
    sequence = "MVLSPADKTNVKAAW"
    mutation = "A123T"
    wt_path = "wt.pdb"
    mut_path = "mut.pdb"
    
    # Single prediction
    result = pipeline.predict_single_mutation(sequence, mutation, wt_path, mut_path)
    print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")
    
    # Ensemble prediction
    ensemble_result = pipeline.ensemble_prediction(sequence, mutation, wt_path, mut_path)
    print(f"Ensemble prediction: {ensemble_result['ensemble_prediction']} (confidence: {ensemble_result['ensemble_confidence']:.3f})")
    
    # Model performance
    performance = pipeline.get_model_performance()
    for model_name, perf in performance.items():
        print(f"{model_name}: CV Score = {perf['cv_score']:.3f}")


if __name__ == "__main__":
    main()
