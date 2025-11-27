"""
High-accuracy ML models for mutation impact prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import joblib
import json

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Optional imports - handle missing dependencies gracefully
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None


class MLModelTrainer:
    """Trains and evaluates multiple ML models for mutation impact prediction."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.feature_importance = {}
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data for ML models."""
        # Separate features and labels
        feature_cols = [col for col in df.columns if col not in ['label', 'mutation', 'clinical_significance']]
        X = df[feature_cols].values
        y = df['label'].values
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y_encoded, feature_cols
    
    def train_random_forest(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Train Random Forest model with hyperparameter tuning."""
        print("Training Random Forest...")
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Grid search with cross-validation
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X, y)
        
        best_rf = grid_search.best_estimator_
        
        # Feature importance
        feature_importance = dict(zip(feature_names, best_rf.feature_importances_))
        
        return {
            'model': best_rf,
            'params': grid_search.best_params_,
            'feature_importance': feature_importance,
            'cv_score': grid_search.best_score_
        }
    
    def train_xgboost(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Train XGBoost model with hyperparameter tuning."""
        print("Training XGBoost...")
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        # Grid search with cross-validation
        xgb_model = xgb.XGBClassifier(random_state=42)
        grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X, y)
        
        best_xgb = grid_search.best_estimator_
        
        # Feature importance
        feature_importance = dict(zip(feature_names, best_xgb.feature_importances_))
        
        return {
            'model': best_xgb,
            'params': grid_search.best_params_,
            'feature_importance': feature_importance,
            'cv_score': grid_search.best_score_
        }
    
    def train_neural_network(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Train Neural Network model with hyperparameter tuning."""
        print("Training Neural Network...")
        
        # Hyperparameter grid
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
        
        # Grid search with cross-validation
        nn = MLPClassifier(random_state=42, max_iter=1000)
        grid_search = GridSearchCV(nn, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X, y)
        
        best_nn = grid_search.best_estimator_
        
        return {
            'model': best_nn,
            'params': grid_search.best_params_,
            'feature_importance': {},  # NN doesn't have direct feature importance
            'cv_score': grid_search.best_score_
        }
    
    def train_ensemble(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Train ensemble of all models."""
        print("Training ensemble model...")
        
        # Train individual models
        rf_result = self.train_random_forest(X, y, feature_names)
        xgb_result = self.train_xgboost(X, y, feature_names)
        nn_result = self.train_neural_network(X, y, feature_names)
        
        # Create ensemble
        from sklearn.ensemble import VotingClassifier
        ensemble = VotingClassifier([
            ('rf', rf_result['model']),
            ('xgb', xgb_result['model']),
            ('nn', nn_result['model'])
        ], voting='soft')
        
        ensemble.fit(X, y)
        
        return {
            'model': ensemble,
            'individual_models': {
                'random_forest': rf_result,
                'xgboost': xgb_result,
                'neural_network': nn_result
            },
            'cv_score': np.mean([rf_result['cv_score'], xgb_result['cv_score'], nn_result['cv_score']])
        }
    
    def evaluate_models(self, X: np.ndarray, y: np.ndarray, models: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all trained models."""
        results = {}
        
        for name, model_info in models.items():
            model = model_info['model']
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
            
            # Predictions
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrics
            auc_score = roc_auc_score(y, y_pred_proba) if y_pred_proba is not None else 0.0
            
            results[name] = {
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'auc_score': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
        
        return results
    
    def save_models(self, models: Dict[str, Any], feature_names: List[str]):
        """Save trained models and metadata."""
        for name, model_info in models.items():
            # Save model
            model_path = self.model_dir / f"{name}_model.joblib"
            joblib.dump(model_info['model'], model_path)
            
            # Save metadata
            metadata = {
                'feature_names': feature_names,
                'label_encoder_classes': self.label_encoder.classes_.tolist(),
                'scaler_mean': self.scaler.mean_.tolist(),
                'scaler_scale': self.scaler.scale_.tolist(),
                'cv_score': model_info.get('cv_score', 0.0),
                'feature_importance': model_info.get('feature_importance', {})
            }
            
            metadata_path = self.model_dir / f"{name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"Models saved to {self.model_dir}")
    
    def train_all_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train all models and return results."""
        print("Preparing data...")
        X, y, feature_names = self.prepare_data(df)
        
        print(f"Training data shape: {X.shape}")
        print(f"Features: {feature_names}")
        
        # Train individual models
        models = {}
        models['random_forest'] = self.train_random_forest(X, y, feature_names)
        models['xgboost'] = self.train_xgboost(X, y, feature_names)
        models['neural_network'] = self.train_neural_network(X, y, feature_names)
        
        # Train ensemble
        models['ensemble'] = self.train_ensemble(X, y, feature_names)
        
        # Evaluate models
        results = self.evaluate_models(X, y, models)
        
        # Save models
        self.save_models(models, feature_names)
        
        return results


def main():
    """Example usage of ML model training."""
    # Load training data
    df = pd.read_csv("data/cache/training_dataset.csv")
    
    # Train models
    trainer = MLModelTrainer()
    results = trainer.train_all_models(df)
    
    # Print results
    for name, result in results.items():
        print(f"\n{name.upper()}:")
        print(f"  CV Score: {result['cv_mean']:.3f} ± {result['cv_std']:.3f}")
        print(f"  AUC Score: {result['auc_score']:.3f}")


if __name__ == "__main__":
    main()
