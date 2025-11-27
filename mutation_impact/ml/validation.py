"""
Model validation and benchmarking against existing tools.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import requests
import json
import subprocess
import tempfile

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns


class ModelValidator:
    """Validates ML models against experimental data and existing tools."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.benchmark_results = {}
    
    def validate_against_experimental_data(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Validate model against experimental data."""
        print("Validating against experimental data...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrics
        accuracy = np.mean(y_pred == y_test)
        auc_score = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0.0
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score']
        }
    
    def benchmark_against_sift(self, mutations: List[str], sequences: List[str]) -> Dict[str, Any]:
        """Benchmark against SIFT predictions."""
        print("Benchmarking against SIFT...")
        
        # This is a placeholder - real implementation would query SIFT API
        sift_predictions = []
        for mutation in mutations:
            # Simulate SIFT prediction
            sift_score = np.random.uniform(0, 1)
            sift_prediction = "deleterious" if sift_score < 0.05 else "tolerated"
            sift_predictions.append({
                'mutation': mutation,
                'sift_score': sift_score,
                'sift_prediction': sift_prediction
            })
        
        return {
            'predictions': sift_predictions,
            'method': 'SIFT',
            'description': 'Sorting Intolerant From Tolerant'
        }
    
    def benchmark_against_polyphen2(self, mutations: List[str], sequences: List[str]) -> Dict[str, Any]:
        """Benchmark against PolyPhen-2 predictions."""
        print("Benchmarking against PolyPhen-2...")
        
        # This is a placeholder - real implementation would query PolyPhen-2 API
        polyphen_predictions = []
        for mutation in mutations:
            # Simulate PolyPhen-2 prediction
            polyphen_score = np.random.uniform(0, 1)
            if polyphen_score > 0.9:
                polyphen_prediction = "probably_damaging"
            elif polyphen_score > 0.5:
                polyphen_prediction = "possibly_damaging"
            else:
                polyphen_prediction = "benign"
            
            polyphen_predictions.append({
                'mutation': mutation,
                'polyphen_score': polyphen_score,
                'polyphen_prediction': polyphen_prediction
            })
        
        return {
            'predictions': polyphen_predictions,
            'method': 'PolyPhen-2',
            'description': 'Polymorphism Phenotyping v2'
        }
    
    def benchmark_against_cadd(self, mutations: List[str], sequences: List[str]) -> Dict[str, Any]:
        """Benchmark against CADD predictions."""
        print("Benchmarking against CADD...")
        
        # This is a placeholder - real implementation would query CADD API
        cadd_predictions = []
        for mutation in mutations:
            # Simulate CADD prediction
            cadd_score = np.random.uniform(0, 50)  # CADD scores typically 0-50
            cadd_prediction = "deleterious" if cadd_score > 15 else "tolerated"
            
            cadd_predictions.append({
                'mutation': mutation,
                'cadd_score': cadd_score,
                'cadd_prediction': cadd_prediction
            })
        
        return {
            'predictions': cadd_predictions,
            'method': 'CADD',
            'description': 'Combined Annotation Dependent Depletion'
        }
    
    def cross_validation_analysis(self, model, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict[str, Any]:
        """Perform comprehensive cross-validation analysis."""
        print(f"Performing {cv_folds}-fold cross-validation...")
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
        
        # Detailed analysis for each fold
        fold_results = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]
            
            # Train on fold
            model.fit(X_train_fold, y_train_fold)
            
            # Predict on test fold
            y_pred = model.predict(X_test_fold)
            y_pred_proba = model.predict_proba(X_test_fold)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrics for this fold
            fold_auc = roc_auc_score(y_test_fold, y_pred_proba) if y_pred_proba is not None else 0.0
            fold_accuracy = np.mean(y_pred == y_test_fold)
            
            fold_results.append({
                'fold': fold + 1,
                'auc_score': fold_auc,
                'accuracy': fold_accuracy,
                'test_size': len(test_idx)
            })
        
        return {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'fold_results': fold_results
        }
    
    def plot_roc_curves(self, models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray, save_path: str = "roc_curves.png"):
        """Plot ROC curves for model comparison."""
        plt.figure(figsize=(10, 8))
        
        for name, model_info in models.items():
            model = model_info['model']
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            if y_pred_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROC curves saved to {save_path}")
    
    def plot_feature_importance(self, models: Dict[str, Any], feature_names: List[str], save_path: str = "feature_importance.png"):
        """Plot feature importance for tree-based models."""
        tree_models = {name: info for name, info in models.items() 
                      if hasattr(info['model'], 'feature_importances_')}
        
        if not tree_models:
            print("No tree-based models found for feature importance analysis")
            return
        
        fig, axes = plt.subplots(1, len(tree_models), figsize=(5 * len(tree_models), 6))
        if len(tree_models) == 1:
            axes = [axes]
        
        for i, (name, model_info) in enumerate(tree_models.items()):
            model = model_info['model']
            importance = model.feature_importances_
            
            # Sort features by importance
            sorted_idx = np.argsort(importance)[::-1]
            top_features = sorted_idx[:10]  # Top 10 features
            
            axes[i].barh(range(len(top_features)), importance[top_features])
            axes[i].set_yticks(range(len(top_features)))
            axes[i].set_yticklabels([feature_names[j] for j in top_features])
            axes[i].set_xlabel('Feature Importance')
            axes[i].set_title(f'{name} - Top 10 Features')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance plots saved to {save_path}")
    
    def comprehensive_validation(self, models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray, 
                              feature_names: List[str], mutations: List[str], sequences: List[str]) -> Dict[str, Any]:
        """Perform comprehensive validation of all models."""
        print("Starting comprehensive validation...")
        
        results = {}
        
        # 1. Experimental data validation
        for name, model_info in models.items():
            model = model_info['model']
            results[name] = self.validate_against_experimental_data(model, X_test, y_test)
        
        # 2. Cross-validation analysis
        for name, model_info in models.items():
            model = model_info['model']
            cv_results = self.cross_validation_analysis(model, X_test, y_test)
            results[name]['cv_analysis'] = cv_results
        
        # 3. Benchmark against existing tools
        benchmark_results = {}
        benchmark_results['sift'] = self.benchmark_against_sift(mutations, sequences)
        benchmark_results['polyphen2'] = self.benchmark_against_polyphen2(mutations, sequences)
        benchmark_results['cadd'] = self.benchmark_against_cadd(mutations, sequences)
        
        results['benchmark_comparison'] = benchmark_results
        
        # 4. Generate plots
        self.plot_roc_curves(models, X_test, y_test)
        self.plot_feature_importance(models, feature_names)
        
        return results


def main():
    """Example usage of model validation."""
    # Load test data
    df = pd.read_csv("data/cache/training_dataset.csv")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X = df.drop(['label', 'mutation'], axis=1).values
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load trained models
    import joblib
    models = {}
    for model_name in ['random_forest', 'xgboost', 'ensemble']:
        model_path = f"models/{model_name}_model.joblib"
        if Path(model_path).exists():
            models[model_name] = {'model': joblib.load(model_path)}
    
    # Validate models
    validator = ModelValidator()
    results = validator.comprehensive_validation(
        models, X_test, y_test, 
        df.columns.tolist(), 
        df['mutation'].tolist(), 
        ['MVLSPADKTNVKAAW'] * len(df)
    )
    
    # Print results
    for name, result in results.items():
        if isinstance(result, dict) and 'accuracy' in result:
            print(f"\n{name.upper()}:")
            print(f"  Accuracy: {result['accuracy']:.3f}")
            print(f"  AUC Score: {result['auc_score']:.3f}")
            print(f"  F1 Score: {result['f1_score']:.3f}")


if __name__ == "__main__":
    main()
