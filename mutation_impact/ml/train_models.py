"""
Training script for high-accuracy ML models.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json

from mutation_impact.ml.data_sources import TrainingDataCollector
from mutation_impact.ml.feature_engineering import AdvancedFeatureExtractor
from mutation_impact.ml.models import MLModelTrainer
from mutation_impact.ml.validation import ModelValidator


def main():
    parser = argparse.ArgumentParser(description="Train ML models for mutation impact prediction")
    parser.add_argument("--data-dir", default="data", help="Directory for training data")
    parser.add_argument("--model-dir", default="models", help="Directory for trained models")
    parser.add_argument("--collect-data", action="store_true", help="Collect training data from sources")
    parser.add_argument("--train-models", action="store_true", help="Train ML models")
    parser.add_argument("--validate-models", action="store_true", help="Validate trained models")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    
    args = parser.parse_args()
    
    if args.all:
        args.collect_data = True
        args.train_models = True
        args.validate_models = True
    
    # Step 1: Collect training data
    if args.collect_data:
        print("=" * 60)
        print("STEP 1: Collecting training data")
        print("=" * 60)
        
        collector = TrainingDataCollector(args.data_dir)
        dataset = collector.create_training_dataset()
        
        print(f"Collected {len(dataset)} training examples")
        print(f"Features: {list(dataset.columns)}")
        print(f"Label distribution: {dataset['label'].value_counts().to_dict()}")
    
    # Step 2: Train models
    if args.train_models:
        print("\n" + "=" * 60)
        print("STEP 2: Training ML models")
        print("=" * 60)
        
        # Load training data
        data_path = Path(args.data_dir) / "cache" / "training_dataset.csv"
        if not data_path.exists():
            print("Training data not found. Run with --collect-data first.")
            return
        
        df = pd.read_csv(data_path)
        
        # Train models
        trainer = MLModelTrainer(args.model_dir)
        results = trainer.train_all_models(df)
        
        # Print results
        print("\nModel Performance:")
        for name, result in results.items():
            print(f"  {name.upper()}:")
            print(f"    CV Score: {result['cv_mean']:.3f} ± {result['cv_std']:.3f}")
            print(f"    AUC Score: {result['auc_score']:.3f}")
    
    # Step 3: Validate models
    if args.validate_models:
        print("\n" + "=" * 60)
        print("STEP 3: Validating models")
        print("=" * 60)
        
        # Load test data
        data_path = Path(args.data_dir) / "cache" / "training_dataset.csv"
        if not data_path.exists():
            print("Training data not found. Run with --collect-data first.")
            return
        
        df = pd.read_csv(data_path)
        
        # Split data for validation
        from sklearn.model_selection import train_test_split
        X = df.drop(['label', 'mutation'], axis=1)
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Load trained models
        import joblib
        models = {}
        model_dir = Path(args.model_dir)
        
        for model_file in model_dir.glob("*_model.joblib"):
            model_name = model_file.stem.replace("_model", "")
            try:
                model = joblib.load(model_file)
                models[model_name] = {'model': model}
                print(f"Loaded model: {model_name}")
            except Exception as e:
                print(f"Could not load {model_name}: {e}")
        
        if models:
            # Validate models
            validator = ModelValidator(args.model_dir)
            results = validator.comprehensive_validation(
                models, X_test.values, y_test.values, 
                X.columns.tolist(), 
                df['mutation'].tolist(), 
                ['MVLSPADKTNVKAAW'] * len(df)
            )
            
            # Print validation results
            print("\nValidation Results:")
            for name, result in results.items():
                if isinstance(result, dict) and 'accuracy' in result:
                    print(f"  {name.upper()}:")
                    print(f"    Accuracy: {result['accuracy']:.3f}")
                    print(f"    AUC Score: {result['auc_score']:.3f}")
                    print(f"    F1 Score: {result['f1_score']:.3f}")
        else:
            print("No trained models found. Run with --train-models first.")
    
    print("\n" + "=" * 60)
    print("Training pipeline completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
