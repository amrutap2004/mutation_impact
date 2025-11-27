"""
Comprehensive testing script for ML prediction accuracy.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Add the project to Python path
sys.path.insert(0, str(Path(__file__).parent))

from mutation_impact.ml.data_sources import TrainingDataCollector
from mutation_impact.ml.feature_engineering import AdvancedFeatureExtractor
from mutation_impact.ml.models import MLModelTrainer
from mutation_impact.ml.validation import ModelValidator
from mutation_impact.ml.pipeline import ProductionMLPipeline
from mutation_impact.input_module.parser import load_sequence, parse_mutation
from mutation_impact.structure.retrieval import fetch_rcsb_pdb
from mutation_impact.structure.modeling import build_mutant_structure_stub
from mutation_impact.features.interfaces import compute_basic_features


def create_test_dataset():
    """Create a comprehensive test dataset with known outcomes."""
    print("Creating test dataset...")
    
    # Create test cases with expected outcomes
    test_cases = [
        # Known pathogenic mutations
        {"sequence": "MVLSPADKTNVKAAW", "mutation": "A123T", "expected": "Harmful", "pdb_id": "1CRN"},
        {"sequence": "MVLSPADKTNVKAAW", "mutation": "K4E", "expected": "Harmful", "pdb_id": "1CRN"},
        {"sequence": "MVLSPADKTNVKAAW", "mutation": "D2N", "expected": "Harmful", "pdb_id": "1CRN"},
        
        # Known neutral mutations
        {"sequence": "MVLSPADKTNVKAAW", "mutation": "A1V", "expected": "Neutral", "pdb_id": "1CRN"},
        {"sequence": "MVLSPADKTNVKAAW", "mutation": "T3S", "expected": "Neutral", "pdb_id": "1CRN"},
        {"sequence": "MVLSPADKTNVKAAW", "mutation": "V5I", "expected": "Neutral", "pdb_id": "1CRN"},
    ]
    
    return test_cases


def test_rule_based_accuracy():
    """Test the current rule-based classifier accuracy."""
    print("\n" + "="*60)
    print("TESTING RULE-BASED CLASSIFIER ACCURACY")
    print("="*60)
    
    from mutation_impact.classifier.model import HarmfulnessClassifier
    from mutation_impact.features.interfaces import compute_basic_features
    
    test_cases = create_test_dataset()
    correct_predictions = 0
    total_predictions = 0
    
    results = []
    
    for i, case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {case['mutation']} in {case['sequence'][:10]}...")
        
        try:
            # Parse mutation
            mutation = parse_mutation(case['mutation'])
            
            # Get structure (use a small test PDB)
            wt_path = fetch_rcsb_pdb(case['pdb_id'])
            mut_path = build_mutant_structure_stub(wt_path, case['sequence'], mutation, force_naive=True)
            
            # Compute features
            features = compute_basic_features(case['sequence'], mutation, wt_path, mut_path)
            
            # Make prediction
            classifier = HarmfulnessClassifier()
            prediction = classifier.predict(features)
            
            # Check accuracy
            is_correct = prediction['label'] == case['expected']
            if is_correct:
                correct_predictions += 1
            
            total_predictions += 1
            
            result = {
                'mutation': case['mutation'],
                'expected': case['expected'],
                'predicted': prediction['label'],
                'confidence': prediction['confidence'],
                'correct': is_correct,
                'features': features
            }
            results.append(result)
            
            print(f"  Expected: {case['expected']}")
            print(f"  Predicted: {prediction['label']} (confidence: {prediction['confidence']:.3f})")
            print(f"  Correct: {'✓' if is_correct else '✗'}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\nRule-based Classifier Results:")
    print(f"  Accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
    print(f"  Correct predictions: {correct_predictions}")
    print(f"  Total predictions: {total_predictions}")
    
    return results, accuracy


def test_ml_pipeline_accuracy():
    """Test ML pipeline accuracy if models are available."""
    print("\n" + "="*60)
    print("TESTING ML PIPELINE ACCURACY")
    print("="*60)
    
    # Check if models exist
    model_dir = Path("models")
    if not model_dir.exists() or not list(model_dir.glob("*_model.joblib")):
        print("No trained ML models found. Training models first...")
        train_ml_models()
    
    try:
        # Initialize ML pipeline
        pipeline = ProductionMLPipeline("models/")
        
        test_cases = create_test_dataset()
        correct_predictions = 0
        total_predictions = 0
        
        results = []
        
        for i, case in enumerate(test_cases):
            print(f"\nTest Case {i+1}: {case['mutation']} in {case['sequence'][:10]}...")
            
            try:
                # Parse mutation
                mutation = parse_mutation(case['mutation'])
                
                # Get structure
                wt_path = fetch_rcsb_pdb(case['pdb_id'])
                mut_path = build_mutant_structure_stub(wt_path, case['sequence'], mutation, force_naive=True)
                
                # Make ML prediction
                result = pipeline.predict_single_mutation(
                    sequence=case['sequence'],
                    mutation=case['mutation'],
                    wt_path=wt_path,
                    mut_path=mut_path,
                    model_name="ensemble"
                )
                
                # Check accuracy
                is_correct = result['prediction'] == case['expected']
                if is_correct:
                    correct_predictions += 1
                
                total_predictions += 1
                
                print(f"  Expected: {case['expected']}")
                print(f"  Predicted: {result['prediction']} (confidence: {result['confidence']:.3f})")
                print(f"  Model: {result['model_used']}")
                print(f"  Correct: {'✓' if is_correct else '✗'}")
                
                results.append({
                    'mutation': case['mutation'],
                    'expected': case['expected'],
                    'predicted': result['prediction'],
                    'confidence': result['confidence'],
                    'model': result['model_used'],
                    'correct': is_correct
                })
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        # Calculate accuracy
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"\nML Pipeline Results:")
        print(f"  Accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
        print(f"  Correct predictions: {correct_predictions}")
        print(f"  Total predictions: {total_predictions}")
        
        return results, accuracy
        
    except Exception as e:
        print(f"ML pipeline test failed: {e}")
        return [], 0.0


def train_ml_models():
    """Train ML models for testing."""
    print("\n" + "="*60)
    print("TRAINING ML MODELS")
    print("="*60)
    
    try:
        # Create training data
        collector = TrainingDataCollector("data")
        dataset = collector.create_training_dataset()
        
        # Train models
        trainer = MLModelTrainer("models")
        results = trainer.train_all_models(dataset)
        
        print("ML models trained successfully!")
        return True
        
    except Exception as e:
        print(f"Failed to train ML models: {e}")
        return False


def benchmark_against_baseline():
    """Benchmark against simple baseline methods."""
    print("\n" + "="*60)
    print("BENCHMARKING AGAINST BASELINE METHODS")
    print("="*60)
    
    test_cases = create_test_dataset()
    
    # Simple baseline: random prediction
    random_correct = 0
    for case in test_cases:
        random_pred = np.random.choice(["Harmful", "Neutral"])
        if random_pred == case['expected']:
            random_correct += 1
    
    random_accuracy = random_correct / len(test_cases)
    print(f"Random baseline accuracy: {random_accuracy:.3f}")
    
    # Simple baseline: always predict "Harmful"
    harmful_correct = sum(1 for case in test_cases if case['expected'] == "Harmful")
    harmful_accuracy = harmful_correct / len(test_cases)
    print(f"Always 'Harmful' baseline accuracy: {harmful_accuracy:.3f}")
    
    return {
        'random': random_accuracy,
        'always_harmful': harmful_accuracy
    }


def main():
    """Run comprehensive accuracy testing."""
    print("🧬 MUTATION IMPACT PREDICTION ACCURACY TESTING")
    print("="*60)
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Test rule-based classifier
    rule_results, rule_accuracy = test_rule_based_accuracy()
    
    # Test ML pipeline
    ml_results, ml_accuracy = test_ml_pipeline_accuracy()
    
    # Benchmark against baselines
    baseline_results = benchmark_against_baseline()
    
    # Summary
    print("\n" + "="*60)
    print("ACCURACY TESTING SUMMARY")
    print("="*60)
    print(f"Rule-based Classifier: {rule_accuracy:.3f}")
    print(f"ML Pipeline: {ml_accuracy:.3f}")
    print(f"Random Baseline: {baseline_results['random']:.3f}")
    print(f"Always 'Harmful' Baseline: {baseline_results['always_harmful']:.3f}")
    
    # Improvement analysis
    if ml_accuracy > rule_accuracy:
        improvement = ml_accuracy - rule_accuracy
        print(f"\n🎉 ML Pipeline shows {improvement:.3f} improvement over rule-based!")
    else:
        print(f"\n⚠️  ML Pipeline needs more training data or better features")
    
    # Feature analysis
    if rule_results:
        print(f"\n📊 FEATURE ANALYSIS:")
        feature_names = list(rule_results[0]['features'].keys())
        for feature in feature_names:
            values = [r['features'][feature] for r in rule_results]
            print(f"  {feature}: mean={np.mean(values):.3f}, std={np.std(values):.3f}")
    
    print(f"\n✅ Testing completed!")


if __name__ == "__main__":
    main()
