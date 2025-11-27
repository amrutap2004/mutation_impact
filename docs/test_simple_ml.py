"""
Simple ML accuracy test without complex training.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))

from mutation_impact.classifier.model import HarmfulnessClassifier
from mutation_impact.features.interfaces import compute_basic_features


def test_improved_classifier():
    """Test with improved feature weights and thresholds."""
    print("🧬 TESTING IMPROVED CLASSIFIER ACCURACY")
    print("="*60)
    
    # Test cases with expected outcomes
    test_cases = [
        {"mutation": "K4E", "expected": "Harmful", "features": {"rmsd": 1.2, "delta_sasa": 150, "delta_hbonds": -2, "blosum62": -1, "delta_hydrophobicity": 3.5, "conservation_score": 0.9}},
        {"mutation": "D2N", "expected": "Harmful", "features": {"rmsd": 0.8, "delta_sasa": 100, "delta_hbonds": -1, "blosum62": 1, "delta_hydrophobicity": 0.0, "conservation_score": 0.85}},
        {"mutation": "P7A", "expected": "Harmful", "features": {"rmsd": 2.1, "delta_sasa": 200, "delta_hbonds": -3, "blosum62": -1, "delta_hydrophobicity": 3.4, "conservation_score": 0.95}},
        {"mutation": "A1V", "expected": "Neutral", "features": {"rmsd": 0.1, "delta_sasa": 20, "delta_hbonds": 0, "blosum62": 0, "delta_hydrophobicity": 2.4, "conservation_score": 0.3}},
        {"mutation": "T3S", "expected": "Neutral", "features": {"rmsd": 0.2, "delta_sasa": 15, "delta_hbonds": 0, "blosum62": 1, "delta_hydrophobicity": -0.1, "conservation_score": 0.4}},
        {"mutation": "V5I", "expected": "Neutral", "features": {"rmsd": 0.1, "delta_sasa": 10, "delta_hbonds": 0, "blosum62": 3, "delta_hydrophobicity": 0.3, "conservation_score": 0.35}},
        {"mutation": "L6I", "expected": "Neutral", "features": {"rmsd": 0.2, "delta_sasa": 25, "delta_hbonds": 0, "blosum62": 2, "delta_hydrophobicity": 0.7, "conservation_score": 0.4}},
    ]
    
    correct = 0
    total = 0
    results = []
    
    print(f"Testing {len(test_cases)} mutations with realistic features...\n")
    
    for i, case in enumerate(test_cases):
        print(f"Test {i+1}: {case['mutation']}")
        print(f"  Expected: {case['expected']}")
        
        try:
            # Use provided features
            features = case['features']
            
            # Make prediction
            classifier = HarmfulnessClassifier()
            prediction = classifier.predict(features)
            
            # Check result
            is_correct = prediction['label'] == case['expected']
            if is_correct:
                correct += 1
            total += 1
            
            result = {
                'mutation': case['mutation'],
                'expected': case['expected'],
                'predicted': prediction['label'],
                'confidence': prediction['confidence'],
                'correct': is_correct,
                'features': features
            }
            results.append(result)
            
            print(f"  Predicted: {prediction['label']} (confidence: {prediction['confidence']:.3f})")
            print(f"  Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
            
            # Show key features
            print(f"  Key features: RMSD={features['rmsd']:.1f}, ΔSASA={features['delta_sasa']:.0f}, ΔH-bonds={features['delta_hbonds']:.0f}")
            print()
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    
    print("="*60)
    print("ACCURACY RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"Correct: {correct}/{total}")
    
    # Detailed results
    print(f"\nDETAILED RESULTS:")
    for result in results:
        status = "✓" if result['correct'] else "✗"
        print(f"  {status} {result['mutation']}: {result['expected']} -> {result['predicted']} (conf: {result['confidence']:.3f})")
    
    # Feature analysis
    print(f"\nFEATURE ANALYSIS:")
    feature_names = ['rmsd', 'delta_sasa', 'delta_hbonds', 'blosum62', 'delta_hydrophobicity', 'conservation_score']
    for feature in feature_names:
        values = [r['features'][feature] for r in results]
        print(f"  {feature}: mean={np.mean(values):.3f}, std={np.std(values):.3f}")
    
    # Confusion matrix
    print(f"\nCONFUSION MATRIX:")
    true_harmful = sum(1 for r in results if r['expected'] == 'Harmful')
    true_neutral = sum(1 for r in results if r['expected'] == 'Neutral')
    pred_harmful = sum(1 for r in results if r['predicted'] == 'Harmful')
    pred_neutral = sum(1 for r in results if r['predicted'] == 'Neutral')
    
    correct_harmful = sum(1 for r in results if r['expected'] == 'Harmful' and r['predicted'] == 'Harmful')
    correct_neutral = sum(1 for r in results if r['expected'] == 'Neutral' and r['predicted'] == 'Neutral')
    
    print(f"  True Harmful: {true_harmful}, Predicted Harmful: {pred_harmful}")
    print(f"  True Neutral: {true_neutral}, Predicted Neutral: {pred_neutral}")
    print(f"  Correct Harmful: {correct_harmful}")
    print(f"  Correct Neutral: {correct_neutral}")
    
    # Precision and Recall
    precision_harmful = correct_harmful / pred_harmful if pred_harmful > 0 else 0
    recall_harmful = correct_harmful / true_harmful if true_harmful > 0 else 0
    precision_neutral = correct_neutral / pred_neutral if pred_neutral > 0 else 0
    recall_neutral = correct_neutral / true_neutral if true_neutral > 0 else 0
    
    print(f"\nPRECISION & RECALL:")
    print(f"  Harmful - Precision: {precision_harmful:.3f}, Recall: {recall_harmful:.3f}")
    print(f"  Neutral - Precision: {precision_neutral:.3f}, Recall: {recall_neutral:.3f}")
    
    # F1 scores
    f1_harmful = 2 * (precision_harmful * recall_harmful) / (precision_harmful + recall_harmful) if (precision_harmful + recall_harmful) > 0 else 0
    f1_neutral = 2 * (precision_neutral * recall_neutral) / (precision_neutral + recall_neutral) if (precision_neutral + recall_neutral) > 0 else 0
    
    print(f"\nF1 SCORES:")
    print(f"  Harmful: {f1_harmful:.3f}")
    print(f"  Neutral: {f1_neutral:.3f}")
    print(f"  Macro F1: {(f1_harmful + f1_neutral) / 2:.3f}")
    
    return accuracy, results


def test_feature_importance():
    """Test which features are most important."""
    print(f"\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Test with different feature combinations
    base_features = {"rmsd": 0.0, "delta_sasa": 0.0, "delta_hbonds": 0.0, "blosum62": 0.0, "delta_hydrophobicity": 0.0, "conservation_score": 0.5}
    
    feature_tests = [
        ("RMSD", {"rmsd": 2.0}),
        ("ΔSASA", {"delta_sasa": 200}),
        ("ΔH-bonds", {"delta_hbonds": -3}),
        ("BLOSUM62", {"blosum62": -3}),
        ("ΔHydrophobicity", {"delta_hydrophobicity": 4.0}),
        ("Conservation", {"conservation_score": 0.9}),
    ]
    
    classifier = HarmfulnessClassifier()
    
    print("Testing individual feature importance:")
    for name, feature_delta in feature_tests:
        test_features = base_features.copy()
        test_features.update(feature_delta)
        
        prediction = classifier.predict(test_features)
        print(f"  {name:15}: {prediction['label']:8} (confidence: {prediction['confidence']:.3f})")


def benchmark_against_baselines():
    """Compare against various baselines."""
    print(f"\n" + "="*60)
    print("BASELINE COMPARISON")
    print("="*60)
    
    # Random baseline
    np.random.seed(42)
    random_correct = 0
    test_cases = 7  # Same as our test
    for _ in range(test_cases):
        random_pred = np.random.choice(["Harmful", "Neutral"])
        # Assume 3 harmful, 4 neutral in our test
        expected = "Harmful" if random_correct < 3 else "Neutral"
        if random_pred == expected:
            random_correct += 1
    
    random_accuracy = random_correct / test_cases
    print(f"Random baseline: {random_accuracy:.3f} ({random_accuracy*100:.1f}%)")
    
    # Always predict majority class
    majority_accuracy = 4/7  # 4 neutral out of 7
    print(f"Majority class (Neutral): {majority_accuracy:.3f} ({majority_accuracy*100:.1f}%)")
    
    # Simple rule: if RMSD > 1.0, then Harmful
    rule_accuracy = 0.0  # Would need to test with actual RMSD values
    print(f"Simple RMSD rule: {rule_accuracy:.3f} (not tested)")
    
    return random_accuracy, majority_accuracy


if __name__ == "__main__":
    # Test improved classifier
    accuracy, results = test_improved_classifier()
    
    # Test feature importance
    test_feature_importance()
    
    # Benchmark against baselines
    random_acc, majority_acc = benchmark_against_baselines()
    
    # Final summary
    print(f"\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"🎯 Improved Classifier Accuracy: {accuracy:.1%}")
    print(f"🎲 Random Baseline: {random_acc:.1%}")
    print(f"📊 Majority Class Baseline: {majority_acc:.1%}")
    
    if accuracy > random_acc:
        improvement = accuracy - random_acc
        print(f"✅ Classifier is {improvement:.1%} better than random!")
    else:
        print(f"⚠️  Classifier needs improvement to beat random baseline")
    
    if accuracy > majority_acc:
        print(f"🎉 Classifier beats majority class baseline!")
    else:
        print(f"🔧 Classifier should at least beat majority class baseline")
    
    print(f"\n💡 Key insights:")
    print(f"   - Current accuracy: {accuracy:.1%}")
    print(f"   - Best improvement: Use realistic structural features")
    print(f"   - Next steps: Train ML models on larger datasets")
    print(f"   - Goal: Achieve >80% accuracy with real experimental data")
