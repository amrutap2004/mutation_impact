"""
Comprehensive accuracy testing with better test cases and analysis.
"""

import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))

from mutation_impact.classifier.model import HarmfulnessClassifier
from mutation_impact.features.interfaces import compute_basic_features
from mutation_impact.input_module.parser import parse_mutation
from mutation_impact.structure.retrieval import fetch_rcsb_pdb
from mutation_impact.structure.modeling import build_mutant_structure_stub


def create_better_test_cases():
    """Create test cases with more realistic mutations."""
    return [
        # Known pathogenic mutations (should be Harmful)
        {"sequence": "MVLSPADKTNVKAAW", "mutation": "K4E", "expected": "Harmful", "reason": "Charge change K->E"},
        {"sequence": "MVLSPADKTNVKAAW", "mutation": "D2N", "expected": "Harmful", "reason": "Charge change D->N"},
        {"sequence": "MVLSPADKTNVKAAW", "mutation": "A123T", "expected": "Harmful", "reason": "Large size change A->T"},
        
        # Known neutral mutations (should be Neutral)
        {"sequence": "MVLSPADKTNVKAAW", "mutation": "A1V", "expected": "Neutral", "reason": "Similar size A->V"},
        {"sequence": "MVLSPADKTNVKAAW", "mutation": "T3S", "expected": "Neutral", "reason": "Similar properties T->S"},
        {"sequence": "MVLSPADKTNVKAAW", "mutation": "V5I", "expected": "Neutral", "reason": "Similar hydrophobicity V->I"},
        
        # Edge cases
        {"sequence": "MVLSPADKTNVKAAW", "mutation": "L6I", "expected": "Neutral", "reason": "Very similar L->I"},
        {"sequence": "MVLSPADKTNVKAAW", "mutation": "P7A", "expected": "Harmful", "reason": "Proline disruption P->A"},
    ]


def test_with_analysis():
    """Test accuracy with detailed analysis."""
    print("🧬 COMPREHENSIVE MUTATION IMPACT PREDICTION TESTING")
    print("="*70)
    
    test_cases = create_better_test_cases()
    results = []
    correct = 0
    total = 0
    
    print(f"Testing {len(test_cases)} mutations with detailed analysis...\n")
    
    for i, case in enumerate(test_cases):
        print(f"Test {i+1}: {case['mutation']} - {case['reason']}")
        print(f"  Expected: {case['expected']}")
        
        try:
            # Parse mutation
            mutation = parse_mutation(case['mutation'])
            
            # Get structure
            wt_path = fetch_rcsb_pdb("1CRN")
            mut_path = build_mutant_structure_stub(wt_path, case['sequence'], mutation, force_naive=True)
            
            # Compute features
            features = compute_basic_features(case['sequence'], mutation, wt_path, mut_path)
            
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
                'features': features,
                'reason': case['reason']
            }
            results.append(result)
            
            print(f"  Predicted: {prediction['label']} (confidence: {prediction['confidence']:.3f})")
            print(f"  Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
            
            # Feature analysis
            print(f"  Key features:")
            print(f"    RMSD: {features.get('rmsd', 0):.3f}")
            print(f"    ΔSASA: {features.get('delta_sasa', 0):.3f}")
            print(f"    ΔH-bonds: {features.get('delta_hbond_count', 0):.3f}")
            print(f"    BLOSUM62: {features.get('blosum62', 0):.3f}")
            print(f"    ΔHydrophobicity: {features.get('delta_hydrophobicity', 0):.3f}")
            print()
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    
    print("="*70)
    print("ACCURACY ANALYSIS")
    print("="*70)
    print(f"Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"Correct: {correct}/{total}")
    
    # Detailed analysis
    print(f"\nDETAILED RESULTS:")
    for result in results:
        status = "✓" if result['correct'] else "✗"
        print(f"  {status} {result['mutation']}: {result['expected']} -> {result['predicted']} (conf: {result['confidence']:.3f})")
    
    # Feature analysis
    print(f"\nFEATURE ANALYSIS:")
    feature_names = ['rmsd', 'delta_sasa', 'delta_hbond_count', 'blosum62', 'delta_hydrophobicity']
    for feature in feature_names:
        values = [r['features'].get(feature, 0) for r in results]
        print(f"  {feature}: mean={np.mean(values):.3f}, std={np.std(values):.3f}, range=[{np.min(values):.3f}, {np.max(values):.3f}]")
    
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
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    if accuracy < 0.7:
        print(f"  ⚠️  Current accuracy ({accuracy:.1%}) is below 70%")
        print(f"  🔧 Consider:")
        print(f"     - Adding more structural features (minimization, better SASA)")
        print(f"     - Training ML models on larger datasets")
        print(f"     - Improving feature engineering")
        print(f"     - Using ensemble methods")
    
    if np.mean([r['features'].get('rmsd', 0) for r in results]) < 0.1:
        print(f"  🔧 All RMSD values are near zero - consider enabling minimization")
    
    if np.mean([r['features'].get('delta_sasa', 0) for r in results]) < 10:
        print(f"  🔧 SASA changes are small - consider installing freesasa for better calculations")
    
    return accuracy, results


def benchmark_against_random():
    """Compare against random baseline."""
    print(f"\n" + "="*70)
    print("BASELINE COMPARISON")
    print("="*70)
    
    test_cases = create_better_test_cases()
    
    # Random baseline
    np.random.seed(42)
    random_correct = 0
    for case in test_cases:
        random_pred = np.random.choice(["Harmful", "Neutral"])
        if random_pred == case['expected']:
            random_correct += 1
    
    random_accuracy = random_correct / len(test_cases)
    print(f"Random baseline accuracy: {random_accuracy:.3f} ({random_accuracy*100:.1f}%)")
    
    # Always predict majority class
    harmful_count = sum(1 for case in test_cases if case['expected'] == 'Harmful')
    neutral_count = len(test_cases) - harmful_count
    majority_class = 'Harmful' if harmful_count > neutral_count else 'Neutral'
    majority_correct = max(harmful_count, neutral_count)
    majority_accuracy = majority_correct / len(test_cases)
    
    print(f"Majority class ({majority_class}) accuracy: {majority_accuracy:.3f} ({majority_accuracy*100:.1f}%)")
    
    return random_accuracy, majority_accuracy


if __name__ == "__main__":
    # Run comprehensive test
    accuracy, results = test_with_analysis()
    
    # Run baseline comparison
    random_acc, majority_acc = benchmark_against_random()
    
    # Final summary
    print(f"\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"🎯 Current Pipeline Accuracy: {accuracy:.1%}")
    print(f"🎲 Random Baseline: {random_acc:.1%}")
    print(f"📊 Majority Class Baseline: {majority_acc:.1%}")
    
    if accuracy > random_acc:
        improvement = accuracy - random_acc
        print(f"✅ Pipeline is {improvement:.1%} better than random!")
    else:
        print(f"⚠️  Pipeline needs improvement to beat random baseline")
    
    if accuracy > majority_acc:
        print(f"🎉 Pipeline beats majority class baseline!")
    else:
        print(f"🔧 Pipeline should at least beat majority class baseline")
