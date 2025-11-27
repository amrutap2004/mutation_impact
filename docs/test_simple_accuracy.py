"""
Simple accuracy test for the current pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from mutation_impact.classifier.model import HarmfulnessClassifier
from mutation_impact.features.interfaces import compute_basic_features
from mutation_impact.input_module.parser import parse_mutation
from mutation_impact.structure.retrieval import fetch_rcsb_pdb
from mutation_impact.structure.modeling import build_mutant_structure_stub


def test_simple_accuracy():
    """Test accuracy with simple test cases."""
    print("🧬 Testing Mutation Impact Prediction Accuracy")
    print("="*50)
    
    # Test cases with expected outcomes
    test_cases = [
        {"sequence": "MVLSPADKTNVKAAW", "mutation": "A123T", "expected": "Harmful"},
        {"sequence": "MVLSPADKTNVKAAW", "mutation": "K4E", "expected": "Harmful"}, 
        {"sequence": "MVLSPADKTNVKAAW", "mutation": "D2N", "expected": "Harmful"},
        {"sequence": "MVLSPADKTNVKAAW", "mutation": "A1V", "expected": "Neutral"},
        {"sequence": "MVLSPADKTNVKAAW", "mutation": "T3S", "expected": "Neutral"},
    ]
    
    correct = 0
    total = 0
    
    print(f"Testing {len(test_cases)} mutations...")
    
    for i, case in enumerate(test_cases):
        print(f"\nTest {i+1}: {case['mutation']}")
        
        try:
            # Parse mutation
            mutation = parse_mutation(case['mutation'])
            print(f"  Parsed: {mutation}")
            
            # Get structure
            print(f"  Fetching structure...")
            wt_path = fetch_rcsb_pdb("1CRN")
            mut_path = build_mutant_structure_stub(wt_path, case['sequence'], mutation, force_naive=True)
            print(f"  Structure created: {mut_path}")
            
            # Compute features
            print(f"  Computing features...")
            features = compute_basic_features(case['sequence'], mutation, wt_path, mut_path)
            print(f"  Features: {features}")
            
            # Make prediction
            classifier = HarmfulnessClassifier()
            prediction = classifier.predict(features)
            
            # Check result
            is_correct = prediction['label'] == case['expected']
            if is_correct:
                correct += 1
            total += 1
            
            print(f"  Expected: {case['expected']}")
            print(f"  Predicted: {prediction['label']} (confidence: {prediction['confidence']:.3f})")
            print(f"  Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    print(f"\n" + "="*50)
    print(f"ACCURACY RESULTS:")
    print(f"  Correct: {correct}/{total}")
    print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    if accuracy >= 0.8:
        print(f"  🎉 EXCELLENT accuracy!")
    elif accuracy >= 0.6:
        print(f"  ✅ GOOD accuracy!")
    else:
        print(f"  ⚠️  Needs improvement")
    
    return accuracy


if __name__ == "__main__":
    test_simple_accuracy()
