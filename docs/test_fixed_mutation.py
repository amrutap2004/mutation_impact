"""
Test the fixed sequence-mutation combination.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from mutation_impact.input_module.parser import load_sequence, parse_mutation, validate_mutation_against_sequence
from mutation_impact.structure.retrieval import fetch_rcsb_pdb
from mutation_impact.structure.modeling import build_mutant_structure_stub
from mutation_impact.features.interfaces import compute_basic_features
from mutation_impact.classifier.model import HarmfulnessClassifier

def test_fixed_mutation():
    """Test the fixed sequence-mutation combination."""
    print("🧬 Testing Fixed Sequence-Mutation Combination")
    print("="*60)
    
    # Fixed inputs
    sequence = "MVLSPADKTNVKAAW"
    mutation_str = "S4E"
    
    print(f"Sequence: {sequence}")
    print(f"Mutation: {mutation_str}")
    print()
    
    try:
        # Parse and validate
        print("1. Parsing mutation...")
        mutation = parse_mutation(mutation_str)
        print(f"   Parsed: {mutation}")
        
        print("2. Validating against sequence...")
        validate_mutation_against_sequence(sequence, mutation)
        print("   ✅ Validation passed!")
        
        print("3. Fetching structure...")
        wt_path = fetch_rcsb_pdb("1CRN")
        print(f"   Structure: {wt_path}")
        
        print("4. Building mutant structure...")
        mut_path = build_mutant_structure_stub(wt_path, sequence, mutation, force_naive=True)
        print(f"   Mutant: {mut_path}")
        
        print("5. Computing features...")
        features = compute_basic_features(sequence, mutation, wt_path, mut_path)
        print(f"   Features: {len(features)} computed")
        
        print("6. Making prediction...")
        classifier = HarmfulnessClassifier()
        prediction = classifier.predict(features)
        print(f"   Prediction: {prediction['label']} (confidence: {prediction['confidence']:.3f})")
        
        print("\n" + "="*60)
        print("✅ SUCCESS! Fixed sequence-mutation combination works!")
        print("="*60)
        print(f"🎯 Use these values in your web interface:")
        print(f"   Sequence: {sequence}")
        print(f"   Mutation: {mutation_str}")
        print(f"   PDB ID: 1CRN")
        print(f"   Result: {prediction['label']} (confidence: {prediction['confidence']:.1%})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_fixed_mutation()
