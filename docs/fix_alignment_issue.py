"""
Fix sequence-to-structure alignment issues.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from mutation_impact.input_module.parser import load_sequence, parse_mutation, validate_mutation_against_sequence
from mutation_impact.structure.retrieval import fetch_rcsb_pdb
from mutation_impact.structure.modeling import build_mutant_structure_stub
from mutation_impact.features.interfaces import compute_basic_features
from mutation_impact.classifier.model import HarmfulnessClassifier

def test_with_force_naive():
    """Test with force_naive option to bypass alignment issues."""
    print("🔧 Testing with Force Naive Option")
    print("="*60)
    
    # Input values
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
        
        print("4. Building mutant structure with force_naive=True...")
        print("   (This bypasses sequence-to-structure alignment issues)")
        mut_path = build_mutant_structure_stub(wt_path, sequence, mutation, force_naive=True)
        print(f"   ✅ Mutant structure created: {mut_path}")
        
        print("5. Computing features...")
        features = compute_basic_features(sequence, mutation, wt_path, mut_path)
        print(f"   ✅ Features computed: {len(features)} features")
        
        print("6. Making prediction...")
        classifier = HarmfulnessClassifier()
        prediction = classifier.predict(features)
        print(f"   ✅ Prediction: {prediction['label']} (confidence: {prediction['confidence']:.3f})")
        
        print("\n" + "="*60)
        print("✅ SUCCESS! Force naive option works!")
        print("="*60)
        print(f"🎯 Web Interface Settings:")
        print(f"   Sequence: {sequence}")
        print(f"   Mutation: {mutation_str}")
        print(f"   PDB ID: 1CRN")
        print(f"   ✅ Force naive mapping: ENABLED")
        print(f"   Result: {prediction['label']} (confidence: {prediction['confidence']:.1%})")
        
        # Show feature details
        print(f"\n📊 Feature Details:")
        print(f"   RMSD: {features.get('rmsd', 0):.3f} Å")
        print(f"   ΔSASA: {features.get('delta_sasa', 0):.1f} Å²")
        print(f"   ΔH-bonds: {features.get('delta_hbond_count', 0)}")
        print(f"   BLOSUM62: {features.get('blosum62', 0)}")
        print(f"   ΔHydrophobicity: {features.get('delta_hydrophobicity', 0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def explain_force_naive():
    """Explain what force_naive does."""
    print("\n" + "="*60)
    print("🔍 What is Force Naive Mapping?")
    print("="*60)
    print("Force naive mapping bypasses sequence-to-structure alignment")
    print("and uses a simple 1-to-1 mapping based on sequence position.")
    print()
    print("When to use:")
    print("  ✅ When sequence alignment fails")
    print("  ✅ When structure has gaps or missing residues")
    print("  ✅ For quick testing and prototyping")
    print("  ✅ When you trust the sequence position mapping")
    print()
    print("Limitations:")
    print("  ⚠️  May not work if structure is very different from sequence")
    print("  ⚠️  Assumes 1-to-1 correspondence between sequence and structure")
    print("  ⚠️  May not be accurate for structures with large insertions/deletions")
    print()
    print("For production use:")
    print("  🔧 Consider using aligned sequences")
    print("  🔧 Use structure-specific sequence mapping")
    print("  🔧 Validate results with known mutations")

if __name__ == "__main__":
    success = test_with_force_naive()
    if success:
        explain_force_naive()
