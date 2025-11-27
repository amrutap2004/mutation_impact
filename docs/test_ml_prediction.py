#!/usr/bin/env python3
"""
Test ML model prediction for deleterious mutations.
"""

from mutation_impact.classifier.simple_ml_only import SimpleMLOnlyClassifier
from mutation_impact.input_module.parser import load_sequence, parse_mutation
from mutation_impact.structure.retrieval import fetch_rcsb_pdb
from mutation_impact.structure.modeling import build_mutant_structure_stub

def test_ml_prediction():
    """Test ML prediction for deleterious mutation."""
    
    # Test the deleterious mutation K8E
    sequence = 'MVLSPADKTNVKAAW'
    mutation = 'K8E'
    print(f'🧬 Testing: {sequence} with mutation {mutation}')
    print('='*60)
    
    try:
        # Get structures
        print('📁 Fetching PDB structure...')
        wt_path = fetch_rcsb_pdb('1CRN')
        
        print('🔄 Building mutant structure...')
        mut_obj = parse_mutation(mutation)
        mut_path = build_mutant_structure_stub(wt_path, sequence, mut_obj, force_naive=True)
        
        # Test ML prediction
        print('🤖 Making ML prediction...')
        classifier = SimpleMLOnlyClassifier('models/')
        result = classifier.predict(sequence, mutation, str(wt_path), str(mut_path), 'ensemble')
        
        print('\n🎯 RESULTS:')
        print(f'   Prediction: {result["label"]}')
        print(f'   Confidence: {result["confidence"]:.1%}')
        print(f'   Model used: {result["model_used"]}')
        print(f'   Feature quality: {result["feature_quality"]:.1%}')
        
        # Expected: Should be "Harmful" with high confidence
        if result["label"] == "Harmful":
            print('\n✅ SUCCESS: ML model correctly predicted harmful mutation!')
        else:
            print('\n❌ ISSUE: ML model predicted neutral for clearly harmful mutation')
            
        return result
        
    except Exception as e:
        print(f'\n❌ ERROR: {e}')
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_ml_prediction()
