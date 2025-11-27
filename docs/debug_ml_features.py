#!/usr/bin/env python3
"""
Debug ML features to understand why predictions are incorrect.
"""

from mutation_impact.classifier.simple_ml_only import SimpleMLOnlyClassifier
from mutation_impact.input_module.parser import load_sequence, parse_mutation
from mutation_impact.structure.retrieval import fetch_rcsb_pdb
from mutation_impact.structure.modeling import build_mutant_structure_stub
from mutation_impact.features.interfaces import compute_basic_features

def debug_ml_features():
    """Debug ML features for deleterious mutation."""
    
    # Test the deleterious mutation K8E
    sequence = 'MVLSPADKTNVKAAW'
    mutation = 'K8E'
    print(f'🧬 Debugging features for: {sequence} with mutation {mutation}')
    print('='*60)
    
    try:
        # Get structures
        print('📁 Fetching PDB structure...')
        wt_path = fetch_rcsb_pdb('1CRN')
        
        print('🔄 Building mutant structure...')
        mut_obj = parse_mutation(mutation)
        mut_path = build_mutant_structure_stub(wt_path, sequence, mut_obj, force_naive=True)
        
        # Compute features manually
        print('🔬 Computing features...')
        features = compute_basic_features(sequence, mut_obj, wt_path, mut_path)
        
        print('\n📊 COMPUTED FEATURES:')
        for key, value in features.items():
            print(f'   {key}: {value}')
        
        # Load model metadata to see expected features
        import json
        with open('models/ensemble_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print('\n🎯 EXPECTED FEATURES:')
        for feature in metadata['feature_names']:
            value = features.get(feature, 'MISSING')
            print(f'   {feature}: {value}')
        
        # Test ML prediction with debug
        print('\n🤖 ML PREDICTION DEBUG:')
        classifier = SimpleMLOnlyClassifier('models/')
        
        # Get the features that the ML model will use
        ml_features = classifier._extract_basic_features(sequence, mutation, str(wt_path), str(mut_path))
        
        print('🔍 ML MODEL FEATURES:')
        for key, value in ml_features.items():
            print(f'   {key}: {value}')
        
        # Make prediction
        result = classifier.predict(sequence, mutation, str(wt_path), str(mut_path), 'ensemble')
        
        print(f'\n🎯 FINAL PREDICTION:')
        print(f'   Label: {result["label"]}')
        print(f'   Confidence: {result["confidence"]:.1%}')
        
        # Analyze why it might be wrong
        print(f'\n🔍 ANALYSIS:')
        print(f'   RMSD: {ml_features.get("rmsd", 0)} (>0.1 suggests structural change)')
        print(f'   ΔSASA: {ml_features.get("delta_sasa", 0)} (>10 suggests surface change)')
        print(f'   ΔH-bonds: {ml_features.get("delta_hbond_count", 0)} (≠0 suggests H-bond change)')
        print(f'   BLOSUM62: {ml_features.get("blosum62", 0)} (<0 suggests unfavorable)')
        print(f'   ΔHydrophobicity: {ml_features.get("delta_hydrophobicity", 0)} (large values suggest change)')
        print(f'   Conservation: {ml_features.get("conservation_score", 0)} (>0.7 suggests conserved)')
        
        return result
        
    except Exception as e:
        print(f'\n❌ ERROR: {e}')
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    debug_ml_features()
