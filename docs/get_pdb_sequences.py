#!/usr/bin/env python3
"""
Get sequences for different PDB structures for testing.
"""

import gemmi
from mutation_impact.structure.retrieval import fetch_rcsb_pdb

def get_pdb_sequence(pdb_id):
    """Get the sequence for a PDB structure."""
    
    print(f"🧬 Getting sequence for {pdb_id}")
    
    try:
        # Fetch PDB structure
        pdb_path = fetch_rcsb_pdb(pdb_id)
        struct = gemmi.read_structure(str(pdb_path))
        
        # Get first chain
        chain = struct[0][0]  # First model, first chain
        
        aa_map = {
            "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", 
            "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", 
            "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", 
            "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
        }
        
        sequence = ""
        for res in chain:
            if res.name in aa_map:
                sequence += aa_map[res.name]
        
        print(f"   Chain: {chain.name}")
        print(f"   Length: {len(sequence)} residues")
        print(f"   Sequence: {sequence}")
        
        # Find some interesting residues for mutations
        mutations = []
        
        # Find K residues (for K->E mutations)
        k_positions = [i+1 for i, aa in enumerate(sequence) if aa == 'K']
        if k_positions:
            mutations.append(f"K{k_positions[0]}E")
        
        # Find first residue for size change
        if sequence:
            mutations.append(f"{sequence[0]}1W")
        
        # Find P residues for proline disruption
        p_positions = [i+1 for i, aa in enumerate(sequence) if aa == 'P']
        if p_positions:
            mutations.append(f"P{p_positions[0]}A")
        
        # Conservative change - find A residues
        a_positions = [i+1 for i, aa in enumerate(sequence) if aa == 'A']
        if a_positions:
            mutations.append(f"A{a_positions[0]}V")
        
        print(f"   Suggested mutations: {', '.join(mutations[:3])}")
        
        return {
            'pdb_id': pdb_id,
            'chain': chain.name,
            'sequence': sequence,
            'length': len(sequence),
            'mutations': mutations[:3]
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def main():
    """Get sequences for multiple PDB structures."""
    
    print("🧪 GETTING PDB SEQUENCES FOR TESTING")
    print("="*60)
    
    # List of interesting PDB structures
    pdb_list = [
        "1CRN",  # Crambin (small protein)
        "1UBQ",  # Ubiquitin (regulatory protein)
        "1LYZ",  # Lysozyme (enzyme)
        "1MBO",  # Myoglobin (oxygen binding)
        "1ZNI",  # Insulin (hormone)
    ]
    
    results = []
    
    for pdb_id in pdb_list:
        result = get_pdb_sequence(pdb_id)
        if result:
            results.append(result)
        print()
    
    # Generate test cases
    print("🎯 GENERATED TEST CASES:")
    print("="*60)
    
    for i, result in enumerate(results):
        print(f"\n📋 Test Case {i+1}: {result['pdb_id']}")
        print(f"```")
        print(f"Sequence: {result['sequence']}")
        print(f"Mutation: {result['mutations'][0] if result['mutations'] else 'N/A'}")
        print(f"ID: {result['pdb_id']}")
        print(f"Expected: Harmful")
        print(f"```")

if __name__ == "__main__":
    main()
