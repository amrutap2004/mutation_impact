#!/usr/bin/env python3
"""
Check the actual sequence of 1CRN PDB structure.
"""

import gemmi
from mutation_impact.structure.retrieval import fetch_rcsb_pdb

def check_1crn_sequence():
    """Check the actual sequence of 1CRN."""
    
    print("🧬 CHECKING 1CRN SEQUENCE")
    print("="*60)
    
    # Get 1CRN structure
    wt_path = fetch_rcsb_pdb("1CRN")
    struct = gemmi.read_structure(str(wt_path))
    
    # Extract sequence from chain A
    chain_a = struct[0]['A']
    
    aa_map = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", 
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", 
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", 
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
    }
    
    sequence = ""
    residue_info = []
    
    for res in chain_a:
        if res.name in aa_map:
            aa = aa_map[res.name]
            sequence += aa
            residue_info.append({
                'pos': res.seqid.num,
                'name': res.name,
                'aa': aa
            })
    
    print(f"📊 1CRN Chain A Information:")
    print(f"   Total residues: {len(residue_info)}")
    print(f"   Sequence: {sequence}")
    print(f"   Length: {len(sequence)}")
    
    print(f"\n📋 Residue details (first 15):")
    for i, info in enumerate(residue_info[:15]):
        print(f"   {info['pos']:2d}: {info['name']} ({info['aa']})")
    
    print(f"\n🔍 Comparison with test input:")
    test_sequence = "MVLSPADKTNVKAAW"
    print(f"   Test input: {test_sequence}")
    print(f"   1CRN actual: {sequence}")
    print(f"   Match: {'✅' if sequence.startswith(test_sequence) else '❌'}")
    
    if not sequence.startswith(test_sequence):
        print(f"\n💡 ISSUE IDENTIFIED:")
        print(f"   The test sequence 'MVLSPADKTNVKAAW' doesn't match 1CRN!")
        print(f"   When you specify K8E, it's trying to mutate position 8 in the PDB,")
        print(f"   which is actually '{sequence[7] if len(sequence) > 7 else 'N/A'}', not 'K'.")
        
        print(f"\n🔧 SOLUTIONS:")
        print(f"   1. Use the correct 1CRN sequence: {sequence[:15]}...")
        print(f"   2. Or use a different PDB that matches your test sequence")
        print(f"   3. Or find the correct position of K in the 1CRN sequence")
        
        # Find K positions in 1CRN
        k_positions = [i+1 for i, aa in enumerate(sequence) if aa == 'K']
        if k_positions:
            print(f"   4. K residues in 1CRN are at positions: {k_positions}")
            print(f"      Try mutations like K{k_positions[0]}E instead of K8E")
        else:
            print(f"   4. No K residues found in 1CRN sequence")

if __name__ == "__main__":
    check_1crn_sequence()
