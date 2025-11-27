#!/usr/bin/env python3
"""
Check if mutations are actually being applied to the structure files.
"""

import gemmi

def check_structure_mutations():
    """Check if mutations are correctly applied in structure files."""
    
    print("🔍 CHECKING STRUCTURE MUTATIONS")
    print("="*60)
    
    # Load structures
    wt_path = r"C:\Users\akash\.mutation_impact\1CRN.pdb"
    mut_path = r"C:\Users\akash\.mutation_impact\1CRN_K8E.pdb"
    
    try:
        # Load structures using gemmi
        wt_struct = gemmi.read_structure(wt_path)
        mut_struct = gemmi.read_structure(mut_path)
        
        print(f"📊 Structure info:")
        print(f"   WT chains: {len(wt_struct[0])}")
        print(f"   MUT chains: {len(mut_struct[0])}")
        
        # Check chain A residues
        wt_chain = wt_struct[0]['A']
        mut_chain = mut_struct[0]['A']
        
        print(f"   WT chain A residues: {len(wt_chain)}")
        print(f"   MUT chain A residues: {len(mut_chain)}")
        
        # Look for residue 8 (K8E mutation)
        print(f"\n🔍 Checking residue 8 (K8E mutation):")
        
        # Find residue 8 in both structures
        wt_res8 = None
        mut_res8 = None
        
        for res in wt_chain:
            if res.seqid.num == 8:
                wt_res8 = res
                break
        
        for res in mut_chain:
            if res.seqid.num == 8:
                mut_res8 = res
                break
        
        if wt_res8 and mut_res8:
            print(f"   WT residue 8: {wt_res8.name} (seqid: {wt_res8.seqid})")
            print(f"   MUT residue 8: {mut_res8.name} (seqid: {mut_res8.seqid})")
            
            if wt_res8.name != mut_res8.name:
                print(f"   ✅ Mutation applied: {wt_res8.name} → {mut_res8.name}")
            else:
                print(f"   ❌ No mutation detected: both are {wt_res8.name}")
        else:
            print(f"   ⚠️  Could not find residue 8 in structures")
            
            # List all residues to see what's available
            print(f"\n📋 Available residues in WT:")
            for i, res in enumerate(wt_chain):
                if i < 10:  # Show first 10
                    print(f"     {res.seqid.num}: {res.name}")
            
            print(f"\n📋 Available residues in MUT:")
            for i, res in enumerate(mut_chain):
                if i < 10:  # Show first 10
                    print(f"     {res.seqid.num}: {res.name}")
        
        # Check sequence alignment
        print(f"\n🧬 Sequence comparison:")
        wt_seq = ""
        mut_seq = ""
        
        for res in wt_chain:
            if res.name in ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]:
                aa_map = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
                wt_seq += aa_map.get(res.name, "X")
        
        for res in mut_chain:
            if res.name in ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]:
                aa_map = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
                mut_seq += aa_map.get(res.name, "X")
        
        print(f"   WT sequence:  {wt_seq}")
        print(f"   MUT sequence: {mut_seq}")
        print(f"   Expected:     MVLSPADETNVKAAW (K8E)")
        
        # Check position 8 specifically
        if len(wt_seq) >= 8 and len(mut_seq) >= 8:
            print(f"\n🎯 Position 8 check:")
            print(f"   WT pos 8: {wt_seq[7] if len(wt_seq) > 7 else 'N/A'}")
            print(f"   MUT pos 8: {mut_seq[7] if len(mut_seq) > 7 else 'N/A'}")
            print(f"   Expected: E")
            
            if len(mut_seq) > 7 and mut_seq[7] == 'E':
                print(f"   ✅ Mutation correctly applied at position 8")
            else:
                print(f"   ❌ Mutation NOT applied correctly at position 8")
        
    except Exception as e:
        print(f"❌ Error loading structures: {e}")

if __name__ == "__main__":
    check_structure_mutations()
