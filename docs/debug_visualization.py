#!/usr/bin/env python3
"""
Debug the 3D visualization to see why it shows the same structures.
"""

import os
from pathlib import Path
from mutation_impact.input_module.parser import load_sequence, parse_mutation
from mutation_impact.structure.retrieval import fetch_rcsb_pdb
from mutation_impact.structure.modeling import build_mutant_structure_stub
from mutation_impact.features.interfaces import compute_basic_features
from mutation_impact.reporting.report import render_html_report

def debug_visualization():
    """Debug the 3D visualization issue."""
    
    print("🔍 DEBUGGING 3D VISUALIZATION")
    print("="*60)
    
    # Test different mutations to see if structures are different
    test_cases = [
        {"sequence": "MVLSPADKTNVKAAW", "mutation": "K8E", "description": "K→E charge change"},
        {"sequence": "MVLSPADKTNVKAAW", "mutation": "A13W", "description": "A→W size change"},
        {"sequence": "MVLSPADKTNVKAAW", "mutation": "S4F", "description": "S→F hydrophobicity change"},
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n🧬 Test Case {i+1}: {case['description']}")
        print(f"   Mutation: {case['mutation']}")
        
        try:
            # Parse mutation and get structures
            sequence = case["sequence"]
            mutation = parse_mutation(case["mutation"])
            
            # Get wild-type structure
            wt_path = fetch_rcsb_pdb("1CRN")
            print(f"   WT structure: {wt_path}")
            
            # Build mutant structure
            mut_path = build_mutant_structure_stub(wt_path, sequence, mutation, force_naive=True)
            print(f"   Mutant structure: {mut_path}")
            
            # Check if files exist and are different
            if os.path.exists(wt_path) and os.path.exists(mut_path):
                wt_size = os.path.getsize(wt_path)
                mut_size = os.path.getsize(mut_path)
                
                print(f"   WT file size: {wt_size} bytes")
                print(f"   Mutant file size: {mut_size} bytes")
                
                # Read first few lines to check if they're different
                with open(wt_path, 'r') as f:
                    wt_lines = f.readlines()[:10]
                
                with open(mut_path, 'r') as f:
                    mut_lines = f.readlines()[:10]
                
                # Check if structures are actually different
                differences = 0
                for j, (wt_line, mut_line) in enumerate(zip(wt_lines, mut_lines)):
                    if wt_line != mut_line:
                        differences += 1
                        if differences <= 3:  # Show first 3 differences
                            print(f"   Diff line {j+1}:")
                            print(f"     WT:  {wt_line.strip()}")
                            print(f"     MUT: {mut_line.strip()}")
                
                if differences == 0:
                    print(f"   ❌ ISSUE: WT and mutant structures are identical!")
                else:
                    print(f"   ✅ Structures are different ({differences} line differences)")
                
                # Check specific residue in the structure
                print(f"   🔍 Checking residue {mutation['position']} in structures...")
                
                # Look for the mutated residue in both files
                wt_residue_lines = [line for line in wt_lines if f"A   {mutation['position']:>3}" in line and "ATOM" in line]
                mut_residue_lines = [line for line in mut_lines if f"A   {mutation['position']:>3}" in line and "ATOM" in line]
                
                if wt_residue_lines and mut_residue_lines:
                    print(f"   WT residue {mutation['position']}: {wt_residue_lines[0].strip()}")
                    print(f"   MUT residue {mutation['position']}: {mut_residue_lines[0].strip()}")
                    
                    # Check if residue type changed
                    wt_res_type = wt_residue_lines[0][17:20].strip()
                    mut_res_type = mut_residue_lines[0][17:20].strip()
                    
                    expected_mut_type = {"K": "LYS", "E": "GLU", "A": "ALA", "W": "TRP", "S": "SER", "F": "PHE"}.get(mutation['to_res'], mutation['to_res'])
                    
                    print(f"   WT residue type: {wt_res_type}")
                    print(f"   MUT residue type: {mut_res_type}")
                    print(f"   Expected MUT type: {expected_mut_type}")
                    
                    if mut_res_type == expected_mut_type:
                        print(f"   ✅ Residue type correctly changed")
                    else:
                        print(f"   ❌ Residue type NOT changed correctly")
                else:
                    print(f"   ⚠️  Could not find residue {mutation['position']} in structure files")
                
            else:
                print(f"   ❌ Structure files not found")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n🎯 SUMMARY:")
    print(f"If structures are identical, the issue is in structure generation.")
    print(f"If structures are different but visualization is same, the issue is in the web viewer.")

def test_visualization_generation():
    """Test the visualization generation process."""
    
    print(f"\n" + "="*60)
    print("🌐 TESTING VISUALIZATION GENERATION")
    print("="*60)
    
    # Test with K8E mutation
    sequence = "MVLSPADKTNVKAAW"
    mutation = parse_mutation("K8E")
    
    # Get structures
    wt_path = fetch_rcsb_pdb("1CRN")
    mut_path = build_mutant_structure_stub(wt_path, sequence, mutation, force_naive=True)
    
    # Compute features
    features = compute_basic_features(sequence, mutation, wt_path, mut_path)
    
    print(f"📊 Features computed:")
    print(f"   WT path: {features.get('wt_structure_path')}")
    print(f"   MUT path: {features.get('mut_structure_path')}")
    print(f"   Chain: {features.get('mut_chain')}")
    print(f"   Residue ID: {features.get('mut_resid')}")
    print(f"   Mutation: {features.get('mutation')}")
    
    # Generate report
    dummy_prediction = {"label": "Harmful", "confidence": 0.66}
    report_html = render_html_report(features, dummy_prediction)
    
    # Save report for inspection
    with open("debug_visualization_report.html", "w") as f:
        f.write(report_html)
    
    print(f"📄 Report saved to debug_visualization_report.html")
    
    # Check if base64 data is being generated correctly
    from mutation_impact.reporting.report import _file_to_base64_text
    
    wt_data = _file_to_base64_text(str(wt_path))
    mut_data = _file_to_base64_text(str(mut_path))
    
    print(f"📊 Base64 data:")
    print(f"   WT data length: {len(wt_data.get('b64', ''))} chars")
    print(f"   MUT data length: {len(mut_data.get('b64', ''))} chars")
    print(f"   WT extension: {wt_data.get('ext')}")
    print(f"   MUT extension: {mut_data.get('ext')}")
    
    if wt_data.get('b64') == mut_data.get('b64'):
        print(f"   ❌ ISSUE: WT and MUT base64 data are identical!")
    else:
        print(f"   ✅ WT and MUT base64 data are different")

if __name__ == "__main__":
    debug_visualization()
    test_visualization_generation()
