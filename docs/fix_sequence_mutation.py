"""
Fix sequence-mutation mismatch issues.
"""

def analyze_sequence_mutation(sequence, mutation):
    """Analyze sequence and mutation for mismatches."""
    print(f"🧬 Sequence-Mutation Analysis")
    print(f"="*50)
    print(f"Sequence: {sequence}")
    print(f"Mutation: {mutation}")
    print()
    
    # Parse mutation
    from_res = mutation[0]
    position = int(mutation[1:-1])
    to_res = mutation[-1]
    
    print(f"Mutation details:")
    print(f"  From: {from_res}")
    print(f"  Position: {position}")
    print(f"  To: {to_res}")
    print()
    
    # Check sequence at position
    if position <= len(sequence):
        actual_residue = sequence[position - 1]
        print(f"Sequence at position {position}: {actual_residue}")
        
        if actual_residue == from_res:
            print(f"✅ MATCH: Sequence and mutation are compatible")
            return True
        else:
            print(f"❌ MISMATCH: Expected {from_res}, found {actual_residue}")
            return False
    else:
        print(f"❌ ERROR: Position {position} is beyond sequence length {len(sequence)}")
        return False

def suggest_fixes(sequence, mutation):
    """Suggest fixes for sequence-mutation mismatches."""
    print(f"\n🔧 Suggested Fixes:")
    print(f"="*50)
    
    from_res = mutation[0]
    position = int(mutation[1:-1])
    to_res = mutation[-1]
    actual_residue = sequence[position - 1]
    
    print(f"Option 1: Change mutation to match sequence")
    print(f"  Current: {mutation}")
    print(f"  Suggested: {actual_residue}{position}{to_res}")
    print(f"  Example: {actual_residue}{position}E")
    print()
    
    print(f"Option 2: Change sequence to match mutation")
    print(f"  Current: {sequence}")
    new_sequence = sequence[:position-1] + from_res + sequence[position:]
    print(f"  Suggested: {new_sequence}")
    print()
    
    print(f"Option 3: Use a different position")
    print(f"  Find positions with {from_res}:")
    for i, res in enumerate(sequence, 1):
        if res == from_res:
            print(f"    Position {i}: {res}")
    print()

def main():
    """Main analysis function."""
    # Your current input
    sequence = "MVLSPADKTNVKAAW"
    mutation = "K4E"
    
    print("🔍 Analyzing your input...")
    print()
    
    # Analyze the mismatch
    is_compatible = analyze_sequence_mutation(sequence, mutation)
    
    if not is_compatible:
        suggest_fixes(sequence, mutation)
        
        print(f"🎯 Recommended Solutions:")
        print(f"="*50)
        print(f"1. Use mutation: S4E (matches your sequence)")
        print(f"2. Use sequence: MVLKPADKTNVKAAW (matches your mutation)")
        print(f"3. Use mutation: K10E (K is at position 10 in your sequence)")
        print()
        
        print(f"💡 For testing, I recommend:")
        print(f"   Sequence: MVLSPADKTNVKAAW")
        print(f"   Mutation: S4E")
        print(f"   This will test S→E mutation at position 4")

if __name__ == "__main__":
    main()
