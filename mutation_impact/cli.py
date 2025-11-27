import argparse
from mutation_impact.input_module.parser import load_sequence, parse_mutation, validate_mutation_against_sequence


def main() -> None:
	parser = argparse.ArgumentParser(description="Validate protein sequence and mutation notation (e.g., A123T)")
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument("--seq", type=str, help="Raw amino acid sequence (single-letter)")
	group.add_argument("--fasta", type=str, help="Path to FASTA file with one sequence")
	parser.add_argument("--mut", type=str, required=True, help="Mutation in p.X123Y format, e.g., A123T")
	args = parser.parse_args()

	sequence = load_sequence(raw_sequence=args.seq, fasta_path=args.fasta)
	mut = parse_mutation(args.mut)
	validate_mutation_against_sequence(sequence, mut)
	print("Validation successful:")
	print(f"\tLength: {len(sequence)} aa")
	print(f"\tMutation: {mut['from_res']}{mut['position']}{mut['to_res']}")


if __name__ == "__main__":
	main()
