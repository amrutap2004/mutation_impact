import pytest

from mutation_impact.input_module import load_sequence, parse_mutation, validate_mutation_against_sequence
from mutation_impact.structure.retrieval import validate_pdb_id, validate_sequence_vs_pdb_length


def test_invalid_mutation_format():
	# Does not match required regex (wrong residue + missing position)
	with pytest.raises(ValueError, match="Invalid mutation format. Use format R23K."):
		parse_mutation("Z23K")


def test_invalid_chars_in_sequence():
	# Contains space and invalid residue "B"
	with pytest.raises(ValueError, match="Protein sequence contains invalid characters."):
		load_sequence(raw_sequence="ACDE BFGHIKB")


def test_mutation_out_of_range():
	seq = "ACDEFGHIK"
	# Position 20 exceeds sequence length
	mut = parse_mutation("A20K")
	with pytest.raises(ValueError, match="Mutation position exceeds sequence length."):
		validate_mutation_against_sequence(seq, mut)


def test_mutation_residue_mismatch():
	seq = "ACDEFGHIK"  # position 3 = D
	mut = parse_mutation("K3A")  # from_res = K, position = 3
	with pytest.raises(
		ValueError,
		match=r"Sequence residue 'D' at position 3 does not match mutation-from residue 'K'\.",
	):
		validate_mutation_against_sequence(seq, mut)


def test_invalid_pdb_id_format():
	with pytest.raises(ValueError, match="Invalid PDB ID format."):
		validate_pdb_id("ABCD")  # must start with digit


def test_sequence_structure_length_mismatch(monkeypatch):
	# Fake PDB FASTA sequence of length 100
	def fake_fetch_fasta(pdb_id: str) -> str:  # noqa: ANN001
		return "A" * 100

	monkeypatch.setattr(
		"mutation_impact.structure.retrieval.fetch_rcsb_fasta_sequence",
		fake_fetch_fasta,
	)

	# User sequence length differs by >20% (e.g., 200 vs 100)
	user_sequence = "A" * 200

	with pytest.raises(ValueError, match="Provided sequence does not match structure \(length mismatch\)."):
		validate_sequence_vs_pdb_length(user_sequence, "1ABC")


