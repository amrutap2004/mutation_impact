import re
from typing import Dict, Optional

try:
	from Bio import SeqIO  # type: ignore
except Exception:  # pragma: no cover
	SeqIO = None  # type: ignore

# Valid standard amino acids (1-letter). Only the 20 standard amino acids.
_VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _normalize_sequence(raw: str) -> str:
	seq = raw.strip().upper().replace("\n", "").replace("\r", "")
	return seq


def load_sequence(*, raw_sequence: Optional[str] = None, fasta_path: Optional[str] = None) -> str:
	"""Load a protein sequence from raw string or single-entry FASTA file.

	Exactly one of raw_sequence or fasta_path must be provided.
	"""
	if (raw_sequence is None) == (fasta_path is None):
		raise ValueError("Provide exactly one of raw_sequence or fasta_path")
	if raw_sequence is not None:
		sequence = _normalize_sequence(raw_sequence)
	else:
		if SeqIO is None:
			raise ImportError("biopython is required to read FASTA files")
		records = list(SeqIO.parse(fasta_path, "fasta"))  # type: ignore[arg-type]
		if len(records) == 0:
			raise ValueError("No sequences found in FASTA file")
		if len(records) > 1:
			raise ValueError("FASTA contains multiple sequences; provide a single sequence")
		sequence = _normalize_sequence(str(records[0].seq))

	if not sequence:
		raise ValueError("Sequence is empty after normalization")
	invalid = [c for c in sequence if c not in _VALID_AA]
	if invalid:
		# Backend rule: reject any non-standard amino acids / invalid chars
		raise ValueError("Protein sequence contains invalid characters.")
	return sequence


_MUT_RE = re.compile(
	r"^(?P<from>[ACDEFGHIKLMNPQRSTVWY])(?P<pos>[1-9][0-9]*)(?P<to>[ACDEFGHIKLMNPQRSTVWY])$"
)


def parse_mutation(mutation_str: str) -> Dict[str, object]:
	"""Parse a point mutation like A123T.

	Returns a dict with from_res, position (int), to_res.
	"""
	m = _MUT_RE.match(mutation_str.strip())
	if not m:
		# Backend rule: strict mutation format validation
		raise ValueError("Invalid mutation format. Use format R23K.")
	from_res = m.group("from").upper()
	to_res = m.group("to").upper()
	position = int(m.group("pos"))
	return {"from_res": from_res, "position": position, "to_res": to_res}


def validate_mutation_against_sequence(sequence: str, mutation: Dict[str, object]) -> None:
	"""Validate that mutation fits the sequence: position bounds and from_res match.
	Raises ValueError on mismatch.
	"""
	pos = int(mutation["position"])  # type: ignore[index]
	if pos < 1 or pos > len(sequence):
		# Backend rule: mutation position must be within sequence bounds
		raise ValueError("Mutation position exceeds sequence length.")
	seq_res = sequence[pos - 1]
	if seq_res != mutation["from_res"]:  # type: ignore[index]
		# Backend rule: residue at position must match mutation-from residue
		raise ValueError(
			f"Sequence residue '{seq_res}' at position {pos} does not match mutation-from residue '{mutation['from_res']}'."
		)
