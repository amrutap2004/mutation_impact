import os
import pathlib
import re
from typing import Optional

import requests

_PDB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
_PDB_FASTA_URL = "https://www.rcsb.org/fasta/entry/{pdb_id}"
_ALPHAFOLD_URL = "https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v{version}.pdb"
_CACHE_DIR = pathlib.Path(os.getenv("MUT_IMPACT_CACHE", pathlib.Path.home() / ".mutation_impact"))

_PDB_ID_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}$")


def _ensure_cache_dir() -> pathlib.Path:
	_CACHE_DIR.mkdir(parents=True, exist_ok=True)
	return _CACHE_DIR


def validate_pdb_id(pdb_id: str) -> str:
	"""Validate PDB ID format according to backend rules.

	Returns the normalized (upper-case) ID or raises ValueError.
	"""
	pid = pdb_id.strip()
	if not _PDB_ID_RE.match(pid):
		# Backend rule: strict PDB ID validation
		raise ValueError("Invalid PDB ID format.")
	return pid.upper()


def fetch_rcsb_pdb(pdb_id: str, *, cache: bool = True) -> pathlib.Path:
	pid = validate_pdb_id(pdb_id)
	cache_dir = _ensure_cache_dir()
	out_path = cache_dir / f"{pid}.pdb"
	if cache and out_path.exists():
		return out_path
	url = _PDB_DOWNLOAD_URL.format(pdb_id=pid)
	r = requests.get(url, timeout=60)
	r.raise_for_status()
	out_path.write_bytes(r.content)
	return out_path


def fetch_rcsb_fasta_sequence(pdb_id: str) -> str:
	"""Fetch the primary FASTA sequence for a PDB entry from RCSB."""
	pid = validate_pdb_id(pdb_id)
	url = _PDB_FASTA_URL.format(pdb_id=pid)
	r = requests.get(url, timeout=60)
	r.raise_for_status()
	fasta_text = r.text
	lines = [ln.strip() for ln in fasta_text.splitlines() if ln and not ln.startswith(">")]
	seq = "".join(lines).upper()
	return seq


def validate_sequence_vs_pdb_length(sequence: str, pdb_id: str, *, tolerance: float = 0.2) -> None:
	"""Validate that user-provided sequence length matches PDB sequence within tolerance.

	Backend rule: reject if length differs by more than 20%.
	"""
	pdb_seq = fetch_rcsb_fasta_sequence(pdb_id)
	if not pdb_seq:
		# If RCSB did not return a sequence, we can't apply this check safely.
		return

	user_len = len(sequence)
	pdb_len = len(pdb_seq)
	if pdb_len == 0:
		return

	diff = abs(user_len - pdb_len)
	if diff > tolerance * pdb_len:
		raise ValueError("Provided sequence does not match structure (length mismatch).")


def fetch_alphafold_model(uniprot_id: str, *, version: Optional[int] = None, cache: bool = True) -> pathlib.Path:
	uid = uniprot_id.strip().upper()
	ver = version if version is not None else 4
	cache_dir = _ensure_cache_dir()
	out_path = cache_dir / f"AF-{uid}-v{ver}.pdb"
	if cache and out_path.exists():
		return out_path
	url = _ALPHAFOLD_URL.format(uniprot=uid, version=ver)
	r = requests.get(url, timeout=60)
	r.raise_for_status()
	out_path.write_bytes(r.content)
	return out_path
