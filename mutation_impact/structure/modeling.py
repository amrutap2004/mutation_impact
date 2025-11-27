import math
import pathlib
import tempfile
import uuid
import time
from typing import Dict, List, Optional, Tuple

import gemmi  # type: ignore


_BACKBONE_ATOMS = {"N", "CA", "C", "O"}
_STANDARD_RESIDUES = {
	"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
	"LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}

_ONE_TO_THREE = {
	"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
	"G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
	"P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

_THREE_TO_ONE = {v: k for k, v in _ONE_TO_THREE.items()}


def _extract_structure_polymer_sequence(struct: gemmi.Structure) -> List[Tuple[gemmi.Residue, str]]:
	seq: List[Tuple[gemmi.Residue, str]] = []
	for model in struct:
		for chain in model:
			for res in chain:
				if res.name in _STANDARD_RESIDUES:
					one = _THREE_TO_ONE.get(res.name, "X")
					seq.append((res, one))
	return seq


def _align_seq_to_structure(raw_sequence: str, struct: gemmi.Structure) -> Dict[int, gemmi.Residue]:
	"""Return mapping from 1-based sequence index to gemmi.Residue using a simple global alignment.

	If alignment fails, fall back to naive 1:1 mapping until min length.
	"""
	seq_pairs = _extract_structure_polymer_sequence(struct)
	struct_seq = "".join(one for _, one in seq_pairs)
	seq = raw_sequence
	# Simple Needleman-Wunsch with match=1, mismatch=-1, gap=-1
	n, m = len(seq), len(struct_seq)
	score = [[0]*(m+1) for _ in range(n+1)]
	ptr = [[0]*(m+1) for _ in range(n+1)]  # 1:diag, 2:up, 3:left
	for i in range(1, n+1):
		score[i][0] = -i
		ptr[i][0] = 2
	for j in range(1, m+1):
		score[0][j] = -j
		ptr[0][j] = 3
	for i in range(1, n+1):
		for j in range(1, m+1):
			match = score[i-1][j-1] + (1 if seq[i-1] == struct_seq[j-1] else -1)
			up = score[i-1][j] - 1
			left = score[i][j-1] - 1
			best = match
			p = 1
			if up > best:
				best = up
				p = 2
			if left > best:
				best = left
				p = 3
			score[i][j] = best
			ptr[i][j] = p
	# Traceback
	i, j = n, m
	mapping: Dict[int, gemmi.Residue] = {}
	while i > 0 and j > 0:
		p = ptr[i][j]
		if p == 1:
			# aligned
			if seq[i-1] == struct_seq[j-1]:
				res = seq_pairs[j-1][0]
				mapping[i] = res
			i -= 1
			j -= 1
		elif p == 2:
			i -= 1
		else:
			j -= 1
	# Fallback fill: naive mapping up to min length if empty
	if not mapping:
		k = min(n, len(seq_pairs))
		for idx in range(1, k+1):
			mapping[idx] = seq_pairs[idx-1][0]
	return mapping


def _vec(a: gemmi.Atom) -> gemmi.Vec3:
	return gemmi.Vec3(a.pos.x, a.pos.y, a.pos.z)


def _place_cb(n: gemmi.Atom, ca: gemmi.Atom, c: gemmi.Atom) -> gemmi.Vec3:
	r_ca = _vec(ca)
	r_n = _vec(n)
	r_c = _vec(c)
	n_dir = (r_n - r_ca).normalized()
	c_dir = (r_c - r_ca).normalized()
	bis = (n_dir + c_dir)
	if bis.length() == 0:
		bis = n_dir
	perp = n_dir.cross(c_dir)
	if perp.length() == 0:
		perp = gemmi.Vec3(1.0, 0.0, 0.0)
	cb_dir = (-0.6 * bis.normalized() + 0.8 * perp.normalized())
	if cb_dir.length() == 0:
		cb_dir = gemmi.Vec3(0.0, 0.0, 1.0)
	cb_dir = cb_dir.normalized()
	length = 1.53
	return r_ca + cb_dir * length


def _mutate_residue_keep_backbone(res: gemmi.Residue, to_three_letter: str) -> None:
	res.name = to_three_letter
	backbone_atoms: Dict[str, gemmi.Atom] = {}
	to_remove: List[gemmi.Atom] = []
	for at in res:
		name_stripped = at.name.strip()
		if name_stripped in _BACKBONE_ATOMS:
			backbone_atoms[name_stripped] = at
		else:
			to_remove.append(at)
	for at in to_remove:
		alt = at.altloc if at.altloc and at.altloc != "\x00" else ""
		try:
			res.remove_atom(at.name, alt, at.element)
		except Exception:
			pass
	
	# Add side chain atoms based on target residue
	if to_three_letter != "GLY" and "CA" in backbone_atoms and "N" in backbone_atoms and "C" in backbone_atoms:
		# Place CB atom
		cb_pos = _place_cb(backbone_atoms["N"], backbone_atoms["CA"], backbone_atoms["C"])
		cb = gemmi.Atom()
		cb.name = " CB "
		cb.element = gemmi.Element("C")
		cb.occ = 1.0
		cb.b_iso = 20.0
		cb.pos = gemmi.Position(cb_pos.x, cb_pos.y, cb_pos.z)
		res.add_atom(cb)
		
		# Add additional side chain atoms for complex residues
		if to_three_letter == "TRP":
			# Add CG, CD1, CD2, NE1, CE2, CE3, CZ2, CZ3, CH2 atoms for TRP
			# This is a simplified approach - in practice, you'd use proper geometry
			ca_pos = backbone_atoms["CA"].pos
			cb_pos = cb.pos
			
			# Calculate direction from CA to CB
			cb_dir = gemmi.Position(cb_pos.x - ca_pos.x, cb_pos.y - ca_pos.y, cb_pos.z - ca_pos.z)
			cb_length = math.sqrt(cb_dir.x**2 + cb_dir.y**2 + cb_dir.z**2)
			if cb_length > 0:
				cb_dir.x /= cb_length
				cb_dir.y /= cb_length
				cb_dir.z /= cb_length
			
			# Add CG atom
			cg_pos = gemmi.Position(cb_pos.x + cb_dir.x * 1.5, cb_pos.y + cb_dir.y * 1.5, cb_pos.z + cb_dir.z * 1.5)
			cg = gemmi.Atom()
			cg.name = " CG "
			cg.element = gemmi.Element("C")
			cg.occ = 1.0
			cg.b_iso = 20.0
			cg.pos = cg_pos
			res.add_atom(cg)
			
			# Add CD1 atom (simplified)
			cd1_pos = gemmi.Position(cg_pos.x + 1.0, cg_pos.y, cg_pos.z)
			cd1 = gemmi.Atom()
			cd1.name = " CD1"
			cd1.element = gemmi.Element("C")
			cd1.occ = 1.0
			cd1.b_iso = 20.0
			cd1.pos = cd1_pos
			res.add_atom(cd1)
			
			# Add CD2 atom (simplified)
			cd2_pos = gemmi.Position(cg_pos.x - 1.0, cg_pos.y, cg_pos.z)
			cd2 = gemmi.Atom()
			cd2.name = " CD2"
			cd2.element = gemmi.Element("C")
			cd2.occ = 1.0
			cd2.b_iso = 20.0
			cd2.pos = cd2_pos
			res.add_atom(cd2)
			
		elif to_three_letter == "GLU":
			# Add CG, CD, OE1, OE2 atoms for GLU
			ca_pos = backbone_atoms["CA"].pos
			cb_pos = cb.pos
			
			# Calculate direction from CA to CB
			cb_dir = gemmi.Position(cb_pos.x - ca_pos.x, cb_pos.y - ca_pos.y, cb_pos.z - ca_pos.z)
			cb_length = math.sqrt(cb_dir.x**2 + cb_dir.y**2 + cb_dir.z**2)
			if cb_length > 0:
				cb_dir.x /= cb_length
				cb_dir.y /= cb_length
				cb_dir.z /= cb_length
			
			# Add CG atom
			cg_pos = gemmi.Position(cb_pos.x + cb_dir.x * 1.5, cb_pos.y + cb_dir.y * 1.5, cb_pos.z + cb_dir.z * 1.5)
			cg = gemmi.Atom()
			cg.name = " CG "
			cg.element = gemmi.Element("C")
			cg.occ = 1.0
			cg.b_iso = 20.0
			cg.pos = cg_pos
			res.add_atom(cg)
			
			# Add CD atom
			cd_pos = gemmi.Position(cg_pos.x + cb_dir.x * 1.5, cg_pos.y + cb_dir.y * 1.5, cg_pos.z + cb_dir.z * 1.5)
			cd = gemmi.Atom()
			cd.name = " CD "
			cd.element = gemmi.Element("C")
			cd.occ = 1.0
			cd.b_iso = 20.0
			cd.pos = cd_pos
			res.add_atom(cd)
			
			# Add OE1 atom
			oe1_pos = gemmi.Position(cd_pos.x + 1.0, cd_pos.y, cd_pos.z)
			oe1 = gemmi.Atom()
			oe1.name = " OE1"
			oe1.element = gemmi.Element("O")
			oe1.occ = 1.0
			oe1.b_iso = 20.0
			oe1.pos = oe1_pos
			res.add_atom(oe1)
			
			# Add OE2 atom
			oe2_pos = gemmi.Position(cd_pos.x - 1.0, cd_pos.y, cd_pos.z)
			oe2 = gemmi.Atom()
			oe2.name = " OE2"
			oe2.element = gemmi.Element("O")
			oe2.occ = 1.0
			oe2.b_iso = 20.0
			oe2.pos = oe2_pos
			res.add_atom(oe2)


def build_mutant_structure_stub(wt_structure_path: pathlib.Path, sequence: str, mutation: Dict[str, object], *, force_naive: bool = False) -> pathlib.Path:
	wt_path = pathlib.Path(wt_structure_path)
	if not wt_path.exists():
		raise FileNotFoundError(str(wt_path))
	from_res = str(mutation["from_res"]).upper()
	to_res = str(mutation["to_res"]).upper()
	pos = int(mutation["position"])  # 1-based
	to_three = _ONE_TO_THREE.get(to_res)
	if to_three is None:
		raise ValueError(f"Unsupported target residue: {to_res}")
	st = gemmi.read_structure(str(wt_path))
	pairs = _extract_structure_polymer_sequence(st)
	mapping = _align_seq_to_structure(sequence, st)
	res = mapping.get(pos)
	if res is None and force_naive and pairs:
		# naive 1:1 mapping fallback
		idx = pos - 1
		if 0 <= idx < len(pairs):
			res = pairs[idx][0]
	if res is None:
		raise ValueError(
			f"Could not map sequence position {pos} to a structure residue. Structure may have gaps; consider providing aligned structures or use force_naive."
		)
	_mutate_residue_keep_backbone(res, to_three)
	mut_code = f"{from_res}{pos}{to_res}"
	
	# Create unique filename to avoid caching issues between different test runs
	unique_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
	timestamp = int(time.time() * 1000) % 100000  # Use milliseconds for uniqueness
	out_path = wt_path.with_name(wt_path.stem + f"_{mut_code}_{unique_id}_{timestamp}" + wt_path.suffix)
	
	io = st
	io.remove_empty_chains()
	if wt_path.suffix.lower() == ".cif" or wt_path.suffix.lower() == ".mmcif":
		gemmi.cif.write_minimal_pdbx(io, str(out_path))
	else:
		io.write_minimal_pdb(str(out_path))
	return out_path


def minimize_with_openmm(structure_path: pathlib.Path) -> pathlib.Path:
	"""Optionally minimize the structure using OpenMM if available.

	Requires packages: openmm, pdbfixer. Leaves the input unmodified and writes a *_min.pdb.
	"""
	try:
		from openmm import unit
		from openmm import LocalEnergyMinimizer
		from openmm.app import PDBFile, Modeller, ForceField, Simulation
		from openmm.openmm import LangevinIntegrator
		from pdbfixer import PDBFixer
	except Exception as exc:  # pragma: no cover
		raise ImportError("OpenMM and pdbfixer are required for minimization. Install: pip install openmm pdbfixer") from exc

	in_path = pathlib.Path(structure_path)
	
	# Create unique filename to avoid caching issues
	unique_id = str(uuid.uuid4())[:8]
	timestamp = int(time.time() * 1000) % 100000
	out_path = in_path.with_name(in_path.stem + f"_min_{unique_id}_{timestamp}" + in_path.suffix)
	# Fix missing atoms/hydrogens minimally
	fixer = PDBFixer(filename=str(in_path))
	fixer.findMissingResidues()
	fixer.findMissingAtoms()
	fixer.addMissingAtoms()
	fixer.addMissingHydrogens(pH=7.0)
	# Convert to OpenMM objects
	with open(out_path, "w", encoding="utf-8") as fh:
		PDBFile.writeFile(fixer.topology, fixer.positions, fh)
	pdb = PDBFile(str(out_path))
	mod = Modeller(pdb.topology, pdb.positions)
	ff = ForceField("amber14-all.xml", "amber14/tip3p.xml")
	system = ff.createSystem(mod.topology, nonbondedMethod=None, constraints=None)
	integrator = LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.004*unit.picoseconds)
	sim = Simulation(mod.topology, system, integrator)
	sim.context.setPositions(mod.positions)
	LocalEnergyMinimizer.minimize(sim.context, tolerance=10.0*unit.kilojoule_per_mole, maxIterations=500)
	state = sim.context.getState(getPositions=True)
	with open(out_path, "w", encoding="utf-8") as fh:
		PDBFile.writeFile(mod.topology, state.getPositions(), fh)
	return out_path


def get_residue_for_sequence_position(wt_structure_path: pathlib.Path, sequence: str, position_1based: int, *, force_naive: bool = False) -> Tuple[str, str]:
	"""Return (chain_id, resid_label) for mapped residue at a given sequence position.

	resid_label is e.g. 'A 50' or includes insertion code if present.
	"""
	st = gemmi.read_structure(str(wt_structure_path))
	pairs = _extract_structure_polymer_sequence(st)
	mapping = _align_seq_to_structure(sequence, st)
	res = mapping.get(position_1based)
	if res is None and force_naive and pairs:
		idx = position_1based - 1
		if 0 <= idx < len(pairs):
			res = pairs[idx][0]
	if res is None:
		raise ValueError("Could not map sequence position to structure residue")
	chain = res.chain.name if hasattr(res, 'chain') and res.chain is not None else ""
	resid = f"{res.seqid}"
	return chain, resid


def cleanup_temporary_files(directory: pathlib.Path, pattern: str = "*_mut_*", max_age_hours: int = 24) -> int:
	"""Clean up temporary mutant structure files older than max_age_hours.
	
	Args:
		directory: Directory to clean up
		pattern: File pattern to match (default: mutant files)
		max_age_hours: Maximum age in hours before cleanup
		
	Returns:
		Number of files cleaned up
	"""
	import os
	import glob
	
	if not directory.exists():
		return 0
		
	current_time = time.time()
	max_age_seconds = max_age_hours * 3600
	cleaned_count = 0
	
	# Find files matching the pattern
	files = glob.glob(str(directory / pattern))
	
	for file_path in files:
		try:
			file_stat = os.stat(file_path)
			file_age = current_time - file_stat.st_mtime
			
			if file_age > max_age_seconds:
				os.remove(file_path)
				cleaned_count += 1
		except (OSError, FileNotFoundError):
			# File might have been deleted already or permission issues
			continue
	
	return cleaned_count


def cleanup_mutation_cache(cache_dir: Optional[pathlib.Path] = None) -> int:
	"""Clean up mutation-related temporary files from cache directory.
	
	Args:
		cache_dir: Cache directory to clean (default: system cache dir)
		
	Returns:
		Number of files cleaned up
	"""
	if cache_dir is None:
		cache_dir = pathlib.Path.home() / ".mutation_impact"
	
	total_cleaned = 0
	
	# Clean up mutant files
	total_cleaned += cleanup_temporary_files(cache_dir, "*_mut_*", max_age_hours=1)
	
	# Clean up minimized files
	total_cleaned += cleanup_temporary_files(cache_dir, "*_min_*", max_age_hours=1)
	
	return total_cleaned
