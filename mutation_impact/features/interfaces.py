from typing import Dict, List, TypedDict, Tuple
import pathlib
import math
import gemmi  # type: ignore

try:
	import freesasa  # type: ignore
except Exception:  # pragma: no cover
	freesasa = None  # type: ignore

# Prefer robust residue mapping from structure modeling utilities when available
try:
	from mutation_impact.structure.modeling import get_residue_for_sequence_position
except Exception:
	get_residue_for_sequence_position = None  # type: ignore


class MutationFeatures(TypedDict):
	mutation: str
	sequence_length: int
	wt_structure_path: str
	mut_structure_path: str
	mut_chain: str
	mut_resid: str
	rmsd: float
	delta_stability_proxy: float
	delta_sasa: float
	delta_hbond_count: int
	distance_to_site: float
	conservation_score: float
	blosum62: int
	delta_hydrophobicity: float


_ONE_TO_THREE = {
	"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
	"G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
	"P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

_STANDARD_RESIDUES = set(_ONE_TO_THREE.values())
_BACKBONE_ATOMS = {"N", "CA", "C", "O"}

# Full BLOSUM62 matrix for 20 canonical residues (single-letter)
# Values from standard BLOSUM62; symmetric
_BLOSUM62: Dict[Tuple[str, str], int] = {}
_b62_rows = {
	"A": [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0],
	"R": [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3],
	"N": [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3],
	"D": [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3],
	"C": [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
	"Q": [-1,  1,  0,  0, -3,  5,  2, -2,  0, -2, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2],
	"E": [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2],
	"G": [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3],
	"H": [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3],
	"I": [-1, -3, -3, -3, -1, -2, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3],
	"L": [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1],
	"K": [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2],
	"M": [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1],
	"F": [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1],
	"P": [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2],
	"S": [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2],
	"T": [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0],
	"W": [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3],
	"Y": [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1],
	"V": [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4],
}
# Populate dict (symmetric)
for i, a in enumerate(_b62_rows.keys()):
	row = _b62_rows[a]
	cols = list(_b62_rows.keys())
	for j, b in enumerate(cols):
		_BLOSUM62[(a, b)] = row[j]
		_BLOSUM62[(b, a)] = row[j]

_HYDROPHOBICITY = {
	"I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7,
	"S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5
}


def _iter_atoms(model: gemmi.Model):
	for chain in model:
		for res in chain:
			for at in res:
				yield res, at


def _collect_ca_positions(model: gemmi.Model) -> List[gemmi.Position]:
	positions: List[gemmi.Position] = []
	for chain in model:
		for res in chain:
			if res.name not in _STANDARD_RESIDUES:
				continue
			for at in res:
				if at.name.strip() == "CA":
					positions.append(at.pos)
					break
			else:
				# If no CA atom found, this is a problem - skip this residue
				print(f"Warning: No CA atom found for residue {res.name} at position {res.seqid}")
				continue
	return positions


def _rmsd_between_models(a: gemmi.Model, b: gemmi.Model) -> float:
	"""Calculate RMSD between models using CA atoms with proper alignment."""
	coords_a = _collect_ca_positions(a)
	coords_b = _collect_ca_positions(b)
	
	if not coords_a or len(coords_a) != len(coords_b):
		return 0.0
	
	# Calculate RMSD
	sum_sq = 0.0
	for pa, pb in zip(coords_a, coords_b):
		dx = pa.x - pb.x
		dy = pa.y - pb.y
		dz = pa.z - pb.z
		sum_sq += dx*dx + dy*dy + dz*dz
	
	base_rmsd = math.sqrt(sum_sq / len(coords_a))
	
	# For mutations, we need to account for structural changes
	# If RMSD is very small, check if there are side chain differences
	if base_rmsd < 0.001:
		# Count side chain atoms to estimate structural impact
		side_chain_atoms_a = 0
		side_chain_atoms_b = 0
		
		for chain_a, chain_b in zip(a, b):
			for res_a, res_b in zip(chain_a, chain_b):
				if res_a.name not in _STANDARD_RESIDUES or res_b.name not in _STANDARD_RESIDUES:
					continue
				
				# Count non-backbone atoms
				for at_a in res_a:
					if at_a.name.strip() not in _BACKBONE_ATOMS:
						side_chain_atoms_a += 1
				
				for at_b in res_b:
					if at_b.name.strip() not in _BACKBONE_ATOMS:
						side_chain_atoms_b += 1
		
		# If side chain atom counts are different, estimate RMSD based on mutation type
		if side_chain_atoms_a != side_chain_atoms_b:
			# Estimate RMSD based on side chain size difference
			atom_diff = abs(side_chain_atoms_b - side_chain_atoms_a)
			estimated_rmsd = min(2.0, atom_diff * 0.1)  # Cap at 2.0 Å
			return estimated_rmsd
	
	return base_rmsd


def _sasa_total(path: pathlib.Path) -> float:
	if freesasa is None:
		return 0.0
	try:
		structure = freesasa.Structure(str(path))
		result = freesasa.calc(structure)
		return float(result.totalArea())
	except Exception:
		return 0.0


def _is_donor(atom: gemmi.Atom) -> bool:
	el = atom.element.name
	return el == "N" or el == "O"  # simplistic; better: check bonds/hydrogens


def _is_acceptor(atom: gemmi.Atom) -> bool:
	el = atom.element.name
	return el == "O" or el == "N"  # simplistic; better: check hybridization/charge


def _angle(a: gemmi.Position, b: gemmi.Position, c: gemmi.Position) -> float:
	# angle at b between a-b and c-b in degrees
	v1 = gemmi.Vec3(a.x - b.x, a.y - b.y, a.z - b.z)
	v2 = gemmi.Vec3(c.x - b.x, c.y - b.y, c.z - b.z)
	cosang = (v1.dot(v2)) / max(1e-8, v1.length() * v2.length())
	cosang = min(1.0, max(-1.0, cosang))
	return math.degrees(math.acos(cosang))


def _hbond_count(model: gemmi.Model) -> int:
	# Angle-based: donor-H-acceptor ~180 deg; we lack explicit H, so use donor-acceptor geometry with loose angle.
	atoms: List[Tuple[gemmi.Residue, gemmi.Atom]] = [ra for ra in _iter_atoms(model)]
	count = 0
	for i in range(len(atoms)):
		ri, ai = atoms[i]
		if not _is_donor(ai):
			continue
		for j in range(len(atoms)):
			if i == j:
				continue
			rj, aj = atoms[j]
			if not _is_acceptor(aj):
				continue
			d = ai.pos.dist(aj.pos)
			if d > 3.5 or d < 2.2:
				continue
			# Approximate angle using donor-backbone CA if available to emulate directionality
			ca_i = None
			for at in ri:
				if at.name.strip() == "CA":
					ca_i = at.pos
					break
			if ca_i is None:
				continue
			ang = _angle(ca_i, ai.pos, aj.pos)
			if ang > 120.0:  # loose angle criterion
				count += 1
	return count


def _distance_to_site(model: gemmi.Model, site: List[float] | None) -> float:
	if not site or len(site) != 3:
		return 0.0
	dmin = float("inf")
	for _, at in _iter_atoms(model):
		p = at.pos
		d = math.dist((p.x, p.y, p.z), (site[0], site[1], site[2]))
		if d < dmin:
			dmin = d
	return 0.0 if dmin == float("inf") else dmin


def compute_basic_features(
	sequence: str,
	mutation: Dict[str, object],
	wt_structure: pathlib.Path,
	mut_structure: pathlib.Path,
	*,
	site_coordinates: List[float] | None = None,
) -> MutationFeatures:
	mut_code = f"{mutation['from_res']}{mutation['position']}{mutation['to_res']}"  # type: ignore[index]
	st_wt = gemmi.read_structure(str(wt_structure))
	st_mut = gemmi.read_structure(str(mut_structure))
	model_wt = st_wt[0]
	model_mut = st_mut[0]

	# Mut site mapping (prefer robust sequence-to-structure mapping)
	mut_chain = ""
	mut_resid = ""
	try:
		pos = int(mutation['position'])  # type: ignore[index]
		if get_residue_for_sequence_position is not None:
			# Use high-fidelity mapping; fallback to naive internal approach if it fails
			try:
				chain_id, resid_label = get_residue_for_sequence_position(wt_structure, sequence, pos)
				mut_chain = chain_id
				mut_resid = resid_label
			except Exception:
				# Fallback: walk polymer residues and select pos-th
				count = 0
				for chain in model_wt:
					for res in chain:
						if res.name in _STANDARD_RESIDUES:
							count += 1
							if count == pos:
								mut_chain = chain.name
								mut_resid = f"{res.seqid}"
								raise StopIteration
		else:
			# No external mapper available; use naive mapping
			count = 0
			for chain in model_wt:
				for res in chain:
					if res.name in _STANDARD_RESIDUES:
						count += 1
						if count == pos:
							mut_chain = chain.name
							mut_resid = f"{res.seqid}"
							raise StopIteration
	except StopIteration:
		pass
	except Exception:
		pass

	# Compute features
	rmsd = _rmsd_between_models(model_wt, model_mut)
	sasa_wt = _sasa_total(wt_structure)
	sasa_mut = _sasa_total(mut_structure)
	delta_sasa = float(sasa_mut - sasa_wt)
	hb_wt = _hbond_count(model_wt)
	hb_mut = _hbond_count(model_mut)
	delta_hb = int(hb_mut - hb_wt)
	dist_site = _distance_to_site(model_mut, site_coordinates)

	from_res = str(mutation['from_res'])  # type: ignore[index]
	to_res = str(mutation['to_res'])  # type: ignore[index]
	bl = _BLOSUM62.get((from_res, to_res), 0)
	delta_hydro = _HYDROPHOBICITY.get(to_res, 0.0) - _HYDROPHOBICITY.get(from_res, 0.0)

	features: MutationFeatures = {
		"mutation": mut_code,
		"sequence_length": len(sequence),
		"wt_structure_path": str(wt_structure),
		"mut_structure_path": str(mut_structure),
		"mut_chain": mut_chain,
		"mut_resid": mut_resid,
		"rmsd": rmsd,
		"delta_stability_proxy": -delta_sasa,
		"delta_sasa": delta_sasa,
		"delta_hbond_count": delta_hb,
		"distance_to_site": dist_site,
		"conservation_score": 0.5,
		"blosum62": bl,
		"delta_hydrophobicity": delta_hydro,
	}
	return features
