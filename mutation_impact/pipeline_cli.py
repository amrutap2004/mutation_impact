import argparse
import pathlib
from typing import Optional

from mutation_impact.input_module import load_sequence, parse_mutation, validate_mutation_against_sequence
from mutation_impact.structure import fetch_rcsb_pdb, fetch_alphafold_model
from mutation_impact.structure.retrieval import validate_pdb_id, validate_sequence_vs_pdb_length
from mutation_impact.structure.modeling import build_mutant_structure_stub, cleanup_mutation_cache
from mutation_impact.features import compute_basic_features
from mutation_impact.classifier import HarmfulnessClassifier
from mutation_impact.classifier.simple_ml_only import SimpleMLOnlyClassifier
from mutation_impact.severity import SeverityEstimator
from mutation_impact.reporting import render_html_report

try:
	from mutation_impact.structure.modeling import minimize_with_openmm  # optional
except Exception:  # pragma: no cover
	minimize_with_openmm = None  # type: ignore


def _resolve_wt_structure(
	pdb_id: Optional[str],
	uniprot_id: Optional[str],
) -> pathlib.Path:
	if (pdb_id is None) == (uniprot_id is None):
		raise ValueError("Provide exactly one of --pdb-id or --uniprot-id to obtain WT structure")
	if pdb_id is not None:
		# Validate PDB ID format before fetching
		validate_pdb_id(pdb_id)
		return fetch_rcsb_pdb(pdb_id)
	return fetch_alphafold_model(uniprot_id or "")


def main() -> None:
	parser = argparse.ArgumentParser(description="Run Mutation Impact pipeline and generate HTML report")
	seq_group = parser.add_mutually_exclusive_group(required=True)
	seq_group.add_argument("--seq", type=str, help="Raw amino acid sequence (single-letter)")
	seq_group.add_argument("--fasta", type=str, help="Path to FASTA file with one sequence")
	parser.add_argument("--mut", type=str, required=True, help="Mutation in p.X123Y format, e.g., A123T")
	struct_group = parser.add_mutually_exclusive_group(required=True)
	struct_group.add_argument("--pdb-id", type=str, help="RCSB PDB ID for WT structure (e.g., 1CRN)")
	struct_group.add_argument("--uniprot-id", type=str, help="UniProt ID to fetch AlphaFold model (e.g., P05067)")
	parser.add_argument("--out", type=str, default="mutation_impact_report.html", help="Output HTML report path")
	parser.add_argument("--force-naive", action="store_true", help="Fallback to naive residue mapping if alignment fails")
	parser.add_argument("--minimize", action="store_true", help="Relax mutant with OpenMM minimization if available")
	args = parser.parse_args()

	# Clean up any old temporary files to prevent caching issues
	try:
		cleaned_count = cleanup_mutation_cache()
		if cleaned_count > 0:
			print(f"Cleaned up {cleaned_count} temporary files")
	except Exception as cleanup_error:
		print(f"Cleanup warning: {cleanup_error}")

	sequence = load_sequence(raw_sequence=args.seq, fasta_path=args.fasta)
	mutation = parse_mutation(args.mut)
	validate_mutation_against_sequence(sequence, mutation)

	# Additional backend validations when using a PDB structure
	if args.pdb_id:
		# Ensures correct format and sequence–structure length agreement
		validate_pdb_id(args.pdb_id)
		validate_sequence_vs_pdb_length(sequence, args.pdb_id)

	wt_path = _resolve_wt_structure(args.pdb_id, args.uniprot_id)
	mut_path = build_mutant_structure_stub(wt_path, sequence, mutation, force_naive=args.force_naive)

	# Optional OpenMM minimization
	if args.minimize:
		if minimize_with_openmm is None:
			print("[warn] OpenMM minimization requested but not available. Install openmm and pdbfixer.")
		else:
			try:
				mut_path = minimize_with_openmm(mut_path)
			except Exception as exc:  # pragma: no cover
				print(f"[warn] Minimization failed: {exc}")

	features = compute_basic_features(sequence, mutation, wt_path, mut_path)
	
	# Use Simple ML-only classifier for maximum accuracy
	try:
		ml_classifier = SimpleMLOnlyClassifier("models/")
		prediction = ml_classifier.predict(sequence, args.mut, wt_path, mut_path, "ensemble")
		prediction["ml_model"] = True
		prediction["model_used"] = prediction.get("model_used", "ensemble")
		print(f"✅ Using ML model: {prediction.get('model_used', 'ensemble')}")
	except Exception as e:
		raise Exception(f"ML-only prediction failed: {e}. Train models first using create_better_ml_model.py")
	
	severity = SeverityEstimator().estimate(features) if prediction["label"] == "Harmful" else None
	html = render_html_report(features, prediction, severity)
	out_path = pathlib.Path(args.out)
	out_path.write_text(html, encoding="utf-8")
	print(f"Report written to {out_path}")


if __name__ == "__main__":
	main()
