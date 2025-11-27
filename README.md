## Mutation Impact

A modular pipeline to model structural impact of mutations and assess harmfulness.

### Full Windows Setup
See `SETUP_WINDOWS.md` for a step-by-step guide to install Python, build tools, GTK/WeasyPrint runtime, create a virtual environment, install dependencies, and run the web app (`mi-web`).

### Modules
- Input Module: Validate sequences and parse mutations (e.g., A123T).
- Structure Modelling: Retrieve/model WT and mutant structures (PDB/mmCIF/AlphaFold).
- Feature Extraction: Compute RMSD (CA-based), solvent accessibility proxy, hydrogen-bond proxy, BLOSUM62 delta, hydrophobicity delta, site distance.
- Harmfulness Classifier: Predict Harmful/Neutral and confidence.
- Severity & Mode Estimator: Severity and mechanism if Harmful.
- Visualization & Reporting: 3D views and PDF/HTML report.

### Quickstart
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .

# Validate
.\.venv\Scripts\mi-validate --seq MATK --mut A2T

# Run pipeline (AlphaFold)
.\.venv\Scripts\mi-run --seq MKTFFVAI... --mut A123T --uniprot-id P05067 --out report.html
start report.html
```

### Notes on modeling and features
- Mutant modeling uses gemmi to change the residue name, preserve backbone (N, CA, C, O), and place a CB atom with idealized geometry for non-GLY targets. Sidechains beyond CB are not rebuilt yet.
- RMSD is computed across CA atoms between WT and mutant models (same atom ordering assumed from input/modeling).
- Solvent accessibility is approximated via a crude B-factor heuristic (placeholder until true SASA is added).
- Hydrogen bonds are approximated as N–O pairs closer than 3.5 Å (naive, counts many false positives).
- Conservation is currently a fixed placeholder (0.5). Integrating an MSA tool (e.g., HHblits/JackHMMER) would improve this.
- BLOSUM62 and hydrophobicity deltas are included as sequence-level features.

### Programmatic example
```python
from mutation_impact.input_module import load_sequence, parse_mutation, validate_mutation_against_sequence
from mutation_impact.structure import fetch_rcsb_pdb, build_mutant_structure_stub
from mutation_impact.features import compute_basic_features
from mutation_impact.classifier import HarmfulnessClassifier
from mutation_impact.severity import SeverityEstimator
from mutation_impact.reporting import render_html_report

seq = load_sequence(raw_sequence="MKT...")
mut = parse_mutation("A123T")
validate_mutation_against_sequence(seq, mut)
wt_path = fetch_rcsb_pdb("1CRN")
mut_path = build_mutant_structure_stub(wt_path, seq, mut)
features = compute_basic_features(seq, mut, wt_path, mut_path)
pred = HarmfulnessClassifier().predict(features)
sev = SeverityEstimator().estimate(features) if pred["label"] == "Harmful" else None
html = render_html_report(features, pred, sev)
open("report.html", "w", encoding="utf-8").write(html)
```

### License
MIT
