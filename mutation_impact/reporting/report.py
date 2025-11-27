from typing import Dict
from jinja2 import Template
import base64
import os


def _file_to_base64_text(path: str) -> Dict[str, str]:
	try:
		ext = os.path.splitext(path)[1].lower()
		with open(path, "rb") as f:
			data = f.read()
		b64 = base64.b64encode(data).decode("ascii")
		return {"ext": ext.lstrip('.'), "b64": b64}
	except Exception:
		return {"ext": "", "b64": ""}


def _generate_pymol_pml(features: Dict) -> str:
	wt = features.get("wt_structure_path", "")
	mut = features.get("mut_structure_path", "")
	chain = features.get("mut_chain", "")
	resid = features.get("mut_resid", "")
	from_to = features.get("mutation", "")
	resnum = ''.join([c for c in str(resid) if c.isdigit()])
	if not resnum:
		resnum = resid
	return f"""
# PyMOL script to visualize mutation impact (highlight mutant only)
reinitialize
load {wt}, WT
load {mut}, MUT
as cartoon, WT MUT
spectrum b, rainbow, WT
color gray70, MUT
super MUT, WT
select mutSiteMUT, (MUT and chain {chain} and resi {resnum})
show sticks, mutSiteMUT
color tv_red, mutSiteMUT
label mutSiteMUT, "MUT {from_to[-1:]}"
bg_color white
set ray_opaque_background, off
set antialias, 2
set ray_trace_mode, 1
set ray_shadow, off
orient WT
# ray 1200,900; png report.png, dpi=300
"""


_HTML_TMPL = Template(
	"""
	<!doctype html>
	<html>
	<head>
		<meta charset="utf-8">
		<title>Mutation Impact Report</title>
		<script src="https://unpkg.com/ngl@latest/dist/ngl.js"></script>
		<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
		<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
		<style>
			:root {
				--bg-dark: #050b14;
				--bg-panel: #0f1623;
				--primary: #3b82f6;
				--accent: #8b5cf6;
				--text-main: #f8fafc;
				--text-muted: #94a3b8;
				--border: rgba(255, 255, 255, 0.08);
				--success: #22c55e;
				--danger: #ef4444;
				--warning: #f59e0b;
			}
			body { font-family: 'Inter', sans-serif; background: transparent; color: var(--text-main); margin: 0; }
			.report-container { display: flex; flex-direction: column; gap: 2rem; }
			
			/* Result Banner */
			.result-banner {
				background: var(--bg-panel);
				border: 1px solid var(--border);
				border-radius: 12px;
				padding: 2rem;
				text-align: center;
				position: relative;
				overflow: hidden;
			}
			.result-badge {
				display: inline-block;
				padding: 0.5rem 1.5rem;
				border-radius: 50px;
				font-family: 'Outfit', sans-serif;
				font-weight: 700;
				font-size: 1.5rem;
				margin-bottom: 1rem;
			}
			.is-harmful { background: rgba(239, 68, 68, 0.2); color: var(--danger); border: 1px solid rgba(239, 68, 68, 0.3); }
			.is-neutral { background: rgba(34, 197, 94, 0.2); color: var(--success); border: 1px solid rgba(34, 197, 94, 0.3); }
			
			.confidence-meter {
				max-width: 400px;
				margin: 0 auto;
			}
			.meter-label { display: flex; justify-content: space-between; margin-bottom: 0.5rem; font-size: 0.9rem; color: var(--text-muted); }
			.meter-track { height: 8px; background: rgba(255,255,255,0.1); border-radius: 4px; overflow: hidden; }
			.meter-fill { height: 100%; background: var(--primary); border-radius: 4px; transition: width 1s ease; }
			
			/* Grid Layout */
			.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
			@media (max-width: 768px) { .grid-2 { grid-template-columns: 1fr; } }
			
			/* Cards */
			.card {
				background: var(--bg-panel);
				border: 1px solid var(--border);
				border-radius: 12px;
				padding: 1.5rem;
			}
			.card-header {
				font-family: 'Outfit', sans-serif;
				font-size: 1.1rem;
				font-weight: 600;
				margin-bottom: 1rem;
				padding-bottom: 0.75rem;
				border-bottom: 1px solid var(--border);
				display: flex;
				justify-content: space-between;
				align-items: center;
			}
			
			/* 3D Viewer */
			.viewer-container { height: 400px; position: relative; border-radius: 8px; overflow: hidden; border: 1px solid var(--border); }
			.viewer { width: 100%; height: 100%; }
			.viewer-label { position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.7); padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; pointer-events: none; z-index: 10; }
			
			/* Table */
			.data-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
			.data-table th { text-align: left; padding: 0.75rem; color: var(--text-muted); border-bottom: 1px solid var(--border); }
			.data-table td { padding: 0.75rem; border-bottom: 1px solid rgba(255,255,255,0.05); }
			.data-table tr:last-child td { border-bottom: none; }
			
			/* Severity Tags */
			.sev-tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; background: rgba(255,255,255,0.1); margin-right: 4px; }
			
			/* Confidence Factors */
			.factor-grid { display: flex; flex-wrap: wrap; gap: 0.5rem; }
			.factor-tag { background: rgba(59, 130, 246, 0.15); color: #60a5fa; padding: 4px 10px; border-radius: 6px; font-size: 0.85rem; border: 1px solid rgba(59, 130, 246, 0.2); }
		</style>
	</head>
	<body>
		<div class="report-container">
			<!-- Prediction Summary -->
			<div class="result-banner">
				<div class="result-badge {% if prediction.label == 'Harmful' %}is-harmful{% else %}is-neutral{% endif %}">
					{{ prediction.label }}
				</div>
				<div class="confidence-meter">
					<div class="meter-label">
						<span>Model Confidence</span>
						<span>{{ '%.1f' % (prediction.confidence * 100) }}%</span>
					</div>
					<div class="meter-track">
						<div class="meter-fill" style="width: {{ prediction.confidence * 100 }}%; background: {% if prediction.confidence > 0.8 %}var(--success){% elif prediction.confidence > 0.6 %}var(--primary){% else %}var(--warning){% endif %};"></div>
					</div>
				</div>
				{% if prediction.get('enhanced') %}
				<div style="margin-top: 1rem; font-size: 0.9rem; color: var(--text-muted);">
					<i class="fa-solid fa-wand-magic-sparkles" style="color: var(--accent);"></i> Enhanced with {{ prediction.confidence_factors|length }} structural factors
				</div>
				{% endif %}
			</div>

			<!-- 3D Visualization -->
			<div class="card">
				<div class="card-header">
					<span><i class="fa-solid fa-cubes"></i> Structural Impact</span>
					<a id="pymolLink" class="sev-tag" style="text-decoration: none; cursor: pointer;"><i class="fa-solid fa-download"></i> PyMOL Script</a>
				</div>
				<div class="grid-2">
					<div class="viewer-container">
						<div class="viewer-label">Wild-Type</div>
						<div id="viewer-wt" class="viewer"></div>
					</div>
					<div class="viewer-container">
						<div class="viewer-label">Mutant ({{ features.mutation }})</div>
						<div id="viewer-mut" class="viewer"></div>
					</div>
				</div>
			</div>

			<div class="grid-2">
				<!-- Key Metrics -->
				<div class="card">
					<div class="card-header"><i class="fa-solid fa-chart-pie"></i> Quantitative Metrics</div>
					<table class="data-table">
						<tr><th>Metric</th><th>Value</th><th>Impact</th></tr>
						<tr>
							<td>RMSD</td>
							<td>{{ '%.3f' % features.get('rmsd', 0) }} Å</td>
							<td>{% if features.get('rmsd', 0) > 0.5 %}High{% else %}Low{% endif %}</td>
						</tr>
						<tr>
							<td>ΔSASA</td>
							<td>{{ '%.1f' % features.get('delta_sasa', 0) }} Å²</td>
							<td>{% if features.get('delta_sasa', 0)|abs > 50 %}Significant{% else %}Minor{% endif %}</td>
						</tr>
						<tr>
							<td>ΔHydrophobicity</td>
							<td>{{ '%.2f' % features.get('delta_hydrophobicity', 0) }}</td>
							<td>-</td>
						</tr>
						<tr>
							<td>Conservation</td>
							<td>{{ '%.2f' % features.get('conservation_score', 0) }}</td>
							<td>{% if features.get('conservation_score', 0) > 0.7 %}Critical{% else %}Variable{% endif %}</td>
						</tr>
					</table>
				</div>

				<!-- AI Analysis -->
				<div class="card">
					<div class="card-header"><i class="fa-solid fa-brain"></i> Insights</div>
					{% if severity %}
					<div style="margin-bottom: 1.5rem;">
						<div style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.5rem;">Estimated Severity</div>
						<div style="font-size: 1.2rem; font-weight: 600; color: var(--text-main);">{{ severity.severity }}</div>
						<div style="margin-top: 0.5rem;">
							{% for mode in severity.modes %}
							<span class="sev-tag">{{ mode }}</span>
							{% endfor %}
						</div>
					</div>
					{% endif %}
					
					
				</div>
			</div>

				
			
		</div>

		<!-- Scripts -->
		<script>
			(function(){
				var wt = {{ wt_text | tojson }};
				var mut = {{ mut_text | tojson }};
				var wtExt = {{ wt_ext | tojson }} || 'pdb';
				var mutExt = {{ mut_ext | tojson }} || 'pdb';
				var mutChain = {{ features.mut_chain | tojson }};
				var mutResid = {{ features.mut_resid | tojson }};
				var pymolText = {{ pymol_pml | tojson }};
				var fromAA = {{ features.mutation[:1] | tojson }};
				var toAA = {{ features.mutation[-1:] | tojson }};

				var pmlBlob = new Blob([pymolText], {type: 'text/x-pymol'});
				document.getElementById('pymolLink').href = URL.createObjectURL(pmlBlob);

				function b64ToText(b64){ try { return atob(b64); } catch(e){ return ''; } }
				
				function addSiteLabel(comp, chain, resid, text){
					var r = String(resid || '').trim();
					var m = r.match(/^(\\d+)([A-Za-z]?)$/);
					var resToken = m ? (m[1] + (m[2] || '')) : (r.match(/\\d+/) ? r.match(/\\d+/)[0] : '');
					var sel = resToken ? ((chain ? (':' + chain + ' and ') : '') + 'resi ' + resToken) : '';
					if (!sel) return;
					
					comp.addRepresentation('ball+stick', { sele: sel, color: 'red', multipleBond: true });
					comp.addRepresentation('spacefill', { sele: sel, color: 'red', opacity: 0.6 });
					var neighborSele = 'within 5 of (' + sel + ') and not (' + sel + ')';
					comp.addRepresentation('surface', { sele: neighborSele, opacity: 0.2, color: 'yellow', useWorker: false });
					comp.addRepresentation('label', { sele: sel, color: 'black', labelType: 'format', labelFormat: text });
				}

				function loadBlob(stage, text, ext, isMutant){
					var blob = new Blob([text], {type: 'text/plain'});
					stage.loadFile(blob, { ext: ext }).then(function(o){
						if (isMutant) {
							o.addRepresentation('cartoon', { color: 'lightgray' });
							addSiteLabel(o, mutChain, mutResid, 'MUT ' + toAA);
						} else {
							o.addRepresentation('cartoon', { colorScheme: 'chainname' });
						}
						stage.autoView();
					});
				}

				var stageWT = new NGL.Stage('viewer-wt', { backgroundColor: 'white' });
				var stageMUT = new NGL.Stage('viewer-mut', { backgroundColor: 'white' });
				
				window.addEventListener('resize', function(){
					stageWT.handleResize();
					stageMUT.handleResize();
				});

				loadBlob(stageWT, b64ToText(wt), wtExt, false);
				loadBlob(stageMUT, b64ToText(mut), mutExt, true);
			})();
		</script>
	</body>
	</html>
	"""
)


def render_html_report(features: Dict, prediction: Dict, severity: Dict | None = None) -> str:
	wt = _file_to_base64_text(features.get("wt_structure_path", ""))
	mut = _file_to_base64_text(features.get("mut_structure_path", ""))
	pml = _generate_pymol_pml(features)
	return _HTML_TMPL.render(
		features=features,
		prediction=prediction,
		severity=severity,
		wt_text=wt.get("b64", ""),
		mut_text=mut.get("b64", ""),
		wt_ext=wt.get("ext", "pdb"),
		mut_ext=mut.get("ext", "pdb"),
		pymol_pml=pml,
	)
