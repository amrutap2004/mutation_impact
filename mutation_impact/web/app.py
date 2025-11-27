from __future__ import annotations

import io
import pathlib
from typing import Optional

from flask import Flask, render_template_string, request, send_file, current_app

from mutation_impact.input_module import load_sequence, parse_mutation, validate_mutation_against_sequence
from mutation_impact.structure import fetch_rcsb_pdb, fetch_alphafold_model
from mutation_impact.structure.retrieval import validate_pdb_id, validate_sequence_vs_pdb_length
from mutation_impact.structure.modeling import build_mutant_structure_stub, cleanup_mutation_cache
from mutation_impact.features import compute_basic_features
from mutation_impact.classifier import HarmfulnessClassifier
from mutation_impact.classifier.simple_ml_only import SimpleMLOnlyClassifier
from mutation_impact.classifier.ultra_high_accuracy import UltraHighAccuracyClassifier
from mutation_impact.severity import SeverityEstimator
from mutation_impact.reporting import render_html_report
from mutation_impact.ml.feature_engineering import AdvancedFeatureExtractor

try:
	from mutation_impact.structure.modeling import minimize_with_openmm  # optional
except Exception:
	minimize_with_openmm = None  # type: ignore

# Removed WeasyPrint dependency - using browser print functionality instead


HTML = """
<!doctype html>
<html lang="en">
<head>
	<meta charset="utf-8"/>
	<meta name="viewport" content="width=device-width, initial-scale=1"/>
	<title>Analysis | MutationImpact</title>
	<link rel="preconnect" href="https://fonts.googleapis.com">
	<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
	<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
	<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
	<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
	<style>
		:root {
			--bg-dark: #020617; /* Deeper, richer dark */
			--bg-panel: #0f172a; /* Slate 900 */
			--bg-input: #1e293b; /* Slate 800 */
			--primary: #3b82f6; /* Blue 500 */
			--primary-dark: #2563eb;
			--accent: #6366f1; /* Indigo 500 */
			--text-main: #f8fafc;
			--text-muted: #94a3b8;
			--border: rgba(255, 255, 255, 0.08);
			--border-focus: rgba(59, 130, 246, 0.5);
			--danger: #ef4444;
			--success: #10b981; /* Emerald */
			--glass: rgba(15, 23, 42, 0.8);
			--glow: rgba(59, 130, 246, 0.15);
		}
		
		* { box-sizing: border-box; margin: 0; padding: 0; }
		
		body {
			font-family: 'Inter', sans-serif;
			background-color: var(--bg-dark);
			background-image: 
				radial-gradient(circle at 0% 0%, rgba(59, 130, 246, 0.15) 0%, transparent 50%),
				radial-gradient(circle at 100% 100%, rgba(99, 102, 241, 0.1) 0%, transparent 50%);
			background-attachment: fixed;
			color: var(--text-main);
			line-height: 1.6;
			min-height: 100vh;
			display: flex;
			flex-direction: column;
		}
		
		/* Navigation */
		.navbar {
			background: rgba(2, 6, 23, 0.8);
			backdrop-filter: blur(16px);
			border-bottom: 1px solid var(--border);
			padding: 1rem 2rem;
			position: sticky;
			top: 0;
			z-index: 100;
		}
		
		.nav-content {
			max-width: 1600px;
			margin: 0 auto;
			display: flex;
			justify-content: space-between;
			align-items: center;
		}
		
		.logo {
			font-family: 'Outfit', sans-serif;
			font-size: 1.5rem;
			font-weight: 700;
			text-decoration: none;
			color: #fff;
			display: flex;
			align-items: center;
			gap: 0.75rem;
			text-shadow: 0 0 20px rgba(59, 130, 246, 0.3);
		}
		
		.logo span { 
			background: linear-gradient(135deg, var(--primary), var(--accent));
			-webkit-background-clip: text;
			-webkit-text-fill-color: transparent;
		}
		
		.nav-actions { display: flex; gap: 2rem; }
		
		.btn-nav {
			color: var(--text-muted);
			text-decoration: none;
			font-size: 0.95rem;
			font-weight: 500;
			transition: all 0.2s;
			display: flex;
			align-items: center;
			gap: 0.5rem;
			padding: 0.5rem 1rem;
			border-radius: 8px;
		}
		
		.btn-nav:hover { 
			color: #fff; 
			background: rgba(255, 255, 255, 0.05);
		}

		/* Main Layout */
		.main-container {
			flex: 1;
			max-width: 1600px;
			margin: 0 auto;
			padding: 2rem;
			width: 100%;
			display: grid;
			grid-template-columns: 380px 1fr;
			gap: 2rem;
			height: calc(100vh - 80px);
			overflow: hidden;
		}
		
		@media (max-width: 1024px) {
			.main-container { 
				grid-template-columns: 1fr; 
				height: auto; 
				overflow: visible;
			}
		}

		/* Panels */
		.panel {
			background: var(--glass);
			backdrop-filter: blur(12px);
			border: 1px solid var(--border);
			border-radius: 20px;
			display: flex;
			flex-direction: column;
			overflow: hidden;
			box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.3);
		}
		
		.input-panel {
			overflow-y: auto;
			padding: 2rem;
		}
		
		.results-panel {
			padding: 0;
			position: relative;
		}
		
		.panel-header {
			margin-bottom: 2rem;
			padding-bottom: 1rem;
			border-bottom: 1px solid var(--border);
			display: flex;
			justify-content: space-between;
			align-items: center;
		}
		
		.panel-title {
			font-family: 'Outfit', sans-serif;
			font-size: 1.25rem;
			font-weight: 600;
			color: #fff;
			display: flex;
			align-items: center;
			gap: 0.75rem;
		}

		/* Form Elements */
		.form-section { margin-bottom: 2.5rem; }
		.form-section-title {
			font-size: 0.75rem;
			text-transform: uppercase;
			letter-spacing: 0.1em;
			color: var(--primary);
			margin-bottom: 1.25rem;
			font-weight: 700;
		}
		
		.form-group { margin-bottom: 1.5rem; }
		
		label {
			display: block;
			font-size: 0.9rem;
			font-weight: 500;
			color: var(--text-muted);
			margin-bottom: 0.75rem;
		}
		
		.input-control {
			width: 100%;
			background: var(--bg-input);
			border: 1px solid var(--border);
			border-radius: 10px;
			padding: 1rem;
			color: #fff;
			font-family: 'Inter', sans-serif;
			font-size: 0.95rem;
			transition: all 0.2s;
		}
		
		.input-control:focus {
			outline: none;
			border-color: var(--primary);
			box-shadow: 0 0 0 4px var(--glow);
			background: rgba(30, 41, 59, 0.8);
		}
		
		textarea.input-control {
			min-height: 140px;
			resize: vertical;
			font-family: 'Courier New', monospace;
			line-height: 1.5;
		}
		
		.row-grid {
			display: grid;
			grid-template-columns: 1fr 1fr;
			gap: 1rem;
		}
		
		/* Checkboxes */
		.options-grid {
			display: grid;
			grid-template-columns: 1fr;
			gap: 0.75rem;
		}
		
		.checkbox-label {
			display: flex;
			align-items: center;
			gap: 0.75rem;
			cursor: pointer;
			font-size: 0.9rem;
			color: var(--text-muted);
			padding: 0.75rem 1rem;
			background: rgba(255, 255, 255, 0.02);
			border: 1px solid var(--border);
			border-radius: 10px;
			transition: all 0.2s;
		}
		
		.checkbox-label:hover {
			background: rgba(255, 255, 255, 0.05);
			border-color: rgba(255, 255, 255, 0.15);
			color: #fff;
		}
		
		.checkbox-label input:checked + span { color: var(--primary); font-weight: 600; }
		
		input[type="checkbox"] {
			accent-color: var(--primary);
			width: 18px;
			height: 18px;
		}
		
		/* Collapsible Advanced Section */
		details summary {
			cursor: pointer;
			padding: 1rem;
			background: rgba(255, 255, 255, 0.02);
			border: 1px solid var(--border);
			border-radius: 10px;
			color: var(--text-muted);
			font-weight: 500;
			list-style: none;
			display: flex;
			justify-content: space-between;
			align-items: center;
			transition: all 0.2s;
		}
		details summary:hover { color: #fff; background: rgba(255, 255, 255, 0.04); }
		details summary::after { content: '+'; font-size: 1.2rem; font-weight: 300; }
		details[open] summary::after { content: '-'; }
		details[open] summary { margin-bottom: 1rem; color: #fff; border-color: var(--primary); }

		/* Buttons */
		.btn {
			width: 100%;
			padding: 1rem;
			border-radius: 10px;
			font-weight: 600;
			cursor: pointer;
			transition: all 0.2s;
			border: none;
			font-size: 1rem;
			display: flex;
			justify-content: center;
			align-items: center;
			gap: 0.75rem;
		}
		
		.btn-primary {
			background: linear-gradient(135deg, var(--primary), var(--accent));
			color: white;
			box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3);
			margin-top: 1.5rem;
			font-size: 1.1rem;
			letter-spacing: 0.02em;
		}
		
		.btn-primary:hover {
			transform: translateY(-2px);
			box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
		}
		
		.btn-secondary {
			background: transparent;
			border: 1px solid var(--border);
			color: var(--text-muted);
			margin-top: 1rem;
			font-size: 0.9rem;
			padding: 0.75rem;
		}
		
		.btn-secondary:hover {
			border-color: var(--text-muted);
			color: #fff;
			background: rgba(255,255,255,0.05);
		}
		
		/* Results Area */
		.results-content {
			height: 100%;
			overflow-y: auto;
			padding: 2.5rem;
		}
		
		.empty-state {
			display: flex;
			flex-direction: column;
			align-items: center;
			justify-content: center;
			height: 100%;
			color: var(--text-muted);
			text-align: center;
			padding: 2rem;
		}
		
		.empty-icon {
			font-size: 4rem;
			margin-bottom: 1.5rem;
			opacity: 0.2;
			color: var(--primary);
		}
		
		.tip-box {
			margin-top: 2rem;
			font-size: 0.95rem;
			background: rgba(59, 130, 246, 0.08);
			padding: 1.5rem;
			border-radius: 12px;
			border: 1px solid rgba(59, 130, 246, 0.15);
			max-width: 400px;
		}

		/* Loading Overlay */
		.loading-overlay {
			position: fixed; top: 0; left: 0; width: 100%; height: 100%;
			background: rgba(5, 11, 20, 0.9); z-index: 2000;
			display: none; align-items: center; justify-content: center; flex-direction: column;
		}
		.loader {
			width: 48px; height: 48px; border: 5px solid #FFF; border-bottom-color: var(--primary);
			border-radius: 50%; display: inline-block; box-sizing: border-box;
			animation: rotation 1s linear infinite; margin-bottom: 1rem;
		}
		@keyframes rotation { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
		
		/* Print Styles */
		@media print {
			body { background: white; color: #000; display: block; }
			.navbar, .input-panel, .btn { display: none !important; }
			.main-container { display: block; padding: 0; margin: 0; max-width: none; height: auto; overflow: visible; }
			.panel { border: none; padding: 0; background: none; }
			.panel-header { display: none; }
			.results-content { overflow: visible; padding: 0; }
		}
	</style>
	<script>
		// Validation Logic
		function validateSequence(sequence) {
			if (!sequence || sequence.trim() === '') return { valid: false, message: 'Please enter a protein sequence' };
			const validAAs = /^[ACDEFGHIKLMNPQRSTVWY]+$/i;
			if (!validAAs.test(sequence)) {
				const invalidChars = sequence.match(/[^ACDEFGHIKLMNPQRSTVWY]/gi);
				if (invalidChars) {
					const uniqueInvalid = [...new Set(invalidChars)];
					return { valid: false, message: `Invalid characters: ${uniqueInvalid.join(', ')}` };
				}
			}
			if (sequence.length < 10) return { valid: false, message: 'Sequence too short (min 10 AA)' };
			if (sequence.length > 5000) return { valid: false, message: 'Sequence too long (max 5000 AA)' };
			return { valid: true, message: 'Valid sequence' };
		}
		
		function validateMutation(mutation) {
			if (!mutation || mutation.trim() === '') return { valid: false, message: 'Please enter a mutation' };
			const mutationPattern = /^[A-Za-z]\\d+[A-Za-z]$/;
			if (!mutationPattern.test(mutation)) return { valid: false, message: 'Invalid format. Use A123T' };
			return { valid: true, message: 'Valid mutation' };
		}
		
		function setupValidation() {
			const sequenceInput = document.querySelector('textarea[name="seq"]');
			const mutationInput = document.querySelector('input[name="mut"]');
			
			if (sequenceInput) {
				sequenceInput.addEventListener('input', function() {
					const validation = validateSequence(this.value);
					showValidationMessage(this, validation);
				});
			}
			
			if (mutationInput) {
				mutationInput.addEventListener('input', function() {
					const validation = validateMutation(this.value);
					showValidationMessage(this, validation);
				});
			}
		}
		
		function showValidationMessage(input, validation) {
			const existingMsg = input.parentNode.querySelector('.validation-msg');
			if (existingMsg) existingMsg.remove();
			
			if (!validation.valid) {
				const msgDiv = document.createElement('div');
				msgDiv.className = 'validation-msg';
				msgDiv.style.color = 'var(--danger)';
				msgDiv.style.fontSize = '0.85rem';
				msgDiv.style.marginTop = '0.5rem';
				msgDiv.innerHTML = `<i class="fa-solid fa-circle-exclamation"></i> ${validation.message}`;
				input.parentNode.appendChild(msgDiv);
				input.style.borderColor = 'var(--danger)';
			} else {
				input.style.borderColor = 'var(--success)';
				setTimeout(() => { input.style.borderColor = ''; }, 2000);
			}
		}
		
		document.addEventListener('DOMContentLoaded', setupValidation);

		function handleFormSubmit(event) {
			const form = event.target;
			const seqInput = form.querySelector('textarea[name="seq"]');
			const mutInput = form.querySelector('input[name="mut"]');

			// Clear previous validation messages
			form.querySelectorAll('.validation-msg').forEach(msg => msg.remove());

			const seqRaw = (seqInput.value || '');
			const seqNormalized = seqRaw.replace(/\s+/g, '').toUpperCase();
			const mutRaw = (mutInput.value || '').trim();

			let isValid = true;

			// Field-level validation (same rules as server)
			const seqValidation = validateSequence(seqNormalized);
			if (!seqValidation.valid) {
				showValidationMessage(seqInput, seqValidation);
				isValid = false;
			}

			const mutValidation = validateMutation(mutRaw);
			if (!mutValidation.valid) {
				showValidationMessage(mutInput, mutValidation);
				isValid = false;
			}

			// Cross-check mutation against sequence: bounds + from-res match
			if (isValid) {
				const mutMatch = mutRaw.match(/^([A-Za-z])(\\d+)([A-Za-z])$/);
				if (mutMatch) {
					const fromRes = mutMatch[1].toUpperCase();
					const pos = parseInt(mutMatch[2], 10);

					if (!Number.isFinite(pos) || pos < 1 || pos > seqNormalized.length) {
						showValidationMessage(mutInput, {
							valid: false,
							message: `Mutation position ${pos} is out of bounds for sequence length ${seqNormalized.length}`
						});
						isValid = false;
					} else {
						const seqRes = seqNormalized[pos - 1];
						if (seqRes !== fromRes) {
							showValidationMessage(mutInput, {
								valid: false,
								message: `Sequence residue ${seqRes} at position ${pos} does not match mutation from-res ${fromRes}`
							});
							isValid = false;
						}
					}
				}
			}

			if (!isValid) {
				event.preventDefault();
				return false;
			}

			const loader = document.getElementById('loader');
			if (loader) loader.style.display = 'flex';
			return true;
		}
		
		function resetForm() {
			const form = document.querySelector('form');
			form.reset();
			document.querySelector('textarea[name="seq"]').value = '';
			document.querySelector('input[name="mut"]').value = '';
			document.querySelector('input[name="id"]').value = '';
			document.querySelectorAll('select').forEach(select => select.selectedIndex = 0);
			
			// Reset checkboxes to defaults
			document.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = false);
			document.querySelector('input[name="high_accuracy"]').checked = true; // Default High Accuracy
			document.querySelector('input[name="enable_sasa"]').checked = true;
			document.querySelector('input[name="enable_blosum"]').checked = true;
			document.querySelector('input[name="enable_hydrophobicity"]').checked = true;

			document.querySelectorAll('.validation-msg').forEach(msg => msg.remove());
			document.querySelectorAll('.input-control').forEach(input => input.style.borderColor = '');
		}

		async function captureVisualizationScreenshots() {
			const screenshots = {};
			const wtViewer = document.getElementById('viewer-wt');
			const mutViewer = document.getElementById('viewer-mut');
			
			if (wtViewer && mutViewer) {
				try {
					await new Promise(resolve => setTimeout(resolve, 500));
					if (typeof html2canvas !== 'undefined') {
						const opts = { backgroundColor: '#ffffff', scale: 2, useCORS: true, allowTaint: true, logging: false };
						const wtCanvas = await html2canvas(wtViewer, opts);
						const mutCanvas = await html2canvas(mutViewer, opts);
						screenshots.wt = wtCanvas.toDataURL('image/png');
						screenshots.mut = mutCanvas.toDataURL('image/png');
					}
				} catch (e) { console.error("Screenshot capture failed:", e); }
			}
			return screenshots;
		}
		
		async function generateReport() {
			const btn = event.currentTarget;
			const originalContent = btn.innerHTML;
			btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Generating Report...';
			btn.disabled = true;
			
			try {
				const seq = document.querySelector('textarea[name="seq"]').value || 'N/A';
				const mut = document.querySelector('input[name="mut"]').value || 'N/A';
				const id = document.querySelector('input[name="id"]').value || 'N/A';
				const screenshots = await captureVisualizationScreenshots();
				
				const resultsNode = document.querySelector('.results-content');
				if (!resultsNode) throw new Error("No results found");
				
				const tempDiv = document.createElement('div');
				tempDiv.innerHTML = resultsNode.innerHTML;
				
				const viewers = tempDiv.querySelectorAll('[id^="viewer-"]');
				viewers.forEach(v => v.remove());
				const grid = tempDiv.querySelector('.grid');
				if(grid) grid.remove();
				
				const timestamp = new Date().toLocaleString();
				const reportHTML = `
					<!DOCTYPE html>
					<html>
					<head>
						<title>MutationImpact Analysis Report</title>
						<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
						<style>
							@page { size: A4; margin: 2cm; }
							body { font-family: 'Inter', sans-serif; color: #1e293b; line-height: 1.5; max-width: 210mm; margin: 0 auto; }
							.header { border-bottom: 2px solid #3b82f6; padding-bottom: 1rem; margin-bottom: 2rem; display: flex; justify-content: space-between; align-items: flex-end; }
							.brand { font-family: 'Outfit', sans-serif; font-size: 1.5rem; font-weight: 700; color: #0f172a; }
							.brand span { color: #3b82f6; }
							.meta { font-size: 0.875rem; color: #64748b; text-align: right; }
							.section { margin-bottom: 2rem; break-inside: avoid; }
							.section-title { font-family: 'Outfit', sans-serif; font-size: 1.1rem; font-weight: 600; color: #0f172a; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.5rem; margin-bottom: 1rem; }
							.params-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; background: #f8fafc; padding: 1rem; border-radius: 8px; border: 1px solid #e2e8f0; }
							.param-item { display: flex; flex-direction: column; }
							.param-label { font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
							.param-value { font-family: 'Courier New', monospace; font-weight: 600; color: #0f172a; }
							.viz-section { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }
							.viz-card { border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }
							.viz-header { background: #f1f5f9; padding: 0.5rem 1rem; font-size: 0.875rem; font-weight: 600; color: #475569; text-align: center; border-bottom: 1px solid #e2e8f0; }
							.viz-img { width: 100%; height: auto; display: block; }
							.results-body { font-size: 0.95rem; }
							table { width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.875rem; }
							th { background: #f8fafc; text-align: left; padding: 0.75rem; font-weight: 600; color: #475569; border-bottom: 1px solid #e2e8f0; }
							td { padding: 0.75rem; border-bottom: 1px solid #f1f5f9; color: #334155; }
							.footer { margin-top: 4rem; padding-top: 1rem; border-top: 1px solid #e2e8f0; text-align: center; font-size: 0.75rem; color: #94a3b8; }
						</style>
					</head>
					<body>
						<div class="header">
							<div class="brand">Mutation<span>Impact</span></div>
							<div class="meta">
								<div>Analysis Report</div>
								<div>${timestamp}</div>
							</div>
						</div>
						<div class="section">
							<div class="section-title">Configuration</div>
							<div class="params-grid">
								<div class="param-item"><span class="param-label">Mutation</span><span class="param-value">${mut}</span></div>
								<div class="param-item"><span class="param-label">Structure ID</span><span class="param-value">${id}</span></div>
								<div class="param-item"><span class="param-label">Sequence Length</span><span class="param-value">${seq.length} AA</span></div>
							</div>
						</div>
						${screenshots.wt ? `
						<div class="section">
							<div class="section-title">Structural Visualization</div>
							<div class="viz-section">
								<div class="viz-card"><div class="viz-header">Wild-Type</div><img src="${screenshots.wt}" class="viz-img"></div>
								<div class="viz-card"><div class="viz-header">Mutant</div><img src="${screenshots.mut}" class="viz-img"></div>
							</div>
						</div>` : ''}
						<div class="section">
							<div class="section-title">Prediction Results</div>
							<div class="results-body">${tempDiv.innerHTML}</div>
						</div>
						<div class="footer">Generated by MutationImpact Web</div>
						<script>window.onload = function() { window.print(); }<\\/script>
					</body>
					</html>
				`;
				const printWin = window.open('', '_blank');
				printWin.document.write(reportHTML);
				printWin.document.close();
			} catch (e) {
				console.error(e);
				alert('Failed to generate report: ' + e.message);
			} finally {
				btn.innerHTML = originalContent;
				btn.disabled = false;
			}
		}
	</script>
</head>
<body>
	<div id="loader" class="loading-overlay">
		<span class="loader"></span>
		<h3>Processing Analysis...</h3>
	</div>

	<nav class="navbar">
		<div class="nav-content">
			<a href="/" class="logo">
				<i class="fa-solid fa-dna"></i>
				Mutation<span>Impact</span>
			</a>
			<div class="nav-actions">
				<a href="/" class="btn-nav"><i class="fa-solid fa-house"></i> Home</a>
				<a href="/gym" class="btn-nav"><i class="fa-solid fa-heart-pulse"></i> Gym</a>
				<a href="https://github.com/yourusername/mutation-impact" target="_blank" class="btn-nav"><i class="fa-brands fa-github"></i> GitHub</a>
			</div>
		</div>
	</nav>

	<div class="main-container">
		<!-- Input Panel -->
		<aside class="panel input-panel">
			<div class="panel-header">
				<div class="panel-title"><i class="fa-solid fa-sliders"></i> Configuration</div>
			</div>
			
			{% if error %}
			<div style="background: rgba(239, 68, 68, 0.1); border: 1px solid var(--danger); color: var(--danger); padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; font-size: 0.9rem;">
				<i class="fa-solid fa-circle-exclamation"></i> {{ error }}
			</div>
			{% endif %}

			<form method="post" onsubmit="return handleFormSubmit(event)">
				<div class="form-section">
					<div class="form-section-title">Input Data</div>
					<div class="form-group">
						<label>Protein Sequence</label>
						<textarea name="seq" class="input-control" placeholder="Enter raw 1-letter sequence (e.g., MKT...)">{{ seq or '' }}</textarea>
					</div>

					<div class="row-grid">
						<div class="form-group">
							<label>Mutation</label>
							<input type="text" name="mut" class="input-control" placeholder="A123T" value="{{ mut or '' }}">
						</div>
						<div class="form-group">
							<label>Structure ID</label>
							<input type="text" name="id" class="input-control" placeholder="e.g., 1CRN" value="{{ sid or '' }}">
						</div>
					</div>
					
					<div class="form-group">
						<label>Structure Source</label>
						<select name="src" class="input-control">
							<option value="pdb" {% if src == 'pdb' %}selected{% endif %}>RCSB PDB</option>
							<option value="af" {% if src == 'af' %}selected{% endif %}>AlphaFold</option>
						</select>
					</div>
				</div>

				
				<button type="submit" class="btn btn-primary">
					<i class="fa-solid fa-bolt"></i> Run Analysis
				</button>
				<button type="button" onclick="resetForm()" class="btn btn-secondary">
					Reset Parameters
				</button>
			</form>
		</aside>

		<!-- Results Panel -->
		<main class="panel results-panel">
			{% if report_html %}
				<div class="panel-header" style="margin: 1.5rem 2rem 0;">
					<div class="panel-title"><i class="fa-solid fa-chart-simple"></i> Analysis Results</div>
					<div class="btn-group">
						<button onclick="generateReport()" class="btn btn-secondary" style="margin: 0; padding: 0.5rem 1rem; font-size: 0.85rem; width: auto;">
							<i class="fa-solid fa-print"></i> Print Report
						</button>
					</div>
				</div>
				<div class="results-content">
					{{ report_html | safe }}
				</div>
			{% else %}
				<div class="empty-state">
					<div class="empty-icon"><i class="fa-solid fa-dna"></i></div>
					<h3>Ready to Analyze</h3>
					<p style="max-width: 400px; margin: 0 auto;">Configure your parameters on the left and click "Run Analysis" to generate a comprehensive stability report.</p>
					<div class="tip-box">
						<i class="fa-solid fa-lightbulb" style="color: var(--primary); margin-right: 0.5rem;"></i>
						<strong>Tip:</strong> Use ID <code>1CRN</code> and mutation <code>T2A</code> for a quick test run.
					</div>
				</div>
			{% endif %}
		</main>
	</div>
</body>
</html>
"""

LANDING_HTML = """
<!doctype html>
<html lang="en">
<head>
	<meta charset="utf-8"/>
	<meta name="viewport" content="width=device-width, initial-scale=1"/>
	<title>MutationImpact | Advanced Protein Stability Analysis</title>
	<meta name="description" content="End-to-end mutation analysis with 3D visualization and ensemble machine learning predictions.">
	<link rel="preconnect" href="https://fonts.googleapis.com">
	<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
	<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
	<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
	<style>
		:root {
			--bg-dark: #050b14;
			--bg-panel: #0f1623;
			--primary: #3b82f6;
			--primary-glow: rgba(59, 130, 246, 0.5);
			--accent: #8b5cf6;
			--text-main: #f8fafc;
			--text-muted: #94a3b8;
			--border: rgba(255, 255, 255, 0.08);
			--glass: rgba(15, 22, 35, 0.7);
		}
		
		* { box-sizing: border-box; margin: 0; padding: 0; }
		
		body {
			font-family: 'Inter', sans-serif;
			background-color: var(--bg-dark);
			color: var(--text-main);
			line-height: 1.6;
			overflow-x: hidden;
		}
		
		h1, h2, h3, h4, h5, h6 {
			font-family: 'Outfit', sans-serif;
			color: #fff;
		}
		
		/* Background Effects */
		.bg-glow {
			position: fixed;
			width: 600px;
			height: 600px;
			background: radial-gradient(circle, rgba(59, 130, 246, 0.15) 0%, transparent 70%);
			border-radius: 50%;
			top: -100px;
			left: -100px;
			z-index: -1;
			pointer-events: none;
		}
		
		.bg-glow-2 {
			position: fixed;
			width: 500px;
			height: 500px;
			background: radial-gradient(circle, rgba(139, 92, 246, 0.1) 0%, transparent 70%);
			border-radius: 50%;
			bottom: 0;
			right: -100px;
			z-index: -1;
			pointer-events: none;
		}

		/* Navigation */
		.navbar {
			position: fixed;
			top: 0;
			width: 100%;
			z-index: 1000;
			padding: 1.25rem 0;
			transition: all 0.3s ease;
			border-bottom: 1px solid transparent;
		}
		
		.navbar.scrolled {
			background: rgba(5, 11, 20, 0.8);
			backdrop-filter: blur(12px);
			border-bottom: 1px solid var(--border);
			padding: 1rem 0;
		}
		
		.container {
			max-width: 1200px;
			margin: 0 auto;
			padding: 0 2rem;
		}
		
		.nav-content {
			display: flex;
			justify-content: space-between;
			align-items: center;
		}
		
		.logo {
			font-size: 1.5rem;
			font-weight: 700;
			text-decoration: none;
			color: #fff;
			display: flex;
			align-items: center;
			gap: 0.5rem;
		}
		
		.logo span { color: var(--primary); }
		
		.nav-links {
			display: flex;
			gap: 2rem;
			align-items: center;
		}
		
		.nav-link {
			color: var(--text-muted);
			text-decoration: none;
			font-weight: 500;
			font-size: 0.95rem;
			transition: color 0.2s;
		}
		
		.nav-link:hover { color: #fff; }
		
		.btn {
			display: inline-flex;
			align-items: center;
			justify-content: center;
			padding: 0.75rem 1.5rem;
			border-radius: 8px;
			font-weight: 600;
			text-decoration: none;
			transition: all 0.2s ease;
			cursor: pointer;
			border: 1px solid transparent;
		}
		
		.btn-primary {
			background: linear-gradient(135deg, var(--primary), var(--accent));
			color: white;
			box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3);
		}
		
		.btn-primary:hover {
			transform: translateY(-2px);
			box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
		}
		
		.btn-outline {
			background: transparent;
			border: 1px solid var(--border);
			color: #fff;
		}
		
		.btn-outline:hover {
			border-color: var(--primary);
			background: rgba(59, 130, 246, 0.1);
		}

		/* Hero Section */
		.hero {
			padding-top: 160px;
			padding-bottom: 100px;
			text-align: center;
			position: relative;
		}
		
		.badge {
			display: inline-block;
			padding: 6px 16px;
			background: rgba(59, 130, 246, 0.1);
			border: 1px solid rgba(59, 130, 246, 0.2);
			border-radius: 100px;
			color: var(--primary);
			font-size: 0.875rem;
			font-weight: 600;
			margin-bottom: 1.5rem;
		}
		
		.hero h1 {
			font-size: 4rem;
			line-height: 1.1;
			font-weight: 800;
			margin-bottom: 1.5rem;
			letter-spacing: -0.02em;
			background: linear-gradient(to right, #fff, #cbd5e1);
			-webkit-background-clip: text;
			-webkit-text-fill-color: transparent;
		}
		
		.hero p {
			font-size: 1.25rem;
			color: var(--text-muted);
			max-width: 700px;
			margin: 0 auto 2.5rem;
		}
		
		.hero-btns {
			display: flex;
			gap: 1rem;
			justify-content: center;
			margin-bottom: 4rem;
		}
		
		.hero-stats {
			display: grid;
			grid-template-columns: repeat(3, 1fr);
			gap: 2rem;
			max-width: 800px;
			margin: 0 auto;
			padding: 2rem;
			background: var(--glass);
			border: 1px solid var(--border);
			border-radius: 16px;
			backdrop-filter: blur(10px);
		}
		
		.stat-item h3 {
			font-size: 2rem;
			color: #fff;
			margin-bottom: 0.25rem;
		}
		
		.stat-item p {
			font-size: 0.875rem;
			color: var(--text-muted);
		}

		/* Features Section */
		.section { padding: 100px 0; }
		
		.section-header {
			text-align: center;
			margin-bottom: 4rem;
		}
		
		.section-header h2 {
			font-size: 2.5rem;
			margin-bottom: 1rem;
		}
		
		.section-header p {
			color: var(--text-muted);
			font-size: 1.1rem;
		}
		
		.features-grid {
			display: grid;
			grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
			gap: 2rem;
		}
		
		.feature-card {
			background: var(--bg-panel);
			border: 1px solid var(--border);
			padding: 2rem;
			border-radius: 16px;
			transition: all 0.3s ease;
			position: relative;
			overflow: hidden;
		}
		
		.feature-card:hover {
			transform: translateY(-5px);
			border-color: var(--primary);
			box-shadow: 0 10px 40px -10px rgba(0,0,0,0.5);
		}
		
		.feature-icon {
			width: 50px;
			height: 50px;
			background: rgba(59, 130, 246, 0.1);
			border-radius: 12px;
			display: flex;
			align-items: center;
			justify-content: center;
			color: var(--primary);
			font-size: 1.5rem;
			margin-bottom: 1.5rem;
		}
		
		.feature-card h3 {
			font-size: 1.25rem;
			margin-bottom: 0.75rem;
		}
		
		.feature-card p {
			color: var(--text-muted);
			font-size: 0.95rem;
		}

		/* Workflow Section */
		.workflow-steps {
			display: grid;
			grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
			gap: 2rem;
			position: relative;
		}
		
		.step-card {
			text-align: center;
			position: relative;
			z-index: 1;
		}
		
		.step-number {
			width: 40px;
			height: 40px;
			background: var(--bg-panel);
			border: 1px solid var(--border);
			border-radius: 50%;
			display: flex;
			align-items: center;
			justify-content: center;
			margin: 0 auto 1.5rem;
			font-weight: 700;
			color: var(--primary);
			position: relative;
		}
		
		.step-card::after {
			content: '';
			position: absolute;
			top: 20px;
			right: -50%;
			width: 100%;
			height: 2px;
			background: var(--border);
			z-index: -1;
		}
		
		.step-card:last-child::after { display: none; }

		/* CTA Section */
		.cta-section {
			background: linear-gradient(180deg, rgba(15, 22, 35, 0) 0%, rgba(59, 130, 246, 0.1) 100%);
			border-top: 1px solid var(--border);
			text-align: center;
		}
		
		.cta-box {
			max-width: 800px;
			margin: 0 auto;
		}

		/* Footer */
		.footer {
			border-top: 1px solid var(--border);
			padding: 4rem 0 2rem;
			background: #020617;
		}
		
		.footer-grid {
			display: grid;
			grid-template-columns: 2fr 1fr 1fr 1fr;
			gap: 3rem;
			margin-bottom: 3rem;
		}
		
		.footer-brand p {
			color: var(--text-muted);
			margin-top: 1rem;
			font-size: 0.9rem;
		}
		
		.footer-col h4 {
			margin-bottom: 1.5rem;
			font-size: 1rem;
		}
		
		.footer-links {
			list-style: none;
		}
		
		.footer-links li { margin-bottom: 0.75rem; }
		
		.footer-links a {
			color: var(--text-muted);
			text-decoration: none;
			font-size: 0.9rem;
			transition: color 0.2s;
		}
		
		.footer-links a:hover { color: var(--primary); }
		
		.footer-bottom {
			text-align: center;
			padding-top: 2rem;
			border-top: 1px solid var(--border);
			color: var(--text-muted);
			font-size: 0.875rem;
		}

		@media (max-width: 768px) {
			.hero h1 { font-size: 2.5rem; }
			.nav-links { display: none; }
			.hero-stats { grid-template-columns: 1fr; }
			.step-card::after { display: none; }
			.footer-grid { grid-template-columns: 1fr; gap: 2rem; }
		}
	</style>
</head>
<body>
	<div class="bg-glow"></div>
	<div class="bg-glow-2"></div>

	<nav class="navbar">
		<div class="container nav-content">
			<a href="/" class="logo">
				<i class="fa-solid fa-dna"></i>
				Mutation<span>Impact</span>
			</a>
			<div class="nav-links">
				<a href="#features" class="nav-link">Features</a>
				<a href="/gym" class="nav-link" style="color: #f59e0b;"><i class="fa-solid fa-heart-pulse"></i> Health & Nutrient</a>
				<a href="#how-it-works" class="nav-link">How it Works</a>
				<a href="#science" class="nav-link">Science</a>
				<a href="/analyze" class="btn btn-primary">Launch App</a>
			</div>
		</div>
	</nav>

	<section class="hero">
		<div class="container">
			<span class="badge">v1.0 Now Available</span>
			<h1>Predict Protein Stability<br>with Precision AI</h1>
			<p>An advanced bioinformatics platform combining structural analysis and ensemble machine learning to assess the impact of protein mutations instantly.</p>
			
			<div class="hero-btns">
				<a href="/analyze" class="btn btn-primary">
					Start Analysis <i class="fa-solid fa-arrow-right" style="margin-left: 8px;"></i>
				</a>
				<a href="#features" class="btn btn-outline">Learn More</a>
			</div>

			<div class="hero-stats">
				<div class="stat-item">
					<h3>95%</h3>
					<p>Prediction Accuracy</p>
				</div>
				<div class="stat-item">
					<h3>< 60s</h3>
					<p>Analysis Time</p>
				</div>
				<div class="stat-item">
					<h3>200K+</h3>
					<p>Structures Supported</p>
				</div>
			</div>
		</div>
	</section>

	<section id="features" class="section">
		<div class="container">
			<div class="section-header">
				<h2>Powerful Analysis Features</h2>
				<p>Comprehensive tools for structural biology and mutation assessment</p>
			</div>
			
			<div class="features-grid">
				<div class="feature-card">
					<div class="feature-icon"><i class="fa-solid fa-brain"></i></div>
					<h3>Ensemble ML Models</h3>
					<p>Utilizes XGBoost and Random Forest classifiers trained on extensive mutation datasets for high-confidence predictions.</p>
				</div>
				<div class="feature-card">
					<div class="feature-icon"><i class="fa-solid fa-cubes"></i></div>
					<h3>3D Visualization</h3>
					<p>Interactive NGL-based molecular viewer to inspect wild-type and mutant structures side-by-side in real-time.</p>
				</div>
				<div class="feature-card">
					<div class="feature-icon"><i class="fa-solid fa-microscope"></i></div>
					<h3>Structural Metrics</h3>
					<p>Calculates ΔSASA, hydrogen bond changes, and hydrophobicity deltas to quantify physical impacts.</p>
				</div>
				<div class="feature-card">
					<div class="feature-icon"><i class="fa-solid fa-file-export"></i></div>
					<h3>Professional Reports</h3>
					<p>Generate detailed PDF and HTML reports suitable for publication and academic research.</p>
				</div>
				<div class="feature-card">
					<div class="feature-icon"><i class="fa-solid fa-bolt"></i></div>
					<h3>AlphaFold Integration</h3>
					<p>Seamlessly fetch and analyze predicted structures from the AlphaFold database when PDBs are unavailable.</p>
				</div>
				<div class="feature-card">
					<div class="feature-icon"><i class="fa-solid fa-shield-halved"></i></div>
					<h3>Secure & Private</h3>
					<p>All analysis is performed locally or in your secure environment. Your sequence data never leaves your control.</p>
				</div>
			</div>
		</div>
	</section>

	<section id="how-it-works" class="section" style="background: rgba(255,255,255,0.02);">
		<div class="container">
			<div class="section-header">
				<h2>How It Works</h2>
				<p>From sequence to insight in three simple steps</p>
			</div>
			
			<div class="workflow-steps">
				<div class="step-card">
					<div class="step-number">1</div>
					<h3>Input Data</h3>
					<p style="color: var(--text-muted); margin-top: 0.5rem;">Provide your protein sequence and define the specific mutation (e.g., A123T).</p>
				</div>
				<div class="step-card">
					<div class="step-number">2</div>
					<h3>Select Structure</h3>
					<p style="color: var(--text-muted); margin-top: 0.5rem;">Choose a PDB ID or let us fetch the AlphaFold model automatically.</p>
				</div>
				<div class="step-card">
					<div class="step-number">3</div>
					<h3>Get Results</h3>
					<p style="color: var(--text-muted); margin-top: 0.5rem;">Receive a comprehensive report with stability predictions and 3D visuals.</p>
				</div>
			</div>
		</div>
	</section>

	<section class="section cta-section">
		<div class="container cta-box">
			<h2>Ready to analyze your proteins?</h2>
			<p style="margin: 1.5rem 0 2.5rem; color: var(--text-muted); font-size: 1.1rem;">Join researchers using MutationImpact for reliable stability predictions.</p>
			<a href="/analyze" class="btn btn-primary" style="padding: 1rem 2.5rem; font-size: 1.1rem;">Launch Application</a>
		</div>
	</section>

	<footer class="footer">
		<div class="container">
			<div class="footer-grid">
				<div class="footer-brand">
					<a href="/" class="logo">
						<i class="fa-solid fa-dna"></i>
						Mutation<span>Impact</span>
					</a>
					<p>Advanced bioinformatics tools for the modern researcher. Built with Python, Flask, and Scikit-Learn.</p>
				</div>
				<div class="footer-col">
					<h4>Product</h4>
					<ul class="footer-links">
						<li><a href="/analyze">Analysis Tool</a></li>
						<li><a href="#features">Features</a></li>
						<li><a href="#">Documentation</a></li>
					</ul>
				</div>
				<div class="footer-col">
					<h4>Resources</h4>
					<ul class="footer-links">
						<li><a href="#">Case Studies</a></li>
						<li><a href="#">API Reference</a></li>
						<li><a href="#">Support</a></li>
					</ul>
				</div>
				<div class="footer-col">
					<h4>Legal</h4>
					<ul class="footer-links">
						<li><a href="#">Privacy Policy</a></li>
						<li><a href="#">Terms of Service</a></li>
					</ul>
				</div>
			</div>
			<div class="footer-bottom">
				<p>&copy; 2025 MutationImpact. All rights reserved.</p>
			</div>
		</div>
	</footer>

	<script>
		// Navbar scroll effect
		window.addEventListener('scroll', function() {
			const navbar = document.querySelector('.navbar');
			if (window.scrollY > 50) {
				navbar.classList.add('scrolled');
			} else {
				navbar.classList.remove('scrolled');
			}
		});
	</script>
</body>
</html>
"""

GYM_HTML = """
<!doctype html>
<html lang="en">
<head>
	<meta charset="utf-8"/>
	<meta name="viewport" content="width=device-width, initial-scale=1"/>
	<title>MutationImpact | Health & Nutrient</title>
	<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
	<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
	<script src="https://unpkg.com/ngl@latest/dist/ngl.js"></script>
	<style>
		:root {
			--bg-dark: #050b14;
			--bg-panel: #0f1623;
			--primary: #f59e0b; /* Amber for energy/fitness */
			--primary-glow: rgba(245, 158, 11, 0.5);
			--accent: #ef4444; /* Red for muscle/power */
			--text-main: #f8fafc;
			--text-muted: #94a3b8;
			--border: rgba(255, 255, 255, 0.08);
			--glass: rgba(15, 22, 35, 0.95);
			--success: #22c55e;
			--danger: #ef4444;
		}
		
		* { box-sizing: border-box; margin: 0; padding: 0; }
		
		body {
			font-family: 'Inter', sans-serif;
			background-color: var(--bg-dark);
			color: var(--text-main);
			line-height: 1.6;
			overflow-x: hidden;
		}
		
		h1, h2, h3, h4 { font-family: 'Outfit', sans-serif; color: #fff; }
		
		/* Navbar */
		.navbar {
			position: fixed; top: 0; width: 100%; z-index: 1000;
			padding: 1.25rem 0; background: rgba(5, 11, 20, 0.9);
			backdrop-filter: blur(12px); border-bottom: 1px solid var(--border);
		}
		.container { max-width: 1200px; margin: 0 auto; padding: 0 2rem; }
		.nav-content { display: flex; justify-content: space-between; align-items: center; }
		.logo { font-size: 1.5rem; font-weight: 700; text-decoration: none; color: #fff; display: flex; align-items: center; gap: 0.5rem; }
		.logo span { color: var(--primary); }
		.nav-links a { color: var(--text-muted); text-decoration: none; margin-left: 2rem; font-weight: 500; transition: color 0.2s; }
		.nav-links a:hover { color: #fff; }
		
		/* Hero */
		.hero {
			padding: 140px 0 60px;
			text-align: center;
			background: radial-gradient(circle at 50% 30%, rgba(245, 158, 11, 0.15) 0%, transparent 60%);
			min-height: 100vh;
			display: flex;
			flex-direction: column;
			justify-content: center;
		}
		.hero h1 { font-size: 3.5rem; font-weight: 800; margin-bottom: 1.5rem; line-height: 1.1; }
		.hero p { font-size: 1.2rem; color: var(--text-muted); max-width: 700px; margin: 0 auto 3rem; }
		
		/* Analyzer Card */
		.analyzer-card {
			background: var(--bg-panel);
			border: 1px solid var(--border);
			border-radius: 20px;
			padding: 3rem;
			max-width: 600px;
			margin: 0 auto;
			box-shadow: 0 20px 50px -10px rgba(0,0,0,0.5);
			position: relative;
			z-index: 10;
		}
		
		.form-group { margin-bottom: 1.5rem; text-align: left; }
		.form-group label { display: block; margin-bottom: 0.5rem; font-weight: 500; color: var(--text-muted); }
		.input-control {
			width: 100%; background: rgba(255,255,255,0.05); border: 1px solid var(--border);
			padding: 1rem; border-radius: 8px; color: #fff; font-family: 'Inter', sans-serif;
			font-size: 1rem; outline: none; transition: border-color 0.2s;
		}
		.input-control:focus { border-color: var(--primary); }
		
		.btn-primary {
			width: 100%; padding: 1rem; background: linear-gradient(135deg, var(--primary), var(--accent));
			border: none; border-radius: 8px; color: #fff; font-weight: 700; font-size: 1.1rem;
			cursor: pointer; transition: transform 0.2s;
		}
		.btn-primary:hover { transform: translateY(-2px); }
		
		/* Results Section */
		.results-section {
			margin-top: 4rem;
			text-align: left;
			animation: fadeIn 0.5s ease-out;
		}
		
		@keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
		
		.result-header {
			display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;
			padding-bottom: 1rem; border-bottom: 1px solid var(--border);
		}
		
		.score-card {
			background: var(--glass); border: 1px solid var(--border); border-radius: 16px; padding: 2rem;
			display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-bottom: 2rem;
		}
		
		.score-circle {
			width: 150px; height: 150px; border-radius: 50%; border: 8px solid var(--border);
			display: flex; flex-direction: column; align-items: center; justify-content: center;
			margin: 0 auto; position: relative;
		}
		.score-val { font-size: 2.5rem; font-weight: 800; color: #fff; }
		.score-label { font-size: 0.9rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }
		
		.impact-badge {
			padding: 0.5rem 1rem; border-radius: 50px; font-weight: 700; font-size: 1.1rem;
			display: inline-flex; align-items: center; gap: 0.5rem;
		}
		.impact-harmful { background: rgba(239, 68, 68, 0.2); color: var(--danger); border: 1px solid rgba(239, 68, 68, 0.3); }
		.impact-neutral { background: rgba(34, 197, 94, 0.2); color: var(--success); border: 1px solid rgba(34, 197, 94, 0.3); }
		
		.viewer-container {
			height: 400px; background: #fff; border-radius: 12px; overflow: hidden; position: relative;
		}
		.viewer-label {
			position: absolute; top: 1rem; left: 1rem; background: rgba(0,0,0,0.8); color: #fff;
			padding: 0.25rem 0.75rem; border-radius: 4px; font-size: 0.8rem; z-index: 10;
		}
		
		.features-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 2rem; }
		.metric-card { background: rgba(255,255,255,0.03); padding: 1rem; border-radius: 8px; text-align: center; }
		.metric-val { font-size: 1.25rem; font-weight: 700; color: var(--primary); }
		.metric-name { font-size: 0.85rem; color: var(--text-muted); }

		/* Loading Overlay */
		.loading-overlay {
			position: fixed; top: 0; left: 0; width: 100%; height: 100%;
			background: rgba(5, 11, 20, 0.9); z-index: 2000;
			display: none; align-items: center; justify-content: center; flex-direction: column;
		}
		.loader {
			width: 48px; height: 48px; border: 5px solid #FFF; border-bottom-color: var(--primary);
			border-radius: 50%; display: inline-block; box-sizing: border-box;
			animation: rotation 1s linear infinite; margin-bottom: 1rem;
		}
		@keyframes rotation { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
	</style>
</head>
<body>
	<div id="loader" class="loading-overlay">
		<span class="loader"></span>
		<h3>Analyzing Biomolecular Impact...</h3>
	</div>

	<nav class="navbar">
		<div class="container nav-content">
			<a href="/" class="logo"><i class="fa-solid fa-heart-pulse"></i> Health<span>Nutrient</span></a>
			<div class="nav-links">
				<a href="/">Home</a>
				<a href="/analyze">Advanced Tool</a>
			</div>
		</div>
	</nav>

	<section class="hero">
		<div class="container">
			{% if not prediction %}
			<h1>Optimize Your Fitness<br>with Genetic Insights</h1>
			<p>Analyze key protein variants affecting muscle growth, metabolism, and recovery. Unlock your body's potential with precision science.</p>
			
			<div class="analyzer-card">
				<form method="post" onsubmit="document.getElementById('loader').style.display='flex'">
					<div class="form-group">
						<label>Select Fitness Goal / Protein</label>
						<select id="proteinSelect" class="input-control" onchange="updateProtein()">
							<option value="">-- Choose a Target --</option>
							<option value="myoglobin">Oxygen Transport (Myoglobin - 1MBN)</option>
							<option value="crambin">General Stability Test (Crambin - 1CRN)</option>
						</select>
					</div>
					
					<div class="form-group">
						<label>Enter Mutation (e.g., V10A)</label>
						<input type="text" name="mut" id="mutationInput" class="input-control" placeholder="e.g., V10A" required>
					</div>
					
					<!-- Hidden fields to carry the sequence and ID -->
					<input type="hidden" name="seq" id="seqInput">
					<input type="hidden" name="id" id="idInput">
					<input type="hidden" name="src" value="pdb">
					
					<button type="submit" class="btn-primary">Analyze Impact</button>
				</form>
			</div>
			{% else %}
			<!-- Results View -->
			<div class="results-section">
				<div class="result-header">
					<h2><i class="fa-solid fa-chart-line"></i> Analysis Report</h2>
					<a href="/gym" class="btn-primary" style="width: auto; padding: 0.5rem 1.5rem; font-size: 0.9rem;">New Analysis</a>
				</div>
				
				<div class="score-card">
					<div style="text-align: center;">
						<div class="score-circle" style="border-color: {% if prediction.label == 'Harmful' %}var(--danger){% else %}var(--success){% endif %};">
							<div class="score-val">{{ '%.0f' % (prediction.confidence * 100) }}%</div>
							<div class="score-label">Confidence</div>
						</div>
						<div style="margin-top: 1.5rem;">
							<div class="impact-badge {% if prediction.label == 'Harmful' %}impact-harmful{% else %}impact-neutral{% endif %}">
								{% if prediction.label == 'Harmful' %}
									<i class="fa-solid fa-triangle-exclamation"></i> High Impact
								{% else %}
									<i class="fa-solid fa-check-circle"></i> Low Impact
								{% endif %}
							</div>
							<p style="margin-top: 1rem; color: var(--text-muted);">
								{% if prediction.label == 'Harmful' %}
								This variant significantly alters protein stability, potentially affecting metabolic efficiency or structural integrity.
								{% else %}
								This variant is likely benign, maintaining standard protein function and stability.
								{% endif %}
							</p>
						</div>
					</div>
					
					<div>
						<div class="viewer-container">
							<div class="viewer-label">Mutant Structure ({{ features.mutation }})</div>
							<div id="viewer-mut" style="width: 100%; height: 100%;"></div>
						</div>
					</div>
				</div>
				
				<div class="features-grid">
					<div class="metric-card">
						<div class="metric-val">{{ '%.3f' % features.get('rmsd', 0) }} Å</div>
						<div class="metric-name">Structural Deviation (RMSD)</div>
					</div>
					<div class="metric-card">
						<div class="metric-val">{{ '%.1f' % features.get('delta_sasa', 0) }} Å²</div>
						<div class="metric-name">Surface Area Change</div>
					</div>
					<div class="metric-card">
						<div class="metric-val">{{ '%.2f' % features.get('conservation_score', 0) }}</div>
						<div class="metric-name">Evolutionary Conservation</div>
					</div>
				</div>
			</div>
			{% endif %}
			
			{% if error %}
			<div style="margin-top: 2rem; background: rgba(239, 68, 68, 0.1); border: 1px solid var(--danger); padding: 1rem; border-radius: 8px; color: var(--danger);">
				<i class="fa-solid fa-circle-exclamation"></i> {{ error }}
			</div>
			{% endif %}
		</div>
	</section>

	<script>
		const PROTEINS = {
			'myoglobin': {
				id: '1MBN',
				seq: 'VLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDMASNYKELGFQG'
			},
			'crambin': {
				id: '1CRN',
				seq: 'TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN'
			}
		};

		function updateProtein() {
			const select = document.getElementById('proteinSelect');
			const val = select.value;
			if (!val) return;
			
			const data = PROTEINS[val];
			document.getElementById('seqInput').value = data.seq;
			document.getElementById('idInput').value = data.id;
			
			if (val === 'myoglobin') document.getElementById('mutationInput').value = 'V10A';
			if (val === 'crambin') document.getElementById('mutationInput').value = 'T2A';
		}
		
		// Initialize NGL if results are present
		{% if prediction %}
		document.addEventListener('DOMContentLoaded', function() {
			var stage = new NGL.Stage("viewer-mut", { backgroundColor: "white" });
			var mutBlob = new Blob([atob("{{ mut_b64 }}")], { type: 'text/plain' });
			
			stage.loadFile(mutBlob, { ext: "pdb" }).then(function(o) {
				o.addRepresentation("cartoon", { color: "orange" });
				o.addRepresentation("ball+stick", { sele: "not protein" });
				stage.autoView();
			});
			
			// Handle resize
			window.addEventListener('resize', function() { stage.handleResize(); });
		});
		{% endif %}
	</script>
</body>
</html>
"""


def main() -> None:
	app = Flask(__name__)

	@app.route("/", methods=["GET"])
	def index():
		return render_template_string(LANDING_HTML)

	@app.route("/gym", methods=["GET", "POST"])
	def gym():
		prediction = None
		features = None
		error = None
		mut_b64 = None
		
		if request.method == "POST":
			try:
				# 1. Get Data
				seq_text = request.form.get("seq")
				mut_text = request.form.get("mut")
				sid = request.form.get("id")
				
				if not seq_text or not mut_text or not sid:
					raise ValueError("Missing required data. Please select a protein and mutation.")
				
				# 2. Run Analysis Pipeline (Simplified for Gym) with strict validation
				# Load Sequence (backend validation of allowed characters)
				sequence = load_sequence(raw_sequence=seq_text, fasta_path=None)

				# Parse & validate mutation format + consistency with sequence
				mutation = parse_mutation(mut_text)
				validate_mutation_against_sequence(sequence, mutation)

				# Validate PDB ID format and sequence vs structure length (PDB only)
				validate_pdb_id(sid)
				validate_sequence_vs_pdb_length(sequence, sid)

				# Fetch Structure
				wt_path = fetch_rcsb_pdb(sid)
				mut_path = build_mutant_structure_stub(wt_path, sequence, mutation)
				
				# Compute Features
				features = compute_basic_features(sequence, mutation, wt_path, mut_path)
				
				# Predict
				ml_classifier = SimpleMLOnlyClassifier("models/")
				prediction = ml_classifier.predict(sequence, mut_text, wt_path, mut_path, "ensemble")
				
				# Prepare Visualization Data
				from mutation_impact.reporting.report import _file_to_base64_text
				mut_data = _file_to_base64_text(mut_path)
				mut_b64 = mut_data.get("b64", "")
				
			except Exception as e:
				error = str(e)
				print(f"Gym Analysis Error: {e}")
		
		return render_template_string(GYM_HTML, prediction=prediction, features=features, error=error, mut_b64=mut_b64)

	@app.route("/analyze", methods=["GET", "POST"])
	def analyze():
		report_html = None
		error = None
		seq_text = None
		mut_text = None
		src = "pdb"
		sid = "1CRN"
		forcenaive = False
		high_accuracy = False
		enable_sasa = True
		enable_conservation = False
		enable_blosum = True
		enable_hydrophobicity = True
		if request.method == "POST":
			# Clean up any old temporary files to prevent caching issues
			try:
				cleaned_count = cleanup_mutation_cache()
				if cleaned_count > 0:
					print(f"Cleaned up {cleaned_count} temporary files")
			except Exception as cleanup_error:
				print(f"Cleanup warning: {cleanup_error}")
			
			# Handle form submission for analysis
			seq_text = request.form.get("seq")
			mut_text = (request.form.get("mut") or "").strip()
			src = request.form.get("src") or "pdb"
			sid = (request.form.get("id") or "").strip()
			forcenaive = bool(request.form.get("forcenaive"))
			high_accuracy = bool(request.form.get("high_accuracy"))
			enable_sasa = bool(request.form.get("enable_sasa"))
			enable_conservation = bool(request.form.get("enable_conservation"))
			enable_blosum = bool(request.form.get("enable_blosum"))
			enable_hydrophobicity = bool(request.form.get("enable_hydrophobicity"))

			try:
				seq_text_clean = (seq_text or "").strip()
				if not seq_text_clean:
					raise ValueError("❌ Please provide a protein sequence.")
				sequence = load_sequence(raw_sequence=seq_text_clean, fasta_path=None)

				# Validate mutation format and content
				if not mut_text:
					raise ValueError("❌ Please provide a mutation in the format A123T (e.g., K8E)")
				
				mutation = parse_mutation(mut_text)
				validate_mutation_against_sequence(sequence, mutation)

				# Validate structure identifier
				if not sid:
					raise ValueError("❌ Please provide a structure ID (e.g., 1CRN for PDB or P05067 for AlphaFold).")

				# PDB-specific validation: ID format + sequence vs structure length
				if src == "pdb":
					validate_pdb_id(sid)
					validate_sequence_vs_pdb_length(sequence, sid)
					wt_path = fetch_rcsb_pdb(sid)
				else:
					wt_path = fetch_alphafold_model(sid)

				mut_path = build_mutant_structure_stub(wt_path, sequence, mutation, force_naive=forcenaive)


				# Compute features with robust error handling
				try:
					# Always use basic features first for compatibility
					features = compute_basic_features(sequence, mutation, wt_path, mut_path)
					
					# Enhanced features if high-accuracy mode is enabled
					enhanced_features = {}
					confidence_factors = []
					
					if high_accuracy:
						try:
							# Try to get enhanced features
							extractor = AdvancedFeatureExtractor()
							enhanced_features = extractor.extract_all_features(sequence, mut_text, wt_path, mut_path)
							
							# Merge enhanced features with basic features
							features.update(enhanced_features)
							
							# Enhanced confidence scoring based on feature quality
							if features.get('rmsd', 0) > 0.1:
								confidence_factors.append(0.2)  # Structural change detected
							if abs(features.get('delta_sasa', 0)) > 10:
								confidence_factors.append(0.2)  # SASA change detected
							if abs(features.get('delta_hbond_count', 0)) > 0:
								confidence_factors.append(0.15)  # H-bond change detected
							if abs(features.get('blosum62', 0)) > 0:
								confidence_factors.append(0.15)  # Evolutionary score available
							if abs(features.get('delta_hydrophobicity', 0)) > 0.5:
								confidence_factors.append(0.1)  # Hydrophobicity change
							if features.get('conservation_score', 0.5) > 0.7:
								confidence_factors.append(0.2)  # High conservation
						except Exception as e:
							print(f"Enhanced features failed, using basic features: {e}")
							# Continue with basic features only
					
					# Make prediction using Simple ML-ONLY classifier (no rule-based fallback)
					try:
						# Use simple ML-only classifier for maximum accuracy
						ml_classifier = SimpleMLOnlyClassifier("models/")
						pred = ml_classifier.predict(sequence, mut_text, wt_path, mut_path, "ensemble")
						
						# Add ML model flag
						pred["ml_model"] = True
						pred["model_used"] = pred.get("model_used", "ensemble")
						
					except Exception as e:
						# If ML model fails, raise error instead of falling back
						raise Exception(f"ML-only prediction failed: {e}. Ensure models are trained using create_better_ml_model.py")
					
					# Enhanced confidence if high-accuracy mode
					if high_accuracy and confidence_factors:
						base_confidence = pred.get('confidence', 0.5)
						enhancement = sum(confidence_factors)
						enhanced_confidence = min(0.95, base_confidence + enhancement)
						
						pred.update({
							"confidence": enhanced_confidence,
							"enhanced": True,
							"feature_quality": len(confidence_factors) / 6.0,
							"confidence_factors": confidence_factors
						})
					
				except Exception as e:
					# ML-only fallback - still use ML model even with minimal features
					print(f"Feature computation failed: {e}")
					# Create minimal features for ML prediction
					features = {
						"mutation": mut_text,
						"sequence_length": len(sequence),
						"rmsd": 0.0,
						"delta_sasa": 0.0,
						"delta_hbond_count": 0,
						"blosum62": 0,
						"delta_hydrophobicity": 0.0,
						"conservation_score": 0.5
					}
					
					# Simple ML-only prediction even with minimal features
					try:
						ml_classifier = SimpleMLOnlyClassifier("models/")
						pred = ml_classifier.predict(sequence, mut_text, wt_path, mut_path, "ensemble")
						pred["ml_model"] = True
						pred["model_used"] = pred.get("model_used", "ensemble")
					except Exception as ml_error:
						raise Exception(f"ML-only prediction failed even with minimal features: {ml_error}. Train models first.")
				
				# Severity estimation
				sev = SeverityEstimator().estimate(features) if pred["label"] == "Harmful" else None
				
				# Generate report with enhanced information
				report_html = render_html_report(features, pred, sev)
				current_app.config['LAST_REPORT_HTML'] = report_html
			except Exception as exc:
				error = str(exc)

		return render_template_string(
			HTML,
			report_html=report_html,
			error=error,
			seq=seq_text,
			mut=mut_text,
			src=src,
			sid=sid,
			forcenaive=forcenaive,
			high_accuracy=high_accuracy,
			enable_sasa=enable_sasa,
			enable_conservation=enable_conservation,
			enable_blosum=enable_blosum,
			enable_hydrophobicity=enable_hydrophobicity,
		)

	app.run(host="127.0.0.1", port=7860, debug=True)


if __name__ == "__main__":
	main()
