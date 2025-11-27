#!/usr/bin/env python3
"""
Test the web interface logic directly to see what's happening.
"""

from mutation_impact.input_module.parser import load_sequence, parse_mutation, validate_mutation_against_sequence
from mutation_impact.structure.retrieval import fetch_rcsb_pdb
from mutation_impact.structure.modeling import build_mutant_structure_stub
from mutation_impact.features.interfaces import compute_basic_features
from mutation_impact.classifier.simple_ml_only import SimpleMLOnlyClassifier
from mutation_impact.severity.estimator import SeverityEstimator
from mutation_impact.reporting.report import render_html_report

def test_web_direct():
    """Test the exact web interface logic."""
    
    print("🌐 TESTING WEB INTERFACE LOGIC DIRECTLY")
    print("="*60)
    
    # Simulate exact web interface inputs
    sequence_text = 'MVLSPADKTNVKAAW'
    mut_text = 'K8E'
    src = 'pdb'
    sid = '1CRN'
    forcenaive = True
    high_accuracy = False
    
    print(f"Inputs:")
    print(f"  Sequence: {sequence_text}")
    print(f"  Mutation: {mut_text}")
    print(f"  Structure: {src} {sid}")
    print(f"  Force naive: {forcenaive}")
    print(f"  High accuracy: {high_accuracy}")
    
    try:
        # Step 1: Load and validate sequence (as web interface does)
        sequence = load_sequence(raw_sequence=sequence_text, fasta_path=None)
        mutation = parse_mutation(mut_text)
        validate_mutation_against_sequence(sequence, mutation)
        print(f"✅ Sequence and mutation validated")
        
        # Step 2: Get structures (as web interface does)
        wt_path = fetch_rcsb_pdb(sid)
        mut_path = build_mutant_structure_stub(wt_path, sequence, mutation, force_naive=forcenaive)
        print(f"✅ Structures built")
        
        # Step 3: Compute features (as web interface does)
        features = compute_basic_features(sequence, mutation, wt_path, mut_path)
        print(f"✅ Basic features computed: {len(features)} features")
        
        # Step 4: ML prediction (as web interface does)
        print(f"\n🤖 Making ML prediction...")
        ml_classifier = SimpleMLOnlyClassifier("models/")
        pred = ml_classifier.predict(sequence, mut_text, str(wt_path), str(mut_path), "ensemble")
        
        # Add ML model flag (as web interface does)
        pred["ml_model"] = True
        pred["model_used"] = pred.get("model_used", "ensemble")
        
        print(f"ML Prediction Result:")
        print(f"  Label: {pred['label']}")
        print(f"  Confidence: {pred['confidence']:.1%}")
        print(f"  Model: {pred['model_used']}")
        print(f"  ML Model Flag: {pred['ml_model']}")
        print(f"  Feature Quality: {pred['feature_quality']:.1%}")
        
        # Step 5: Severity estimation (as web interface does)
        sev = SeverityEstimator().estimate(features) if pred["label"] == "Harmful" else None
        print(f"✅ Severity estimated: {sev is not None}")
        
        # Step 6: Generate report (as web interface does)
        report_html = render_html_report(features, pred, sev)
        print(f"✅ Report generated: {len(report_html)} characters")
        
        # Check if the report contains the right prediction
        if "Harmful" in report_html:
            print(f"✅ Report contains 'Harmful' prediction")
        elif "Neutral" in report_html:
            print(f"❌ Report contains 'Neutral' prediction")
        else:
            print(f"⚠️  Report prediction unclear")
        
        # Save the report for inspection
        with open("test_report_k8e.html", "w") as f:
            f.write(report_html)
        print(f"📄 Report saved to test_report_k8e.html")
        
        return pred
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_web_direct()
    if result:
        print(f"\n🎯 FINAL RESULT:")
        print(f"   The web interface logic predicts: {result['label']}")
        print(f"   With confidence: {result['confidence']:.1%}")
        if result['label'] == 'Harmful':
            print(f"   ✅ SUCCESS: K8E correctly predicted as harmful!")
        else:
            print(f"   ❌ ISSUE: K8E still predicted as neutral")
