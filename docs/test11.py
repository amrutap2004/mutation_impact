from mutation_impact.ml import ProductionMLPipeline

# Initialize with trained models
pipeline = ProductionMLPipeline("models/")

# Single prediction with ML model
result = pipeline.predict_single_mutation(
    sequence="MVLSPADKTNVKAAW",
    mutation={"from_res": "A", "position": 123, "to_res": "T"},  # dict, not string
    wt_path="wt.pdb",
    mut_path="mut.pdb",
    model_name="ensemble"  # Use best model
)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Model: {result['model_used']}")
