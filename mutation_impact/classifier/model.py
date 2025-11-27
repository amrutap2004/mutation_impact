from typing import TypedDict, Optional
import joblib
from pathlib import Path


class HarmfulnessPrediction(TypedDict):
	label: str  # "Harmful" | "Neutral"
	confidence: float


class HarmfulnessClassifier:
	"""Classifier with ML model support and rule-based fallback.

	Features used: RMSD, negative stability proxy, BLOSUM penalty, |Δhydrophobicity|.
	"""

	def __init__(self, model_path: Optional[str] = None):
		self.model_path = model_path
		self.ml_model = None
		self.scaler = None
		self.label_encoder = None
		self.feature_names = []
		
		# Try to load ML model
		if model_path and Path(model_path).exists():
			self._load_ml_model()

	def _load_ml_model(self):
		"""Load trained ML model if available."""
		try:
			self.ml_model = joblib.load(self.model_path)
			print(f"Loaded ML model from {self.model_path}")
		except Exception as e:
			print(f"Could not load ML model: {e}")
			self.ml_model = None

	def predict(self, features: dict) -> HarmfulnessPrediction:
		"""Predict using ML model if available, otherwise use rule-based approach."""
		if self.ml_model is not None:
			return self._predict_ml(features)
		else:
			return self._predict_rule_based(features)

	def _predict_ml(self, features: dict) -> HarmfulnessPrediction:
		"""Predict using trained ML model."""
		try:
			# Convert features to array
			feature_array = [[features.get(name, 0.0) for name in self.feature_names]]
			
			# Scale features if scaler is available
			if self.scaler is not None:
				feature_array = self.scaler.transform(feature_array)
			
			# Make prediction
			prediction = self.ml_model.predict(feature_array)[0]
			probability = self.ml_model.predict_proba(feature_array)[0] if hasattr(self.ml_model, 'predict_proba') else None
			
			# Decode prediction
			if self.label_encoder is not None:
				label = self.label_encoder.inverse_transform([prediction])[0]
			else:
				label = "Harmful" if prediction == 1 else "Neutral"
			
			# Calculate confidence
			confidence = max(probability) if probability is not None else 0.5
			
			return {"label": label, "confidence": confidence}
		except Exception as e:
			print(f"ML prediction failed, falling back to rule-based: {e}")
			return self._predict_rule_based(features)

	def _predict_rule_based(self, features: dict) -> HarmfulnessPrediction:
		"""Predict using rule-based approach."""
		rmsd = float(features.get("rmsd", 0.0))
		delta_stability = float(features.get("delta_stability_proxy", 0.0))
		blosum = float(features.get("blosum62", 0.0))
		delta_hydro = float(features.get("delta_hydrophobicity", 0.0))

		# Normalize feature magnitudes to ~[0,1]
		rmsd_norm = max(0.0, min(1.0, rmsd / 2.0))  # ~2Å -> strong impact
		destab_norm = max(0.0, min(1.0, (-delta_stability) / 5.0))  # more negative -> higher risk
		blosum_penalty = max(0.0, min(1.0, (-blosum) / 4.0))  # strong negative substitution -> higher risk
		hydro_norm = max(0.0, min(1.0, abs(delta_hydro) / 3.0))  # large hydrophobicity shift -> higher risk

		# Weighted risk score in [0,1]
		score = (
			0.35 * rmsd_norm +
			0.35 * destab_norm +
			0.20 * blosum_penalty +
			0.10 * hydro_norm
		)
		score = max(0.0, min(1.0, score))

		# Decision boundary
		label = "Harmful" if score >= 0.5 else "Neutral"

		# Confidence: distance from boundary with floors
		margin = abs(score - 0.5) * 2.0  # 0..1
		if label == "Harmful":
			confidence = max(0.6, 0.5 + 0.5 * margin)
		else:
			confidence = max(0.2, 0.1 + 0.7 * margin)
		confidence = min(0.99, max(0.01, confidence))

		return {"label": label, "confidence": confidence}
