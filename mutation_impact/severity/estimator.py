from typing import List, Literal, TypedDict

SeverityLevel = Literal["Low", "Medium", "High"]
Mode = Literal["stability_loss", "binding_disruption", "functional_site_disruption"]


class SeverityPrediction(TypedDict):
	severity: SeverityLevel
	modes: List[Mode]


class SeverityEstimator:
	def estimate(self, features: dict) -> SeverityPrediction:
		rmsd = float(features.get("rmsd", 0.0))
		delta_stability = float(features.get("delta_stability_proxy", 0.0))
		distance_to_site = float(features.get("distance_to_site", 999.0))
		conservation = float(features.get("conservation_score", 0.0))
		blosum = float(features.get("blosum62", 0.0))
		delta_hydro = float(features.get("delta_hydrophobicity", 0.0))

		modes: List[Mode] = []
		if delta_stability < -1.0 or abs(delta_hydro) > 2.0:
			modes.append("stability_loss")
		if distance_to_site < 5.0 or rmsd > 1.0:
			modes.append("binding_disruption")
		if conservation > 0.8 or blosum < -1:
			modes.append("functional_site_disruption")

		risk_score = (
			0.4 * rmsd + 0.4 * max(0.0, -delta_stability) + (0.2 if distance_to_site < 5.0 else 0.0) + 0.1 * abs(delta_hydro)
		)
		severity: SeverityLevel
		if risk_score > 1.6:
			severity = "High"
		elif risk_score > 0.8:
			severity = "Medium"
		else:
			severity = "Low"

		return {"severity": severity, "modes": modes or ["stability_loss"]}
