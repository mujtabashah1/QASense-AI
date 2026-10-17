"""Normalize completed RiskNet inference for multimodal evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class RiskNetAdapter:
    def __init__(self, backend: Any | None = None) -> None:
        if backend is None:
            from src.risknet.inference import RiskNet
            backend = RiskNet()
        self.backend = backend

    def predict(self, file: str | Path, metrics: dict[str, float]) -> dict[str, Any]:
        path = str(file)
        if not path.strip():
            raise ValueError("file must be nonempty")
        result = self.backend.predict_risk(metrics)
        return {
            "schema_version": "1.0", "module": "risknet", "file": path,
            "defect_probability": float(result["defect_probability"]),
            "priority": str(result["risk_level"]).lower(),
            "qa_priority": result["qa_priority"], "prediction": result["prediction"],
            "risk_factors": result.get("top_risk_factors", []),
            "model_path": "models/risknet/risknet_v1.joblib",
            "limitation": "Risk is a file/module prioritization score and does not prove a defect.",
        }
