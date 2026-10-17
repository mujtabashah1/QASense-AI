"""Deterministic multimodal evidence normalization without learned fusion claims."""

from __future__ import annotations

from typing import Any


def build_multimodal_finding(
    title: str, severity: str, *, risk: dict[str, Any] | None = None,
    code: dict[str, Any] | None = None, runtime: dict[str, Any] | None = None,
    visual: dict[str, Any] | None = None, recommendations: list[str] | None = None,
) -> dict[str, Any]:
    finding = {
        "schema_version": "1.0", "title": title.strip(), "severity": severity.lower(),
        "risk": risk, "code": code, "runtime": runtime, "visual": visual,
        "recommendations": recommendations or [],
        "fusion_method": "deterministic_evidence_normalization",
    }
    validate_multimodal_finding(finding)
    return finding


def validate_multimodal_finding(value: dict[str, Any]) -> None:
    if not isinstance(value, dict):
        raise ValueError("finding must be an object")
    if not value.get("title"):
        raise ValueError("title is required")
    if value.get("severity") not in {"info", "low", "medium", "high", "critical"}:
        raise ValueError("invalid severity")
    if value.get("fusion_method") != "deterministic_evidence_normalization":
        raise ValueError("unsupported fusion method")
    for field in ("risk", "code", "runtime", "visual"):
        if value.get(field) is not None and not isinstance(value[field], dict):
            raise ValueError(f"{field} must be an object or null")
    if not isinstance(value.get("recommendations"), list) or not all(isinstance(item, str) for item in value["recommendations"]):
        raise ValueError("recommendations must be a string array")
