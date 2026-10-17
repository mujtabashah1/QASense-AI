"""Canonical JSON-serializable QASense product evidence contract."""

from __future__ import annotations

import json
from typing import Any

SCHEMA_VERSION = "1.0"
SEVERITIES = ("critical", "high", "medium", "low", "info")
RUN_STATUSES = ("queued", "running", "completed", "partial", "failed")


def empty_analysis(run_id: str, target: dict[str, Any]) -> dict[str, Any]:
    if not run_id.strip():
        raise ValueError("run_id is required")
    return {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "status": "running",
        "target": target, "timing": {}, "qa_score": None,
        "risk": {"status": "unavailable", "items": []},
        "code": {"status": "unavailable", "items": []},
        "runtime": {"status": "unavailable"},
        "visual": {"status": "unavailable"},
        "findings": [], "recommendations": [], "test_suggestions": [],
        "regression_priority": [], "artifacts": {}, "errors": [],
    }


def validate_analysis(value: dict[str, Any]) -> None:
    required = {"schema_version", "run_id", "status", "target", "timing", "qa_score", "risk", "code", "runtime", "visual", "findings", "recommendations", "test_suggestions", "regression_priority", "artifacts", "errors"}
    missing = required - set(value)
    if missing:
        raise ValueError(f"Missing analysis fields: {sorted(missing)}")
    if value["schema_version"] != SCHEMA_VERSION or value["status"] not in RUN_STATUSES:
        raise ValueError("Unsupported schema version or run status")
    for field in ("findings", "recommendations", "test_suggestions", "regression_priority", "errors"):
        if not isinstance(value[field], list):
            raise ValueError(f"{field} must be an array")
    for finding in value["findings"]:
        if finding.get("severity") not in SEVERITIES:
            raise ValueError(f"Invalid finding severity: {finding.get('severity')}")
        confidence = finding.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError("Finding confidence must be between 0 and 1")
        if not isinstance(finding.get("evidence"), list) or not finding["evidence"]:
            raise ValueError("Every finding must preserve evidence provenance")
    if not isinstance(value["qa_score"], int) or not 0 <= value["qa_score"] <= 100:
        raise ValueError("qa_score must be an integer between 0 and 100")
    json.dumps(value)


def evidence_ref(kind: str, source: str, summary: str, **details: Any) -> dict[str, Any]:
    return {"kind": kind, "source": source, "summary": summary, "details": details}
