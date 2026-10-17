import pytest

from src.contracts.evidence import build_multimodal_finding, validate_multimodal_finding


def test_builds_explicit_deterministic_contract():
    result = build_multimodal_finding("Checkout failure", "high", runtime={"console_errors": 1})
    assert result["fusion_method"] == "deterministic_evidence_normalization"
    assert result["runtime"]["console_errors"] == 1


def test_rejects_claimed_unknown_fusion():
    with pytest.raises(ValueError):
        validate_multimodal_finding({"title": "x", "severity": "high", "fusion_method": "ai", "recommendations": []})
