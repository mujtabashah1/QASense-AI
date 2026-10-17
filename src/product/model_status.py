"""Read-only status and integrity information for QASense models."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "risknet": ("production", ROOT / "models/risknet/risknet_v1.joblib", "b32efa3e83707f01bdf29bd7dbcaebe4d484aa9f81f44bf890b9a739deac913c"),
    "codelocator_v2": ("production", ROOT / "models/codelocator_v2/line_localizer/model.safetensors", "9459bfa6b343fc749e5b7653df0f0def005d8781c89388d49466f83e6c0ec288"),
    "visionqa_v1": ("production", ROOT / "models/visionqa/atomblock_yolo11n/best.pt", "7f527e18271067e2c3ed1c85b557b8b794646b996fed2128bd5a06f8c7234f57"),
    "visionqa_v2": ("experimental", ROOT / "models/visionqa/visionqa_v2_mixed14/candidate.pt", "dd86b81d5faf5f018940564ddb344a8ea7caf328b7b567161937e40ab7f7866d"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_model_status(verify_hashes: bool = False) -> dict[str, Any]:
    output = {}
    for name, (status, path, expected) in MODELS.items():
        exists = path.is_file()
        item = {"status": status, "available": exists, "path": str(path.relative_to(ROOT)), "expected_sha256": expected, "bytes": path.stat().st_size if exists else None}
        if verify_hashes and exists:
            actual = _sha256(path)
            item.update({"sha256": actual, "integrity": actual == expected})
        output[name] = item
    output["visionqa_v2"]["recommended_experimental_threshold"] = 0.0625
    output["visionqa_v2"]["limitations"] = ["select unresolved", "tiny-object recall weak", "mobile recall weak", "higher false-positive load"]
    return {"schema_version": "1.0", "models": output}
