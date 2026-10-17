"""Fingerprint completed QASense models and reports without modifying them."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "risknet_model": "models/risknet/risknet_v1.joblib",
    "risknet_report": "reports/risknet/RISKNET_MODEL_REPORT.md",
    "codelocator_v2_model": "models/codelocator_v2/line_localizer/model.safetensors",
    "codelocator_v2_report": "reports/codelocator_v2/CODELOCATOR_V2_MODEL_REPORT.md",
    "visionqa_model": "models/visionqa/atomblock_yolo11n/best.pt",
    "visionqa_report": "reports/visionqa/VISIONQA_ATOMBLOCK_REPORT.md",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    records = {}
    for name, relative in FILES.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        records[name] = {"path": relative, "bytes": stat.st_size, "sha256": digest(path), "last_write_time_ns": stat.st_mtime_ns}
    result = {
        "schema_version": "1.0", "captured_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Read-only integrity fingerprint for completed project artifacts before integration work.",
        "artifacts": records,
    }
    output = ROOT / "reports/state/protected_artifact_manifest.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
