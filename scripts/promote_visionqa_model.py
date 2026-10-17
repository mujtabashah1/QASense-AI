"""Validate and atomically promote a selected VisionQA checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs" / "visionqa"))

from ultralytics import YOLO  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--destination",
        type=Path,
        default=ROOT / "models" / "visionqa" / "atomblock_yolo11n" / "best.pt",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Selected checkpoint does not exist: {source}")

    model = YOLO(str(source))
    class_names = {int(key): value for key, value in model.names.items()}
    if len(class_names) != 14:
        raise ValueError(f"Expected 14 classes, checkpoint has {len(class_names)}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    if sha256(source) != sha256(temporary):
        temporary.unlink(missing_ok=True)
        raise OSError("Checkpoint hash mismatch after copy")
    temporary.replace(destination)

    manifest = {
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(source),
        "stable_checkpoint": str(destination),
        "sha256": sha256(destination),
        "size_bytes": destination.stat().st_size,
        "class_count": len(class_names),
        "classes": class_names,
    }
    manifest_path = destination.parent / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
