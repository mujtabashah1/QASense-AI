"""Verify the selected portable QASense inference assets without training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "risknet": {"path": "models/risknet/risknet_v1.joblib", "sha256": "b32efa3e83707f01bdf29bd7dbcaebe4d484aa9f81f44bf890b9a739deac913c", "status": "production"},
    "codelocator_v2": {"path": "models/codelocator_v2/line_localizer/model.safetensors", "sha256": "9459bfa6b343fc749e5b7653df0f0def005d8781c89388d49466f83e6c0ec288", "status": "production"},
    "visionqa_v1": {"path": "models/visionqa/atomblock_yolo11n/best.pt", "sha256": "7f527e18271067e2c3ed1c85b557b8b794646b996fed2128bd5a06f8c7234f57", "status": "production"},
    "visionqa_v2": {"path": "models/visionqa/visionqa_v2_mixed14/candidate.pt", "sha256": "dd86b81d5faf5f018940564ddb344a8ea7caf328b7b567161937e40ab7f7866d", "status": "experimental", "threshold": 0.0625},
}
CODELOCATOR_SUPPORT = ("config.json", "evaluation.json", "tokenizer.json", "tokenizer_config.json", "training_metadata.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path) -> dict:
    results: dict[str, dict] = {}
    ok = True
    for name, spec in ASSETS.items():
        path = root / spec["path"]
        exists = path.is_file()
        actual = sha256(path) if exists else None
        pointer = exists and path.stat().st_size < 1024 and path.read_bytes().startswith(b"version https://git-lfs")
        valid = exists and not pointer and actual == spec["sha256"]
        results[name] = {**spec, "exists": exists, "bytes": path.stat().st_size if exists else None, "actual_sha256": actual, "lfs_pointer_only": pointer, "valid": valid}
        ok = ok and valid
    support_root = root / "models/codelocator_v2/line_localizer"
    support = {name: (support_root / name).is_file() for name in CODELOCATOR_SUPPORT}
    ok = ok and all(support.values())
    return {"root": str(root), "ready": ok, "models": results, "codelocator_support": support}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = verify(args.root.resolve())
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for name, item in result["models"].items():
            print(f"{name}: {'READY' if item['valid'] else 'INVALID'} ({item['bytes']} bytes)")
        print(f"CodeLocator support files: {'READY' if all(result['codelocator_support'].values()) else 'INCOMPLETE'}")
        print(f"Portable model set: {'READY' if result['ready'] else 'NOT READY'}")
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
