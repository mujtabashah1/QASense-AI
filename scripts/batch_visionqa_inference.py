"""Run VisionQA over a JSON manifest while loading the detector once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.visionqa.inference import VisionQADetector  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    detector = VisionQADetector(device=args.device, confidence=args.confidence)
    for index, item in enumerate(manifest["cases"], 1):
        case_dir = Path(item["case_dir"])
        for phase in ("baseline", "mutated"):
            output = detector.predict(case_dir / f"{phase}.png")
            (case_dir / f"visionqa_{phase}.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
        if index % 10 == 0 or index == len(manifest["cases"]):
            print(f"VisionQA {index}/{len(manifest['cases'])}", flush=True)


if __name__ == "__main__":
    main()
