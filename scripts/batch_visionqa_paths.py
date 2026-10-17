"""Run the stable VisionQA checkpoint once over an explicit path manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.visionqa.inference import DEFAULT_CHECKPOINT, VisionQADetector  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="JSON containing items with image and output paths")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    detector = VisionQADetector(
        checkpoint=args.checkpoint if args.checkpoint is not None else DEFAULT_CHECKPOINT,
        device=args.device,
        confidence=args.confidence,
    )
    for index, item in enumerate(payload["items"], 1):
        result = detector.predict(item["image"])
        output = Path(item["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if index % 25 == 0 or index == len(payload["items"]):
            print(f"VisionQA {index}/{len(payload['items'])}", flush=True)


if __name__ == "__main__":
    main()
