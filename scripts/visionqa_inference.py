"""CLI wrapper for the reusable QASense VisionQA inference interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.visionqa.inference import DEFAULT_CHECKPOINT, VisionQADetector  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screenshot", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=1500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    detector = VisionQADetector(
        checkpoint=args.checkpoint,
        device=args.device,
        imgsz=args.imgsz,
        confidence=args.confidence,
        iou=args.iou,
        max_det=args.max_det,
    )
    payload = detector.predict(args.screenshot)
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
