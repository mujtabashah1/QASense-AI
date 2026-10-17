"""Capture browser evidence, run VisionQA, and match DOM elements to detections."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.playwright_runtime.collector import PlaywrightEvidenceCollector, atomic_json  # noqa: E402
from src.visionqa.matching import match_dom_to_detections  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("url")
    parser.add_argument("--run-id"); parser.add_argument("--browser-channel", default="msedge")
    parser.add_argument("--width", type=int, default=1440); parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--confidence", type=float, default=0.25); parser.add_argument("--device", default="0")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--headed", action="store_true"); args = parser.parse_args()
    collector = PlaywrightEvidenceCollector(browser_channel=args.browser_channel, headless=not args.headed, viewport=(args.width, args.height))
    metadata = collector.capture(args.url, run_id=args.run_id); run_dir = ROOT / "runs" / "playwright" / metadata["run_id"]
    command = [str(ROOT / ".venv-visionqa" / "Scripts" / "python.exe"), str(ROOT / "scripts" / "visionqa_inference.py"), str(run_dir / "viewport.png"), "--confidence", str(args.confidence), "--device", args.device, "--output", str(run_dir / "visionqa.json")]
    if args.checkpoint:
        command.extend(["--checkpoint", str(args.checkpoint.resolve())])
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode:
        atomic_json(run_dir / "visionqa_error.json", {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr})
        raise RuntimeError(f"VisionQA subprocess failed; see {run_dir / 'visionqa_error.json'}")
    vision = json.loads((run_dir / "visionqa.json").read_text(encoding="utf-8")); dom = json.loads((run_dir / "dom.json").read_text(encoding="utf-8"))["elements"]
    matching = match_dom_to_detections(dom, vision["detections"]); atomic_json(run_dir / "dom_vision_matches.json", matching)
    metadata["artifacts"]["visionqa"] = str(run_dir / "visionqa.json"); metadata["artifacts"]["dom_vision_matches"] = str(run_dir / "dom_vision_matches.json")
    metadata["visionqa_summary"] = {"detection_count": vision["detection_count"], "counts_by_class": vision["counts_by_class"], "inference_time_ms": vision["inference_time_ms"], "match_count": matching["match_count"]}
    atomic_json(run_dir / "metadata.json", metadata); print(json.dumps(metadata, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
