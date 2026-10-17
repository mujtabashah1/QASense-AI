"""Run the deterministic local QASense product demo end to end."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.product.orchestrator import QASenseOrchestrator  # noqa: E402


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--run-id"); parser.add_argument("--device",default="0"); parser.add_argument("--browser-channel",default="msedge"); args=parser.parse_args()
    config=json.loads((ROOT/"configs/demo.json").read_text(encoding="utf-8")); fixture=(ROOT/"tests/fixtures/runtime_fixture.html").resolve().as_uri(); viewport=config["viewport"]
    result=QASenseOrchestrator(browser_channel=args.browser_channel,device=args.device).run(
        fixture,run_id=args.run_id,viewport=(viewport["width"],viewport["height"]),source_file=ROOT/config["source_file"],risk_metrics=config["risk_metrics"])
    print(json.dumps({"run_id":result["run_id"],"status":result["status"],"findings":len(result["findings"]),"recommendations":len(result["recommendations"]),"test_suggestions":len(result["test_suggestions"]),"regression_priorities":len(result["regression_priority"]),"timing":result["timing"],"errors":result["errors"],"analysis":result["artifacts"]["analysis"]},indent=2)); return 0 if result["status"] in {"completed","partial"} and result["runtime"].get("status")=="available" and result["visual"].get("status")=="available" else 1


if __name__=="__main__": raise SystemExit(main())
