"""Replay deterministic local application states through Playwright and VisionQA."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.playwright_runtime.collector import PlaywrightEvidenceCollector  # noqa: E402
from src.visionqa.matching import _compatible, _geometry, _xyxy, match_dom_to_detections  # noqa: E402


EXPECTED = [
    {"selector": "html > body > nav", "name": "primary navigation", "classes": ["block-nav", "block"]},
    {"selector": "html > body > nav > a:nth-of-type(1)", "name": "checkout link", "classes": ["link"]},
    {"selector": "html > body > nav > a:nth-of-type(2)", "name": "status link", "classes": ["link"]},
    {"selector": "#form", "name": "checkout form", "classes": ["block-form", "block"]},
    {"selector": "#email", "name": "email input", "classes": ["input"]},
    {"selector": "#agree", "name": "terms checkbox", "classes": ["checkbox", "input"]},
    {"selector": "#ship", "name": "shipping radio", "classes": ["radio", "input"]},
    {"selector": "#country", "name": "country select", "classes": ["select", "input"]},
    {"selector": "#submit", "name": "submit button", "classes": ["button"]},
    {"selector": "html > body > main > img", "name": "product image", "classes": ["image"]},
    {"selector": "html > body > main > table", "name": "item table", "classes": ["block-table", "block"]},
]


def states(url: str) -> list[dict]:
    base = {"application": "QASense local runtime fixture", "url": url, "route": "/runtime_fixture.html", "expected_stable_elements": EXPECTED, "expected_dynamic_elements": [], "authentication_required": False}
    return [
        {**base, "state_id": "desktop-default", "viewport": {"width": 1440, "height": 900}, "setup_actions": []},
        {**base, "state_id": "desktop-empty-email", "viewport": {"width": 1440, "height": 900}, "setup_actions": [{"action": "fill", "selector": "#email", "value": ""}]},
        {**base, "state_id": "desktop-terms-unchecked", "viewport": {"width": 1440, "height": 900}, "setup_actions": [{"action": "uncheck", "selector": "#agree"}]},
        {**base, "state_id": "desktop-navigation-focus", "viewport": {"width": 1440, "height": 900}, "setup_actions": [{"action": "focus", "selector": "nav a:first-of-type"}]},
        {**base, "state_id": "tablet-default", "viewport": {"width": 768, "height": 1024}, "setup_actions": []},
        {**base, "state_id": "mobile-default", "viewport": {"width": 390, "height": 844}, "setup_actions": []},
    ]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review_run(run_dir: Path, expected: list[dict]) -> dict:
    dom = json.loads((run_dir / "dom.json").read_text(encoding="utf-8"))["elements"]
    vision = json.loads((run_dir / "visionqa.json").read_text(encoding="utf-8"))
    matching = match_dom_to_detections(dom, vision["detections"])
    (run_dir / "dom_vision_matches.json").write_text(json.dumps(matching, indent=2) + "\n", encoding="utf-8")
    match_by_dom = {row["dom_index"]: row for row in matching["matches"]}
    ambiguity_by_dom: dict[int, list[dict]] = {}
    for row in matching["ambiguous_candidates"]:
        ambiguity_by_dom.setdefault(row["dom_index"], []).append(row)
    decisions, counts = [], Counter()
    for expected_item in expected:
        dom_index = next((i for i, item in enumerate(dom) if item["selector"] == expected_item["selector"]), None)
        if dom_index is None:
            decision = "missed_detection"; detail = "expected DOM element absent"
        elif dom_index in match_by_dom:
            match = match_by_dom[dom_index]
            confidence = match["evidence"]["detection_confidence"]
            decision = "low_confidence_correct_detection" if confidence < 0.4 else "correct_detection"
            detail = match
            # Count extra compatible boxes strongly overlapping the same element.
            overlapping = 0
            for detection in vision["detections"]:
                compatible, _ = _compatible(dom[dom_index], detection["class_name"])
                iou, containment, _ = _geometry(_xyxy(dom[dom_index]), _xyxy(detection))
                if compatible and iou >= 0.5 and containment >= 0.65: overlapping += 1
            if overlapping > 1:
                counts["duplicate_detections"] += overlapping - 1
        elif dom_index in ambiguity_by_dom:
            decision = "wrong_class"; detail = ambiguity_by_dom[dom_index]
            counts["semantic_conflicts"] += 1
        else:
            decision = "missed_detection"; detail = "no compatible or strong-overlap detection"
        counts[decision] += 1
        decisions.append({**expected_item, "decision": decision, "evidence": detail})
    counts["reviewed_elements"] = len(expected)
    counts["correctly_detected_elements"] = counts["correct_detection"] + counts["low_confidence_correct_detection"]
    counts["match_rate"] = counts["correctly_detected_elements"] / len(expected)
    errors = json.loads((run_dir / "errors.json").read_text(encoding="utf-8"))
    return {"counts": dict(counts), "decisions": decisions, "visionqa": {"detection_count": vision["detection_count"], "counts_by_class": vision["counts_by_class"], "inference_time_ms": vision["inference_time_ms"]}, "dom_count": len(dom), "matching": {"match_count": matching["match_count"], "semantic_conflicts": matching["ambiguous_candidate_count"]}, "runtime_errors": errors, "screenshot_sha256": digest(run_dir / "viewport.png")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    url = (ROOT / "tests/fixtures/runtime_fixture.html").resolve().as_uri()
    selected = states(url)
    manifest_path = ROOT / "reports/integration/real_state_manifest.json"
    manifest = {"schema_version": "1.0", "created_at": datetime.now(timezone.utc).isoformat(), "target_type": "realistic_existing_local_fixture_fallback", "reason": "No separate local application or configured deterministic target exists in the repository.", "repeat_count": args.repeats, "states": selected}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    batch_items, run_records = [], []
    for state in selected:
        viewport = state["viewport"]
        collector = PlaywrightEvidenceCollector(viewport=(viewport["width"], viewport["height"]))
        for repeat in range(1, args.repeats + 1):
            run_id = f"real-state-{state['state_id']}-r{repeat}-20261013"
            metadata = collector.capture(url, run_id=run_id, setup_actions=state["setup_actions"])
            run_dir = ROOT / "runs/playwright" / run_id
            batch_items.append({"image": str(run_dir / "viewport.png"), "output": str(run_dir / "visionqa.json")})
            run_records.append({"state_id": state["state_id"], "repeat": repeat, "run_id": run_id, "run_dir": str(run_dir), "metadata": metadata})
    batch_manifest = ROOT / "reports/integration/real_state_inference_manifest.json"
    batch_manifest.write_text(json.dumps({"items": batch_items}, indent=2) + "\n", encoding="utf-8")
    command = [str(ROOT / ".venv-visionqa/Scripts/python.exe"), str(ROOT / "scripts/batch_visionqa_paths.py"), str(batch_manifest), "--device", "0"]
    subprocess.run(command, cwd=ROOT, check=True)
    by_state: dict[str, list[dict]] = {}
    for run in run_records:
        review = review_run(Path(run["run_dir"]), next(row["expected_stable_elements"] for row in selected if row["state_id"] == run["state_id"]))
        run["review"] = review
        by_state.setdefault(run["state_id"], []).append(run)
    repeatability = {}
    for state_id, rows in by_state.items():
        signatures = [{"detections": row["review"]["visionqa"]["detection_count"], "classes": row["review"]["visionqa"]["counts_by_class"], "matches": row["review"]["matching"]["match_count"], "conflicts": row["review"]["matching"]["semantic_conflicts"], "dom_count": row["review"]["dom_count"], "screenshot_sha256": row["review"]["screenshot_sha256"], "runtime_error_counts": {"page_error_count": len(row["review"]["runtime_errors"]["page_errors"]), "console_error_count": row["review"]["runtime_errors"]["console_error_count"], "request_failure_count": row["review"]["runtime_errors"]["request_failure_count"], "http_error_count": row["review"]["runtime_errors"]["http_error_count"], "page_error_messages": [item["message"] for item in row["review"]["runtime_errors"]["page_errors"]]}} for row in rows]
        fields = ("detections", "classes", "matches", "conflicts", "dom_count", "screenshot_sha256", "runtime_error_counts")
        stability = {field: all(item[field] == signatures[0][field] for item in signatures[1:]) for field in fields}
        repeatability[state_id] = {"stable": all(stability.values()), "field_stability": stability, "runs": signatures}
    aggregate = Counter()
    for row in run_records:
        aggregate.update({key: value for key, value in row["review"]["counts"].items() if key != "match_rate"})
    aggregate["match_rate"] = aggregate["correctly_detected_elements"] / aggregate["reviewed_elements"]
    output = {"schema_version": "1.0", "generated_at": datetime.now(timezone.utc).isoformat(), "manifest": str(manifest_path), "state_count": len(selected), "run_count": len(run_records), "repeatability": repeatability, "aggregate_review": dict(aggregate), "runs": run_records, "limitation": "Existing local fixture fallback; this is broader state/viewport validation but not an external production application benchmark."}
    result_path = ROOT / "reports/integration/real_state_evaluation.json"
    result_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"states": len(selected), "runs": len(run_records), "stable_states": sum(item["stable"] for item in repeatability.values()), "aggregate": dict(aggregate)}, indent=2))


if __name__ == "__main__":
    main()
