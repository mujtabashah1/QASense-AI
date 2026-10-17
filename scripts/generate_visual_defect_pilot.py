"""Generate a deterministic 108-case visual-defect pilot with Playwright."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.playwright_runtime.dom import DOM_EXTRACTION_SCRIPT
from src.visual_defects.mutations import MUTATION_TYPES, apply_mutation_sync, revert_mutation_sync

VIEWPORTS = {"desktop": (1440, 900), "tablet": (768, 1024), "mobile": (390, 844)}
TARGETS = ("#submit", "#email", "#country")
AFFECTED = {"#submit": "#email", "#email": "#country", "#country": "#submit"}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def element_by_selector(elements: list[dict], selector: str) -> dict | None:
    return next((item for item in elements if item["selector"] == selector), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=(ROOT / "tests/fixtures/runtime_fixture.html").resolve().as_uri())
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/visual_defects_generated/pilot_v2")
    parser.add_argument("--channel", default="msedge")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Pilot output already contains files: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cases: list[dict] = []
    failures: list[dict] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.channel, headless=True)
        for viewport_name, (width, height) in VIEWPORTS.items():
            context = browser.new_context(viewport={"width": width, "height": height}, device_scale_factor=1)
            page = context.new_page()
            for defect_type in MUTATION_TYPES:
                for target_index, target_selector in enumerate(TARGETS, 1):
                    case_id = f"{viewport_name}-{defect_type}-{target_index:02d}"
                    case_dir = output / case_id
                    case_dir.mkdir()
                    try:
                        page.goto(args.url, wait_until="load")
                        page.wait_for_timeout(50)
                        before = page.evaluate(DOM_EXTRACTION_SCRIPT)
                        before_document = page.evaluate("() => ({width:document.documentElement.scrollWidth,height:document.documentElement.scrollHeight})")
                        page.screenshot(path=case_dir / "baseline.png", full_page=False)
                        mutation = apply_mutation_sync(page, defect_type, target_selector, AFFECTED[target_selector])
                        page.wait_for_timeout(20)
                        after = page.evaluate(DOM_EXTRACTION_SCRIPT)
                        after_document = page.evaluate("() => ({width:document.documentElement.scrollWidth,height:document.documentElement.scrollHeight})")
                        page.screenshot(path=case_dir / "mutated.png", full_page=False)
                        revert = revert_mutation_sync(page)
                        reverted = page.evaluate(DOM_EXTRACTION_SCRIPT)
                        before_target, reverted_target = element_by_selector(before, target_selector), element_by_selector(reverted, target_selector)
                        reversal_ok = bool(before_target and reverted_target and before_target["bbox_xyxy"] == reverted_target["bbox_xyxy"])
                        ground_truth = {
                            "schema_version": "1.0", "case_id": case_id, "defect_type": defect_type,
                            "target_selector": target_selector, "affected_selector": AFFECTED[target_selector],
                            "severity": "major", "viewport": {"name": viewport_name, "width": width, "height": height},
                            "baseline": str(case_dir / "baseline.png"), "mutated": str(case_dir / "mutated.png"),
                            "expected_region": mutation["expected_region"], "mutation": mutation,
                            "reversal": {**revert, "geometry_restored": reversal_ok},
                        }
                        write_json(case_dir / "dom_before.json", {"elements": before, "document_size": before_document})
                        write_json(case_dir / "dom_after.json", {"elements": after, "document_size": after_document})
                        write_json(case_dir / "ground_truth.json", ground_truth)
                        cases.append({"case_id": case_id, "case_dir": str(case_dir), "defect_type": defect_type, "target_selector": target_selector, "viewport": viewport_name, "reversal_ok": reversal_ok})
                    except Exception as exc:
                        failures.append({"case_id": case_id, "error": repr(exc)})
            context.close()
        browser.close()
    manifest = {
        "schema_version": "1.0", "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_url": args.url, "controlled_synthetic_benchmark": True,
        "case_count": len(cases), "failure_count": len(failures), "cases": cases, "failures": failures,
        "limitation": "Controlled CSS/DOM mutations on one fixture measure pipeline behavior, not real-world defect accuracy.",
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "cases": len(cases), "failures": failures, "reversal_failures": sum(not item["reversal_ok"] for item in cases)}, indent=2))


if __name__ == "__main__":
    main()
