"""Capture curated README screenshots from the real local QASense product.

The script never creates analysis data. It selects completed Web, Code, and
Full Analysis runs already exposed by the local API, renders the actual
dashboard, and fails if the page reports console or uncaught browser errors.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]


def api_json(base_url: str, path: str) -> dict:
    with urllib.request.urlopen(f"{base_url}{path}", timeout=10) as response:
        return json.load(response)


def completed_runs_by_type(base_url: str) -> dict[str, str]:
    runs = api_json(base_url, "/api/runs").get("runs", [])
    selected: dict[str, str] = {}
    for item in runs:
        analysis_type = item.get("analysis_type") or item.get("target", {}).get(
            "analysis_type"
        )
        if (
            analysis_type in {"web", "code", "full"}
            and item.get("status") in {"completed", "partial"}
            and analysis_type not in selected
        ):
            selected[analysis_type] = item["run_id"]
    missing = {"web", "code", "full"} - selected.keys()
    if missing:
        raise RuntimeError(f"No completed saved run available for: {sorted(missing)}")
    return selected


def settle(page) -> None:
    page.wait_for_load_state("domcontentloaded")
    page.locator("#overview-loading").wait_for(state="hidden", timeout=15_000)
    page.wait_for_timeout(350)


def show_saved_run(page, run_id: str, tab: str) -> None:
    page.evaluate("([runId, tabName]) => showRun(runId, tabName)", [run_id, tab])
    page.locator("#view-run.active").wait_for(state="visible", timeout=15_000)
    page.wait_for_timeout(250)


def verify_responsive(page, base_url: str) -> dict[str, dict[str, int]]:
    checks: dict[str, dict[str, int]] = {}
    for label, width in (("desktop", 1440), ("laptop", 1024), ("tablet", 768), ("mobile", 390)):
        page.set_viewport_size({"width": width, "height": 900})
        page.goto(f"{base_url}/#overview")
        settle(page)
        widest = 0
        for view in ("overview", "web", "code", "full"):
            page.evaluate("viewName => navigate(viewName, false)", view)
            page.locator(f"#view-{view}.active").wait_for(state="visible", timeout=5_000)
            overflow = page.evaluate(
                "() => Math.max(0, document.documentElement.scrollWidth - window.innerWidth)"
            )
            widest = max(widest, int(overflow))
            if view != "overview":
                button = page.locator(f"#view-{view} .analyze-button")
                if not button.is_visible():
                    raise RuntimeError(f"{label}: {view} analysis action is not visible")
        if widest:
            raise RuntimeError(f"{label}: horizontal overflow is {widest}px")
        checks[label] = {"width": width, "horizontal_overflow_px": widest}
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "screenshots")
    parser.add_argument("--channel", default="msedge")
    args = parser.parse_args()

    if api_json(args.base_url, "/api/health").get("status") != "ok":
        raise RuntimeError(f"QASense is not healthy at {args.base_url}")
    selected = completed_runs_by_type(args.base_url)
    args.output.mkdir(parents=True, exist_ok=True)

    console_errors: list[str] = []
    page_errors: list[str] = []
    api_errors: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.channel, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.on(
            "console",
            lambda message: console_errors.append(message.text)
            if message.type == "error"
            else None,
        )
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.on(
            "response",
            lambda response: api_errors.append(f"{response.status} {response.url}")
            if "/api/" in response.url and response.status >= 400
            else None,
        )

        page.goto(f"{args.base_url}/#overview")
        settle(page)
        page.screenshot(path=args.output / "01-dashboard.png")

        show_saved_run(page, selected["web"], "summary")
        page.screenshot(path=args.output / "02-web-analysis.png")

        show_saved_run(page, selected["code"], "evidence")
        source = page.locator("#run-evidence .source-evidence")
        if source.count():
            source.scroll_into_view_if_needed(timeout=5_000)
        page.screenshot(path=args.output / "03-code-analysis.png")

        show_saved_run(page, selected["full"], "summary")
        page.screenshot(path=args.output / "04-full-analysis.png")

        show_saved_run(page, selected["full"], "evidence")
        screenshot = page.locator("#run-evidence .screenshot-card")
        if screenshot.count():
            screenshot.scroll_into_view_if_needed(timeout=5_000)
        page.screenshot(path=args.output / "05-evidence-details.png")

        evidence_image = page.locator("#evidence-image")
        if evidence_image.count():
            evidence_image.wait_for(state="visible", timeout=5_000)
            loaded = evidence_image.evaluate(
                "image => image.complete && image.naturalWidth > 0 && image.naturalHeight > 0"
            )
            if not loaded:
                raise RuntimeError("Evidence screenshot did not load")

        show_saved_run(page, selected["full"], "findings")
        first_finding = page.locator("#run-findings [data-finding-id]").first
        if first_finding.count():
            first_finding.click()
            page.locator("#finding-drawer.open").wait_for(state="visible", timeout=5_000)
            page.locator("#drawer-close").click()
        responsive = verify_responsive(page, args.base_url)
        browser.close()

    if console_errors or page_errors or api_errors:
        raise RuntimeError(
            json.dumps(
                {
                    "console_errors": console_errors,
                    "page_errors": page_errors,
                    "api_errors": api_errors,
                },
                indent=2,
            )
        )
    print(
        json.dumps(
            {
                "base_url": args.base_url,
                "viewport": "1440x900",
                "runs": selected,
                "screenshots": sorted(path.name for path in args.output.glob("*.png")),
                "responsive": responsive,
                "console_errors": 0,
                "page_errors": 0,
                "api_errors": 0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
