"""Reusable Playwright browser evidence collector."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

from .dom import DOM_EXTRACTION_SCRIPT

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = ROOT / "runs" / "playwright"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_run_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex[:8]}"


class PlaywrightEvidenceCollector:
    def __init__(self, runs_root: str | Path = DEFAULT_RUNS, browser_channel: str = "msedge",
                 headless: bool = True, viewport: tuple[int, int] = (1440, 900)) -> None:
        self.runs_root = Path(runs_root).resolve()
        self.browser_channel, self.headless, self.viewport = browser_channel, headless, viewport

    def capture(self, url: str, run_id: str | None = None, wait_until: str = "load",
                settle_ms: int = 500, timeout_ms: int = 30_000,
                setup_actions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if not re.match(r"^(https?|file|data):", url, flags=re.IGNORECASE):
            raise ValueError("URL must use http, https, file, or data scheme")
        run_id = run_id or make_run_id(); run_dir = self.runs_root / run_id
        if run_dir.exists(): raise FileExistsError(f"Run ID already exists: {run_id}")
        run_dir.mkdir(parents=True)
        console: list[dict[str, Any]] = []; page_errors: list[dict[str, Any]] = []
        request_failures: list[dict[str, Any]] = []; http_errors: list[dict[str, Any]] = []
        started_at = utc_now()
        with sync_playwright() as playwright:
            browser: Browser = playwright.chromium.launch(channel=self.browser_channel, headless=self.headless)
            context: BrowserContext = browser.new_context(viewport={"width": self.viewport[0], "height": self.viewport[1]}, device_scale_factor=1)
            context.tracing.start(screenshots=True, snapshots=True, sources=True)
            page: Page = context.new_page()
            page.on("console", lambda message: console.append({"timestamp": utc_now(), "type": message.type, "text": message.text, "location": message.location}))
            page.on("pageerror", lambda error: page_errors.append({"timestamp": utc_now(), "name": error.name, "message": error.message}))
            page.on("requestfailed", lambda request: request_failures.append({"timestamp": utc_now(), "url": request.url, "method": request.method, "resource_type": request.resource_type, "failure": request.failure}))
            page.on("response", lambda response: http_errors.append({"timestamp": utc_now(), "url": response.url, "status": response.status, "status_text": response.status_text, "request_method": response.request.method, "resource_type": response.request.resource_type}) if response.status >= 400 else None)
            navigation_response = page.goto(url, wait_until=wait_until, timeout=timeout_ms)
            for action in setup_actions or []:
                selector = action.get("selector")
                kind = action.get("action")
                locator = page.locator(selector) if selector else None
                if kind == "fill": locator.fill(str(action.get("value", "")))
                elif kind == "check": locator.check()
                elif kind == "uncheck": locator.uncheck()
                elif kind == "click": locator.click()
                elif kind == "focus": locator.focus()
                else: raise ValueError(f"Unsupported setup action: {action!r}")
            if settle_ms: page.wait_for_timeout(settle_ms)
            final_url, title, viewport = page.url, page.title(), page.viewport_size
            page.screenshot(path=run_dir / "viewport.png", full_page=False)
            page.screenshot(path=run_dir / "full_page.png", full_page=True)
            dom = page.evaluate(DOM_EXTRACTION_SCRIPT)
            timing = page.evaluate("""() => { const nav=performance.getEntriesByType('navigation')[0]; const paint=Object.fromEntries(performance.getEntriesByType('paint').map(x=>[x.name,x.startTime])); return nav ? {dns_ms:nav.domainLookupEnd-nav.domainLookupStart,connect_ms:nav.connectEnd-nav.connectStart,ttfb_ms:nav.responseStart-nav.requestStart,dom_content_loaded_ms:nav.domContentLoadedEventEnd,load_event_ms:nav.loadEventEnd,transfer_size_bytes:nav.transferSize,encoded_body_size_bytes:nav.encodedBodySize,decoded_body_size_bytes:nav.decodedBodySize,paints_ms:paint}:{paints_ms:paint}; }""")
            document_size = page.evaluate("() => ({width: document.documentElement.scrollWidth, height: document.documentElement.scrollHeight})")
            context.tracing.stop(path=run_dir / "trace.zip"); browser.close()
        network = {"failed_requests": request_failures, "http_error_responses": http_errors}
        errors = {"page_errors": page_errors, "console_error_count": sum(item["type"] == "error" for item in console), "request_failure_count": len(request_failures), "http_error_count": len(http_errors)}
        metadata = {"schema_version": "1.0", "run_id": run_id, "started_at": started_at, "finished_at": utc_now(), "requested_url": url, "url": final_url, "title": title,
            "navigation_status": navigation_response.status if navigation_response else None, "viewport": viewport, "document_size": document_size,
            "browser_channel": self.browser_channel, "headless": self.headless, "setup_actions": setup_actions or [], "timing": timing,
            "artifacts": {"viewport_screenshot": str(run_dir / "viewport.png"), "full_page_screenshot": str(run_dir / "full_page.png"), "dom": str(run_dir / "dom.json"), "console": str(run_dir / "console.json"), "network": str(run_dir / "network.json"), "errors": str(run_dir / "errors.json"), "trace": str(run_dir / "trace.zip")}}
        atomic_json(run_dir / "dom.json", {"elements": dom, "count": len(dom)})
        atomic_json(run_dir / "console.json", {"messages": console, "count": len(console)})
        atomic_json(run_dir / "network.json", network); atomic_json(run_dir / "errors.json", errors)
        atomic_json(run_dir / "metadata.json", metadata); return metadata
