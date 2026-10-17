import json
from pathlib import Path

from src.playwright_runtime.collector import PlaywrightEvidenceCollector


ROOT = Path(__file__).resolve().parents[2]


def test_real_edge_capture_collects_required_evidence(tmp_path):
    fixture = (ROOT / "tests" / "fixtures" / "runtime_fixture.html").resolve().as_uri()
    collector = PlaywrightEvidenceCollector(runs_root=tmp_path, browser_channel="msedge", viewport=(1000, 700))
    metadata = collector.capture(fixture, run_id="capture-test", settle_ms=300)
    run = tmp_path / "capture-test"
    for name in ("metadata.json", "viewport.png", "full_page.png", "dom.json", "console.json", "network.json", "errors.json", "trace.zip"):
        assert (run / name).is_file() and (run / name).stat().st_size > 0
    dom = json.loads((run / "dom.json").read_text(encoding="utf-8"))
    errors = json.loads((run / "errors.json").read_text(encoding="utf-8"))
    assert metadata["title"] == "QASense Runtime Fixture"
    assert dom["count"] >= 10
    assert any(row["selector"] == "#submit" and row["role"] == "button" for row in dom["elements"])
    assert errors["console_error_count"] >= 1
    assert len(errors["page_errors"]) >= 1
    assert errors["request_failure_count"] >= 1
