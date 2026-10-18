"""End-to-end QASense product orchestration with partial-result recovery."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

from src.playwright_runtime.collector import PlaywrightEvidenceCollector, atomic_json, make_run_id
from src.visionqa.matching import match_dom_to_detections
from src.visual_defects.runtime import inspect_runtime_visuals

from .contracts import empty_analysis
from .fusion import build_analysis
from .source_analysis import build_source_manifest

ROOT = Path(__file__).resolve().parents[2]
PRODUCT_RUNS = ROOT / "runs" / "product"
PLAYWRIGHT_RUNS = ROOT / "runs" / "playwright"


def _load(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _subprocess_json(command: list[str], output: Path, cwd: Path = ROOT) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout or "subprocess failed")[-4000:])
    if not output.is_file():
        raise RuntimeError(f"Expected output was not created: {output}")
    return _load(output, {})


def annotate_detections(image_path: Path, detections: list[dict[str, Any]], output: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image); font = ImageFont.load_default()
    palette = ((35, 99, 235), (5, 150, 105), (220, 38, 38), (147, 51, 234), (234, 88, 12))
    for item in detections:
        box = tuple(round(v) for v in item["bbox_xyxy"]); color = palette[item["class_id"] % len(palette)]
        draw.rectangle(box, outline=color, width=3)
        label = f"{item['class_name']} {item['confidence']:.2f}"
        text_box = draw.textbbox((box[0], box[1]), label, font=font)
        draw.rectangle((text_box[0]-2, text_box[1]-2, text_box[2]+2, text_box[3]+2), fill=color)
        draw.text((box[0], box[1]), label, fill="white", font=font)
    image.save(output)


class QASenseOrchestrator:
    def __init__(self, runs_root: str | Path = PRODUCT_RUNS, browser_channel: str = "msedge", device: str = "0") -> None:
        self.runs_root = Path(runs_root).resolve(); self.browser_channel = browser_channel; self.device = device

    def run(self, url: str | None = None, *, run_id: str | None = None,
            viewport: tuple[int, int] = (1000, 700), source_file: str | Path | None = None,
            source_path: str | Path | None = None, risk_metrics: dict[str, float] | None = None,
            analysis_type: str | None = None, max_source_files: int = 20,
            progress_callback: Callable[[str, str], None] | None = None) -> dict[str, Any]:
        analysis_type = analysis_type or ("full" if source_file or source_path else "web")
        if analysis_type not in {"web", "code", "full"}:
            raise ValueError("analysis_type must be web, code, or full")
        if analysis_type in {"web", "full"} and not str(url or "").strip():
            raise ValueError("url is required for Web and Full Analysis")
        selected_source = source_path or source_file
        if analysis_type in {"code", "full"} and not selected_source:
            raise ValueError("source_path is required for Code and Full Analysis")

        def progress(stage: str, message: str) -> None:
            if progress_callback:
                progress_callback(stage, message)

        started = time.perf_counter(); run_id = run_id or f"product-{make_run_id()}"; run_dir = self.runs_root / run_id
        if run_dir.exists():
            raise FileExistsError(run_dir)
        run_dir.mkdir(parents=True)
        result = empty_analysis(run_id, {
            "analysis_type": analysis_type, "url": url if analysis_type in {"web", "full"} else None,
            "viewport": {"width": viewport[0], "height": viewport[1]},
            "source_path": str(selected_source) if selected_source else None,
        })
        result["started_at"] = datetime.now(timezone.utc).isoformat()

        capture_id = f"{run_id}-browser"; capture_dir = PLAYWRIGHT_RUNS / capture_id
        if analysis_type in {"web", "full"}:
            progress("browser", "Capturing browser, network, console, and DOM evidence")
            capture_started = time.perf_counter()
            try:
                metadata = PlaywrightEvidenceCollector(browser_channel=self.browser_channel, viewport=viewport).capture(str(url), run_id=capture_id, settle_ms=400)
                result["timing"]["playwright_ms"] = round((time.perf_counter()-capture_started)*1000, 3)
                console = _load(capture_dir/"console.json", {}).get("messages", [])
                network = _load(capture_dir/"network.json", {})
                errors = _load(capture_dir/"errors.json", {})
                dom = _load(capture_dir/"dom.json", {"elements": []})
                result["runtime"] = {"status": "available", "metadata": metadata, "console": console,
                    "page_errors": errors.get("page_errors", []), "failed_requests": network.get("failed_requests", []),
                    "http_error_responses": network.get("http_error_responses", [])}
                result["artifacts"].update({"viewport_screenshot": str(capture_dir/"viewport.png"), "full_page_screenshot": str(capture_dir/"full_page.png"), "dom": str(capture_dir/"dom.json"), "console": str(capture_dir/"console.json"), "network": str(capture_dir/"network.json"), "trace": str(capture_dir/"trace.zip")})
            except Exception as exc:
                result["timing"]["playwright_ms"] = round((time.perf_counter()-capture_started)*1000, 3)
                result["errors"].append({"module": "playwright", "message": str(exc)})
                dom = {"elements": []}

            progress("vision", "Analyzing visible UI elements and DOM agreement")
            vision_started = time.perf_counter()
            screenshot = capture_dir / "viewport.png"
            try:
                vision_path = run_dir / "visionqa.json"
                vision = _subprocess_json([
                    str(ROOT/".venv-visionqa/Scripts/python.exe"), str(ROOT/"scripts/visionqa_inference.py"), str(screenshot),
                    "--checkpoint", str(ROOT/"models/visionqa/atomblock_yolo11n/best.pt"), "--confidence", "0.25", "--device", self.device, "--output", str(vision_path),
                ], vision_path)
                matching = match_dom_to_detections(dom.get("elements", []), vision.get("detections", []))
                defects = inspect_runtime_visuals(dom, {"width": viewport[0], "height": viewport[1]}, matching)
                atomic_json(run_dir/"dom_vision_matches.json", matching); atomic_json(run_dir/"visual_rules.json", {"defects": defects})
                annotated = run_dir/"visionqa_annotated.png"; annotate_detections(screenshot, vision.get("detections", []), annotated)
                result["visual"] = {"status": "available", "visionqa": vision, "matching": matching, "defects": defects}
                result["artifacts"].update({"visionqa": str(vision_path), "dom_vision_matches": str(run_dir/"dom_vision_matches.json"), "visual_rules": str(run_dir/"visual_rules.json"), "annotated_screenshot": str(annotated)})
                result["timing"]["visionqa_total_ms"] = round((time.perf_counter()-vision_started)*1000, 3)
                result["timing"]["visionqa_inference_ms"] = vision.get("inference_time_ms")
            except Exception as exc:
                result["timing"]["visionqa_total_ms"] = round((time.perf_counter()-vision_started)*1000, 3)
                result["errors"].append({"module": "visionqa", "message": str(exc)})

        if analysis_type in {"code", "full"}:
            source_started = time.perf_counter()
            try:
                if risk_metrics is not None and Path(selected_source).resolve().is_file():
                    manifest = {"source_root": str(Path(selected_source).resolve()), "supported_language": "java", "files": [{"file": str(Path(selected_source).resolve()), "metrics": risk_metrics, "metrics_extractor": "provided"}]}
                else:
                    manifest = build_source_manifest(selected_source, max_files=max_source_files)
                manifest_path = run_dir / "source_manifest.json"; atomic_json(manifest_path, manifest)
                result["artifacts"]["source_manifest"] = str(manifest_path)
                result["target"]["source_file_count"] = len(manifest["files"])
            except Exception as exc:
                manifest = {"files": []}
                result["errors"].append({"module": "source", "message": str(exc)})

            if manifest["files"]:
                progress("risk", f"Prioritizing {len(manifest['files'])} source file(s) by defect risk")
                try:
                    input_path = run_dir/"risknet_input.json"; output_path = run_dir/"risknet.json"
                    atomic_json(input_path, {"items": [{"file": item["file"], "metrics": item["metrics"]} for item in manifest["files"]]})
                    risk = _subprocess_json([str(ROOT/".venv-risknet/Scripts/python.exe"), str(ROOT/"scripts/product_model_inference.py"), "risknet", str(input_path), str(output_path)], output_path)
                    result["risk"] = {
                        "status": "available", "items": risk["items"],
                        "metrics_extractor": manifest["files"][0].get("metrics_extractor"),
                        "metrics_limitation": manifest["files"][0].get("metrics_limitation"),
                    }
                    result["artifacts"]["risknet"] = str(output_path)
                except Exception as exc:
                    result["errors"].append({"module": "risknet", "message": str(exc)})

                progress("code", f"Ranking suspicious lines in {len(manifest['files'])} source file(s)")
                try:
                    input_path = run_dir/"codelocator_input.json"; output_path = run_dir/"codelocator.json"
                    atomic_json(input_path, {"items": [{"file": item["file"], "top_k": 5} for item in manifest["files"]]})
                    code = _subprocess_json([str(ROOT/".venv-codelocator/Scripts/python.exe"), str(ROOT/"scripts/product_model_inference.py"), "codelocator", str(input_path), str(output_path)], output_path)
                    result["code"] = {"status": "available", "items": code["items"]}
                    result["artifacts"]["codelocator"] = str(output_path)
                except Exception as exc:
                    result["errors"].append({"module": "codelocator", "message": str(exc)})
            result["timing"]["source_analysis_ms"] = round((time.perf_counter() - source_started) * 1000, 3)

        progress("fusion", "Combining evidence and preparing QA actions")
        result = build_analysis(result)
        result["finished_at"] = datetime.now(timezone.utc).isoformat(); result["timing"]["total_ms"] = round((time.perf_counter()-started)*1000,3)
        result["artifacts"]["analysis"] = str(run_dir/"analysis.json")
        atomic_json(run_dir/"analysis.json", result)
        return result
