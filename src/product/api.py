"""Dependency-free local API and dashboard server for QASense."""

from __future__ import annotations

import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .model_status import get_model_status
from .orchestrator import PRODUCT_RUNS, QASenseOrchestrator, ROOT

DASHBOARD = Path(__file__).with_name("dashboard")


class RunRegistry:
    def __init__(self, orchestrator: QASenseOrchestrator | None = None) -> None:
        self.orchestrator = orchestrator or QASenseOrchestrator(); self.live: dict[str, dict[str, Any]] = {}; self.lock = threading.Lock()

    def _analysis_path(self, run_id: str) -> Path:
        if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in run_id):
            raise ValueError("Invalid run id")
        return PRODUCT_RUNS / run_id / "analysis.json"

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self.lock:
            live = self.live.get(run_id)
        if live and live.get("status") in {"queued", "running", "failed"}:
            return live
        path = self._analysis_path(run_id)
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else live

    def list(self) -> list[dict[str, Any]]:
        found = {}
        if PRODUCT_RUNS.exists():
            for path in PRODUCT_RUNS.glob("*/analysis.json"):
                try:
                    item=json.loads(path.read_text(encoding="utf-8")); found[item["run_id"]]=item
                except (OSError,json.JSONDecodeError,KeyError):
                    continue
        with self.lock: found.update(self.live)
        return sorted(({"run_id":v["run_id"],"status":v["status"],"target":v.get("target",{}),"started_at":v.get("started_at"),"finding_count":len(v.get("findings",[]))} for v in found.values()),key=lambda x:x.get("started_at") or "",reverse=True)

    def start(self, payload: dict[str, Any]) -> str:
        url=str(payload.get("url","")).strip()
        if not url: raise ValueError("url is required")
        run_id=str(payload.get("run_id") or "").strip() or None
        if run_id is None:
            from src.playwright_runtime.collector import make_run_id
            run_id=f"product-{make_run_id()}"
        self._analysis_path(run_id)
        viewport=payload.get("viewport") or {}; size=(int(viewport.get("width",1000)),int(viewport.get("height",700)))
        state={"schema_version":"1.0","run_id":run_id,"status":"queued","target":{"url":url,"viewport":{"width":size[0],"height":size[1]}},"findings":[]}
        with self.lock: self.live[run_id]=state
        def execute() -> None:
            with self.lock: self.live[run_id]={**state,"status":"running"}
            try:
                result=self.orchestrator.run(url,run_id=run_id,viewport=size,source_file=payload.get("source_file"),risk_metrics=payload.get("risk_metrics"))
                with self.lock: self.live[run_id]=result
            except Exception as exc:
                with self.lock: self.live[run_id]={**state,"status":"failed","errors":[{"module":"orchestrator","message":str(exc)}]}
        threading.Thread(target=execute,name=f"qasense-{run_id}",daemon=True).start(); return run_id


class QASenseHandler(BaseHTTPRequestHandler):
    registry = RunRegistry()
    server_version = "QASense/1.0"

    def _json(self, value: Any, status: int = 200) -> None:
        body=json.dumps(value,ensure_ascii=False).encode(); self.send_response(status); self.send_header("Content-Type","application/json; charset=utf-8"); self.send_header("Content-Length",str(len(body))); self.send_header("Cache-Control","no-store"); self.end_headers(); self.wfile.write(body)

    def _static(self, path: Path) -> None:
        if not path.is_file() or DASHBOARD.resolve() not in path.resolve().parents:
            self.send_error(404); return
        body=path.read_bytes(); self.send_response(200); self.send_header("Content-Type",mimetypes.guess_type(path.name)[0] or "application/octet-stream"); self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self) -> None:
        route=unquote(urlparse(self.path).path)
        if route in {"/","/index.html"}: return self._static(DASHBOARD/"index.html")
        if route.startswith("/assets/"): return self._static(DASHBOARD/route.removeprefix("/assets/"))
        if route=="/api/health": return self._json({"status":"ok","schema_version":"1.0"})
        if route=="/api/models/status": return self._json(get_model_status(False))
        if route=="/api/runs": return self._json({"runs":self.registry.list()})
        parts=[p for p in route.split("/") if p]
        if len(parts)>=3 and parts[:2]==["api","runs"]:
            run=self.registry.get(parts[2])
            if run is None: return self._json({"error":"run not found"},404)
            if len(parts)==3: return self._json(run)
            if parts[3]=="evidence":
                return self._json({"risk":run.get("risk"),"code":run.get("code"),"runtime":run.get("runtime"),"visual":run.get("visual"),"artifacts":run.get("artifacts",{})})
            field={"findings":"findings","recommendations":"recommendations","tests":"test_suggestions","regression":"regression_priority"}.get(parts[3])
            if field: return self._json({field:run.get(field,[] if field!="artifacts" else {})})
            if parts[3]=="artifact" and len(parts)==5:
                artifact=run.get("artifacts",{}).get(parts[4]); path=Path(artifact).resolve() if artifact else None
                if path and path.is_file() and (ROOT.resolve()==path or ROOT.resolve() in path.parents):
                    body=path.read_bytes(); self.send_response(200); self.send_header("Content-Type",mimetypes.guess_type(path.name)[0] or "application/octet-stream"); self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body); return
            return self._json({"error":"unknown run resource"},404)
        self.send_error(404)

    def do_POST(self) -> None:
        if urlparse(self.path).path!="/api/runs": return self._json({"error":"not found"},404)
        try:
            length=int(self.headers.get("Content-Length","0")); payload=json.loads(self.rfile.read(length) or b"{}")
            run_id=self.registry.start(payload); self._json({"run_id":run_id,"status":"queued"},HTTPStatus.ACCEPTED)
        except (ValueError,json.JSONDecodeError) as exc: self._json({"error":str(exc)},400)

    def log_message(self, format: str, *args: Any) -> None:
        return


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    print(f"QASense dashboard: http://{host}:{port}")
    ThreadingHTTPServer((host,port),QASenseHandler).serve_forever()
