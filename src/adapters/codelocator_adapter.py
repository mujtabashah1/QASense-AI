"""Normalize completed CodeLocator v2 inference for multimodal evidence."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


class CodeLocatorAdapter:
    def __init__(self, backend: Any | None = None) -> None:
        if backend is None:
            source_root = str(Path(__file__).resolve().parents[1])
            if source_root not in sys.path:
                sys.path.insert(0, source_root)
            from codelocator_v2.inference import CodeLocatorV2
            backend = CodeLocatorV2()
        self.backend = backend

    def locate(self, file: str | Path, code: str | None = None, top_k: int = 5) -> dict[str, Any]:
        path = Path(file)
        if code is None:
            if not path.is_file():
                raise FileNotFoundError(path)
            code = path.read_text(encoding="utf-8")
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        result = self.backend.locate(code, top_k=top_k)
        return {
            "schema_version": "1.0", "module": "codelocator_v2", "file": str(file),
            "suspicious_lines": [
                {"line": int(row["line_number"]), "score": float(row["score"]), "rank": rank, "code": row.get("code", "")}
                for rank, row in enumerate(result.get("suspicious_lines", []), 1)
            ],
            "model_path": "models/codelocator_v2/line_localizer",
            "limitation": result.get("limitation", "Scores rank suspiciousness and do not prove an exact defect."),
        }
