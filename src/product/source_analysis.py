"""Safe source discovery and deterministic Java metrics for product inference."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

SUPPORTED_SUFFIXES = {".java"}
EXCLUDED_DIRECTORIES = {
    ".git", ".idea", ".vscode", ".venv", "build", "dist", "node_modules",
    "out", "target", "vendor", "runs", "datasets",
}
RISKNET_FEATURES = (
    "loc", "v(g)", "ev(g)", "iv(g)", "n", "v", "l", "d", "i", "e", "b", "t",
    "lOCode", "lOComment", "lOBlank", "locCodeAndComment", "uniq_Op", "uniq_Opnd",
    "total_Op", "total_Opnd", "branchCount",
)

_OPERATOR_RE = re.compile(
    r"(?:>>>=|<<=|>>=|==|!=|<=|>=|&&|\|\||\+\+|--|\+=|-=|\*=|/=|%=|->|::|<<|>>>|>>|"
    r"[+\-*/%=&|^~!<>?:])"
)
_TOKEN_RE = re.compile(r'"(?:\\.|[^"\\])*"|\b\d+(?:\.\d+)?\b|\b[A-Za-z_$][\w$]*\b')
_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class",
    "const", "continue", "default", "do", "double", "else", "enum", "extends", "final",
    "finally", "float", "for", "goto", "if", "implements", "import", "instanceof", "int",
    "interface", "long", "native", "new", "package", "private", "protected", "public", "return",
    "short", "static", "strictfp", "super", "switch", "synchronized", "this", "throw", "throws",
    "transient", "try", "void", "volatile", "while", "true", "false", "null", "record", "var",
}


def discover_source_files(source: str | Path, max_files: int = 20) -> list[Path]:
    """Return deterministic, supported source files without following build/data trees."""
    if not isinstance(max_files, int) or not 1 <= max_files <= 100:
        raise ValueError("max_files must be between 1 and 100")
    root = Path(source).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Source path does not exist: {root}")
    if root.is_file():
        if root.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError("Code Analysis currently supports Java source files")
        return [root]
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if any(
            part.lower() in EXCLUDED_DIRECTORIES or part.lower().startswith(".venv")
            for part in path.relative_to(root).parts[:-1]
        ):
            continue
        candidates.append(path.resolve())
    return sorted(candidates, key=lambda item: str(item).casefold())[:max_files]


def _strip_comments(source: str) -> tuple[str, list[tuple[bool, bool]]]:
    """Strip Java comments while retaining line structure and line-type evidence."""
    output: list[str] = []
    kinds: list[tuple[bool, bool]] = []
    in_block = False
    for original in source.splitlines():
        index = 0
        code: list[str] = []
        comment = False
        in_string: str | None = None
        while index < len(original):
            char = original[index]
            pair = original[index:index + 2]
            if in_block:
                comment = True
                if pair == "*/":
                    in_block = False
                    index += 2
                else:
                    index += 1
                continue
            if in_string:
                code.append(char)
                if char == "\\" and index + 1 < len(original):
                    code.append(original[index + 1]); index += 2; continue
                if char == in_string:
                    in_string = None
                index += 1; continue
            if pair == "//":
                comment = True; break
            if pair == "/*":
                comment = True; in_block = True; index += 2; continue
            if char in {'"', "'"}:
                in_string = char
            code.append(char); index += 1
        cleaned = "".join(code)
        output.append(cleaned)
        kinds.append((bool(cleaned.strip()), comment))
    return "\n".join(output), kinds


def extract_risknet_metrics(path: str | Path) -> dict[str, float]:
    """Compute the exact 21-feature RiskNet contract from Java source.

    The lexical Halstead/complexity values are a deterministic runtime estimate,
    not a replacement for the original CK-based research extractor.
    """
    source_path = Path(path).resolve()
    source = source_path.read_text(encoding="utf-8")
    cleaned, line_kinds = _strip_comments(source)
    operators = _OPERATOR_RE.findall(cleaned)
    tokens = [token for token in _TOKEN_RE.findall(cleaned) if token not in _KEYWORDS]
    unique_operators, unique_operands = set(operators), set(tokens)
    n1, n2 = len(unique_operators), len(unique_operands)
    total_operators, total_operands = len(operators), len(tokens)
    vocabulary = n1 + n2
    length = total_operators + total_operands
    volume = length * math.log2(vocabulary) if vocabulary and length else 0.0
    difficulty = (n1 / 2.0) * (total_operands / n2) if n2 else 0.0
    level = 1.0 / difficulty if difficulty else 0.0
    effort = difficulty * volume
    intelligence = volume / difficulty if difficulty else 0.0
    branch_count = len(re.findall(r"\b(?:if|for|while|case|catch)\b|&&|\|\||\?", cleaned))
    code_lines = sum(code for code, _ in line_kinds)
    comment_lines = sum(comment for _, comment in line_kinds)
    blank_lines = sum(not code and not comment for code, comment in line_kinds)
    mixed_lines = sum(code and comment for code, comment in line_kinds)
    cyclomatic = float(branch_count + 1)
    metrics = {
        "loc": float(len(source.splitlines())), "v(g)": cyclomatic, "ev(g)": cyclomatic,
        "iv(g)": cyclomatic, "n": float(vocabulary), "v": float(volume), "l": float(level),
        "d": float(difficulty), "i": float(intelligence), "e": float(effort),
        "b": float(volume / 3000.0), "t": float(effort / 18.0),
        "lOCode": float(code_lines), "lOComment": float(comment_lines),
        "lOBlank": float(blank_lines), "locCodeAndComment": float(mixed_lines),
        "uniq_Op": float(n1), "uniq_Opnd": float(n2), "total_Op": float(total_operators),
        "total_Opnd": float(total_operands), "branchCount": float(branch_count),
    }
    if tuple(metrics) != RISKNET_FEATURES:
        raise RuntimeError("RiskNet runtime feature order changed unexpectedly")
    return metrics


def build_source_manifest(source: str | Path, max_files: int = 20) -> dict[str, Any]:
    root = Path(source).expanduser().resolve()
    files = discover_source_files(root, max_files=max_files)
    if not files:
        raise ValueError("No supported Java source files were found")
    return {
        "source_root": str(root),
        "supported_language": "java",
        "requested_max_files": max_files,
        "files": [
            {
                "file": str(path),
                "relative_path": path.name if root.is_file() else str(path.relative_to(root)),
                "metrics": extract_risknet_metrics(path),
                "metrics_extractor": "java_static_metrics_v1",
                "metrics_limitation": "Runtime lexical estimate aligned to the 21-feature contract; it is not the original CK extractor.",
            }
            for path in files
        ],
    }
