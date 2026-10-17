"""Summarize Ultralytics cache findings without modifying AtomBlock data."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "datasets" / "processed" / "visionqa_atomblock"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "visionqa" / "metrics" / "loader_audit.json",
    )
    args = parser.parse_args()
    payload = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset.resolve()),
        "dataset_modified": False,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        cache_path = args.dataset.resolve() / split / "labels.cache"
        if not cache_path.exists():
            payload["splits"][split] = {"cache_exists": False}
            continue
        cache = np.load(cache_path, allow_pickle=True).item()
        messages = [ANSI.sub("", message) for message in cache.get("msgs", [])]
        corrupt = [message for message in messages if "corrupt image/label" in message]
        duplicates = [message for message in messages if "duplicate labels removed" in message]
        payload["splits"][split] = {
            "cache_exists": True,
            "loader_results": list(cache.get("results", [])),
            "duplicate_rows_removed_in_memory": sum(
                int(match.group(1))
                for message in duplicates
                if (match := re.search(r"(\d+) duplicate labels removed", message))
            ),
            "files_with_duplicate_rows": len(duplicates),
            "corrupt_files_excluded": corrupt,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
