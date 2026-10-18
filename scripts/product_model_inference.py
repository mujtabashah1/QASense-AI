"""Run a protected QASense model adapter inside its validated environment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))


def atomic_json(path: Path, value: dict) -> None:
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    if len(sys.argv)!=4 or sys.argv[1] not in {"risknet","codelocator"}:
        raise SystemExit("usage: product_model_inference.py {risknet|codelocator} INPUT OUTPUT")
    module,input_path,output_path=sys.argv[1],Path(sys.argv[2]),Path(sys.argv[3]); payload=json.loads(input_path.read_text(encoding="utf-8"))
    items = payload.get("items")
    if module=="risknet" and items is not None:
        from src.adapters.risknet_adapter import RiskNetAdapter
        adapter = RiskNetAdapter()
        result = {"items": [adapter.predict(item["file"], item["metrics"]) for item in items]}
    elif module=="codelocator" and items is not None:
        from src.adapters.codelocator_adapter import CodeLocatorAdapter
        adapter = CodeLocatorAdapter()
        result = {"items": [adapter.locate(item["file"], top_k=int(item.get("top_k", 5))) for item in items]}
    elif module=="risknet":
        from src.adapters.risknet_adapter import RiskNetAdapter
        result=RiskNetAdapter().predict(payload["file"],payload["metrics"])
    else:
        from src.adapters.codelocator_adapter import CodeLocatorAdapter
        result=CodeLocatorAdapter().locate(payload["file"],top_k=int(payload.get("top_k",5)))
    atomic_json(output_path,result); print(json.dumps(result)); return 0


if __name__=="__main__": raise SystemExit(main())
