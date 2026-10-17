"""Launch the local QASense API and dashboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.product.api import serve  # noqa: E402

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--host",default="127.0.0.1"); parser.add_argument("--port",type=int,default=8765); args=parser.parse_args(); serve(args.host,args.port)
