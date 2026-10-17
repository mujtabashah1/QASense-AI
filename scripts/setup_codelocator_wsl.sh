#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/QASense-AI
python3 -m venv .venv-codelocator-linux
.venv-codelocator-linux/bin/python -m pip install --upgrade pip
.venv-codelocator-linux/bin/python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
.venv-codelocator-linux/bin/python -m pip install -r requirements-codelocator.txt
.venv-codelocator-linux/bin/python scripts/run_codelocator_pipeline.py
