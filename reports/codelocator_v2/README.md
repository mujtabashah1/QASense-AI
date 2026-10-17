# CodeLocator v2 reproduction

CodeLocator v1 remains separate and unchanged.

```powershell
.\.venv-codelocator\Scripts\python.exe scripts/prepare_codelocator_v2.py
.\.venv-codelocator\Scripts\python.exe scripts/benchmark_codelocator_v2.py
.\.venv-codelocator\Scripts\python.exe scripts/benchmark_codelocator_v2_inference.py
.\.venv-codelocator\Scripts\python.exe scripts/train_codelocator_v2.py
.\.venv-codelocator\Scripts\python.exe scripts/infer_codelocator_v2.py --file Example.java
.\.venv-codelocator\Scripts\python.exe -m pytest tests/codelocator -q
```

Training resumes automatically from `models/codelocator_v2/checkpoints/latest.pt` when configuration and split hashes match. The pre-training gate cannot be bypassed when its scientific checks fail.
