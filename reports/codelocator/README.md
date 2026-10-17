# CodeLocator reproduction and artifacts

Run commands from `C:\QASense-AI`. RiskNet uses a separate environment and is unchanged.

```powershell
python -m venv .venv-codelocator
.\.venv-codelocator\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv-codelocator\Scripts\python.exe -m pip install -r requirements-codelocator.txt
.\.venv-codelocator\Scripts\python.exe scripts/run_codelocator_pipeline.py
```

Individual stages:

CodeBERT layer indices are zero-based: embeddings and encoder layers **0–7 are frozen**; only encoder layers **8–11 and the task heads** (`classifier.dense`, `classifier.out_proj`) train. This applies to both bug-type classification and binary line-candidate localization. The optimizer excludes frozen parameters. Exact policies and parameter counts are recorded in each model's `training_metadata.json` and `metrics/*_layer_policy.json`.

Training retains the full prepared data and unchanged leakage-safe splits, up to three epochs, validation macro-F1 selection for classification, validation MRR selection for localization, effective batch size 16 through gradient accumulation, and CUDA FP16 mixed precision. Localization reports Top-1, Top-3, Top-5 and MRR. CUDA OOM recovery reduces microbatch size or token length before any dataset reduction; the implemented recovery does not reduce dataset rows.

```powershell
.\.venv-codelocator\Scripts\python.exe scripts/preprocess_codelocator.py
.\.venv-codelocator\Scripts\python.exe scripts/prepare_codebert.py
.\.venv-codelocator\Scripts\python.exe scripts/train_codelocator.py
.\.venv-codelocator\Scripts\python.exe scripts/infer_codelocator.py --file example.java
.\.venv-codelocator\Scripts\python.exe -m pytest tests/codelocator -q
```

`--baseline-only` explicitly runs just the classical baseline. `--reuse-baseline` reuses an artifact only when its training-input signature matches the current validated splits. These flags do not make incomplete transformer training count as success.

For an interrupted localizer run that already saved its validation-MRR-selected checkpoint, use:

```powershell
.\.venv-codelocator\Scripts\python.exe scripts/train_codelocator.py --evaluate-localizer-only
```

This mode validates the recorded processed-file checksums, evaluates the existing checkpoint on the untouched validation and test localization splits, writes both evaluation artifacts, updates execution status, and regenerates the report. It performs no training, preprocessing, or split generation. The evidence and recovery decision for the October 9 interruption are recorded in `LOCALIZER_RECOVERY_AUDIT.md`.

`CODELOCATOR_MODEL_REPORT.md` records results and execution status. `metrics/execution_status.json` identifies failed stages with the real exception. The data-quality gate fails closed if inputs, label mappings, splits or recorded checksums are invalid. Pretrained compatibility checks do not establish trained-model quality.

Processed JSONL files live under `datasets/processed/codelocator`. Raw files stay intact. The model stores use separate `baseline`, `bug_type_classifier` and `line_localizer` directories. Large weights and datasets are ignored by Git, while reports and manifests remain reviewable.

For Windows Application Control blocks, use a user Ubuntu WSL2 environment rather than cycling through package downgrades. Install `python3-venv`, then run:

```bash
bash /mnt/c/QASense-AI/scripts/setup_codelocator_wsl.sh
```

Inference returns no overall defect probability because repair-category classification does not learn defective versus nondefective programs. Localization appears only after a trained localizer exists, and its scores rank suspiciousness rather than prove exact faulty lines. Whole-file repair classification is outside the fragment training distribution.
