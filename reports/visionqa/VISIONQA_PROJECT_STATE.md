# VisionQA Persistent Project State

Last updated: 2026-10-13 PDT after runtime integration and external-dataset audits completed.

This file is the durable handoff for future sessions. For live epoch progress,
run `scripts/get_visionqa_training_status.py`. It writes the canonical snapshot
to `reports/visionqa/metrics/training_live_snapshot.json`. The trainer also
updates `training_status.json` after every completed train/validation epoch.

## QASense development flow

1. RiskNet was completed first and ranks source modules by defect risk.
2. CodeLocator v1 exposed a severe positional shortcut. CodeLocator v2 rebuilt
   natural candidate contexts, passed leakage and position-baseline gates, and
   became the preferred source localizer.
3. VisionQA is the active module. Its current scope is AtomBlock UI-element
   detection only; WebUI, Google Screen Annotation, Playwright mutations, and
   multimodal fusion remain future work.
4. After VisionQA training, the validation-selected detector will be evaluated
   once on the held-out test split, promoted to a stable model path, and exposed
   through a screenshot-to-JSON inference interface.

This order preserves module boundaries and prevents test-set selection leakage.

## Protected completed modules

- RiskNet is complete. Selected model: `models/risknet/risknet_v1.joblib`.
- CodeLocator v2 is complete. Selected model: `models/codelocator_v2/line_localizer`.
- Do not retrain, overwrite, or modify RiskNet or CodeLocator artifacts for VisionQA work.

## VisionQA dataset

- Dataset YAML: `datasets/processed/visionqa_atomblock/data.yaml`
- Prepared counts: train 5,422 image/label pairs; val 1,388; test 1,389.
- Classes: 14.
- Label syntax/range audit: PASS.
- Ultralytics can decode 5,420 training images. Two source PNGs are truncated in
  both the original and processed datasets and are excluded by the loader. The
  files were preserved unchanged. Details: `reports/visionqa/metrics/loader_audit.json`.
- Ultralytics removes exact duplicate label rows in memory. Source labels remain unchanged.
- Splits and `data.yaml` remain unchanged.

## Verified environment

- Environment: `.venv-visionqa`
- Python: 3.14.8
- Torch: 2.14.1+cu126
- TorchVision: 0.29.1+cu126
- Ultralytics: 8.4.176
- CUDA runtime used by Torch: 12.6
- GPU: NVIDIA T1200 Laptop GPU, 4,096 MiB
- CUDA tensor operation and YOLO11n GPU inference: PASS
- Canonical environment record: `reports/visionqa/metrics/environment.json`

## Completed runs

- Smoke: `runs/visionqa/atomblock_yolo11n_smoke_b2`
  - 1 epoch, batch 2, image size 640, seed 42, max_det 1500.
  - mAP50 0.21222; mAP50-95 0.13816.
  - Full record: `reports/visionqa/metrics/smoke_test.json`.
- Benchmark: `runs/visionqa/atomblock_yolo11n_benchmark_b4`
  - 3 epochs, batch 4, image size 640, seed 42, max_det 1500.
  - Final validation mAP50 0.27688; mAP50-95 0.19073.
  - Full record: `reports/visionqa/metrics/benchmark.json`.
- Batch 2 was selected for the main run because its measured epoch was faster
  than batch 4 despite using less VRAM.

## Completed main run

- Run: `runs/visionqa/atomblock_yolo11n_main_b2`
- Configuration: pretrained YOLO11n, 30 epochs maximum, batch 2, image size 640,
  device 0, workers 2, seed 42, patience 10, max_det 1500, save_period 1.
- AMP was requested but Ultralytics disabled it after its GPU/runtime safety check.
- Canonical live state: `reports/visionqa/metrics/training_status.json`.
- Checkpoints: `runs/visionqa/atomblock_yolo11n_main_b2/weights`.
- The run was safely checkpointed after epoch 2, resumed as a hidden detached
  Windows process, and completed all 30 epochs without losing saved work.
- Detached process metadata and disk log paths:
  `reports/visionqa/metrics/training_process.json`.
- The detached finalizer completed checkpoint promotion, one-time held-out test
  evaluation, loader summary, qualitative examples, CUDA inference checks, and
  final report generation. Its final state is
  `reports/visionqa/metrics/postprocess_status.json`.
- Source-of-truth live snapshot:
  `reports/visionqa/metrics/training_live_snapshot.json`.
- Epoch 1 validation: precision 0.43522, recall 0.25283, mAP50 0.22144,
  mAP50-95 0.14656.
- Epoch 2 validation: precision 0.49774, recall 0.29228, mAP50 0.26428,
  mAP50-95 0.17912. Losses also decreased. The run is moving in the expected
  direction and is consistent with the independent three-epoch benchmark.
- Epoch 3 validation: precision 0.45833, recall 0.31222, mAP50 0.27928,
  mAP50-95 0.18956. All training losses fell again. Precision decreased while
  recall and both mAP values improved; validation fitness therefore improved.
- Training completed in approximately 3.78 hours from run-directory creation
  through the final epoch checkpoint. Median epoch time was 432.05 seconds.
- Epoch 28 was selected by validation mAP50-95: precision 0.48437, recall
  0.41114, mAP50 0.39335, and mAP50-95 0.29015.
- Held-out test: precision 0.47383, recall 0.40132, mAP50 0.37755, and
  mAP50-95 0.27825.
- Stable detector: `models/visionqa/atomblock_yolo11n/best.pt`.

Refresh the durable live snapshot with:

```powershell
.\.venv-visionqa\Scripts\python.exe scripts\get_visionqa_training_status.py
```

Start or resume the main run independently of Codex with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\start_visionqa_training.ps1
```

The launcher uses process-scoped execution-policy bypass only. It does not
change the machine-wide PowerShell policy.

Resume an interrupted run with:

```powershell
.\.venv-visionqa\Scripts\python.exe scripts\train_visionqa_yolo.py `
  --resume runs\visionqa\atomblock_yolo11n_main_b2\weights\last.pt `
  --metrics-output reports\visionqa\metrics\training_status.json
```

If `last.pt` was stripped after normal completion, use the latest full-size
`epochN.pt` checkpoint to extend training with optimizer and scheduler state.

## Completed finalization steps

1. Selected epoch 28 by validation fitness.
2. Promoted it to `models/visionqa/atomblock_yolo11n/best.pt` while preserving run artifacts.
3. Evaluated the held-out test split and saved overall and per-class metrics.
4. Saved raw and normalized confusion matrices.
5. Generated matched qualitative examples for correct detections, misses, false
   positives, dense pages, forms, navigation, tables, and rare classes.
6. Verified the stable checkpoint with `scripts/check_visionqa_foundation.py` and
   `scripts/visionqa_inference.py` on CUDA.
7. Completed `reports/visionqa/VISIONQA_ATOMBLOCK_REPORT.md` with measured metrics.

## Reproducible entry points

- `scripts/train_visionqa_yolo.py`
- `scripts/evaluate_visionqa_yolo.py`
- `scripts/generate_visionqa_examples.py`
- `scripts/visionqa_inference.py`
- `scripts/check_visionqa_foundation.py`
- `scripts/inspect_visionqa_state.py`
- `scripts/get_visionqa_training_status.py`
- `scripts/start_visionqa_training.ps1`
- `scripts/finalize_visionqa_after_training.ps1`
- `scripts/start_visionqa_finalizer.ps1`
- `scripts/review_visionqa_training_progress.py`
- `scripts/build_visionqa_report.py`
- `scripts/summarize_visionqa_loader.py`

## Runtime integration and next data decision

The stable detector is integrated with Playwright through `scripts/run_playwright_visionqa.py`. The final verified fixture run is `runs/playwright/integration-final-20261013` and contains screenshots, DOM, console/network/error evidence, a trace, structured VisionQA detections, and semantic DOM/VisionQA matches.

Google Screen Annotation and WebUI were fully audited without training. Google has no local screenshots and uses a different mobile hierarchy. WebUI has valid embedded images and useful DOM/responsive metadata, but its nested DOM-node boxes are not equivalent to AtomBlock classes and exact image duplicates cross its nominal split boundary. The selected action is to keep the detector unchanged and create a human-reviewed, source-grouped WebUI evaluation slice before deciding on fine-tuning.

Controlled `pilot_v2` contains 108 reversible mutations with before/after VisionQA evidence. The deterministic benchmark reaches 1.000 macro precision, recall, and F1 on this known fixture plus unchanged controls. This does not establish real-world visual-defect accuracy and does not justify a new ML model. Full decisions are in `reports/visionqa/VISIONQA_DATASET_STRATEGY.md` and `reports/visual_defects/CONTROLLED_VISUAL_DEFECT_BENCHMARK.md`.

## WebUI transfer readiness update — 2026-10-13

The fixed evaluation slice now contains 399 screens from 133 complete three-viewport groups across 37 sources. Its manifest content SHA-256 is `674ae4c0907b8a1927383d7b0a1a02e285f93cb4ade1ec42becad198b2ec7d45`. All known cross-split exact-image groups were excluded, and invalid, off-viewport, duplicate, and ambiguous-subtype boxes were excluded from scoring without changing WebUI.

At confidence 0.25 and IoU 0.5, full-slice transfer precision is 0.08805, recall 0.03564, and F1 0.05074. Manual review covered 100 screens; 96 were mapping-valid and four were annotation-questionable. The valid reviewed subset measured precision 0.10861, recall 0.02290, and F1 0.03782. Ten classes are scored; icon, checkbox, radio, and generic block remain excluded because their mappings are ambiguous.

The evidence supports future targeted detector fine-tuning on a separate curated WebUI-derived corpus. No training or threshold tuning occurred in this milestone. The frozen evaluation sample groups and known exact-image duplicate groups must be excluded from all future training data. See `WEBUI_TRANSFER_EVALUATION.md` and `NEXT_MODEL_READINESS_DECISION.md`.
