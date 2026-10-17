# QASense model runtime dependency manifest

Verified from the repository's inference loaders on 2026-10-17. Paths are relative to the repository root; no training data is required for product inference.

| Model | Status | Runtime artifact | Required support files | Loader and environment |
|---|---|---|---|---|
| RiskNet | Production | `models/risknet/risknet_v1.joblib` | The joblib bundle contains its feature list, calibrated model, configuration, metadata, and importance data. | `src/risknet/inference.py`; `.venv-risknet`; dependencies in `requirements-risknet.txt` |
| CodeLocator v2 | Production | `models/codelocator_v2/line_localizer/model.safetensors` | `config.json`, `tokenizer.json`, `tokenizer_config.json`, `training_metadata.json`, `evaluation.json` in the same directory | `src/codelocator_v2/inference.py`; `.venv-codelocator`; dependencies in `requirements-codelocator*.txt` |
| VisionQA v1 | Production / rollback | `models/visionqa/atomblock_yolo11n/best.pt` | Self-contained Ultralytics checkpoint with the canonical 14-class schema | `scripts/visionqa_inference.py`; `.venv-visionqa`; VisionQA requirement files |
| VisionQA v2 | Experimental | `models/visionqa/visionqa_v2_mixed14/candidate.pt` | Self-contained 14-class Ultralytics checkpoint; recommended experimental threshold `0.0625` | Ultralytics loader; `.venv-visionqa`; never selected by the production default |

The product orchestrator resolves default paths from the repository root and invokes each model in its validated environment. Datasets, optimizer checkpoints, training runs, and pretrained base-model caches are not needed to restore final inference.

After cloning and pulling LFS objects, run:

```powershell
python scripts\verify_model_assets.py
```
