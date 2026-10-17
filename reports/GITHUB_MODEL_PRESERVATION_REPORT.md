# GitHub model preservation report

Prepared for `mujtabashah1/QASense-AI` on 2026-10-17. Final remote and fresh-clone results are recorded after publication.

| Model | Repository path | Bytes | SHA-256 | Storage | Status |
|---|---|---:|---|---|---|
| RiskNet | `models/risknet/risknet_v1.joblib` | 426,844 | `b32efa3e83707f01bdf29bd7dbcaebe4d484aa9f81f44bf890b9a739deac913c` | Git LFS | Production |
| CodeLocator v2 | `models/codelocator_v2/line_localizer/model.safetensors` | 498,612,824 | `9459bfa6b343fc749e5b7653df0f0def005d8781c89388d49466f83e6c0ec288` | Git LFS | Production |
| VisionQA v1 | `models/visionqa/atomblock_yolo11n/best.pt` | 5,464,794 | `7f527e18271067e2c3ed1c85b557b8b794646b996fed2128bd5a06f8c7234f57` | Git LFS | Production / rollback |
| VisionQA v2 | `models/visionqa/visionqa_v2_mixed14/candidate.pt` | 20,009,359 | `dd86b81d5faf5f018940564ddb344a8ea7caf328b7b567161937e40ab7f7866d` | Git LFS | Experimental; threshold 0.0625 |

CodeLocator v2 additionally requires `config.json`, `evaluation.json`, `tokenizer.json`, `tokenizer_config.json`, and `training_metadata.json` from its `line_localizer` directory. These are normal Git files.

No datasets, optimizer/scheduler checkpoints, training caches, generated runs, or virtual environments are part of the portable repository. A successful `git lfs pull` restores the selected final model binaries; retraining is not required.
