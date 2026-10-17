# QASense AI migration and recovery manifest

Generated after the product-completion milestone on 2026-10-17.

## Required source and configuration

- `src/`, `scripts/`, `tests/`, `configs/`, and `docs/`
- `README.md`, `LICENSE`, `.gitignore`, and `.gitattributes`
- `requirements-product.txt`, `requirements-integration.txt`, `requirements-risknet.txt`, `requirements-codelocator.txt`, `requirements-visionqa-torch.txt`, and `requirements-visionqa.txt`
- `reports/QASENSE_PROJECT_STATE.md` and the selected module/final reports
- `configs/demo.json` and `tests/fixtures/runtime_fixture.html`

## Required demo models

| Model | Bytes | SHA-256 | Distribution |
|---|---:|---|---|
| `models/risknet/risknet_v1.joblib` | 426,844 | `b32efa3e83707f01bdf29bd7dbcaebe4d484aa9f81f44bf890b9a739deac913c` | Normal Git is acceptable |
| `models/codelocator_v2/line_localizer/model.safetensors` | 498,612,824 | `9459bfa6b343fc749e5b7653df0f0def005d8781c89388d49466f83e6c0ec288` | Git LFS required |
| `models/visionqa/atomblock_yolo11n/best.pt` | 5,464,794 | `7f527e18271067e2c3ed1c85b557b8b794646b996fed2128bd5a06f8c7234f57` | Normal Git is acceptable |

The full `models/codelocator_v2/line_localizer` directory is required, including `config.json`, `tokenizer.json`, `tokenizer_config.json`, `training_metadata.json`, and `evaluation.json`. `.gitattributes` configures the large safetensors file for LFS.

## Optional protected research artifacts

- VisionQA v2 experimental candidate: `models/visionqa/visionqa_v2_mixed14/candidate.pt`, SHA-256 `dd86b81d5faf5f018940564ddb344a8ea7caf328b7b567161937e40ab7f7866d`.
- CodeLocator resumable optimizer checkpoints under `models/codelocator_v2/checkpoints/`.
- Historical CodeLocator v1 artifacts and VisionQA pretrained weights.
- Full research reports under `reports/`.

These are not required to run the production demo. Preserve them in the project archive.

## Large data and rebuildable outputs

Archive separately and do not publish through normal Git:

- `datasets/`: raw, processed, frozen evaluation, and generated controlled-defect data.
- `runs/`: training runs, Playwright traces, screenshots, and product demo outputs.
- `.venv-*` and `.venv/`: local environments.
- CodeLocator training states (approximately 730 MB each).

The repository `.gitignore` excludes these paths. No data was deleted during GitHub preparation.

## Recovery order

1. Clone the repository with Git LFS installed.
2. Restore optional research artifacts/datasets from the separate archive when research reproduction is needed.
3. Create `.venv-integration`, `.venv-risknet`, `.venv-codelocator`, and `.venv-visionqa` from the pinned requirement files.
4. Install the VisionQA Torch packages from the official CUDA 12.6 wheel index, then install `requirements-visionqa.txt`.
5. Verify the three required model hashes above.
6. Run the focused tests documented in `docs/DEMO.md`.
7. Run `scripts/run_qasense_demo.py`, then `scripts/serve_qasense.py`.

The datasets and training checkpoints are not required for product inference. They are required only for reproducing research, audits, or training history.
