# QASense CodeLocator v2 model report

## Status

CodeLocator v2 completed three epochs on the position-debiased dataset. Epoch 1 was selected by validation MRR. The learned model beats both the position-only and random baselines on the untouched test split. v1 artifacts remain unchanged.

## v1 positional-bias root cause

v1 first selected a 31-line source window centered on the known bug, with 15 lines before and after it. It then excluded candidates near artificial crop edges. As a result, 94.10% of validation bugs and 94.70% of test bugs occupied the middle relative-position bucket. A midpoint-ranking shortcut achieved test MRR 0.8763 while the learned v1 localizer achieved 0.5309.

## v2 data construction

v2 rejoined the full pre-fix source using the existing exact provenance keys and retained v1 repository-exclusive split membership. It selects the containing method when detected, otherwise the containing block. Regions over 61 lines use deterministic hash-assigned crop positions across five target buckets. Regions with fewer than five eligible candidates are excluded.

Training has one source view with the positive plus up to six executable hard negatives chosen through identifier and operator overlap. Validation and test include every eligible line. Candidate order is deterministic, shuffled, and independent of the label. Model features are restricted to candidate code and shared natural-region code.

## Counts

| Split | Source bugs | Candidate lines | Positives | Negatives | Mean context lines |
|---|---:|---:|---:|---:|---:|
| Train | 5,707 | 38,810 | 5,707 | 33,103 | 30.59 |
| Validation | 980 | 15,975 | 980 | 14,995 | 35.40 |
| Test | 854 | 14,381 | 854 | 13,527 | 37.46 |

## Relative bug-position distribution

Buckets are 0–20%, 20–40%, 40–60%, 60–80%, and 80–100%.

| Split | Bucket percentages | Middle bucket | Exact middle |
|---|---|---:|---:|
| Train | 25.67%, 24.27%, 19.20%, 16.91%, 13.95% | 19.20% | 4.43% |
| Validation | 22.65%, 22.76%, 21.84%, 17.24%, 15.51% | 21.84% | 4.18% |
| Test | 25.18%, 23.07%, 21.90%, 16.98%, 12.88% | 21.90% | 3.75% |

No middle bucket exceeds 40%, and no individual bucket exceeds 45%.

## Pre-training baselines

| Test metric | v1 position-only | v2 position-only | v2 random, five-seed mean | Random standard deviation |
|---|---:|---:|---:|---:|
| Top-1 | 79.57% | 11.36% | 8.15% | 1.31 pp |
| Top-3 | 94.82% | 30.44% | 26.35% | 1.32 pp |
| Top-5 | 96.55% | 48.13% | 44.43% | 1.33 pp |
| MRR | 0.8763 | 0.2826 | 0.2500 | 0.0110 |
| Mean rank | 1.47 | 8.80 | 8.99 | 0.19 |
| Median rank | 1 | 6 | 6.3 | 0.4 |

The position-only test MRR is far below the 0.75 hard-stop threshold and close to random. The position shortcut has been substantially reduced. Position-only retains a small Top-K/MRR advantage over random, so it remains a required comparator during model evaluation.

## Integrity checks

Repository overlap, sample-ID overlap, cross-split model-input overlap, one-positive-per-group, and forbidden-feature checks all passed. Preprocessing excluded 1,689 groups containing indistinguishable duplicate candidate inputs and 510 groups whose model input duplicated an earlier split. Corrected code, patches, labels, target line numbers, candidate indices, and relative-position features are absent from model inputs.

## Training decision

The structural gate passed and the user approved training. CodeLocator v2 was initialized from the v1 epoch-2 model weights. This was a new fine-tuning run and not an optimizer-level continuation of interrupted v1 epoch 3.

## GPU benchmark and configuration

| Batch | Accumulation | Checkpointing | Samples/s | Peak reserved GiB |
|---:|---:|---|---:|---:|
| 2 | 8 | On | 2.679 | 1.031 |
| 4 | 4 | On | 2.885 | 1.039 |
| 8 | 2 | On | 2.932 | 1.092 |
| **8** | **2** | **Off** | **6.006** | **1.242** |

The selected run used physical batch 8, accumulation 2, effective batch 16, FP16, max length 256, learning rate 2e-5, weight decay 0.01, and gradient checkpointing off. Mean benchmark GPU utilization was 98.4%. Estimated maximum runtime was 7.74 hours; actual training plus evaluation took 30,068 seconds (8 hours 21 minutes).

Embeddings and encoder layers 0–7 were frozen. Layers 8–11 and the localization head were trained: 28,943,618 of 124,647,170 parameters (23.22%).

## Resumable training

Atomic checkpoints contain model, optimizer, scheduler, FP16 scaler, epoch and step, RNG state, best validation MRR, history, configuration, and exact split hashes. `latest.pt` was updated every 500 optimizer steps. Permanent epoch snapshots were saved after each epoch. A validation interruption can resume without repeating completed training.

## Epoch history

| Epoch | Loss | Validation Top-1 | Top-3 | Top-5 | MRR | Mean rank | Median rank |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | 0.5274 | **32.65%** | 61.53% | 74.80% | **0.5082** | 4.40 | 3 |
| 2 | 0.4939 | 30.92% | 61.33% | 75.10% | 0.4981 | 4.41 | 3 |
| 3 | 0.4619 | 30.82% | 62.04% | 76.63% | 0.5030 | 4.27 | 3 |

Epoch 1 was selected because validation MRR is primary. Early-stopping patience was reached after epoch 3, also the configured maximum, so training did not stop before the maximum.

## Selected-checkpoint evaluation

| Split | Top-1 | Top-3 | Top-5 | MRR | Mean rank | Median rank |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 32.65% | 61.53% | 74.80% | 0.5082 | 4.40 | 3 |
| Test | **35.60%** | **64.64%** | **77.99%** | **0.5372** | **4.09** | **2** |
| Natural-context test | 34.84% | 64.59% | 78.90% | 0.5326 | 3.88 | 2 |

Test candidate accuracy is 89.91%, macro precision 0.6095, macro recall 0.6546, macro-F1 0.6265, and weighted-F1 0.9077. Bug-line precision is 0.2595, recall 0.3770, and F1 0.3074. Full confusion matrices are in `metrics/evaluation.json`.

## Learned model versus shortcuts

| Metric | v1 learned | v1 position | v2 learned | v2 position | v2 random |
|---|---:|---:|---:|---:|---:|
| Top-1 | 33.47% | 79.57% | **35.60%** | 11.36% | 8.15% |
| Top-3 | 65.04% | 94.82% | **64.64%** | 30.44% | 26.35% |
| Top-5 | 80.52% | 96.55% | **77.99%** | 48.13% | 44.43% |
| MRR | 0.5309 | 0.8763 | **0.5372** | 0.2826 | 0.2500 |
| Mean rank | 3.45 | 1.47 | 4.09 | 8.80 | 8.99 |
| Median rank | 2 | 1 | 2 | 6 | 6.3 |

v2 exceeds position-only by 0.2545 MRR and by 24.24, 34.19, and 29.86 percentage points at Top-1, Top-3, and Top-5. It exceeds random on all ranking metrics. This satisfies the scientific success condition. The v2 test is intentionally harder than v1, so raw results are contextual rather than a like-for-like dataset comparison.

## Inference, recommendation, and limitations

Separate inference is available through `scripts/infer_codelocator_v2.py`; v1 inference remains unchanged. The smoke test loaded the selected checkpoint and returned ranked line number, score, and code. The complete suite passes: 36 tests.

v2 should replace v1 as the preferred localizer when the application adopts natural-region candidate construction. Keep v1 as a rollback until production integration covers large files and real repositories.

ManySStuBs4J contains Java single-statement repairs. Surrounding lines are weak negatives, method/block detection is heuristic, long inputs truncate at 256 tokens, and inference constructs regions without knowing the true bug. Scores rank suspiciousness and do not prove an exact defect. Generalization to multi-line defects, other languages, or whole-repository diagnosis is not established.

## Artifact paths

- Selected model: `models/codelocator_v2/line_localizer/`
- Latest state: `models/codelocator_v2/checkpoints/latest.pt`
- Epoch states: `models/codelocator_v2/checkpoints/epoch_1/`, `epoch_2/`, `epoch_3/`
- Metrics: `reports/codelocator_v2/metrics/evaluation.json`
- Benchmark: `reports/codelocator_v2/metrics/gpu_benchmark.json`
- Status/history: `reports/codelocator_v2/metrics/training_status.json`, `training_history.json`
