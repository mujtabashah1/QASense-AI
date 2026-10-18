# QASense AI Accuracy and Runtime Baseline

Date: 2026-10-18
Scope: read-only model audit plus measured runtime-integration review. No model was trained, fine-tuned, promoted, or modified. No dataset or frozen evaluation split was changed.

## Runtime verification

- Branch: `main`, tracking `origin/main`; working tree was clean at baseline commit `727e7ec`.
- API health: reachable on the existing local product service.
- Saved product run `portfolio-demo-final-20261017`: completed, 10 findings, QA score 58, and zero execution errors.
- RiskNet runtime: Python 3.14.8, joblib 1.6.0, scikit-learn 1.8.0, pandas 3.0.6, NumPy 2.5.3.
- CodeLocator runtime: Python 3.14.8, PyTorch 2.14.1+cu126, Transformers 5.19.0, CUDA available.
- VisionQA runtime: Python 3.14.8, PyTorch 2.14.1+cu126, Ultralytics 8.4.176, CUDA available on NVIDIA T1200 Laptop GPU.
- All four selected model assets load from their documented paths. Protected hashes are recorded below.

## RiskNet

| Item | Verified state |
|---|---|
| Model | `models/risknet/risknet_v1.joblib` |
| Input contract | 21 named static-code metrics in the saved training feature order |
| Preprocessing | Saved pipeline; aliases normalized; missing, extra, nonnumeric, nonfinite, and negative values rejected |
| Output | Calibrated defect probability, decision, risk level, and QA priority |
| Product thresholds | Decision 0.20; risk boundaries 0.30, 0.60, 0.80 |
| Mean precision / recall / F1 | 0.3538 / 0.4464 / 0.3785 |
| ROC-AUC / PR-AUC / MCC | 0.7536 / 0.3639 / 0.2850 |
| Calibration | Sigmoid calibration from grouped out-of-fold predictions; Brier score 0.1070 |

Runtime feature alignment is already defensive and preserves the trained order. The main accuracy limitation is domain/calibration transfer, especially overconfidence in high-probability bins documented by the model report. RiskNet output remains prioritization evidence and must not be treated as a sole acceptance gate.

## CodeLocator v2

| Item | Verified state |
|---|---|
| Model | `models/codelocator_v2/line_localizer/model.safetensors` |
| Input | Java source text, normalized line endings, maximum input size 1 MiB |
| Tokenization | Saved tokenizer; maximum token length 256 per candidate context |
| Line mapping | Candidates retain exact original one-based source lines |
| Ranking output | Suspicious line, rank, raw softmax score, code, and context |
| Test Top-1 / Top-3 / Top-5 | 0.3560 / 0.6464 / 0.7799 |
| Test MRR | 0.5372 |
| Bug-line precision / recall / F1 | 0.2595 / 0.3770 / 0.3074 |
| Natural-context MRR | 0.5326 |

The primary product metric is ranking quality, not imbalanced candidate accuracy. Exact line mapping is present. Remaining risks are Java-only assumptions, long-context truncation, duplicate-line evidence, and interpreting a raw classifier score as a calibrated probability.

## VisionQA v1

| Item | Verified state |
|---|---|
| Production model | `models/visionqa/atomblock_yolo11n/best.pt` |
| Classes | 14 canonical QASense classes |
| Runtime settings | image size 640, confidence 0.25, NMS IoU 0.70, maximum detections 1500 |
| Test precision / recall | 0.4738 / 0.4013 |
| Test mAP50 / mAP50-95 | 0.3775 / 0.2783 |
| Mean inference | 7.78 ms/image in the authoritative AtomBlock evaluation |

Per-class AP50-95: button 0.2589, link 0.1279, input 0.1841, image 0.6534, icon 0.0206, checkbox 0.0100, radio 0.0042, select 0.0951, block-nav 0.4546, block-sidebar 0.5147, block-footer 0.7805, block-form 0.1302, block-table 0.1717, and block 0.4897.

The strongest measured limitations are tiny controls and the icon/checkbox/radio/select classes. VisionQA v2 remains experimental after frozen WebUI confirmation at threshold 0.0625; it is not the production default.

## Current verification

- RiskNet: 27 passed.
- CodeLocator: 36 passed.
- Product, integration, visual-defect, adapter, contract, VisionQA inference, and matcher tests: 21 passed, including 7 matcher tests.
- Total current result: 84 passed, zero test failures.
- The dedicated VisionQA environment does not contain pytest; this is an environment limitation, not a model or application test failure. Dependency-free VisionQA tests run in the integration environment.

## Highest-impact accuracy risks

1. VisionQA v1 has very low standalone recall/AP for radio, checkbox, and icon. Existing real-state evidence contained 18 cases where these native controls were localized as `icon` despite authoritative DOM roles.
2. VisionQA transfer remains weak on tiny objects, mobile layouts, `select`, and source-shifted WebUI pages. V2 remains experimental and no production threshold is changed here.
3. CodeLocator ranks useful candidates but bug-line precision/recall remain modest; score semantics and long-file/context behavior need runtime-focused evaluation before retraining.
4. RiskNet preprocessing is consistent, while calibration and cross-project applicability remain the main limits.

## First retained no-training improvement

The first change is intentionally narrow: when an `icon` detection and a native DOM `checkbox` or `radio` have strong geometry under the existing gate (containment at least 0.65 and IoU at least 0.25), the matcher records a DOM-resolved semantic class. It always preserves `vision_class` and `raw_vision_class` as `icon`. It does not modify detector output, thresholds, NMS, checkpoints, or labels.

On the same 12 saved real-state runs:

| Measure | Before | After | Change |
|---|---:|---:|---:|
| Reviewed DOM elements | 132 | 132 | 0 |
| Vision detections | 104 | 104 | 0 |
| Integrated matches | 70 | 88 | +18 |
| Evidence-assisted match coverage | 53.03% | 66.67% | +13.64 percentage points |
| Strong semantic conflicts | 32 | 14 | -18 |
| Checkbox resolutions | 0 | 8 | +8 |
| Radio resolutions | 0 | 10 | +10 |

All other conflict families remain unresolved: table/button 8, link/image 2, textbox/block 2, and button/block 2. The result improves integrated evidence classification; it is not a claim that the standalone VisionQA benchmark improved.

## Prioritized next work

1. Audit CodeLocator runtime ranking, exact line references, duplicate candidates, long files, and score interpretation on fixed local fixtures.
2. Validate RiskNet feature extraction from real repository inputs against the exact 21-feature contract.
3. Evaluate any VisionQA postprocessing proposal on one unchanged labeled validation asset before changing production defaults.
4. Strengthen DOM/visual matching only for additional patterns supported by labeled evidence; retain raw model outputs throughout.
