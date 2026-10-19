# QASense AI — Canonical Project State

Last updated: 2026-10-17 PDT after the unified 14-class VisionQA v2 experiment completed.

This is the cross-module handoff record. Module reports and machine-readable
metrics remain authoritative for detailed results. Future work should read this
file before modifying models, datasets, splits, or training code.

## Product flow and current boundary

```text
Repository -> RiskNet -> CodeLocator -> QA Intelligence
                                     + Playwright evidence
                                     + VisionQA screenshot evidence
                                     -> future multimodal reasoning and test selection
```

The repository currently has two completed ML modules and one active module:

| Module | State | Selected artifact | Next allowed action |
|---|---|---|---|
| RiskNet | Complete | `models/risknet/risknet_v1.joblib` | Read/integrate only |
| CodeLocator v2 | Complete and preferred | `models/codelocator_v2/line_localizer` | Read/integrate only |
| VisionQA AtomBlock | Complete and integrated; v2 corpus ready | `models/visionqa/atomblock_yolo11n/best.pt` | Future separately authorized YOLO11n fine-tuning; v1 remains rollback |

CodeLocator v1 remains a preserved rollback/reference. It must not replace v2.

## Completed work in order

### 1. RiskNet

- Trained on 17,442 NASA metric rows with leakage-aware grouped validation.
- Selected a calibrated logistic model using equal-project cross-project average
  precision, recall, and MCC criteria.
- Stable model: `models/risknet/risknet_v1.joblib`.
- Report: `reports/risknet/RISKNET_MODEL_REPORT.md`.
- Reproducible inference and tests already exist. Do not retrain for VisionQA.

### 2. CodeLocator

- v1 exposed a positional shortcut caused by its context construction.
- v2 rebuilt natural candidate regions, preserved repository-exclusive splits,
  and passed structural, leakage, position-only, and random-baseline gates.
- Embeddings and encoder layers 0–7 were frozen. Encoder layers 8–11 and the
  localization head were trained: 28,943,618 of 124,647,170 parameters.
- Selected v2 test results: Top-1 35.60%, Top-3 64.64%, Top-5 77.99%, MRR
  0.5372; natural-context MRR 0.5326.
- Stable model: `models/codelocator_v2/line_localizer`.
- Full resumable state: `models/codelocator_v2/checkpoints/latest.pt` and epoch
  directories.
- Report: `reports/codelocator_v2/CODELOCATOR_V2_MODEL_REPORT.md`.
- Do not retrain or overwrite either CodeLocator version during VisionQA work.

### 3. VisionQA AtomBlock foundation

- Original dataset preserved at `datasets/atomblock-webui/yolo-dataset`.
- Prepared dataset: `datasets/processed/visionqa_atomblock`.
- Fixed split membership: 5,422 train, 1,388 validation, 1,389 held-out test
  matched image/label pairs; 14 classes.
- Label integrity audit passes. Two training PNGs are truncated in both source
  and prepared copies, so Ultralytics safely excludes them and processes 5,420
  training images per epoch. Source files and splits remain unchanged.
- Verified environment: Python 3.14.8, torch 2.14.1+cu126, torchvision
  0.29.1+cu126, Ultralytics 8.4.176, CUDA available on NVIDIA T1200 4 GB.
- One-epoch batch-2 smoke test passed.
- Three-epoch batch-4 benchmark passed. Local training was selected based on
  measured epoch time. Batch 2 was selected for the main run because it was
  faster in the measured comparison.
- Main run: pretrained YOLO11n, batch 2, image size 640, seed 42, 30
  epochs, early-stopping patience 10, max detections 1,500, validation each
  epoch, and checkpoint period one epoch.
- AMP was requested. Ultralytics disabled it after its runtime safety test for
  this GPU, so the valid main run uses FP32.
- Training completed all 30 epochs. Epoch 28 was selected by validation
  mAP50-95 and promoted to `models/visionqa/atomblock_yolo11n/best.pt`.
- Held-out test results: precision 0.47383, recall 0.40132, mAP50 0.37755,
  and mAP50-95 0.27825.
- The detached finalizer completed test evaluation, confusion matrices,
  qualitative examples, CUDA inference verification, and the final report.
- Detailed handoff: `reports/visionqa/VISIONQA_PROJECT_STATE.md`.

## Non-negotiable data and evaluation rules

- Do not alter existing train/validation/test membership.
- Use validation metrics for epoch/model selection. Do not use the test split
  to choose a checkpoint or tune hyperparameters.
- Evaluate the held-out test split once after selecting the best validation
  checkpoint.
- Report precision, recall, mAP50, mAP50-95, per-class metrics, and both
  confusion matrices. Discuss rare-class behavior explicitly.
- Preserve valid checkpoints and all source datasets.
- Do not start WebUI or Google Screen Annotation training until the AtomBlock
  detector foundation is complete.

## Recovery and status history

Refresh the live VisionQA snapshot:

```powershell
.\.venv-visionqa\Scripts\python.exe scripts\get_visionqa_training_status.py
```

If future extension training is approved, resume independently of Codex with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\start_visionqa_training.ps1
```

The normal-completion `last.pt` and `best.pt` are inference checkpoints. Full
optimizer/scheduler/scaler state remains in the 30 `epochN.pt` snapshots for an
approved extension. No additional training is currently required.

## Completed integration milestone — 2026-10-13

- Playwright runtime capture now preserves metadata, viewport/full-page screenshots, DOM, console, network, page errors, timing, and `trace.zip` under unique run IDs.
- The stable VisionQA checkpoint has a reusable structured inference interface. DOM/VisionQA matching combines geometry and semantic compatibility while retaining conflicts as ambiguity evidence.
- Verified run: `runs/playwright/integration-final-20261013`; 11 DOM candidates, 10 detections, seven semantic matches, and three semantic conflicts.
- Google Screen Annotation audit: 22,076 physical rows, 520,020 parsed elements, 15 categories, no cross-split ID overlap, and no local screenshots. The files contain 341 fewer rows than the README declares.
- WebUI audit: all 36,807 embedded images decoded; 12,269 complete three-viewport samples and 4,689,704 boxes. Source/sample split isolation passes, while 15 exact-image groups cross splits, 2,986 rows have no boxes, and 16,076 boxes are nonpositive.
- Dataset decision: retain the AtomBlock detector. Use WebUI first for mapped evaluation/responsive geometry and Google annotations for representation/evaluation after Rico images are resolved. No new screen model is approved.
- Controlled `pilot_v2`: 108 reversible positive mutations across 12 types, three viewports, and three target elements, plus 108 unchanged negative-control evaluations. Before/after screenshots, DOM, VisionQA output, and mutation ground truth are preserved. All reversals restored geometry.
- The controlled deterministic baseline reached 1.000 precision, recall, and F1 for every injected class and unchanged controls. This validates this known fixture and does not estimate real-world defect accuracy. No defect ML model is approved.
- Read-only RiskNet and CodeLocator v2 adapters and a deterministic multimodal evidence contract are implemented under `src/adapters` and `src/contracts`.
- Protected model/report fingerprints are in `reports/state/protected_artifact_manifest.json`. Completed models and historical reports were not modified.

Authoritative records are `reports/integration/PLAYWRIGHT_VISIONQA_INTEGRATION.md`, `reports/datasets/GOOGLE_SCREEN_ANNOTATION_AUDIT.md`, `reports/datasets/WEBUI_DATASET_AUDIT.md`, `reports/visionqa/VISIONQA_DATASET_STRATEGY.md`, `reports/visual_defects/CONTROLLED_VISUAL_DEFECT_BENCHMARK.md`, and `reports/integration/MULTIMODAL_CONTRACTS.md`.

## Evaluation/readiness milestone — 2026-10-13

- No standalone application was present, so the repository's most realistic deterministic runtime fixture was used under the required fallback. Six meaningful states were captured twice: desktop default, empty-email validation, unchecked terms, navigation focus, tablet, and mobile. All 12 captures were repeatable for screenshot dimensions, DOM count, detection classes, matches/conflicts, and normalized runtime errors.
- Across the 12 repeated captures, 132 expected elements were reviewed: 70 correct detections, 30 misses, 32 wrong classes/semantic conflicts, and no duplicate detections. The six unique states account for 66 reviewed elements, 35 correct, 15 missed, and 16 wrong-class results. This is fixture evidence, not production-site accuracy.
- The frozen WebUI slice contains 399 screens from 133 complete three-viewport source groups across 37 sources. Manifest content SHA-256: `674ae4c0907b8a1927383d7b0a1a02e285f93cb4ade1ec42becad198b2ec7d45`.
- All known cross-split exact-image groups were excluded. Scoring excludes 69 nonpositive boxes, 62,530 wholly off-viewport boxes, 198 duplicate mapped boxes, and 120 ambiguous input-subtype boxes without modifying the source dataset.
- Ten classes are quantitatively mapped: button, link, input, image, select, block-nav, block-sidebar, block-footer, block-form, and block-table. Icon, checkbox, radio, and generic block are excluded because WebUI does not provide reliable equivalent labels.
- Current VisionQA transfer on all 399 screens at confidence 0.25 and IoU 0.5: precision 0.08805, recall 0.03564, F1 0.05074, and macro-F1 0.04835. The threshold was not tuned on this slice.
- One hundred screens were manually reviewed through ten contact sheets: 96 mapping-valid and four annotation-questionable. On the 96 valid screens, precision was 0.10861, recall 0.02290, and F1 0.03782.
- Responsive analysis covered all 133 complete triplets. Mean detected-class-set Jaccard was 0.39417; 28 responsive-only miss patterns were observed, with newly missing detections increasing from desktop (4) to tablet (14) and mobile (26).
- Weak-class evidence: select and block-table had zero recall; block-form recall was 0.01220; block-sidebar had zero recall. Icon, checkbox, and radio remain unscored because their mappings are ambiguous.
- Largest error buckets were dense-layout misses (2,946), missed small elements (2,842), visual-domain-shift misses (2,759), wrong semantic class (120), low-confidence correct detections (70), and responsive-only misses (28).
- `pilot_v2` remains unchanged and no `pilot_v3` was created. The observed problems are detector domain/taxonomy transfer failures rather than evidence for additional visual-defect mutations.
- Readiness decision: **B — targeted VisionQA fine-tuning is justified**, but no model was trained. Google Screen Annotation remains blocked on missing local Rico screenshots/provenance.
- Authoritative reports: `reports/integration/REAL_APPLICATION_STATE_VALIDATION.md`, `reports/visionqa/WEBUI_TRANSFER_EVALUATION.md`, and `reports/visionqa/NEXT_MODEL_READINESS_DECISION.md`.

## VisionQA v2 WebUI corpus milestone — 2026-10-14

- Created `datasets/processed/visionqa_webui_v2` without changing the source WebUI dataset or training any model.
- The ten-class contiguous schema is button, link, input, image, select, block-nav, block-sidebar, block-footer, block-form, and block-table. It records the original AtomBlock IDs. Icon, checkbox, radio, and generic block remain excluded.
- The frozen 399-screen evaluation slice is preserved. Its 133 source groups and all 15 authoritative cross-split exact-image groups are excluded from corpus construction.
- Eligible complete source groups after frozen/duplicate exclusions: 12,042. Retained: 9,058 positive groups plus 453 deterministically sampled negative-only groups.
- Train: 7,773 groups, 23,319 screens/labels, 221,992 valid boxes, 1,353 negative screens, 35 sources.
- Validation: 1,738 groups, 5,214 screens/labels, 52,104 valid boxes, 357 negative screens, 34 sources.
- Every retained group has complete desktop/tablet/mobile membership within one split. Exact-image-connected groups were consolidated into one split.
- Box exclusions: 2,997,144 fully offscreen; 1,264,695 ambiguous/unmapped; 15,980 nonpositive; 5,393 proven duplicate mapped boxes; 2,328 ambiguous input subtypes. No source annotation was modified.
- All 28,533 candidate images decoded successfully. The final audit found zero corrupt images, invalid label lines, missing/orphan labels, frozen-evaluation leakage, source-group leakage, known-duplicate leakage, or exact-image train/validation leakage.
- Human review covered 300 screens from 300 groups, 35 sources, all viewports, and all ten classes. All supported native-tag mappings were accepted at screen level; no labels were silently corrected.
- Dataset manifest SHA-256: `442c0f3d250f513463ca6188127a31240fab173e781fe630ddc6ffa4a52ced56`.
- Split manifest SHA-256: `4bc01ed12f211408760ca1a67f2db901dc59d9b8bddf1fa5bc55702b37d9add3`.
- Audit result: **PASS**. Training readiness: **YES**. Training performed: **NO**.
- Authoritative records: `reports/visionqa/WEBUI_TRAINING_CORPUS_REPORT.md`, `reports/visionqa/WEBUI_TRAINING_CORPUS_AUDIT.md`, `reports/visionqa/VISIONQA_V2_TRAINING_SPEC.md`, and `reports/visionqa/visionqa_v2_readiness.json`.

## Unified VisionQA v2 experiment — 2026-10-17

- Trained one YOLO11n detector with the canonical 14-class QASense head. The experiment used 23,319 WebUI screens plus all 5,420 decodable AtomBlock rehearsal images; WebUI validation remained separate at 5,214 screens.
- Source-aware partial-label loss supervised ten mapped classes on WebUI and masked icon, checkbox, radio, and generic block there. AtomBlock retained normal supervision for all 14 classes. Cross-image composition was disabled.
- The one-epoch benchmark completed in 31.3 minutes. Stage A completed three epochs with modules 0–10 frozen. Stage B completed nine epochs with modules 0–5 frozen. Main staged training took 3.10 hours in FP32.
- The validation-selected checkpoint was Stage B epoch 9 with supported-class macro-F1 0.11465. Candidate: `models/visionqa/visionqa_v2_mixed14/candidate.pt`; SHA-256 `dd86b81d5faf5f018940564ddb344a8ea7caf328b7b567161937e40ab7f7866d`.
- Frozen WebUI result: precision 0.28521, recall 0.01395, F1 0.02659, and macro-F1 0.05233. Against v1, precision and macro-F1 increased, but recall and overall F1 materially decreased.
- AtomBlock result: precision 0.43786, recall 0.38987, mAP50 0.35590, and mAP50-95 0.25541. This retained 91.79% of v1 mAP50-95 and passed the predeclared catastrophic-forgetting gate.
- Old-class AtomBlock AP50-95: icon 0.02123, checkbox 0.00978, radio 0.00175, and block 0.46320. No old class met the predeclared major-regression rule.
- Promotion decision: **KEEP V2 AS EXPERIMENTAL**. The frozen WebUI F1 gate failed, so Playwright promotion regression was not run and VisionQA v1 remains production/rollback.
- Candidate inference loads on CUDA and exposes all 14 canonical classes. Frozen-set mean latency was 24.33 ms, median 19.36 ms, and p95 28.88 ms over 399 screens.
- All 33 VisionQA tests pass. RiskNet, CodeLocator v2, VisionQA v1, the frozen WebUI manifest, and AtomBlock configuration hashes remain unchanged.
- Authoritative reports: `reports/visionqa/v2/VISIONQA_V2_MODEL_REPORT.md`, `WEBUI_TRANSFER_COMPARISON.md`, `CATASTROPHIC_FORGETTING_ANALYSIS.md`, and `RUNTIME_BENCHMARK.md`.

## Exact next milestone

Perform a validation-only failure and confidence-calibration analysis for the experimental v2 checkpoint, without retraining or reusing the frozen 399-screen labels for tuning. Determine why WebUI validation macro-F1 improved while frozen-slice recall collapsed at confidence 0.25. Prepare one separately authorized next-strategy decision; keep VisionQA v1 in production and do not start another model automatically.

## Superseded corpus-preparation instruction preserved for history

Create a separate source-grouped, human-reviewed WebUI-derived training/validation corpus for the ten supported classes. Exclude every source sample in the frozen 399-screen evaluation manifest and every known exact-image duplicate group. Preserve the fixed evaluation slice for later untouched evaluation. Do not begin that milestone from this evaluation run.

## Superseded sequence preserved for history

1. Run the completed Playwright/VisionQA collector against several real, deterministic application states.
2. Create a 300–500 screen, source-grouped WebUI evaluation slice with human-reviewed AtomBlock mappings, including weak radio, checkbox, icon, select, form, and table classes.
3. Expand the controlled defect dataset across those real page states and retain unchanged negative controls.
4. Re-run the deterministic benchmark and review failures before approving any visual-defect model.
5. Populate the deterministic multimodal contract from linked repository, runtime, and screenshot evidence. Do not train learned fusion until linked outcome labels exist.

## Repository safety status

- RiskNet, CodeLocator, and VisionQA artifacts were read for integration and were not modified. SHA-256 fingerprints are recorded in `reports/state/protected_artifact_manifest.json`.
- Large datasets, run directories, caches, virtual environments, and VisionQA
  checkpoints are excluded by `.gitignore`.
- The Git executable is unavailable on the current PATH, so `git status` could
  not be executed. File-level inspection was used; no reset, deletion, or
  unrelated overwrite was performed.


## VisionQA v2 validation-only failure analysis (2026-10-17)

- VisionQA v2 remains experimental; VisionQA v1 remains production/rollback.
- Analysis used only the 5,214-screen WebUI training-corpus validation split. No training occurred and the frozen 399-screen evaluation was not rerun or inspected.
- Primary mechanism: v2 confidence shifted downward. True-positive median confidence changed from 0.0433 (v1) to 0.0140 (v2).
- Validation best overall F1: 0.1442 at confidence 0.0573; v2 F1 at confidence 0.25: 0.0469.
- At confidence 0.25, v2 produced 2,997 detections and zero detections on 3,895/5,214 screens.
- Secondary findings: tiny/small-object weakness, reduced recall as density rises, mobile weakness, source-specific variation, and complete `select` failure on validation. Localization mismatch was not the main cause.
- Architecture assessment: unified 14-class architecture is not the primary failure; old-domain retention remains 91.79% and validation adaptation is measurable.
- Exactly one recommended next experiment: source-grouped cross-validated global threshold recalibration using persisted validation predictions only. No GPU training; do not reuse the frozen 399-screen set.
- Reports: `reports/visionqa/v2/V2_FAILURE_ANALYSIS.md`, `V2_CONFIDENCE_ANALYSIS.md`, and `V2_NEXT_EXPERIMENT_RECOMMENDATION.md`.


## VisionQA v2 source-grouped threshold calibration (2026-10-17)

- Calibration used persisted predictions for the 5,214-screen WebUI validation split only; no inference or training occurred.
- Five deterministic source-stratified group folds (seed 42) preserved desktop/tablet/mobile source-sample groups without leakage.
- Fold thresholds: 0.0675, 0.0600, 0.0625, 0.0700, 0.0625.
- Threshold mean 0.0645, median 0.0625, standard deviation 0.0037, range 0.0100; classified stable.
- Recommended global threshold: 0.0625.
- Held-out fold mean P/R/F1/macro-F1: 0.1907/0.1150/0.1433/0.1045.
- Out-of-fold aggregate P/R/F1/macro-F1: 0.1904/0.1150/0.1434/0.1031.
- Decision: B. GLOBAL THRESHOLD IMPROVES V2 BUT CLASS-SPECIFIC ISSUES REMAIN. VisionQA v2 remains experimental.
- Next recommended milestone: Authorize one final frozen 399-screen confirmation using the calibrated global threshold. Not executed.
- Frozen 399-screen evaluation remained untouched; protected checkpoints and datasets were not modified.


## VisionQA v2 frozen threshold confirmation (2026-10-17)

- One authorized frozen confirmation evaluated 399 screens / 133 groups using VisionQA v2 candidate SHA-256 `dd86b81d5faf5f018940564ddb344a8ea7caf328b7b567161937e40ab7f7866d` at the preselected confidence threshold 0.0625.
- Frozen manifest canonical SHA-256 remained `674ae4c0907b8a1927383d7b0a1a02e285f93cb4ade1ec42becad198b2ec7d45`. No threshold sweep or tuning occurred.
- Frozen v2@0.0625 P/R/F1/macro-F1: 0.11309/0.05906/0.07759/0.10102.
- V2@0.25 → v2@0.0625: F1 0.02659 → 0.07759; recall 0.01395 → 0.05906; precision 0.28521 → 0.11309.
- Scorable predictions: 284 → 3,033; zero-prediction screens: 180 → 10.
- Remaining limitations: select TP=0, tiny recall 0.0088, mobile recall 0.0354, increased false-positive volume, and source-concentrated gains.
- AtomBlock mAP50-95 retention remains 91.79%; the 14-class architecture and catastrophic-forgetting gate remain intact.
- Final gate: C. WEBUI IMPROVED BUT REMAINS EXPERIMENTAL. VisionQA v1 remains production/rollback; v2 remains experimental and was not promoted.
- Next milestone: Run one select-focused label-quality and visual-separability audit on existing WebUI train/validation data; do not use the frozen evaluation set or retrain during that audit.
- No training, dataset modification, or Playwright promotion regression occurred.

## Product integration milestone (2026-10-17)

- The ML research/training phase is frozen for product completion. No model or dataset was modified. RiskNet, CodeLocator v2, and VisionQA v1 remain production/protected; VisionQA v2 remains experimental and was not promoted.
- Added canonical product evidence schema `1.0`, deterministic multimodal fusion v1, evidence-backed severity/confidence rules, deduplication, QA recommendations, structured test suggestions, and deterministic P0–P3 regression priority.
- Added partial-result handling: unavailable RiskNet, CodeLocator, Playwright, VisionQA, screenshot, source, or network evidence is recorded honestly without discarding available modalities.
- Added the local API and responsive dashboard. The API supports run creation/status/result, findings, evidence, recommendations, tests, regression priority, model status, and safe artifact access.
- Production VisionQA remains `models/visionqa/atomblock_yolo11n/best.pt` at the existing 0.25 inference threshold. The product exposes v2 status but does not load it by default.
- Completed E2E run `runs/product/portfolio-demo-final-20261017/analysis.json`: status completed, all modalities available, QA score 58, 10 findings, 10 recommendations, 10 test suggestions, 10 regression priorities, and no execution errors.
- E2E timing: 138.503 s total; Playwright 2.486 s; VisionQA process path 8.899 s with 3.492 s measured inference; fusion 0.593 ms; action generation 0.127 ms.
- API/dashboard verification passed for the saved run, model registry, dashboard page, and annotated screenshot artifact.
- Final focused validation: 41 tests passed and zero failed. The real demo additionally loaded all three production models successfully.
- GitHub preparation now includes `README.md`, MIT `LICENSE`, architecture/demo docs, four model cards, pinned product requirements, large-artifact ignore rules, Git LFS configuration for CodeLocator v2, and `reports/QASENSE_MIGRATION_MANIFEST.md`.
- Git remains unavailable on PATH, so status/commit verification is the only repository-publication blocker. No push was attempted.
- Future research remains documented but does not block the product: select/tiny/mobile VisionQA work, v2 promotion research, representative real-world visual-defect learning, and learned multimodal fusion.

### Current product entry points

```powershell
.\.venv-integration\Scripts\python.exe scripts\run_qasense_demo.py
.\.venv-integration\Scripts\python.exe scripts\serve_qasense.py
```

Authoritative product records: `reports/QASENSE_FINAL_PRODUCT_REPORT.md`, `reports/QASENSE_MIGRATION_MANIFEST.md`, `docs/ARCHITECTURE.md`, and `docs/DEMO.md`.

## Accuracy baseline and conservative runtime classification improvement (2026-10-18)

- No model was trained, fine-tuned, promoted, or modified. No dataset, split, or frozen evaluation artifact changed.
- Established a reproducible model/runtime baseline in `reports/QASENSE_ACCURACY_BASELINE.md` using authoritative persisted metrics and direct loader/runtime checks.
- The 12 saved real-state runs reproduced 70 DOM/Vision matches and 32 strong semantic conflicts across 132 reviewed elements.
- Eighteen conflicts were a single supported pattern: native DOM checkbox/radio controls localized by VisionQA as `icon` with strong geometric agreement.
- Matcher v3 now preserves the raw `icon` class while recording DOM-resolved `checkbox` or `radio` semantics only when containment is at least 0.65 and IoU is at least 0.25.
- Same-evidence replay: matches 70 -> 88; evidence-assisted coverage 53.03% -> 66.67%; strong conflicts 32 -> 14. Resolutions were 8 checkbox and 10 radio. All other conflict families remain unresolved.
- This is an integration evidence-classification improvement, not a claim of improved standalone VisionQA detector metrics.
- Machine-readable comparison: `reports/accuracy/dom_vision_semantic_resolution.json`.
- Exact next accuracy task: fixed-fixture CodeLocator runtime audit covering line preservation, duplicate candidates, long-file truncation, Top-K ranking, and raw-score interpretation. Do not retrain.

## Web, Code, and Full Analysis product workflows (2026-10-18)

- Added explicit `web`, `code`, and `full` analysis types to the existing asynchronous API and orchestrator. Existing saved runs and evidence contracts remain compatible.
- Web Analysis requests Playwright/runtime and VisionQA evidence only. Code Analysis requests RiskNet and CodeLocator only. Full Analysis requires all four modalities. Run completion now evaluates only the requested modalities.
- Added safe local Java discovery with a default 20-file cap and exclusions for generated/build, dependency, data, environment, and Git directories.
- Added deterministic runtime extraction of RiskNet's exact 21-feature contract. The saved run records that these lexical Halstead/complexity values are an estimate and not the original CK research extractor.
- RiskNet and CodeLocator batch inference loads each protected model once per run. CodeLocator preserves exact original one-based line references.
- The dashboard now leads with three workflow cards and dedicated forms, reports real orchestration stages, labels each run type, and renders risk-ranked files with CodeLocator snippets under progressive disclosure.
- Real verification: Web completed in 5.658 s, Code in 31.590 s, and Full in 44.965 s. Full returned all four modalities, 9 findings/recommendations/tests/priorities, and zero errors.
- UI verification passed at 1440 px and 390 px with no page/console errors and no horizontal overflow.
- No model was trained, fine-tuned, promoted, or modified. No dataset changed. VisionQA v1 remains production and v2 remains experimental.

## GitHub documentation and real product captures (2026-10-18)

- Verified the real local API/dashboard at `http://127.0.0.1:8765` and used completed Web, Code, and Full Analysis runs from the current product implementation.
- Captured five 1440×900 dashboard images under `docs/screenshots`. Capture completed with zero browser-console errors and zero uncaught page errors.
- Reworked `README.md` around the actual workflows, architecture, model roles, datasets, evidence semantics, tested setup, and known limitations. Removed portfolio-oriented wording.
- Added `reports/QASENSE_MODEL_PARAMETER_AUDIT.md`. Artifact-loaded counts are CodeLocator v2 124,647,170 parameters, VisionQA v1 2,608,042, and VisionQA v2 2,608,042. RiskNet remains a calibrated classical pipeline with 21 coefficients, one intercept, and two calibration parameters.
- No model, dataset, split, generated run, or frozen evaluation artifact was modified. VisionQA v1 remains production/rollback; VisionQA v2 remains experimental.

## Analysis-first dashboard redesign (2026-10-19)

- Refined the existing dashboard into a calm, analysis-first product interface while preserving the Web, Code, and Full Analysis workflows and all API contracts.
- Added a two-column product introduction with a real latest-run preview, clearer workflow entry points, compact navigation, severity summaries, and progressive evidence presentation.
- Moved model information to the secondary System Details view. The primary experience now focuses on QA health, findings, evidence, recommendations, and tests.
- Re-captured the five curated 1440×900 product screenshots from the real running application and verified saved Web, Code, and Full Analysis runs.
- Browser verification passed with zero console errors, zero uncaught page errors, and zero failed API responses. Horizontal overflow was zero at 1440, 1024, 768, and 390 pixels.
- Regression validation passed: 16 focused product/integration/adapter tests and 28 broader product/contract/VisionQA/visual-defect tests, with zero failures.
- Protected model packages remained ready and unchanged. No model was trained, promoted, or modified; no dataset or frozen evaluation artifact changed.
