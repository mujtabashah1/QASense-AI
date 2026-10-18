# QASense AI final product report

## Status

**PRODUCT MILESTONE: COMPLETE.** QASense now executes one traceable workflow from repository/application inputs through protected models, browser evidence, visual reasoning, deterministic fusion, QA actions, an API, and a dashboard. No model was trained or changed.

## Completed product modules

- Read-only RiskNet and CodeLocator v2 adapters.
- Playwright screenshot, DOM, console, page-error, network, timing, and trace collection.
- Production VisionQA v1 CUDA inference, annotated screenshot output, and DOM/Vision matching.
- Conservative single-state visual rules plus the preserved pairwise controlled baseline.
- Canonical evidence schema `1.0` with explicit unavailable-modality handling.
- Deterministic fusion v1 with evidence association, severity, confidence support, deduplication, and provenance.
- Evidence-specific recommendations, structured test suggestions, optional selector-safe Playwright skeletons, and regression priority.
- Local asynchronous API, safe artifact endpoints, model-status endpoint, and responsive static dashboard.

## Verified end-to-end demo

Run: `portfolio-demo-final-20261017`.

- Status: completed; all modalities available; zero execution errors.
- QA score: 58/100 using the documented deterministic penalty rule.
- Outputs: 10 findings, 10 recommendations, 10 test suggestions, and 10 regression priorities.
- Artifacts: original/full screenshots, annotated VisionQA screenshot, DOM, console, network, page errors, trace, model evidence, matching evidence, visual rules, and final `analysis.json`.
- Total runtime: 138.503 seconds.
- Playwright: 2.486 seconds.
- VisionQA process path: 8.899 seconds; measured model inference 3.492 seconds.
- Fusion: 0.593 ms; recommendation/test/priority generation: 0.127 ms.
- API/dashboard verification: run, model status, dashboard, and annotated-artifact routes passed.

The fixture intentionally emits errors, so the findings validate runtime collection and rule behavior. This is an integration demonstration, not a real-world accuracy benchmark.

## Tests

The final focused suite passed **41 tests** with zero test failures: 17 integration/product tests and 24 RiskNet tests. Coverage includes adapters, evidence contracts, fusion, severity/confidence rules, deduplication, recommendations, tests, regression scoring, model status, API/dashboard, Playwright capture, VisionQA CUDA inference, DOM matching, visual baseline logic, mutation reversal, and RiskNet model behavior. The real E2E demo additionally loaded RiskNet, CodeLocator v2, and VisionQA v1 successfully.

## Model status

- RiskNet: production / protected.
- CodeLocator v2: production / protected.
- VisionQA v1: production / rollback and the product default.
- VisionQA v2: experimental, preserved, not promoted, and never loaded by the production demo.

## GitHub readiness

Source, tests, documentation, API/dashboard, license, requirements, architecture, model cards, and migration instructions are prepared. Datasets, runs, environments, caches, and non-selected model checkpoints are excluded. CodeLocator v2's 499 MB safetensors file is configured for Git LFS. RiskNet and the 5.5 MB VisionQA v1 checkpoint fit normal Git limits.

The repository uses the project-local portable Git toolchain where needed. `main` is synchronized with `origin`, selected large model artifacts use Git LFS, and generated datasets/runs/environments remain excluded.

## Limitations and deferred research

- VisionQA detects elements rather than defects; v1 has weak small/rare classes and domain-transfer limits.
- VisionQA v2 remains blocked by select, tiny-object, mobile, source-sensitivity, and false-positive concerns.
- The deterministic visual benchmark is controlled synthetic evidence only.
- Real applications may require credentials, stable test accounts, consent handling, and tailored browser setup.
- Learned visual-defect and multimodal fusion models remain deferred until representative labeled data exists.

## Three-workflow product refinement (2026-10-18)

The dashboard and API now expose three separate product workflows without changing a model or dataset:

- Web Analysis uses the browser/runtime and visual stack only. Unrequested code modalities no longer make a successful web run appear partial.
- Code Analysis scans supported local Java source, executes RiskNet and CodeLocator in batches, and presents risk priority beside exact original source lines.
- Full Analysis runs all four evidence modalities and applies the existing deterministic fusion contract. Unsupported correlations are not fabricated.

Verified local runs:

- `product-web-e2e-20261018`: completed in 5.658 seconds; runtime and visual evidence available; 9 findings/actions; zero errors.
- `product-code-e2e-20261018`: completed in 31.590 seconds; one RiskNet result and one CodeLocator result; original lines 7, 5, 3, 4, and 2; zero errors. Model scores remained below the conservative fused-finding gate, so source evidence was shown without inventing a defect finding.
- `product-full-e2e-20261018`: completed in 44.965 seconds; all four modalities available; 9 findings, recommendations, tests, and priorities; zero errors.

Playwright UI verification found three workflow cards, working routes/forms, source evidence with line references, no JavaScript/page errors, and no horizontal overflow at 1440 px or 390 px. Runtime RiskNet metrics use a documented lexical estimate aligned to the exact saved 21-feature schema; it is not described as the original CK extractor.
