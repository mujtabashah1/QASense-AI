# Playwright + VisionQA Integration Report

Generated: 2026-10-13 PDT

## Status

**PASS.** The reusable runtime collector, trained VisionQA detector, DOM extraction, and evidence matcher run end to end without changing any trained model.

Final evidence run: `runs/playwright/integration-final-20261013`.

## Runtime evidence contract

Each run receives a unique ID and writes to `runs/playwright/<run-id>/`:

- `metadata.json`: URL, title, UTC timestamps, viewport, document size, browser, navigation status, and navigation/paint timing.
- `viewport.png` and `full_page.png`.
- `dom.json`: selector, tag, role, text, accessible name, visible/enabled state, bounding boxes, classes, ID, checked state, scroll/client geometry, overflow, clipping, positioning, and z-index.
- `console.json`: all browser console messages, including warnings and errors.
- `network.json`: failed requests and HTTP error responses.
- `errors.json`: page errors and error counts.
- `trace.zip`: Playwright screenshots, snapshots, and source trace.
- `visionqa.json`: structured detector evidence from the stable AtomBlock checkpoint.
- `dom_vision_matches.json`: semantic/geometry matches, unmatched records, and explicit ambiguous conflicts.

## Verified end-to-end run

The fixture run used Edge headless at 1440×900. It captured 11 candidate DOM elements, 10 VisionQA detections, seven semantic matches, and three strong-overlap semantic conflicts. Conflicts were retained as ambiguity evidence rather than accepted as matches. The detector returned one block, one navigation block, two buttons, two icons, two inputs, and two links.

The runtime collectors also captured the fixture's intentional warning, console error, page exception, and failed request. Navigation returned 200 and the Playwright trace was preserved. Cold-process detector time was 3,184.6 ms; this includes Python, CUDA, and model startup and must not be compared with the trained model's steady-state 7.78 ms/image benchmark.

## Matching policy

Primary matches require a compatible DOM tag/role and VisionQA class. Their scores combine IoU, containment, center distance, detection confidence, and semantic compatibility. Strong geometry with a conflicting class is reported separately. Geometry and class compatibility remain evidence rather than guaranteed identity.

## Reproducible entry points

- `src/playwright_runtime/collector.py`
- `src/playwright_runtime/dom.py`
- `src/visionqa/inference.py`
- `src/visionqa/matching.py`
- `scripts/run_playwright_visionqa.py`

The integration environment is `.venv-integration` (Python 3.14, Playwright 1.63.0). VisionQA inference runs in `.venv-visionqa` so its validated Torch/Ultralytics stack remains isolated.

## Limitations

The integration fixture proves collection and contracts. It is not a representative browser corpus. VisionQA still has weak held-out performance for radio, checkbox, icon, select, form, and table classes, which is visible in unmatched and ambiguous runtime evidence. Real sites may require authentication, consent handling, deterministic data setup, and application-specific waiting rules.
