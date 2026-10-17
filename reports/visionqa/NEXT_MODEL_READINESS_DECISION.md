# Next-Model Readiness Decision

Generated: 2026-10-13 PDT

## Decision

**B. Targeted VisionQA fine-tuning is justified. No model was trained in this milestone.**

The fixed WebUI slice shows systematic transfer weakness: full-slice precision 0.08805, recall 0.03564, and F1 0.05074. Manual review of 100 screens retained 96 mapping-valid screens and confirmed precision 0.10861, recall 0.02290, and F1 0.03782. Select, table, and sidebar had zero recall; form and navigation recall were near zero. Real-state validation also degraded sharply on mobile.

WebUI has sufficient candidate volume for the ten supported native-tag mappings, and those classes affect QASense DOM/visual grounding. This meets the readiness criteria for preparing a targeted fine-tuning dataset. It does not justify training directly from raw WebUI annotations.

## Required next model path

Prepare targeted VisionQA fine-tuning on a new curated WebUI-derived training/validation corpus; preserve the 399-screen evaluation slice as evaluation-only.

The next corpus must be source-grouped, filter invalid/off-viewport/nested boxes, undergo human mapping review, and exclude all 133 sample groups and cross-split duplicate groups in the fixed evaluation manifest. The frozen 399 screens must never become training data.

Icon, checkbox, and radio require better semantic labels before inclusion. A separate auxiliary screen-understanding model remains a later option for SVG/container structure, but current evidence first supports targeted detector transfer for the ten defensible classes.

## Controlled defect dataset

`pilot_v2` remains unchanged. The observed failures concern element taxonomy, density, and domain transfer, so no evidence-based `pilot_v3` mutation cases were added.

## Exact next milestone

Create a separate source-grouped, human-reviewed WebUI-derived training/validation corpus for the supported classes, excluding every source sample and exact-image group in webui_eval_manifest.json; benchmark a targeted fine-tune plan without touching the fixed evaluation slice.

This milestone stops before that work, as required.
