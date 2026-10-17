# VisionQA v2 Unified 14-Class Training Specification

> **Current authorized design (2026-10-14):** The earlier ten-class detector
> proposal below is superseded. VisionQA v2 is one YOLO11n detector with the
> canonical 14-class head, initialized from VisionQA v1. Training combines all
> 23,319 WebUI training screens with all 5,420 decodable AtomBlock training
> images. On WebUI images, classification loss is supervised for IDs 0, 1, 2,
> 3, 7, 8, 9, 10, 11, and 12 and masked for unknown IDs 4, 5, 6, and 13. On
> AtomBlock images all 14 classes receive normal supervision. Full details,
> staged freezing, fixed evaluation gates, and exact parameters are recorded in
> `reports/visionqa/v2/training_config.json`. The older proposal is retained
> below as historical context and must not be executed.

No training was executed while creating this specification.

## Objective

The primary objective is to improve cross-domain WebUI transfer on the ten supported classes: button, link, input, image, select, block-nav, block-sidebar, block-footer, block-form, and block-table.

Secondary objectives are higher recall, fewer dense-layout and domain-shift misses, and better performance on select, form, table, sidebar, and navigation. This corpus does not support claims about icon, checkbox, radio, or generic block.

## Fixed baselines

- Frozen WebUI slice: precision 0.08805, recall 0.03564, F1 0.05074, macro-F1 0.04835.
- Original AtomBlock test: precision 0.47383, recall 0.40132, mAP50 0.37755, mAP50-95 0.27825.

## Proposed experiment — do not run in this milestone

1. Initialize YOLO11n from `models/visionqa/atomblock_yolo11n/best.pt` while creating a new ten-class detection head.
2. Train only on `datasets/processed/visionqa_webui_v2/data.yaml`; never use the frozen 399 screens for training, validation, augmentation, mining, mapping, or threshold tuning.
3. Start with image size 640, seed 42, validation each epoch, early stopping, and per-epoch resumable checkpoints. Benchmark safe batch sizes on the T1200 before choosing one.
4. Run an initial maximum of 30 epochs. Select checkpoints only from the new corpus validation metrics, including per-class recall and mAP.
5. After selection, evaluate once on the frozen WebUI transfer slice and regression-test the ten overlapping classes on the original AtomBlock held-out test. Preserve v1's separate 14-class metrics and unsupported-class capability.
6. Keep VisionQA v1 as the production rollback. A v2 checkpoint must not replace it automatically.

## Promotion principle

Promotion requires meaningful improvement over the frozen WebUI baseline, acceptable regression results on the overlapping AtomBlock classes, stable inference, no major supported-class regression, and successful Playwright integration. Final numeric promotion thresholds should be chosen during the training-design milestone before test evaluation, not retrofitted after seeing test results.

## Planned artifact path

`models/visionqa/visionqa_v2_webui_yolo11n/best.pt`

Training readiness recorded by this milestone: **YES**.
