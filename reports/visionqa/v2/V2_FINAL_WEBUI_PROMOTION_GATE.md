# VisionQA v2 final WebUI promotion gate

## Decision

**C. WEBUI IMPROVED BUT REMAINS EXPERIMENTAL**

At the preselected threshold 0.0625, v2 materially exceeds v1 on the frozen WebUI aggregate: F1 0.05074 → 0.07759, macro-F1 0.04835 → 0.10102, recall 0.03564 → 0.05906, and precision 0.08805 → 0.11309. AtomBlock retention remains within the previously accepted gate.

Promotion is withheld because product behavior is not yet broad enough: `select` has zero true positives, tiny-object recall is 0.0088, mobile recall is 0.0354, the scorable prediction volume increases 10.68× relative to v2@0.25, and source-group gains are concentrated.

- Production/rollback model: `models/visionqa/atomblock_yolo11n/best.pt`
- Experimental model: `models/visionqa/visionqa_v2_mixed14/candidate.pt`
- Experimental calibrated threshold: `0.0625`
- Model promotion: **not performed**
- Playwright regression: **not performed**

## Next single milestone

Run one select-focused label-quality and visual-separability audit on existing WebUI train/validation data; do not use the frozen evaluation set or retrain during that audit.
