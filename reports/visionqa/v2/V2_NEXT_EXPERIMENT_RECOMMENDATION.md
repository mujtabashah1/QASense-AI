# VisionQA v2 next experiment recommendation

Run exactly one **source-grouped cross-validated global confidence-threshold recalibration** using the already-persisted predictions from the 5,214-image WebUI validation split.

V2's best validation F1 is 0.1442 near confidence 0.0573, compared with 0.0469 at 0.25. Its true-positive median confidence is 0.0140, and 0.25 leaves 3,895 validation images without any prediction. These measurements make threshold stability the cheapest direct hypothesis test.

Partition source groups into deterministic folds, select one global threshold on calibration folds, and report performance on held-out source folds. The threshold hypothesis passes only if F1 and recall improve consistently across held-out sources without an unacceptable precision loss. Do not use class-specific thresholds in this first test, because the rare classes are too small for stable per-class fitting.

This experiment reuses raw predictions, requires no GPU training, and should take minutes of CPU time—well below 1% of the prior 3.10-hour training cost. It must not rerun or tune against the frozen 399-screen evaluation. External confirmation should use a future prospectively frozen holdout.
