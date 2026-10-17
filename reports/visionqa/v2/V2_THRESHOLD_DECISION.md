# VisionQA v2 global-threshold decision

## Decision

**B. GLOBAL THRESHOLD IMPROVES V2 BUT CLASS-SPECIFIC ISSUES REMAIN**

The five independently selected thresholds are 0.0675, 0.0600, 0.0625, 0.0700, 0.0625. Their standard deviation is 0.0037 and range is 0.0100, satisfying the predeclared stability rule. Cross-validated F1 improves from 0.0469 at confidence 0.25 to 0.1434, while recall improves from 0.0248 to 0.1150. Precision decreases from 0.4311 to 0.1904; the reported F1 and macro-F1 gains account for that cost.

The recommended global threshold is **0.0625**, derived as the median fold-selected threshold. It is not derived from the prior single-split optimum and was not tested on the frozen 399-screen evaluation.

Class-specific problems remain. Unresolved zero-F1 classes: **select**. In particular, `select` has 0 true positives even at the persisted 0.001 prediction floor, so threshold calibration cannot repair that class.

## Next single milestone

Authorize one final frozen 399-screen confirmation using the calibrated global threshold. This step is recommended only; it was not executed here.
