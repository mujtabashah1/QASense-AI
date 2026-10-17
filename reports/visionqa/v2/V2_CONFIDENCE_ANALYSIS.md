# VisionQA v2 confidence analysis

## Setup

Predictions were persisted before analysis thresholds at confidence 0.001. Both models used identical inference settings on the same 5,214 WebUI validation images. Matching uses class-correct IoU >= 0.5.

## Distribution shift

| Model | Supported predictions @0.001 | TP mean | TP median | TP p75 | TP p90 | Best validation F1 threshold |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 738,845 | 0.1834 | 0.0433 | 0.2553 | 0.6505 | 0.1096 |
| v2 | 4,118,347 | 0.0618 | 0.0140 | 0.0821 | 0.1829 | 0.0573 |

V2 creates far more very-low-score proposals. Its TP confidence median is roughly one-third of v1's. Confidence 0.25 lies above the 90th percentile of v2 true-positive scores (0.1829), explaining the sharp recall loss.

## Calibration

The all-prediction ECE is 0.017827 for v1 and 0.000335 for v2. This aggregate is dominated by millions of predictions below 0.1 and must not be read alone. Conditional ECE for predictions >=0.1 is 0.2412 for v1 and 0.0706 for v2. In v2's 0.1–0.5 bins, empirical precision generally exceeds mean confidence, so v2 is under-confident in its useful operating band. V1 is strongly over-confident in the same bands.

## Interpretation boundary

The validation optimum is evidence of threshold sensitivity, not permission to change production behavior. No calibrator was fitted, and the frozen evaluation was not reused. The plots under `figures/` show the PR and fixed-threshold curves.
