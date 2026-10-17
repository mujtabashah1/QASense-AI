# VisionQA v2 frozen-threshold confirmation

## Fixed protocol

VisionQA v2 `candidate.pt` was evaluated once on all 399 screens in the unchanged 133-group frozen WebUI manifest. The confidence threshold was the preselected source-grouped-CV value **0.0625**. Image size 640, NMS IoU 0.7, `max_det=1500`, class mapping, invalid-box filtering, and IoU 0.5 metric matching were unchanged. No threshold sweep, tuning, calibration fitting, training, or dataset modification occurred.

- Candidate SHA-256: `dd86b81d5faf5f018940564ddb344a8ea7caf328b7b567161937e40ab7f7866d`
- Frozen manifest canonical SHA-256: `674ae4c0907b8a1927383d7b0a1a02e285f93cb4ade1ec42becad198b2ec7d45`
- Evaluation passes: 1

## Overall comparison

| Model / threshold | Precision | Recall | F1 | Macro-F1 | Predictions | Zero screens |
|---|---:|---:|---:|---:|---:|---:|
| VisionQA v1 / 0.25 | 0.08805 | 0.03564 | 0.05074 | 0.04835 | 2,351 | not persisted in the original aggregate |
| VisionQA v2 / 0.25 | 0.28521 | 0.01395 | 0.02659 | 0.05233 | 284 | 180 |
| VisionQA v2 / 0.0625 | 0.11309 | 0.05906 | 0.07759 | 0.10102 | 3,033 | 10 |

Against v2 at 0.25, precision changes **-0.17212**, recall **+0.04511**, F1 **+0.05100**, and macro-F1 **+0.04868**. Scorable predictions increase by +2,749 (10.68×). Zero-prediction screens fall by 170, from 45.11% to 2.51%.

## Per-class results

| Class | Predictions | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| button | 739 | 42 | 697 | 2,421 | 0.0568 | 0.0171 | 0.0262 |
| link | 1,223 | 119 | 1,104 | 1,202 | 0.0973 | 0.0901 | 0.0936 |
| input | 199 | 86 | 113 | 741 | 0.4322 | 0.1040 | 0.1676 |
| image | 390 | 49 | 341 | 379 | 0.1256 | 0.1145 | 0.1198 |
| select | 2 | 0 | 2 | 121 | 0.0000 | 0.0000 | 0.0000 |
| block-nav | 122 | 6 | 116 | 236 | 0.0492 | 0.0248 | 0.0330 |
| block-sidebar | 17 | 8 | 9 | 48 | 0.4706 | 0.1429 | 0.2192 |
| block-footer | 13 | 3 | 10 | 19 | 0.2308 | 0.1364 | 0.1714 |
| block-form | 23 | 8 | 15 | 156 | 0.3478 | 0.0488 | 0.0856 |
| block-table | 305 | 22 | 283 | 142 | 0.0721 | 0.1341 | 0.0938 |

`select` remains unresolved: 2 predictions, 0 true positives, and 121 false negatives. Confidence recalibration does not solve select detection.

## Viewport results

| Viewport | Predictions | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| desktop | 1,555 | 0.1042 | 0.0858 | 0.0941 |
| tablet | 988 | 0.1204 | 0.0548 | 0.0754 |
| mobile | 490 | 0.1265 | 0.0354 | 0.0554 |

Mobile recall is 0.0354. It is higher than the overall v2@0.25 recall, but remains substantially below desktop recall 0.0858. Original per-viewport v2@0.25 metrics were not persisted, so no post-hoc threshold comparison is fabricated.

## Object-size results

| Size | Ground truth | Matched | Recall |
|---|---:|---:|---:|
| tiny | 1,250 | 11 | 0.0088 |
| small | 3,170 | 198 | 0.0625 |
| medium | 999 | 86 | 0.0861 |
| large | 389 | 48 | 0.1234 |

Tiny-object recall remains a major limitation at 0.0088.

## Source-group distribution

True positives improve in 47 of 133 groups, remain unchanged in 86, and regress in 0. The ten most-improved groups contribute 67.2% of added true positives. Improvement is real but concentrated rather than uniformly distributed.

## AtomBlock reference

The existing unchanged v2 AtomBlock result is P 0.43786, R 0.38987, mAP50 0.35590, and mAP50-95 0.25541. Retention remains 91.79% of v1 mAP50-95; the catastrophic-forgetting gate remains passed. No AtomBlock evaluation was rerun.
