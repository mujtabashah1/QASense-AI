# VisionQA v2 global-threshold cross-validation

## Scope

This CPU-only experiment used persisted v2 predictions and ground truth for the 5,214-screen WebUI training-corpus validation split. Five folds were formed from 1,738 indivisible source-sample groups using seed 42. Desktop, tablet, and mobile screens from a group always remain together. Sources were balanced across folds where their group counts permitted it. No inference, training, dataset mutation, or frozen-evaluation access occurred.

Each fold selected a global threshold on the other four folds by macro-F1 over the ten supported WebUI classes. Ties used overall F1 and then the higher threshold. The grid was 0.005–0.250 in increments of 0.0025.

## Fold results

| Fold | Selected threshold | Held-out screens | Groups | Precision | Recall | F1 | Macro-F1 | Predictions | Zero screens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0675 | 1,044 | 348 | 0.2006 | 0.1130 | 0.1446 | 0.1037 | 6,201 | 251 |
| 2 | 0.0600 | 1,044 | 348 | 0.1892 | 0.1242 | 0.1500 | 0.1050 | 6,839 | 242 |
| 3 | 0.0625 | 1,044 | 348 | 0.1654 | 0.1089 | 0.1313 | 0.0952 | 6,475 | 233 |
| 4 | 0.0700 | 1,041 | 347 | 0.1942 | 0.1077 | 0.1386 | 0.1137 | 5,838 | 272 |
| 5 | 0.0625 | 1,041 | 347 | 0.2039 | 0.1211 | 0.1520 | 0.1046 | 6,125 | 271 |

Threshold mean: **0.0645**; median: **0.0625**; population standard deviation: **0.0037**; variance: **0.00001350**; range: **0.0600–0.0700**. The predeclared practical rule is stable when standard deviation is at most 0.01 and range is at most 0.02. Result: **stable**.

The recommended threshold is the robust median of independently selected fold thresholds: **0.0625**.

## Cross-validated performance

Held-out fold means: precision **0.1907 ± 0.0136**, recall **0.1150 ± 0.0066**, F1 **0.1433 ± 0.0076**, and macro-F1 **0.1045 ± 0.0059**.

Out-of-fold aggregate: precision **0.1904**, recall **0.1150**, F1 **0.1434**, macro-F1 **0.1031**, and 31,478 predictions. Compared with confidence 0.25, precision changes -0.2407, recall +0.0902, F1 +0.0965, macro-F1 +0.0523, and prediction count +28,481 (10.50×).

Zero-prediction screens fall from 3,895/5,214 (74.70%) to 1,269/5,214 (24.34%), a reduction of 2,626 screens.

## Per-class out-of-fold results

| Class | GT | Predictions | Precision | Recall | CV F1 | F1 @0.25 | Change |
|---|---:|---:|---:|---:|---:|---:|---:|
| button | 8,699 | 2,767 | 0.1373 | 0.0437 | 0.0663 | 0.0190 | +0.0473 |
| link | 29,837 | 20,348 | 0.2021 | 0.1378 | 0.1639 | 0.0536 | +0.1103 |
| input | 2,610 | 741 | 0.2578 | 0.0732 | 0.1140 | 0.0494 | +0.0646 |
| image | 7,251 | 3,053 | 0.2116 | 0.0891 | 0.1254 | 0.0572 | +0.0681 |
| select | 102 | 11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| block-nav | 2,139 | 3,536 | 0.1513 | 0.2501 | 0.1885 | 0.0126 | +0.1760 |
| block-sidebar | 241 | 77 | 0.2338 | 0.0747 | 0.1132 | 0.1103 | +0.0029 |
| block-footer | 441 | 62 | 0.1290 | 0.0181 | 0.0318 | 0.0176 | +0.0142 |
| block-form | 558 | 112 | 0.2321 | 0.0466 | 0.0776 | 0.0506 | +0.0270 |
| block-table | 226 | 771 | 0.0973 | 0.3319 | 0.1505 | 0.1377 | +0.0128 |

## Viewports

| Viewport | Recall @0.25 | CV recall | Recall change | CV precision | CV F1 | Predictions | Zero screens |
|---|---:|---:|---:|---:|---:|---:|---:|
| desktop | 0.0336 | 0.1488 | +0.1152 | 0.1713 | 0.1593 | 15,216 | 280 |
| tablet | 0.0237 | 0.1086 | +0.0849 | 0.2120 | 0.1437 | 9,870 | 429 |
| mobile | 0.0161 | 0.0844 | +0.0682 | 0.2023 | 0.1191 | 6,392 | 560 |

## Object size

| Size | GT | Recall @0.25 | CV-calibrated recall | Change |
|---|---:|---:|---:|---:|
| tiny | 8,442 | 0.0041 | 0.0381 | +0.0340 |
| small | 31,259 | 0.0241 | 0.1151 | +0.0910 |
| medium | 9,620 | 0.0392 | 0.1570 | +0.1178 |
| large | 2,783 | 0.0453 | 0.2019 | +0.1567 |

Lowering the threshold improves every size bucket, but tiny/small recall remains limited; calibration does not solve representation or localization quality.
