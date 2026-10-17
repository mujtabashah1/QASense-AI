# VisionQA v2 failure analysis

## Scope and scientific boundary

This analysis used only the fixed 5,214-screen WebUI training-corpus validation split. It ran v1 and v2 with identical 640-pixel preprocessing, NMS IoU 0.7, `max_det=1500`, and a 0.001 persistence floor. The frozen 399-screen WebUI evaluation was not rerun, relabeled, inspected, or used for threshold selection. No model was trained or modified.

## Main finding

VisionQA v2 did not stop producing candidates. It produced 4,845,989 supported-class candidates at the 0.001 floor versus 738,845 for v1. Its useful candidates moved to much lower confidence: true-positive median confidence fell from 0.0433 to 0.0140. At confidence 0.25, v2 emitted only 2,997 predictions versus v1's 21,090, and 3,895 screens had zero v2 predictions. V2's best validation F1 was 0.1442 at threshold 0.0573; at 0.25 it was 0.0469. Confidence suppression is therefore the primary measured mechanism behind recall collapse at the shared 0.25 operating point.

The gap cannot be assigned to confidence alone. Source response is heterogeneous, `select` has zero true positives at every measured operating point, and tiny/small objects remain weak. The external frozen set's already-recorded aggregate result is consistent with the confidence shift interacting with domain composition, but its labels were not reopened, so exact frozen-source attribution remains unproven.

## Threshold sensitivity

| Threshold | v1 P | v1 R | v1 F1 | v1 preds | v2 P | v2 R | v2 F1 | v2 preds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 5.17% | 7.78% | 6.21% | 78,322 | 15.95% | 12.95% | 14.29% | 42,302 |
| 0.10 | 6.91% | 6.31% | 6.60% | 47,631 | 25.40% | 8.22% | 12.42% | 16,858 |
| 0.15 | 8.12% | 5.37% | 6.47% | 34,470 | 31.86% | 5.08% | 8.76% | 8,302 |
| 0.20 | 9.10% | 4.62% | 6.13% | 26,415 | 37.78% | 3.48% | 6.38% | 4,802 |
| 0.25 | 10.03% | 4.06% | 5.78% | 21,090 | 43.11% | 2.48% | 4.69% | 2,997 |
| 0.30 | 10.95% | 3.60% | 5.42% | 17,139 | 47.86% | 1.71% | 3.31% | 1,866 |
| 0.40 | 12.14% | 2.77% | 4.51% | 11,886 | 55.19% | 0.83% | 1.63% | 781 |
| 0.50 | 13.22% | 2.18% | 3.74% | 8,590 | 58.87% | 0.32% | 0.63% | 282 |

The exact validation optima from the persisted low-floor prediction ranking were v1 F1 0.0664 at 0.1096 and v2 F1 0.1442 at 0.0573. These values are diagnostic; they are not production thresholds.

## Class behavior at confidence 0.25

| Class | GT | v1 F1 | v2 P | v2 R | v2 F1 | v2 best val threshold | v2 best val F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| button | 8,699 | 5.03% | 33.33% | 0.98% | 1.90% | 0.0401 | 7.47% |
| link | 29,837 | 3.18% | 48.68% | 2.84% | 5.36% | 0.0641 | 16.46% |
| input | 2,610 | 8.08% | 46.90% | 2.61% | 4.94% | 0.0331 | 11.89% |
| image | 7,251 | 12.51% | 35.04% | 3.12% | 5.72% | 0.0218 | 14.00% |
| select | 102 | 0.00% | 0.00% | 0.00% | 0.00% | 0.2230 | 0.00% |
| block-nav | 2,139 | 7.99% | 16.47% | 0.65% | 1.26% | 0.0671 | 19.49% |
| block-sidebar | 241 | 3.74% | 48.39% | 6.22% | 11.03% | 0.1590 | 12.20% |
| block-footer | 441 | 3.55% | 30.77% | 0.91% | 1.76% | 0.0037 | 3.43% |
| block-form | 558 | 4.43% | 42.86% | 2.69% | 5.06% | 0.0307 | 10.40% |
| block-table | 226 | 1.68% | 38.00% | 8.41% | 13.77% | 0.1235 | 22.36% |

`select` is a real learned-class failure on this validation set: 102 ground-truth boxes and zero true positives even when scanning the persisted low-floor predictions. `block-table` improves strongly when scored at its validation optimum; `block-sidebar` and `block-form` also improve, while `block-footer` remains weak.

## Size and density

Size buckets use normalized box area: tiny `<0.001`, small `0.001–<0.01`, medium `0.01–<0.1`, and large `>=0.1`.

| Size | GT | v1 recall @0.25 | v2 recall @0.25 |
|---|---:|---:|---:|
| tiny | 8,442 | 1.48% | 0.41% |
| small | 31,259 | 2.99% | 2.41% |
| medium | 9,620 | 6.69% | 3.92% |
| large | 2,783 | 14.77% | 4.53% |

| Density bucket (validation quartiles) | Screens | v1 recall @0.25 | v2 recall @0.25 | v2 predictions |
|---|---:|---:|---:|---:|
| low_le_3 | 1,532 | 6.05% | 2.17% | 281 |
| medium_4_7 | 1,130 | 6.21% | 3.34% | 499 |
| high_8_14 | 1,318 | 4.74% | 2.78% | 896 |
| very_high_gt_14 | 1,234 | 3.16% | 2.18% | 1,321 |

V2 is weakest on tiny objects, but confidence suppression affects every size. Dense pages are additionally difficult: v2 recall drops from 3.34% in the medium-density bucket to 2.18% above the 75th percentile.

## Viewport and source behavior

| Viewport | v1 recall | v1 F1 | v2 recall | v2 F1 |
|---|---:|---:|---:|---:|
| desktop | 6.10% | 7.62% | 3.36% | 6.20% |
| tablet | 3.73% | 5.54% | 2.37% | 4.51% |
| mobile | 2.13% | 3.41% | 1.61% | 3.11% |

Mobile is the weakest viewport for both models. At 0.25, v2's largest measured source regressions were codepen, backstage, github; its largest improvement was `wordpress_themes` (+0.1281 F1). This supports source sensitivity. Train and validation remain closely structured: all three viewports are balanced, density medians are both 7, and they contain 35 and 34 sources respectively. A simple train/validation split imbalance is not supported as the sole cause.

## Classification versus localization

At 0.25, v2 misses include 19,443 correct-class boxes suppressed below the threshold, 136 localization-only failures, 91 wrong-class overlaps, and 30,980 cases with no nearby high-confidence prediction. Localization mismatch is a small minority; confidence and absent/very-low-score candidates dominate. The largest v2 wrong-class pair is `button -> link` with 32 cases.

## Root-cause evidence

| Candidate | Assessment | Evidence |
|---|---|---|
| Confidence shift | Supported, primary | TP median 0.0433→0.0140; best F1 threshold 0.1096→0.0573; v2 prediction count falls 85.8% versus v1 at 0.25. |
| Threshold sensitivity | Supported | v2 F1 is 0.1442 near 0.057 but 0.0469 at 0.25. |
| Source/domain shift | Partially supported | Large source-specific changes exist, including improvement on WordPress themes and regressions on CodePen/GitHub; frozen per-source attribution was deliberately not recomputed. |
| Class imbalance/taxonomy | Supported for specific classes | `select` has only 102 validation boxes and zero TP; structural classes vary materially. |
| Small-object weakness | Supported | v2 tiny recall 0.41% versus large 4.53% at 0.25. |
| Dense-layout conservatism | Supported | Recall falls in the highest-density quartile while few predictions survive. |
| Validation distribution bias | Partially supported | Train/val metadata are similar internally, but the frozen slice is an external source-grouped sample with a different aggregate outcome. |
| Localization mismatch | Not primary | Only 136 v2 misses were isolated localization failures at 0.25. |
| Unified 14-class architecture failure | Not supported | V2 doubles best validation F1 and retained 91.79% of v1 AtomBlock mAP50-95; failures are operating-point and class/domain specific. |

## Conclusion

V2 remains experimental. The 14-class preservation strategy remains technically successful, but production promotion remains rejected because the existing frozen result has lower overall F1. The analysis supports a lower and validation-stable operating threshold as the smallest hypothesis to test next, without retraining.
