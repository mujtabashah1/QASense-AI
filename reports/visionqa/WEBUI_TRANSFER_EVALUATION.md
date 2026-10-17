# WebUI Transfer Evaluation

Generated: 2026-10-13 PDT

## Fixed slice

The frozen evaluation slice contains **399 screens** from **133 source-sample groups** and **37 sources**. Each group retains desktop, tablet, and mobile screenshots. Manifest content SHA-256: `674ae4c0907b8a1927383d7b0a1a02e285f93cb4ade1ec42becad198b2ec7d45`.

All 15 known cross-split exact-image groups were excluded before selection. Nonpositive, fully off-viewport, duplicate mapped, and semantically ambiguous input boxes were excluded from scoring and retained in the manifest. The original dataset was not changed.

## Class mapping

Scored mappings are native `button→button`, `a→link`, `input/textarea→input`, `img→image`, `select→select`, `nav→block-nav`, `aside→block-sidebar`, `footer→block-footer`, `form→block-form`, and `table→block-table`. Generic SVG/i icons, generic containers, and checkbox/radio subtypes are not scored because WebUI does not provide sufficient semantics for reliable mapping.

## Full-slice metrics at IoU 0.50

Precision: **0.08805**  
Recall: **0.03564**  
F1: **0.05074**  
Macro F1: **0.04835**

| Class | GT | Predictions | Precision | Recall | F1 | AP50 |
|---|---:|---:|---:|---:|---:|---:|
| block-footer | 22 | 21 | 0.1429 | 0.1364 | 0.1395 | 0.1386 |
| block-form | 164 | 10 | 0.2000 | 0.0122 | 0.0230 | 0.0078 |
| block-nav | 242 | 102 | 0.0294 | 0.0124 | 0.0174 | 0.0027 |
| block-sidebar | 56 | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| block-table | 164 | 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| button | 2463 | 749 | 0.0668 | 0.0203 | 0.0311 | 0.0032 |
| image | 428 | 510 | 0.0882 | 0.1051 | 0.0959 | 0.0392 |
| input | 827 | 181 | 0.4088 | 0.0895 | 0.1468 | 0.0466 |
| link | 1321 | 702 | 0.0427 | 0.0227 | 0.0297 | 0.0015 |
| select | 121 | 48 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

AP50 is reported because mapped bounding boxes and fixed IoU=0.50 support it. AP50-95 is omitted: taxonomy conversion and DOM-box clipping do not support that stronger localization claim.

## Human-reviewed subset

All **100 selected screens** were visually inspected using ten annotated contact sheets. **96** were accepted for reviewed quantitative scoring and **4** were isolated. Reviewed item decisions: {"model_false_negative": 2475, "model_false_positive": 448, "model_true_positive": 58, "wrong_class": 28, "annotation_questionable": 38}.

On the 96 mapping-valid reviewed screens: precision **0.10861**, recall **0.02290**, F1 **0.03782**. This confirms that the poor full-slice transfer is not only an automated-annotation artifact.

Review workflow: `reports/visionqa/webui_review.html`. Decisions: `reports/visionqa/webui_review_decisions.json`.

## Responsive consistency

All **133** sample groups had complete viewport triplets. Mean Jaccard similarity of detected class sets across desktop/tablet/mobile was **0.394**. There were **28** class-level responsive-only miss patterns; missing occurrences were desktop 4, tablet 14, and mobile 26.

These differences are consistency evidence only. Responsive changes can be intentional and are not labeled as defects.

## Weak classes

`select` had 121 mapped boxes and zero true positives; its median box occupied only 0.31% of the screen, supporting a small-object and domain-shift explanation. `block-form` had 164 boxes and 1.22% recall; missed forms were not generally small, and occurred on dense screens, supporting taxonomy/domain-shift rather than size alone. `block-table` had 164 boxes and zero true positives despite a median 9.17% screen area, strongly indicating domain/taxonomy mismatch rather than small-object size.

Icon, checkbox, and radio were not scored. There were 1,698 SVG boxes and 132 i-tag boxes with ambiguous icon semantics, plus 120 checkbox/radio/switch-like input boxes excluded because input type is absent from WebUI bbox records.

## Error buckets

- dense_layout_miss: **2,946**
- missed_small_element: **2,842**
- visual_domain_shift: **2,759**
- wrong_semantic_class: **120**
- low_confidence_correct: **70**
- responsive_only_miss: **28**
- duplicate_detection: **4**
- annotation_mismatch_or_questionable_screen: **4**

Representative screen IDs are stored in `reports/visionqa/webui_error_analysis.json`.
