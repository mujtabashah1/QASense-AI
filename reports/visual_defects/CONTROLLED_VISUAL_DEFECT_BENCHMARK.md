# Controlled Visual Defect Benchmark

Generated: `2026-10-13T07:25:12.844765+00:00`

## Scope

This benchmark measures deterministic DOM/pixel/VisionQA rules on controlled CSS/DOM mutations. It does not estimate accuracy on real-world visual defects.

## Results

- Mutated positive cases: **108**
- Unchanged negative controls: **108**
- Exact multiclass accuracy: **1.000**
- Positive exact recall: **1.000**
- Defect macro precision: **1.000**
- Defect macro recall: **1.000**
- Defect macro F1: **1.000**

| Class | Support | Precision | Recall | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| clipping | 9 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| hidden_element | 9 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| misalignment | 9 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| missing_element | 9 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| no_defect | 108 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| offscreen_element | 9 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| overflow | 9 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| overlap | 9 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| oversized_element | 9 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| responsive_breakage | 9 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| text_truncation | 9 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| undersized_click_target | 9 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| z_index_issue | 9 | 1.000 | 1.000 | 1.000 | 0 | 0 |

## Decision

Do not train a visual-defect ML model yet. The pilot is large enough to validate contracts but not diverse enough to support generalization; expand with real application page states and review hard cases first.

The pilot has one controlled page fixture and deterministic mutations. Add diverse real application states and human-reviewed failures before training a visual-defect model. Current evidence supports improving the rules and dataset breadth first.

Machine-readable predictions and metrics: `reports/visual_defects/controlled_benchmark.json`.
