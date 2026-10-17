# WebUI Training Corpus Audit

Generated: 2026-10-14T19:32:48.695493+00:00

**AUDIT RESULT: PASS**

## Integrity

- Dataset manifest hash valid: True
- Split manifest hash valid: True
- Frozen evaluation leakage: 0
- Source-group leakage: 0
- Known duplicate-group leakage: 0
- Exact-image train/validation leakage: 0
- Invalid label lines: 0
- Corrupt images: 0

## Split counts

### train

- Groups: 7773
- Images/labels: 23319 / 23319
- Valid boxes: 221992
- Negative screens/groups: 1353 / 578
- Viewports: `{"desktop": 7773, "mobile": 7773, "tablet": 7773}`
- Unique sources: 35

| Class | Boxes | Screens | Groups | Box % |
|---|---:|---:|---:|---:|
| button | 35690 | 10321 | 3732 | 16.08% |
| link | 125627 | 15785 | 5384 | 56.59% |
| input | 13020 | 4237 | 1544 | 5.87% |
| image | 31502 | 9998 | 3520 | 14.19% |
| select | 715 | 468 | 165 | 0.32% |
| block-nav | 9036 | 7948 | 2874 | 4.07% |
| block-sidebar | 923 | 693 | 298 | 0.42% |
| block-footer | 1946 | 1769 | 734 | 0.88% |
| block-form | 2202 | 1972 | 755 | 0.99% |
| block-table | 1331 | 1032 | 362 | 0.60% |

### val

- Groups: 1738
- Images/labels: 5214 / 5214
- Valid boxes: 52104
- Negative screens/groups: 357 / 145
- Viewports: `{"desktop": 1738, "mobile": 1738, "tablet": 1738}`
- Unique sources: 34

| Class | Boxes | Screens | Groups | Box % |
|---|---:|---:|---:|---:|
| button | 8699 | 2326 | 845 | 16.70% |
| link | 29837 | 3641 | 1243 | 57.26% |
| input | 2610 | 866 | 312 | 5.01% |
| image | 7251 | 2312 | 807 | 13.92% |
| select | 102 | 79 | 29 | 0.20% |
| block-nav | 2139 | 1910 | 692 | 4.11% |
| block-sidebar | 241 | 190 | 79 | 0.46% |
| block-footer | 441 | 402 | 167 | 0.85% |
| block-form | 558 | 464 | 174 | 1.07% |
| block-table | 226 | 189 | 69 | 0.43% |

## Source diversity

Extreme source domination flags (>25% of groups): `{"train": [{"source": "github", "group_count": 2906, "screen_count": 8718, "fraction_of_groups": 0.3738582271967065}], "val": [{"source": "github", "group_count": 719, "screen_count": 2157, "fraction_of_groups": 0.41369390103567316}]}`

Full source and per-class viewport distributions are stored in `webui_training_corpus_metrics.json`.
