# WebUI Training Corpus Report

Generated for the VisionQA v2 data-preparation milestone. No model training occurred.

## Corpus design

The source WebUI dataset remains unchanged. The processed corpus uses complete desktop/tablet/mobile source groups, a deterministic seed of 42, a source-stratified split, and exact-image connected-component consolidation. The frozen 399-screen evaluation slice is external and absent from both train and validation.

- Eligible complete candidate groups after frozen/duplicate exclusions: 12,042
- Positive groups retained: 9,058
- Deterministically retained negative-only groups: 453 (5% of positive groups)
- Train: 7773 groups / 23319 screens / 221992 boxes
- Validation: 1738 groups / 5214 screens / 52104 boxes
- Group exclusions: `{"cross_split_exact_duplicate": 94, "frozen_eval_group": 133, "invalid_box_only": 2531}`

## Mapping and cleaning

The ten-class schema is contiguous and maps back to the original AtomBlock IDs. Icon, checkbox, radio, and generic block are excluded. Partially visible boxes are clamped to the image boundary; fully offscreen, nonpositive, duplicate, ambiguous, and unmapped boxes are excluded without modifying the source annotations.

Box exclusions: `{"ambiguous_or_unmapped": 1264695, "duplicate_mapped_box": 5393, "input_subtype_ambiguous": 2328, "nonpositive_box": 15980, "outside_viewport": 2997144}`

## Human review

300 screens from distinct groups were manually reviewed across ['all_supported_classes', 'all_viewports', 'multiple_sources', 'weak_classes', 'density', 'exclusions', 'small_elements', 'large_structural_blocks']. All were accepted for the supported mappings; no annotation was silently corrected. The review is screen-level mapping validation, not exhaustive relabeling of every DOM node.

## Class representation

| Class | Train boxes | Train screens | Train groups | Val boxes | Val screens | Val groups |
|---|---:|---:|---:|---:|---:|---:|
| button | 35690 | 10321 | 3732 | 8699 | 2326 | 845 |
| link | 125627 | 15785 | 5384 | 29837 | 3641 | 1243 |
| input | 13020 | 4237 | 1544 | 2610 | 866 | 312 |
| image | 31502 | 9998 | 3520 | 7251 | 2312 | 807 |
| select | 715 | 468 | 165 | 102 | 79 | 29 |
| block-nav | 9036 | 7948 | 2874 | 2139 | 1910 | 692 |
| block-sidebar | 923 | 693 | 298 | 241 | 190 | 79 |
| block-footer | 1946 | 1769 | 734 | 441 | 402 | 167 |
| block-form | 2202 | 1972 | 755 | 558 | 464 | 174 |
| block-table | 1331 | 1032 | 362 | 226 | 189 | 69 |

## Viewports and sources

- Train viewport screens: `{"desktop": 7773, "mobile": 7773, "tablet": 7773}`; 35 sources.
- Validation viewport screens: `{"desktop": 1738, "mobile": 1738, "tablet": 1738}`; 34 sources.
- Train negative screens/groups: 1353 / 578.
- Validation negative screens/groups: 357 / 145.

## Leakage and readiness

- frozen_evaluation_groups: 0
- known_duplicate_groups: 0
- source_group_across_train_val: 0
- exact_image_hash_across_train_val: 0
- Dataset manifest SHA-256: `442c0f3d250f513463ca6188127a31240fab173e781fe630ddc6ffa4a52ced56`
- Split manifest SHA-256: `4bc01ed12f211408760ca1a67f2db901dc59d9b8bddf1fa5bc55702b37d9add3`
- Corpus audit: **PASS**
- Training readiness: **YES**

Detailed per-source and per-viewport class distributions are in `webui_training_corpus_metrics.json`.
