# VisionQA Dataset Strategy

Generated: 2026-10-13 PDT

## Decision

Keep the trained AtomBlock YOLO11n detector unchanged. Do not merge either audited dataset into its training data yet, and do not train another screen-understanding model in this milestone.

WebUI should first support source-held-out evaluation, responsive geometry, and a human-reviewed label-mapping study. Google Screen Annotation should remain representation/evaluation data until the referenced Rico screenshots and their provenance are locally resolved.

## What each dataset provides

### AtomBlock-WebUI

AtomBlock provides 14 curated web UI detection classes with fixed YOLO train, validation, and test splits. It trained the existing UI-element detector. Its strongest classes are large structural blocks and images; radio, checkbox, icon, select, form, and table remain weak.

### WebUI

The local WebUI package contains 36,807 valid embedded PNG screenshots from 12,269 UI samples, with desktop, tablet, and mobile views for every sample. It includes 4,689,704 typed DOM-node boxes plus HTML, CSS, JavaScript, source, framework, and component metadata. Its source-level splits have no source or sample-ID overlap.

Its labels describe every visible DOM node. The dominant tags are nested containers and text/SVG internals, so direct conversion would create duplicate/nested boxes that do not match AtomBlock's curated taxonomy. The full audit also found 2,986 rows without boxes, 16,076 nonpositive boxes, and 15 exact-image groups that cross split boundaries. These require explicit filtering or grouping before any model evaluation or fine-tuning.

### Google Screen Annotation

The local package contains 22,076 unique mobile screen annotation rows and 520,020 parsed hierarchical elements across 15 categories. It has zero cross-split screen-ID overlap and no malformed annotation rows or invalid parsed boxes. The physical CSVs contain 341 fewer records than the README declares.

No screenshot bytes are present. Rows reference the external Rico screenshot collection by ID. Labels use normalized mobile coordinates, hierarchy, visible text, and semantic classes such as TEXT, PICTOGRAM, LIST_ITEM, BUTTON, IMAGE, TEXT_INPUT, CHECKBOX, and RADIO_BUTTON. This is not the AtomBlock YOLO schema.

## Compatibility map

| Capability | AtomBlock | WebUI | Google Screen Annotation | Decision |
|---|---|---|---|---|
| Web element detection | Direct 14-class ground truth | Partial tag mapping | Mobile semantic mapping only | AtomBlock remains primary |
| Responsive geometry | One screenshot per record | Three linked viewports | Mobile only | Use WebUI |
| DOM/source grounding | No runtime DOM | HTML/CSS/JS and DOM boxes | Text hierarchy only | Use WebUI |
| Screen-language understanding | Limited | GPT descriptions | Rich verified text annotations | Evaluate auxiliary representation later |
| Direct YOLO merge | Native | Unsafe without filtering | Blocked by absent pixels and schema | Do not merge |

Conceptual overlaps exist for button, link/list item, input, image/pictogram, checkbox, radio, select-like controls, navigation, form, and table. They are not label equivalence. WebUI's `div`, `span`, `path`, nested links, and SVG internals conflict with the curated AtomBlock class policy. Google's mobile hierarchy and text descriptions conflict with both web taxonomies.

## Training decision

Current choice: **Option C, use the existing detector unchanged.**

For WebUI, follow Option C with a possible targeted Option A only after:

1. define role-aware mappings and exclusions;
2. remove nonpositive, nested duplicate, and off-viewport boxes;
3. group exact duplicate images before fixed splits;
4. create a small human-reviewed mapping set;
5. measure the current AtomBlock detector on that set;
6. fine-tune only if the measured error represents a product need.

For Google Screen Annotation, use Option C as evaluation/representation data after Rico image resolution. A separate mobile screen-understanding model (Option B) may later be more scientifically coherent than forcing its hierarchy into AtomBlock.

## Recommended next data milestone

Build a mapped WebUI evaluation slice, not a training set: 300–500 source-grouped screens balanced across viewports and the weak AtomBlock classes. Preserve WebUI's test sources and exact-image groups. Measure class coverage and detector transfer before choosing YOLO fine-tuning or an auxiliary representation model.

Authoritative audits:

- `reports/datasets/GOOGLE_SCREEN_ANNOTATION_AUDIT.md`
- `reports/datasets/google_screen_annotation_audit.json`
- `reports/datasets/WEBUI_DATASET_AUDIT.md`
- `reports/datasets/webui_dataset_audit.json`
