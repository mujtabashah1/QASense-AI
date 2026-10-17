# Real Application-State Validation

Generated: 2026-10-13 PDT

## Scope

No separate local application or configured deterministic QA target exists in the repository. This milestone therefore used the existing QASense runtime fixture, as required by the fallback rule. It is a realistic-fixture validation and is not presented as a production-application benchmark.

Six distinct existing states were replayed twice: desktop default, empty email, unchecked terms, focused navigation, tablet default, and mobile default. Every run preserved screenshots, DOM, console/network/page errors, trace, VisionQA output, matching, and ambiguity evidence.

## Results

| State | Detections | Matches | Conflicts | Correct reviewed | Missed | Wrong class |
|---|---:|---:|---:|---:|---:|---:|
| desktop-default | 10 | 7 | 3 | 7 | 1 | 3 |
| desktop-empty-email | 10 | 7 | 3 | 7 | 1 | 3 |
| desktop-terms-unchecked | 9 | 7 | 2 | 7 | 2 | 2 |
| desktop-navigation-focus | 10 | 6 | 4 | 6 | 1 | 4 |
| tablet-default | 8 | 6 | 1 | 6 | 4 | 1 |
| mobile-default | 5 | 2 | 3 | 2 | 6 | 3 |

Across the six primary runs, **66 elements** were reviewed: **35 correct**, **15 missed**, **16 wrong-class**, **16 semantic conflicts**, and **0 duplicate detections**. The reviewed match rate was **53.03%**.

## Repeatability

All **6 states** were stable across two runs for screenshot hash, dimensions, DOM count, detection count, class counts, match count, semantic-conflict count, and normalized console/network/page-error evidence. UTC event timestamps were excluded from equality checks because timestamps are expected to differ.

The fixture intentionally emits one page exception, console errors, and one failed request; those counts and messages were repeatable.

## Findings

Desktop states transferred better than smaller viewports. The mobile state retained 11 DOM candidates but produced only five detections and two semantic matches. Focus styling also changed the predicted class balance, demonstrating that state appearance affects detector output even when DOM structure is unchanged.

Machine-readable manifest: `reports/integration/real_state_manifest.json`. Results: `reports/integration/real_state_evaluation.json`.
