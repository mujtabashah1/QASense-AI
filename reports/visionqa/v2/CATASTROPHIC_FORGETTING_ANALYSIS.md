# Catastrophic Forgetting Analysis

The selected v2 checkpoint was compared with v1 on the untouched AtomBlock test split.

| Class | v1 AP50-95 | v2 AP50-95 | Change | Assessment |
|---|---:|---:|---:|---|
| button | 0.2589 | 0.2484 | -0.0105 | minor regression |
| link | 0.1279 | 0.1262 | -0.0017 | stable |
| input | 0.1841 | 0.1677 | -0.0163 | minor regression |
| image | 0.6534 | 0.6428 | -0.0106 | minor regression |
| icon | 0.0206 | 0.0212 | +0.0007 | stable |
| checkbox | 0.0100 | 0.0098 | -0.0002 | stable |
| radio | 0.0042 | 0.0017 | -0.0025 | stable |
| select | 0.0951 | 0.0810 | -0.0141 | minor regression |
| block-nav | 0.4546 | 0.3991 | -0.0555 | minor regression |
| block-sidebar | 0.5147 | 0.3842 | -0.1305 | minor regression |
| block-footer | 0.7805 | 0.7569 | -0.0235 | minor regression |
| block-form | 0.1302 | 0.0951 | -0.0351 | minor regression |
| block-table | 0.1717 | 0.1784 | +0.0067 | improved |
| block | 0.4897 | 0.4632 | -0.0265 | minor regression |

Final gate: **PASS** under the predeclared materiality rule in `training_config.json`.
