# QASense AI model parameter audit

Audit date: 2026-10-18

This audit loads the selected local inference artifacts without modifying them. Neural-network counts use `sum(p.numel() for p in model.parameters())` on the instantiated model. RiskNet is a calibrated classical pipeline, so its fitted structure is reported separately instead of being presented as a neural-network parameter count.

## Verified artifacts

| Component | Loaded architecture | Exact parameter structure | Classes | Status |
|---|---|---:|---:|---|
| RiskNet | `CalibratedClassifierCV` around an imputer, standard scaler, and logistic regression | 21 coefficients + 1 intercept + 2 sigmoid calibration parameters | Binary risk output | Production |
| CodeLocator v2 | `RobertaForSequenceClassification` | 124,647,170 parameters | 2 ranking labels | Production |
| VisionQA v1 | Ultralytics `DetectionModel` / YOLO11n | 2,608,042 parameters | 14 | Production / rollback |
| VisionQA v2 | Ultralytics `PartialLabelDetectionModel` / YOLO11n | 2,608,042 parameters | 14 | Experimental |

## RiskNet fitted state

RiskNet consumes 21 ordered software-metric features. The fitted logistic classifier has 21 coefficients and one intercept. Sigmoid calibration adds two fitted scalar values. The pipeline also stores 84 preprocessing statistics: 21 imputation statistics and 21 values each for scaler mean, variance, and scale.

The 24 predictive and calibration scalars are not directly comparable to neural-network parameter counts. Serialized file size is not used as a parameter count.

## Neural-network totals

- Production neural models, CodeLocator v2 plus VisionQA v1: **127,255,212 parameters**.
- All selected neural checkpoints, including experimental VisionQA v2: **129,863,254 parameters**.

The second total counts v1 and v2 as separate published checkpoints. It does not imply that both VisionQA models run in the production request path; QASense defaults to v1.

## Scope and method

- CodeLocator was loaded from `models/codelocator_v2/line_localizer` with the local Transformers configuration.
- VisionQA v1 was loaded from `models/visionqa/atomblock_yolo11n/best.pt`.
- VisionQA v2 was loaded from `models/visionqa/visionqa_v2_mixed14/candidate.pt`.
- RiskNet was loaded from `models/risknet/risknet_v1.joblib` and its fitted estimator attributes were inspected.
- Buffers were excluded from neural parameter totals. Each YOLO model contains 15,745 buffer values; CodeLocator contains 1,028 buffer values.
- No model was trained, saved, converted, or overwritten during this audit.
