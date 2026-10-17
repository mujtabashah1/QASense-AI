# VisionQA v2 Model Report

Generated 2026-10-17T17:58:54.270969+00:00.

## Result

**KEEP V2 AS EXPERIMENTAL**. The candidate is one YOLO11n detector with the canonical 14-class head. It was initialized from VisionQA v1 and trained on 23,319 WebUI screens plus all 5,420 decodable AtomBlock rehearsal images. WebUI classes 4, 5, 6, and 13 were masked from negative classification loss; AtomBlock retained normal supervision for all classes.

Selected checkpoint: `C:\QASense-AI\models\visionqa\visionqa_v2_mixed14\candidate.pt`  
SHA-256: `dd86b81d5faf5f018940564ddb344a8ea7caf328b7b567161937e40ab7f7866d`  
Rollback: `C:\QASense-AI\models\visionqa\atomblock_yolo11n\best.pt`

## Training

The one-epoch benchmark completed in 31.3 minutes. Stage A froze model modules 0–10 and trained modules 11–23 for 3 epochs. Stage B froze modules 0–5 and trained modules 6–23 for 9 epochs. The combined main experiment completed 12 epochs in 3.10 hours. Checkpoints include optimizer, scaler, scheduler-equivalent epoch state, arguments, and metrics after every epoch.

The selected epoch came from stage_b epoch 9 using WebUI validation supported-class macro-F1 (0.11465). AMP was requested; the installed Ultralytics safety check disabled it on the T1200, so training ran in FP32.

## Frozen WebUI transfer

| Metric | v1 | v2 | Change |
|---|---:|---:|---:|
| Precision | 0.08805 | 0.28521 | +0.19716 |
| Recall | 0.03564 | 0.01395 | -0.02169 |
| F1 | 0.05074 | 0.02659 | -0.02415 |
| Macro-F1 | 0.04835 | 0.05233 | +0.00398 |

## AtomBlock regression

| Metric | v1 | v2 | Change |
|---|---:|---:|---:|
| Precision | 0.47383 | 0.43786 | -0.03597 |
| Recall | 0.40132 | 0.38987 | -0.01145 |
| mAP50 | 0.37755 | 0.35590 | -0.02165 |
| mAP50-95 | 0.27825 | 0.25541 | -0.02284 |

## Operational result

The 399-screen frozen evaluation measured mean 24.33 ms, median 19.36 ms, and p95 28.88 ms per image including wrapper timing. Model size is 19.08 MiB.

## Scientific limits

WebUI supplies no reliable labels for icon, checkbox, radio, or generic block. Their preservation is evaluated on AtomBlock; frozen WebUI transfer metrics cover only the ten supported mapped classes. The frozen slice was not used for selection or threshold tuning.
