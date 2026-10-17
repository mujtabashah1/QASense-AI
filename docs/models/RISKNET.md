# RiskNet model card

- **Status:** production / protected
- **Purpose:** rank source files or modules by defect risk and QA priority.
- **Artifact:** `models/risknet/risknet_v1.joblib`
- **SHA-256:** `b32efa3e83707f01bdf29bd7dbcaebe4d484aa9f81f44bf890b9a739deac913c`
- **Data:** 17,442 NASA software-metrics rows with leakage-aware grouped validation and cross-project evaluation.

Selected cross-project metrics: precision 0.354, recall 0.446, F1 0.379, ROC-AUC 0.754, PR-AUC 0.364, and MCC 0.285. Held-out average precision was CM1 0.249, JM1 0.327, KC1 0.383, KC2 0.643, and PC1 0.218.

RiskNet estimates prioritization risk from static metrics. It does not prove that a file contains a defect, and projects outside the NASA metric domain may shift performance. The product preserves the probability and its provenance rather than treating it as defect truth.
