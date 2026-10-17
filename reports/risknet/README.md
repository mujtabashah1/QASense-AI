# RiskNet artifacts

Run `scripts/train_risknet.py` with the project-local Python environment to regenerate audit, cleaned data, baseline comparisons, nested cross-project tuning/calibration, plots and the model bundle.

The artifact uses all five projects for the final fit. Evaluation predictions come from held-out splits, never from the full-data final model. Consult `RISKNET_MODEL_REPORT.md` and machine-readable metrics for limitations.

Inference accepts all 21 canonical numeric metrics, or documented aliases. Missing inputs, negative values, nonfinite values and unexpected fields raise `ValueError`; preprocessing imputes missing values in training data, but inference requires complete measurements. Load only trusted model bundles. For the logistic model, `top_risk_factors` contains positive local calibrated log-odds contributions relative to training feature means. These contributions explain the fitted model, not causal relationships. Other model families return explicitly labeled global permutation importance.
