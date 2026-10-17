# QASense Model Adapters and Evidence Contracts

Generated: 2026-10-13 PDT

RiskNet and CodeLocator v2 now have read-only adapters under `src/adapters`. The RiskNet adapter accepts a file identity and the existing metric schema, then returns normalized defect probability, risk priority, QA priority, and factors. The CodeLocator adapter accepts Java source and returns ranked suspicious lines with normalized `line`, `score`, and `rank` fields.

Both adapters load the existing stable artifacts. They do not train, save, or alter model state. Actual smoke inference passed for `risknet_v1.joblib` and the CodeLocator v2 `line_localizer`.

`src/contracts/multimodal_evidence.schema.json` defines the first multimodal finding contract. `src/contracts/evidence.py` validates and constructs it. The contract combines optional RiskNet, CodeLocator, Playwright, and visual evidence while explicitly declaring `deterministic_evidence_normalization`. It does not claim learned fusion.

The next integration step is to populate this contract from a real application run and attach recommendations derived from transparent rules. A fusion model is not justified until real linked code/runtime/visual outcomes exist.
