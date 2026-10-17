# VisionQA v2 Active State

Last reconciled: 2026-10-17.

## Completion update

The experiment completed after this active handoff was created. Stage A completed
three epochs and passed the forgetting gate; Stage B completed nine epochs. Stage B
epoch 9 was selected using WebUI validation macro-F1. The frozen WebUI evaluation
failed the predeclared F1 promotion gate, so the candidate remains experimental and
VisionQA v1 remains production/rollback. See `training_status.json`,
`VISIONQA_V2_MODEL_REPORT.md`, and `promotion_decision.json` for final results.

## Authorized milestone

Train and evaluate one unified YOLO11n VisionQA v2 detector with the canonical
14-class QASense output contract. Do not train a ten-class production model,
YOLO11s, RiskNet, CodeLocator, a visual-defect model, or any unrelated module.

## Completed and verified

- RiskNet, CodeLocator v2, and VisionQA v1 remain protected rollback artifacts.
- The leakage-safe WebUI corpus and frozen 399-screen evaluation slice remain fixed.
- The mixed training corpus contains 23,319 WebUI screens and all 5,420 decodable
  AtomBlock training images; WebUI validation contains 5,214 screens.
- WebUI classification loss supervises IDs 0, 1, 2, 3, 7, 8, 9, 10, 11, and 12.
  IDs 4, 5, 6, and 13 are masked on WebUI and fully supervised on AtomBlock.
- Partial-label, source-tagging, schema, corpus, protected-hash, and resume-state tests pass.
- The full-data one-epoch benchmark completed. Its `epoch0.pt` has full optimizer,
  scaler, stopper, arguments, and result state.
- Same-implementation pretraining baselines exist for WebUI validation and AtomBlock test.

## Active work

- Controller status: `reports/visionqa/v2/training_status.json`.
- Stage A status: `reports/visionqa/v2/stage_a_training_status.json`.
- Stage A log: `reports/visionqa/v2/stage_a.log`.
- Stage A trains three epochs with model modules 0-10 frozen and modules 11-23 trainable.
- A Windows Scheduled Task named `QASense-VisionQA-v2-Recovery` checks every five
  minutes and relaunches the controller if the GPU-owning trainer disappears.
- Every completed epoch writes a full-state `epochN.pt`. An unfinished epoch has
  no recoverable checkpoint and is repeated after interruption.

### Recovery coordination incident (2026-10-17)

Stage A completed all three epochs and passed the AtomBlock forgetting gate. At
the Stage A to Stage B subprocess transition, the recovery lock still referenced
the finished Stage A child PID. The 04:40 scheduled check therefore started a
duplicate controller. The duplicate Stage B process tree was identified by its
04:40 start time and stopped; the original 03:50 controller and its 04:37 Stage B
trainer were retained. The lock now references the persistent original controller
PID rather than an individual stage child. The duplicate run directory is retained
as incident evidence and must not be used for model selection.

## Remaining ordered work

1. Complete Stage A.
2. Evaluate the Stage A checkpoint on the untouched AtomBlock test and enforce
   the predeclared catastrophic-forgetting gate.
3. If the gate passes, run Stage B for at most nine epochs with modules 0-5 frozen.
4. Select one checkpoint using only WebUI validation supported-class macro-F1.
5. Evaluate it once on the frozen 399-screen WebUI set and on AtomBlock test.
6. Run the Playwright integration regression if metric gates pass.
7. Produce final model, transfer, forgetting, runtime, and promotion reports.
8. Verify all protected hashes and update `reports/QASENSE_PROJECT_STATE.md`.

The authoritative exact configuration and gates are in `training_config.json`.
Dynamic progress must always be read from the JSON status files rather than inferred
from this handoff.
