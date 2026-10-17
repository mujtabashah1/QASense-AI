# CodeLocator localizer recovery audit

## Finding

The interrupted run produced a valid validation-MRR-selected checkpoint after epoch 2. Epoch 3 did not complete, and the available artifacts cannot reconstruct its optimizer or scheduler state. Recovery therefore evaluates the saved epoch-2 checkpoint without retraining.

## Evidence

- `localizer_history.json` contains exactly two completed epoch records. Epoch 2 has training loss `0.3845885767723064`.
- `training_metadata.json` records validation MRR `0.5136246338449586`, higher than the epoch-1 checkpoint value `0.5012701225402568`. Its timestamp matches the epoch-2 history update.
- The interrupted-session report says epoch 3 started. There is no durable per-step log, so its exact last step cannot be established from files.
- Epoch 3 did not complete: no third history row exists. The implementation writes a history row after every completed epoch and validation pass.
- Early stopping did not end the run. Epoch 2 improved the selected metric, resetting the stale-epoch counter to zero; patience is two.
- No optimizer, learning-rate scheduler, gradient-scaler, RNG, or current-epoch checkpoint exists. `save_pretrained()` persisted only the best model and tokenizer plus task metadata. Exact epoch-3 continuation is therefore impossible.
- Final evaluation did not complete: `localization.json` and `line_localizer/evaluation.json` were absent, while the parent execution status remained at `line_localizer`.
- The classifier is complete and already evaluated. It must not be retrained.

## Preservation and recovery

The complete checkpoint directory was copied to `models/codelocator/line_localizer_epoch2_best_backup`. The original and backup `model.safetensors` SHA-256 values both equal:

`0e5463f1c7c80915a76297a896772b576771fe92d7b7678c4e3f4dd85e7270f8`

Recovery uses the original untouched validation and test localization JSONL files, the existing `ranking_metrics()` and `classification_metrics()` functions, and the saved model configuration. It performs no preprocessing, splitting, classifier training, localizer training, or dataset reduction.
