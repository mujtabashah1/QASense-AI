# QASense CodeLocator v1 model report

## Objective and execution status

Classify Java repair categories from buggy fragments and, only on verified real context, learn line-level suspiciousness ranking. No universal defect probability is justified by this repair-only dataset.

```json
{
  "status": "trained",
  "stage": "report",
  "completed": [
    "processed_data_validation",
    "baseline",
    "codebert_bug_type",
    "line_localizer"
  ],
  "recovery": {
    "training_performed": false,
    "checkpoint": "saved best after epoch 2"
  }
}
```

## Dataset source and raw schema

Primary: local ManySStuBs4J sstubsLarge. Alternate source context: sstub_input_no_comments. No other QASense datasets or RiskNet files were modified.

```json
{
  "raw_records": 63923,
  "schema": [
    "bugType",
    "fixCommitSHA1",
    "fixCommitParentSHA1",
    "bugFilePath",
    "fixPatch",
    "projectName",
    "bugLineNum",
    "bugNodeStartChar",
    "bugNodeLength",
    "fixLineNum",
    "fixNodeStartChar",
    "fixNodeLength",
    "sourceBeforeFix",
    "sourceAfterFix"
  ]
}
```

## Cleaning methodology, counts and duplicates

Only line endings are normalized. Java case, syntax, punctuation, operators, identifiers, literals and indentation are preserved. Records removed have row-level reasons in tables/dropped_records.csv. Exact standardized duplicate identities include category; conflicting categories for the same fragment are retained as label ambiguity.

```json
{
  "original_rows": 63923,
  "retained": 62394,
  "dropped": 1529,
  "reasons": {
    "exact_standardized_duplicate": 815,
    "identical_buggy_fixed_code": 714
  }
}
```

Raw duplicates: {"exact_duplicate_rows": 815, "duplicate_buggy_statements": 40147, "duplicate_fixed_statements": 39869, "duplicate_pairs": 39250, "duplicate_patches": 37883}. Conflicting-category buggy fragments: 6538.

## Bug types and class imbalance

| Category | Raw count | Raw percentage |
|---|---:|---:|
| CHANGE_IDENTIFIER | 22668 | 35.46% |
| DIFFERENT_METHOD_SAME_ARGS | 10179 | 15.92% |
| CHANGE_NUMERAL | 5447 | 8.52% |
| OVERLOAD_METHOD_MORE_ARGS | 5100 | 7.98% |
| CHANGE_MODIFIER | 5011 | 7.84% |
| LESS_SPECIFIC_IF | 2813 | 4.40% |
| MORE_SPECIFIC_IF | 2381 | 3.72% |
| CHANGE_OPERATOR | 2241 | 3.51% |
| SWAP_BOOLEAN_LITERAL | 1842 | 2.88% |
| OVERLOAD_METHOD_DELETED_ARGS | 1588 | 2.48% |
| CHANGE_CALLER_IN_FUNCTION_CALL | 1504 | 2.35% |
| CHANGE_UNARY_OPERATOR | 1016 | 1.59% |
| CHANGE_OPERAND | 807 | 1.26% |
| SWAP_ARGUMENTS | 612 | 0.96% |
| DELETE_THROWS_EXCEPTION | 508 | 0.79% |
| ADD_THROWS_EXCEPTION | 206 | 0.32% |

Rare raw classes: {'100': [], '50': [], '20': []}. Counts by split are saved in tables/split_class_distribution.csv. Throws-exception additions/deletions have identical source fragments in this dataset; the pair validity rule excludes them, not a frequency-based class merge.

## Context availability and localization design



{
  "alternate_rows": 16899,
  "schemas": [
    {
      "file": "train-00000-of-00002.parquet",
      "columns": [
        "identifier",
        "condition",
        "bug",
        "fix",
        "post_conditional",
        "bugType",
        "buggy_commit",
        "repository_url",
        "bug_timestamp",
        "num_commits_til_fix",
        "originating_commit"
      ],
      "rows": 8450
    },
    {
      "file": "train-00001-of-00002.parquet",
      "columns": [
        "identifier",
        "condition",
        "bug",
        "fix",
        "post_conditional",
        "bugType",
        "buggy_commit",
        "repository_url",
        "bug_timestamp",
        "num_commits_til_fix",
        "originating_commit"
      ],
      "rows": 8449
    }
  ],
  "index_entries": 12620,
  "ambiguous_source_keys": 481,
  "outcomes": {
    "verified_full_source_join": 12505,
    "no_unique_context_join": 48060,
    "line_number_mismatch": 1821,
    "character_offset_mismatch": 8
  },
  "verified_samples": 12505,
  "genuine_context_available": true,
  "localization_scientifically_supported": true,
  "decision": "Only complete pre-fix sources reconstructed from condition + bug + post_conditional, joined by repository, parent commit, category and exact buggy/fixed fragments, with matching line and character offset. Patches are never predictive inputs. No fabricated context."
}

Primary source fields are fragments. Alternate prefix + original bug + suffix reconstruct real pre-fix files. Only exact joins with verified line and character offset are accepted. Candidate inputs center every candidate using the same context rule; neither original bug index nor a patch is exposed. Other context lines are weak negatives: they are not independently proven bug free. Ranking evaluation is conditional on verified Java source context and does not establish full-repository localization.


## Splits and leakage protections

```json
{
  "strategy": "Repository-exclusive GroupShuffleSplit; deterministic seeds 42..121 search balances proportions and label coverage without looking at model performance. Purge evaluation overlaps against train, then validation; all exclusions documented.",
  "seed": 74,
  "requested_seed": 42,
  "pre_purge_sizes": {
    "train": 44157,
    "validation": 9551,
    "test": 8686
  },
  "sizes": {
    "train": 44157,
    "validation": 7165,
    "test": 7004
  },
  "repositories": {
    "train": 443,
    "validation": 88,
    "test": 89
  },
  "split_exclusions": 4068,
  "project_overlap": 0,
  "duplicate_leakage": 0,
  "validated": true
}
```

Repositories are disjoint. Exact original and repaired fragment hashes, pairs, patches and commits are purged across evaluation boundaries. Purging can change split proportions and makes evaluation conditional on nonduplicated examples. Duplicate sample IDs and label coverage are asserted. Split files are SHA256 checked before each training launch.

## Classical baseline and CodeBERT configuration

Baseline: training-only character TF-IDF (3–5 grams, 50,000 features) and balanced Logistic Regression. CodeBERT is initialized from microsoft/codebert-base, never trained from scratch. No corrected code, diff, project or commit metadata is a model feature.

```json
{
  "pretrained_model": "microsoft/codebert-base",
  "max_length": 256,
  "batch_size": 2,
  "gradient_accumulation": 8,
  "epochs": 3,
  "learning_rate": 2e-05,
  "weight_decay": 0.01,
  "warmup_ratio": 0.1,
  "early_stopping_patience": 2,
  "candidate_radius": 3,
  "source_radius": 15,
  "negative_lines_per_sample": 6,
  "train_final_encoder_layers": 4
}
```

## Frozen and trained layers

Layer indices are zero-based. Embeddings and encoder layers 0–7 are frozen. Only encoder layers 8–11 and each task classifier head (dense and out_proj) are trained. The binary candidate classifier is the localization head. Optimizers include only parameters with requires_grad=True. Non-reentrant gradient checkpointing preserves gradients for the upper layers without an embedding gradient hook.

bug_type:

```json
{
  "encoder_layer_indexing": "zero-based",
  "frozen_embeddings": true,
  "frozen_encoder_layers": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7
  ],
  "trained_encoder_layers": [
    8,
    9,
    10,
    11
  ],
  "trained_head": "classifier (dense and out_proj)",
  "trainable_parameters": 28952846,
  "total_parameters": 124656398,
  "policy": "Only final four encoder layers and task head; no training dataset or split reduction"
}
```

localizer:

```json
{
  "encoder_layer_indexing": "zero-based",
  "frozen_embeddings": true,
  "frozen_encoder_layers": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7
  ],
  "trained_encoder_layers": [
    8,
    9,
    10,
    11
  ],
  "trained_head": "classifier (dense and out_proj)",
  "trainable_parameters": 28943618,
  "total_parameters": 124647170,
  "policy": "Only final four encoder layers and task head; no training dataset or split reduction"
}
```

The full prepared datasets and recorded leakage-safe splits remain unchanged. Training uses up to three epochs, effective batches of 16 with token-bounded microbatches and gradient accumulation, and CUDA FP16 mixed precision. Checkpoint selection uses validation macro F1 for bug types and validation MRR for localization; localization evaluation includes Top-1, Top-3, Top-5 and MRR.

## GPU and hardware

```json
{
  "pytorch": "2.14.1+cu126",
  "cuda_available": true,
  "compiled_cuda": "12.6",
  "gpu": "NVIDIA T1200 Laptop GPU",
  "vram_gb": 3.999755859375,
  "capability": [
    7,
    5
  ],
  "cuda_kernel_check": true
}
```

Pretrained compatibility check (not training/evaluation): {"pretrained_weights_loaded": true, "tokenizer_loaded": true, "cpu_forward_shape": [2, 14], "classification_head": "new random initialization; this is a compatibility check, not a trained classifier or evaluation"}.

## Classification results and model comparison

| Model | Split | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | Top-3 |
|---|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | validation | 0.4096 | 0.3239 | 0.4574 | 0.3359 | 0.4222 | 0.8028 |
| TF-IDF | test | 0.3094 | 0.2959 | 0.3520 | 0.2926 | 0.3425 | 0.7390 |
| CodeBERT | validation | 0.4885 | 0.4590 | 0.5368 | 0.4692 | 0.4879 | 0.9070 |
| CodeBERT | test | 0.3844 | 0.3770 | 0.4376 | 0.3864 | 0.4208 | 0.9121 |

![Baseline confusion matrix](figures/baseline_confusion_matrix.png)

## Localization results

```json
{
  "validation": {
    "ranking": {
      "samples": 1983,
      "top_1": 0.31921331316187596,
      "top_3": 0.6192637418053455,
      "top_5": 0.7826525466464952,
      "mrr": 0.5136246338449586,
      "mean_rank": 3.59505799293999,
      "median_rank": 3.0
    },
    "position_only_diagnostic": {
      "samples": 1983,
      "top_1": 0.8290468986384266,
      "top_3": 0.9389813414019162,
      "top_5": 0.9586485123550177,
      "mrr": 0.891586675777296,
      "mean_rank": 1.481089258698941,
      "median_rank": 1.0
    }
  },
  "test": {
    "ranking": {
      "samples": 1679,
      "top_1": 0.334723049434187,
      "top_3": 0.6503871351995235,
      "top_5": 0.8052412150089339,
      "mrr": 0.5309264026083247,
      "mean_rank": 3.4466944609886836,
      "median_rank": 2.0
    },
    "position_only_diagnostic": {
      "samples": 1679,
      "top_1": 0.7957117331745086,
      "top_3": 0.9481834425253127,
      "top_5": 0.9654556283502085,
      "mrr": 0.8763431558845495,
      "mean_rank": 1.4669446098868375,
      "median_rank": 1.0
    }
  }
}
```

Candidate-line classification metrics are retained in `metrics/localization.json`. The position-only diagnostic materially outperforms the learned model on test (MRR 0.8763 versus 0.5309; Top-1 0.7957 versus 0.3347). This demonstrates strong target-position bias caused by the verified-context extraction procedure. Learned localization scores are reported, but they must not be interpreted as evidence that the model beats this positional shortcut or generalizes to arbitrary source files.

Localization sample/candidate counts after exact-input leakage purging: {"train": {"source_samples": 8352, "candidate_lines": 58461}, "validation": {"source_samples": 1983, "candidate_lines": 42268}, "test": {"source_samples": 1679, "candidate_lines": 34782}}. Excluded source groups: 25. Full-radius windows are required away from true file boundaries, so synthetic crop edges cannot identify the target line. The localizer selects its checkpoint by validation MRR.

## Error analysis and final model

Lowest baseline per-class test F1 values (label, F1, support): [('CHANGE_MODIFIER', 0.0, 0.0), ('CHANGE_CALLER_IN_FUNCTION_CALL', 0.11074918566775244, 230.0), ('LESS_SPECIFIC_IF', 0.11916807195053401, 203.0), ('CHANGE_OPERAND', 0.12727272727272726, 63.0), ('OVERLOAD_METHOD_DELETED_ARGS', 0.12849162011173185, 194.0)].

Most frequent baseline confusions (count, actual, predicted): [(719, 'CHANGE_IDENTIFIER', 'LESS_SPECIFIC_IF'), (681, 'DIFFERENT_METHOD_SAME_ARGS', 'LESS_SPECIFIC_IF'), (608, 'CHANGE_IDENTIFIER', 'DIFFERENT_METHOD_SAME_ARGS'), (477, 'CHANGE_IDENTIFIER', 'OVERLOAD_METHOD_MORE_ARGS'), (169, 'CHANGE_IDENTIFIER', 'CHANGE_CALLER_IN_FUNCTION_CALL')].

Model selection uses validation macro F1, never test performance. Selection: {"bug_type_backend": "codebert", "criterion": "validation macro F1", "baseline_validation_macro_f1": 0.33594726603451763, "transformer_evaluation_complete": true, "codebert_validation_macro_f1": 0.46918486286786276}. Per-class scores and full confusion matrices are saved in metrics JSON.

## Interrupted-run recovery

The saved epoch-2 validation-MRR-selected checkpoint was evaluated without retraining. Epoch 3 did not complete and had no optimizer/scheduler state to resume exactly. Recovery provenance:

```json
{
  "recovery_mode": "evaluate_saved_best_checkpoint_only",
  "training_performed": false,
  "checkpoint_sha256": "0e5463f1c7c80915a76297a896772b576771fe92d7b7678c4e3f4dd85e7270f8",
  "selected_after_completed_epoch": 2,
  "selection_metric": "validation MRR",
  "best_validation_score_at_save": 0.5136246338449586,
  "evaluation_splits": {
    "validation": {
      "candidate_lines": 42268,
      "sha256": "291b8e45a3bc82a681f95a73d4b0ade255fd5f5b7a12c272190bc8ffc4da04bf"
    },
    "test": {
      "candidate_lines": 34782,
      "sha256": "3a5e4eab3190e729492f56de533057a8a0c03d684f7a214e60c7ce3388ca9cc9"
    }
  }
}
```

See `LOCALIZER_RECOVERY_AUDIT.md` for the artifact evidence and decision.

## Automated validation

[{"tests": "27", "failures": "0", "errors": "0", "skipped": "0", "time": "13.614"}]. Skipped transformer loading tests mean those models were not trained and are not validated.

## Scientifically relevant limitations

Training examples are Java single-statement repair patterns from ManySStuBs4J, not representative clean versus defective programs. Category confidence is conditional on the repair classification task and is not calibrated defect probability. Multiple labels can describe the same repair. Primary fields are AST fragments; whole-file category inputs are outside that training distribution. Verified-context localization covers a subset selected by available provenance; surrounding lines are weak negatives, not guaranteed bug-free. Candidate scores depend on negative sampling and are suspiciousness scores, not universally calibrated P(defective line). Context selection can induce position artifacts; position-only ranking is evaluated when training completes. CodeBERT truncation can lose relevant context. No claims of universal accuracy for Python, JavaScript, TypeScript, C#, multi-line defects, or exact faulty-line identification are made.

## RiskNet integration

CodeLocator is an independent Python API. A future QASense application can call analyze_code alongside the existing RiskNet interface. RiskNet, frontend, automation and multimodal fusion were not changed.

## Reproduction

From C:\QASense-AI:

```powershell
.\.venv-codelocator\Scripts\python.exe scripts/preprocess_codelocator.py
.\.venv-codelocator\Scripts\python.exe scripts/prepare_codebert.py
.\.venv-codelocator\Scripts\python.exe scripts/train_codelocator.py
.\.venv-codelocator\Scripts\python.exe scripts/infer_codelocator.py
.\.venv-codelocator\Scripts\python.exe -m pytest tests/codelocator -q
```

For Windows native-library blocks, use scripts/setup_codelocator_wsl.sh in a user Ubuntu WSL2 environment; no repeated package downgrades are required.
