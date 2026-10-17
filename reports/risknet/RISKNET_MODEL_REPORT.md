# QASense RiskNet v1 model report

## Objective and dataset overview

Predict defective modules from 21 static metrics. Projects are metadata, never predictors. All 17442 source rows are retained.

## Data quality and class imbalance

See [data audit](data_audit.md) for per-column statistics, invalid tokens, outliers, correlations, duplicates and class counts. Combined findings: {'rows': 17442, 'unique_feature_groups': 11690, 'duplicate_feature_groups': 2532, 'duplicate_feature_rows_beyond_first': 5752, 'conflicting_label_groups': 121, 'cross_project_duplicate_groups': 66, 'cross_project_rows': 1701}

## Preprocessing and leakage precautions

Explicit case-insensitive aliases are collision checked, targets validated, numeric invalid tokens/infinities converted to missing. Training-only median imputation and scaling live inside pipelines. No SMOTE. Exact feature vectors are SHA256 grouped, including missing values. Conflicting labels are preserved. Within-project splits use stratified group folds; LOPO purges training matches to test features. Split indices and purge counts are saved. StratifiedGroupKFold is used instead of StratifiedKFold because duplicate groups must stay together. Class weights use training labels; XGBoost scale_pos_weight is recomputed from labels on every fit, including inner folds.

## Models tested and within/cross-project results

Logistic regression, random forest and XGBoost use class weighting. Baseline decision threshold is 0.5. AP is average precision, not trapezoidal PR area.

| protocol | project | model | precision | recall | f1 | roc_auc | pr_auc | mcc | brier |
|---|---|---|---|---|---|---|---|---|---|
| within | CM1 | logistic | 0.2143 | 0.6000 | 0.3158 | 0.7333 | 0.3038 | 0.2376 | 0.1683 |
| within | CM1 | random_forest | 0.3333 | 0.1000 | 0.1538 | 0.6867 | 0.1959 | 0.1368 | 0.0968 |
| within | CM1 | xgboost | 0.2667 | 0.4000 | 0.3200 | 0.6794 | 0.3785 | 0.2334 | 0.1166 |
| cross | CM1 | logistic | 0.1733 | 0.7143 | 0.2789 | 0.7455 | 0.2477 | 0.2077 | 0.2459 |
| cross | CM1 | random_forest | 0.3077 | 0.0816 | 0.1290 | 0.7151 | 0.2149 | 0.1150 | 0.1087 |
| cross | CM1 | xgboost | 0.2269 | 0.5510 | 0.3214 | 0.7470 | 0.2568 | 0.2417 | 0.1886 |
| within | JM1 | logistic | 0.2991 | 0.5748 | 0.3935 | 0.7302 | 0.3770 | 0.2537 | 0.2010 |
| within | JM1 | random_forest | 0.4266 | 0.2969 | 0.3501 | 0.7349 | 0.3796 | 0.2579 | 0.1309 |
| within | JM1 | xgboost | 0.3017 | 0.6342 | 0.4089 | 0.7412 | 0.3887 | 0.2759 | 0.1967 |
| cross | JM1 | logistic | 0.2715 | 0.6115 | 0.3760 | 0.6972 | 0.3267 | 0.2294 | 0.2319 |
| cross | JM1 | random_forest | 0.3335 | 0.2577 | 0.2908 | 0.6893 | 0.3042 | 0.1784 | 0.1467 |
| cross | JM1 | xgboost | 0.2317 | 0.6034 | 0.3348 | 0.6679 | 0.2944 | 0.1666 | 0.2268 |
| within | KC1 | logistic | 0.3354 | 0.8154 | 0.4753 | 0.8462 | 0.5576 | 0.3888 | 0.1699 |
| within | KC1 | random_forest | 0.5179 | 0.4462 | 0.4793 | 0.8492 | 0.5441 | 0.3943 | 0.1066 |
| within | KC1 | xgboost | 0.3310 | 0.7385 | 0.4571 | 0.8187 | 0.5118 | 0.3548 | 0.1629 |
| cross | KC1 | logistic | 0.3933 | 0.4294 | 0.4106 | 0.7809 | 0.3844 | 0.2975 | 0.1509 |
| cross | KC1 | random_forest | 0.4174 | 0.1472 | 0.2177 | 0.7399 | 0.3207 | 0.1746 | 0.1225 |
| cross | KC1 | xgboost | 0.3673 | 0.4417 | 0.4011 | 0.7645 | 0.3551 | 0.2812 | 0.1410 |
| within | KC2 | logistic | 0.3864 | 0.7727 | 0.5152 | 0.7996 | 0.5602 | 0.3690 | 0.2028 |
| within | KC2 | random_forest | 0.4333 | 0.5909 | 0.5000 | 0.7842 | 0.5537 | 0.3478 | 0.1573 |
| within | KC2 | xgboost | 0.4000 | 0.6364 | 0.4912 | 0.7842 | 0.6046 | 0.3310 | 0.1850 |
| cross | KC2 | logistic | 0.5000 | 0.7944 | 0.6137 | 0.8445 | 0.6318 | 0.5079 | 0.1587 |
| cross | KC2 | random_forest | 0.6585 | 0.2523 | 0.3649 | 0.8060 | 0.5658 | 0.3280 | 0.1273 |
| cross | KC2 | xgboost | 0.4859 | 0.8037 | 0.6056 | 0.8465 | 0.6535 | 0.4984 | 0.1557 |
| within | PC1 | logistic | 0.2245 | 0.7333 | 0.3438 | 0.8272 | 0.2783 | 0.3327 | 0.1623 |
| within | PC1 | random_forest | 0.5714 | 0.2667 | 0.3636 | 0.8211 | 0.3327 | 0.3622 | 0.0563 |
| within | PC1 | xgboost | 0.3125 | 0.3333 | 0.3226 | 0.8111 | 0.3116 | 0.2720 | 0.0696 |
| cross | PC1 | logistic | 0.1658 | 0.4286 | 0.2391 | 0.6848 | 0.2092 | 0.1773 | 0.1813 |
| cross | PC1 | random_forest | 0.2391 | 0.1429 | 0.1789 | 0.7058 | 0.2087 | 0.1389 | 0.0962 |
| cross | PC1 | xgboost | 0.1847 | 0.5974 | 0.2822 | 0.7394 | 0.2283 | 0.2441 | 0.1785 |

## Final model selection and tuning

Selected logistic by equal-project mean cross-project AP, then recall and MCC. Four randomized candidates use three stratified group folds. Nested LOPO repeats tuning using only outer training projects; final artifact refits on all rows. Family selection used the benchmark projects, so reported results retain selection bias. No independent external validation exists.

| protocol | project | model | precision | recall | f1 | roc_auc | pr_auc | mcc | brier |
|---|---|---|---|---|---|---|---|---|---|
| nested_cross_sigmoid | CM1 | logistic | 0.2479 | 0.5918 | 0.3494 | 0.7497 | 0.2491 | 0.2781 | 0.0953 |
| nested_cross_sigmoid | JM1 | logistic | 0.3284 | 0.4684 | 0.3861 | 0.6972 | 0.3267 | 0.2506 | 0.1295 |
| nested_cross_sigmoid | KC1 | logistic | 0.4358 | 0.2914 | 0.3493 | 0.7791 | 0.3829 | 0.2641 | 0.1185 |
| nested_cross_sigmoid | KC2 | logistic | 0.5804 | 0.6075 | 0.5936 | 0.8464 | 0.6431 | 0.4860 | 0.1266 |
| nested_cross_sigmoid | PC1 | logistic | 0.1765 | 0.2727 | 0.2143 | 0.6955 | 0.2176 | 0.1460 | 0.0650 |

## Probability calibration

Sigmoid calibration uses group-separated out-of-fold training predictions, never outer test labels. Weighted raw scores need calibration; raw and calibrated Brier/log loss are saved per project. Isotonic was omitted because small-project positives make it unstable. Calibration transfer to new projects is uncertain.

![Calibration](figures/calibration.png)

## Confusion matrices

Rows are actual classes, columns predicted classes, ordered 0/1.

![Matrices](figures/confusion_matrices.png)

## Feature importance

Permutation AP decrease averaged equally across held-out projects; correlations can dilute importance. This is global importance, not individual causal attribution.

![Importance](figures/feature_importance.png)

          feature  mean_ap_decrease  project_std
              loc          0.208017     0.101698
         total_Op          0.071338     0.036809
      branchCount          0.066848     0.039079
        uniq_Opnd          0.052790     0.027226
          uniq_Op          0.024273     0.012190
       total_Opnd          0.019432     0.024346
                n          0.017046     0.013631
                l          0.013985     0.011336
                i          0.013467     0.016209
             v(g)          0.010863     0.017279
           lOCode          0.006794     0.007017
                d          0.006601     0.009494
                b          0.006281     0.006345
                v          0.005739     0.006019
locCodeAndComment          0.003165     0.010880
            iv(g)          0.002326     0.005614
                e          0.001531     0.003894
                t          0.001410     0.003651
            ev(g)          0.001004     0.003076
          lOBlank         -0.000983     0.002319
        lOComment         -0.009880     0.013416

## Known limitations and QASense integration

NASA labels are historical, duplicates can carry contradictory labels, metrics are correlated, and these results do not establish performance on modern repositories. A single deterministic within-project split does not measure split sensitivity; cross-project standard deviations describe domain instability. Final calibrated probabilities may still drift. Risk thresholds 0.3/0.6/0.8 and priorities P4/P3/P2/P1 are configurable product defaults, not validated scientific cutoffs. Defect classification uses a separate configurable 0.2 threshold, selected before training to prioritize recall because missed defects are expensive. It is a product operating assumption, not an optimized threshold. Metrics at 0.1/0.2/0.3/0.5 are saved for comparison; test labels were not used to choose the threshold. Inference validates all 21 metrics and returns probability, classification and priority. For logistic models, inference returns positive local calibrated log-odds contributions relative to training feature means; other families return explicitly labeled global importance. Contributions explain the fitted function, not causation. Other QASense modules were not changed.

## Reproduction

```powershell
python -m venv .venv-risknet
.\.venv-risknet\Scripts\python.exe -m pip install -r requirements-risknet.txt
.\.venv-risknet\Scripts\python.exe scripts/train_risknet.py
.\.venv-risknet\Scripts\python.exe -m pytest tests/risknet -q
.\.venv-risknet\Scripts\python.exe scripts/infer_risknet.py
```

## Cross-project stability

                 recall                  f1              pr_auc                 mcc          
                   mean       std      mean       std      mean       std      mean       std
model                                                                                        
logistic       0.595641  0.165350  0.383663  0.146266  0.359949  0.166532  0.283958  0.132751
random_forest  0.176359  0.076382  0.236244  0.093070  0.322876  0.144965  0.186990  0.083078
xgboost        0.599461  0.131319  0.389043  0.128435  0.357609  0.172088  0.286420  0.125596

<!-- persisted-summary -->

## Dataset counts

| Project | Rows | Nondefective | Defective | Defect rate |
|---|---:|---:|---:|---:|
| CM1 | 498 | 449 | 49 | 9.84% |
| JM1 | 13204 | 11101 | 2103 | 15.93% |
| KC1 | 2109 | 1783 | 326 | 15.46% |
| KC2 | 522 | 415 | 107 | 20.50% |
| PC1 | 1109 | 1032 | 77 | 6.94% |

## Calibrated operating point and probability quality

The default decision threshold is 0.20. Baselines use 0.50; threshold-free ROC-AUC and AP remain the appropriate ranking comparison. Equal-project nested means: precision=0.3538, recall=0.4464, f1=0.3785, roc_auc=0.7536, pr_auc=0.3639, mcc=0.2850, brier=0.1070.

| Project | Raw weighted Brier | Sigmoid Brier |
|---|---:|---:|
| CM1 | 0.2397 | 0.0953 |
| JM1 | 0.2319 | 0.1295 |
| KC1 | 0.1506 | 0.1185 |
| KC2 | 0.1569 | 0.1266 |
| PC1 | 0.1878 | 0.0650 |

Sigmoid calibration improved Brier score on all five held-out projects. The reliability plot still shows overconfidence in higher probability bins, so high-risk probabilities should not be treated as guarantees. Mean recall is below 50%; RiskNet is suitable as an experimental prioritization signal, not a sole QA acceptance gate. Duplicate weighting and project shifts remain material limitations. The 0.20 threshold was specified before baseline results, not optimized using held-out labels.

## Baseline selection evidence

| Model | Mean cross-project AP | Recall at 0.50 | MCC at 0.50 |
|---|---:|---:|---:|
| logistic | 0.3599 | 0.5956 | 0.2840 |
| xgboost | 0.3576 | 0.5995 | 0.2864 |
| random_forest | 0.3229 | 0.1764 | 0.1870 |

## Final artifact coefficients

These coefficients come from the saved final model. Each is the calibrated log-odds change for a one-standard-deviation feature increase, holding the other metrics fixed. Signs and magnitudes can be unstable with correlated predictors; they are not causal effects. Global held-out permutation importance above is a separate generalization measure.

| Feature | Calibrated log-odds coefficient |
|---|---:|
| loc | 1.1948 |
| v | -1.0247 |
| b | 0.6118 |
| total_Opnd | -0.5945 |
| branchCount | 0.5026 |
| v(g) | -0.4807 |
| total_Op | 0.3833 |
| lOCode | -0.3127 |
| uniq_Opnd | 0.2417 |
| l | -0.2406 |
