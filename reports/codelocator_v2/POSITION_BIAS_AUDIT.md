# CodeLocator v2 position-bias audit

## Root cause in v1

v1 first extracted a 31-line source window centered on the known bug (15 lines before and after). It then removed candidates near artificial crop edges. Consequently, 94.10% of validation and 94.70% of test bugs occupied the middle relative-position bucket. Candidate-specific seven-line model contexts were symmetric too, while the ranking candidate set retained the centered-source construction. The position-only diagnostic therefore ranked the known midpoint and achieved test MRR 0.8763.

## Actual distributions

Buckets are `[0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,1]`.

```json
{
  "v1": {
    "train": {
      "samples": 8352,
      "bucket_counts": [
        163,
        1050,
        5358,
        1405,
        376
      ],
      "bucket_percentages": [
        0.019516283524904216,
        0.1257183908045977,
        0.6415229885057471,
        0.16822318007662834,
        0.045019157088122604
      ],
      "mean": 0.5128186187722494,
      "median": 0.5,
      "standard_deviation": 0.14553414412586937,
      "center_window_percentage": 0.6415229885057471,
      "exact_middle_percentage": 0.13493773946360152
    },
    "validation": {
      "samples": 1983,
      "bucket_counts": [
        0,
        2,
        1868,
        72,
        41
      ],
      "bucket_percentages": [
        0.0,
        0.0010085728693898135,
        0.9420070600100857,
        0.036308623298033284,
        0.020675743822491176
      ],
      "mean": 0.5133319743310489,
      "median": 0.5,
      "standard_deviation": 0.061748881003105535,
      "center_window_percentage": 0.9420070600100857,
      "exact_middle_percentage": 0.6878466969238527
    },
    "test": {
      "samples": 1679,
      "bucket_counts": [
        0,
        3,
        1595,
        58,
        23
      ],
      "bucket_percentages": [
        0.0,
        0.001786777843954735,
        0.9499702203692674,
        0.034544371649791544,
        0.0136986301369863
      ],
      "mean": 0.5094514760131937,
      "median": 0.5,
      "standard_deviation": 0.0536639946473214,
      "center_window_percentage": 0.9499702203692674,
      "exact_middle_percentage": 0.6402620607504467
    }
  },
  "v2": {
    "train": {
      "samples": 5707,
      "bucket_counts": [
        1465,
        1385,
        1096,
        965,
        796
      ],
      "bucket_percentages": [
        0.256702295426669,
        0.24268442263886456,
        0.19204485719292097,
        0.16909059050289119,
        0.13947783423865429
      ],
      "mean": 0.42896159141639584,
      "median": 0.4,
      "standard_deviation": 0.28293256423402424,
      "center_window_percentage": 0.19204485719292097,
      "exact_middle_percentage": 0.04433152269143158
    },
    "validation": {
      "samples": 980,
      "bucket_counts": [
        222,
        223,
        214,
        169,
        152
      ],
      "bucket_percentages": [
        0.22653061224489796,
        0.22755102040816327,
        0.21836734693877552,
        0.17244897959183675,
        0.15510204081632653
      ],
      "mean": 0.4506001108100666,
      "median": 0.4422222222222222,
      "standard_deviation": 0.2756698836051605,
      "center_window_percentage": 0.21836734693877552,
      "exact_middle_percentage": 0.04183673469387755
    },
    "test": {
      "samples": 854,
      "bucket_counts": [
        215,
        197,
        187,
        145,
        110
      ],
      "bucket_percentages": [
        0.25175644028103045,
        0.2306791569086651,
        0.2189695550351288,
        0.16978922716627634,
        0.1288056206088993
      ],
      "mean": 0.42581074113301215,
      "median": 0.4090909090909091,
      "standard_deviation": 0.27472836149862734,
      "center_window_percentage": 0.2189695550351288,
      "exact_middle_percentage": 0.03747072599531616
    }
  }
}
```

## v2 construction

Full pre-fix sources were rejoined using the same exact repository, parent-commit, category, buggy-fragment and fixed-fragment identity checks. v1 split membership was preserved. v2 chooses the containing method when detected, otherwise a containing block. Regions longer than 61 lines use a deterministic hash-assigned crop position across five target buckets. Training keeps one view and six executable hard negatives chosen by identifier/operator similarity; validation and test retain all eligible lines. Candidate rows are deterministically shuffled. Model inputs contain only candidate code plus shared natural-region code.

## Baselines and gate

```json
{
  "counts": {
    "train": {
      "source_bugs": 5707,
      "candidate_lines": 38810,
      "positives": 5707,
      "negatives": 33103,
      "positive_negative_ratio": 0.17240129293417514,
      "context_length_lines": {
        "min": 5,
        "max": 61,
        "mean": 30.585287297088378
      }
    },
    "validation": {
      "source_bugs": 980,
      "candidate_lines": 15975,
      "positives": 980,
      "negatives": 14995,
      "positive_negative_ratio": 0.06535511837279093,
      "context_length_lines": {
        "min": 5,
        "max": 61,
        "mean": 35.39511737089202
      }
    },
    "test": {
      "source_bugs": 854,
      "candidate_lines": 14381,
      "positives": 854,
      "negatives": 13527,
      "positive_negative_ratio": 0.06313299327271384,
      "context_length_lines": {
        "min": 5,
        "max": 61,
        "mean": 37.464501773172934
      }
    }
  },
  "position": {
    "train": {
      "samples": 5707,
      "top_1": 0.1366742596810934,
      "top_3": 0.4249167688803224,
      "top_5": 0.7364639915892763,
      "mrr": 0.3665853129406661,
      "mean_rank": 3.973190818293324,
      "median_rank": 4.0
    },
    "validation": {
      "samples": 980,
      "top_1": 0.10816326530612246,
      "top_3": 0.3193877551020408,
      "top_5": 0.4806122448979592,
      "mrr": 0.283042316311182,
      "mean_rank": 8.58265306122449,
      "median_rank": 6.0
    },
    "test": {
      "samples": 854,
      "top_1": 0.11358313817330211,
      "top_3": 0.3044496487119438,
      "top_5": 0.4812646370023419,
      "mrr": 0.2826210522022098,
      "mean_rank": 8.795081967213115,
      "median_rank": 6.0
    }
  },
  "random": {
    "train": {
      "seeds": [
        42,
        43,
        44,
        45,
        46
      ],
      "runs": [
        {
          "samples": 5707,
          "top_1": 0.15156824951813563,
          "top_3": 0.45049938671806555,
          "top_5": 0.7483791834589101,
          "mrr": 0.38237419376371545,
          "mean_rank": 3.8771683896968634,
          "median_rank": 4.0
        },
        {
          "samples": 5707,
          "top_1": 0.15016646223935518,
          "top_3": 0.4564569826528824,
          "top_5": 0.7499561941475381,
          "mrr": 0.38242008560915175,
          "mean_rank": 3.8613982828105833,
          "median_rank": 4.0
        },
        {
          "samples": 5707,
          "top_1": 0.15139302610828806,
          "top_3": 0.4559313124233398,
          "top_5": 0.7457508323111968,
          "mrr": 0.38400836065984134,
          "mean_rank": 3.8519362186788153,
          "median_rank": 4.0
        },
        {
          "samples": 5707,
          "top_1": 0.15349570702645873,
          "top_3": 0.44681969511126685,
          "top_5": 0.7387418959172946,
          "mrr": 0.38202040935526127,
          "mean_rank": 3.9011739968459787,
          "median_rank": 4.0
        },
        {
          "samples": 5707,
          "top_1": 0.14753811109164183,
          "top_3": 0.43823374802873666,
          "top_5": 0.7476782898195199,
          "mrr": 0.37604195349070063,
          "mean_rank": 3.912388295076222,
          "median_rank": 4.0
        }
      ],
      "mean": {
        "samples": 5707.0,
        "top_1": 0.1508323111967759,
        "top_3": 0.44958822498685824,
        "top_5": 0.746101279130892,
        "mrr": 0.3813730005757341,
        "mean_rank": 3.880813036621692,
        "median_rank": 4.0
      },
      "std": {
        "samples": 0.0,
        "top_1": 0.0019618762925920352,
        "top_3": 0.0067057245763062585,
        "top_5": 0.003919681489183226,
        "mrr": 0.0027525660315576785,
        "mean_rank": 0.02295932219811389,
        "median_rank": 0.0
      }
    },
    "validation": {
      "seeds": [
        42,
        43,
        44,
        45,
        46
      ],
      "runs": [
        {
          "samples": 980,
          "top_1": 0.08673469387755102,
          "top_3": 0.2714285714285714,
          "top_5": 0.44693877551020406,
          "mrr": 0.25532111244697175,
          "mean_rank": 8.846938775510203,
          "median_rank": 6.0
        },
        {
          "samples": 980,
          "top_1": 0.09693877551020408,
          "top_3": 0.2571428571428571,
          "top_5": 0.4326530612244898,
          "mrr": 0.2603632938341151,
          "mean_rank": 8.66530612244898,
          "median_rank": 6.0
        },
        {
          "samples": 980,
          "top_1": 0.09081632653061225,
          "top_3": 0.286734693877551,
          "top_5": 0.46224489795918366,
          "mrr": 0.26408734122701516,
          "mean_rank": 8.596938775510203,
          "median_rank": 6.0
        },
        {
          "samples": 980,
          "top_1": 0.08979591836734693,
          "top_3": 0.26326530612244897,
          "top_5": 0.44081632653061226,
          "mrr": 0.25711499324572745,
          "mean_rank": 8.905102040816326,
          "median_rank": 6.0
        },
        {
          "samples": 980,
          "top_1": 0.07755102040816327,
          "top_3": 0.2612244897959184,
          "top_5": 0.44285714285714284,
          "mrr": 0.2500667576645452,
          "mean_rank": 8.654081632653062,
          "median_rank": 6.0
        }
      ],
      "mean": {
        "samples": 980.0,
        "top_1": 0.08836734693877549,
        "top_3": 0.2679591836734694,
        "top_5": 0.4451020408163265,
        "mrr": 0.2573906996836749,
        "mean_rank": 8.733673469387755,
        "median_rank": 6.0
      },
      "std": {
        "samples": 0.0,
        "top_1": 0.006342967456293899,
        "top_3": 0.010477957267749742,
        "top_5": 0.009753308690080413,
        "mrr": 0.0047261116367061295,
        "mean_rank": 0.1199350789261333,
        "median_rank": 0.0
      }
    },
    "test": {
      "seeds": [
        42,
        43,
        44,
        45,
        46
      ],
      "runs": [
        {
          "samples": 854,
          "top_1": 0.08899297423887588,
          "top_3": 0.28805620608899296,
          "top_5": 0.4672131147540984,
          "mrr": 0.2618031234525799,
          "mean_rank": 8.855971896955504,
          "median_rank": 6.0
        },
        {
          "samples": 854,
          "top_1": 0.09953161592505855,
          "top_3": 0.265807962529274,
          "top_5": 0.45081967213114754,
          "mrr": 0.26339134987604157,
          "mean_rank": 8.711943793911008,
          "median_rank": 6.0
        },
        {
          "samples": 854,
          "top_1": 0.06323185011709602,
          "top_3": 0.2576112412177986,
          "top_5": 0.4309133489461358,
          "mrr": 0.23579700442407486,
          "mean_rank": 8.985948477751757,
          "median_rank": 7.0
        },
        {
          "samples": 854,
          "top_1": 0.0702576112412178,
          "top_3": 0.2540983606557377,
          "top_5": 0.43559718969555034,
          "mrr": 0.24121649331146722,
          "mean_rank": 9.2096018735363,
          "median_rank": 6.0
        },
        {
          "samples": 854,
          "top_1": 0.08548009367681499,
          "top_3": 0.25175644028103045,
          "top_5": 0.43676814988290397,
          "mrr": 0.24790311937933945,
          "mean_rank": 9.185011709601874,
          "median_rank": 6.5
        }
      ],
      "mean": {
        "samples": 854.0,
        "top_1": 0.08149882903981265,
        "top_3": 0.26346604215456676,
        "top_5": 0.4442622950819672,
        "mrr": 0.2500222180887006,
        "mean_rank": 8.98969555035129,
        "median_rank": 6.3
      },
      "std": {
        "samples": 0.0,
        "top_1": 0.013093827399057952,
        "top_3": 0.01318565644989156,
        "top_5": 0.013260316315589772,
        "mrr": 0.010971886692238699,
        "mean_rank": 0.19055145618829256,
        "median_rank": 0.39999999999999997
      }
    }
  },
  "integrity": {
    "repository_overlap": true,
    "sample_id_overlap": true,
    "model_input_overlap": true,
    "duplicate_model_inputs_within_groups": false,
    "one_positive_per_group": true,
    "forbidden_predictive_fields_absent": true,
    "excluded_groups": 2199,
    "exclusion_reasons": {
      "duplicate_model_input_within_candidate_group": 1689,
      "exact_v2_model_input_overlaps_earlier_split": 510
    }
  },
  "gate": {
    "passed": true,
    "training_allowed": false,
    "requires_user_review": true,
    "decision": "STOP_BEFORE_GPU_TRAINING",
    "failures": [],
    "thresholds": {
      "position_test_mrr_max": 0.75,
      "middle_bucket_max": 0.4,
      "single_bucket_max": 0.45
    },
    "counts": {
      "train": {
        "source_bugs": 5707,
        "candidate_lines": 38810,
        "positives": 5707,
        "negatives": 33103,
        "positive_negative_ratio": 0.17240129293417514,
        "context_length_lines": {
          "min": 5,
          "max": 61,
          "mean": 30.585287297088378
        }
      },
      "validation": {
        "source_bugs": 980,
        "candidate_lines": 15975,
        "positives": 980,
        "negatives": 14995,
        "positive_negative_ratio": 0.06535511837279093,
        "context_length_lines": {
          "min": 5,
          "max": 61,
          "mean": 35.39511737089202
        }
      },
      "test": {
        "source_bugs": 854,
        "candidate_lines": 14381,
        "positives": 854,
        "negatives": 13527,
        "positive_negative_ratio": 0.06313299327271384,
        "context_length_lines": {
          "min": 5,
          "max": 61,
          "mean": 37.464501773172934
        }
      }
    },
    "integrity": {
      "repository_overlap": true,
      "sample_id_overlap": true,
      "model_input_overlap": true,
      "duplicate_model_inputs_within_groups": false,
      "one_positive_per_group": true,
      "forbidden_predictive_fields_absent": true,
      "excluded_groups": 2199,
      "exclusion_reasons": {
        "duplicate_model_input_within_candidate_group": 1689,
        "exact_v2_model_input_overlaps_earlier_split": 510
      }
    },
    "position_test": {
      "samples": 854,
      "top_1": 0.11358313817330211,
      "top_3": 0.3044496487119438,
      "top_5": 0.4812646370023419,
      "mrr": 0.2826210522022098,
      "mean_rank": 8.795081967213115,
      "median_rank": 6.0
    },
    "random_test": {
      "samples": 854.0,
      "top_1": 0.08149882903981265,
      "top_3": 0.26346604215456676,
      "top_5": 0.4442622950819672,
      "mrr": 0.2500222180887006,
      "mean_rank": 8.98969555035129,
      "median_rank": 6.3
    }
  }
}
```

GPU training was not started. The gate always stops for user inspection, and it also records any threshold failures.
