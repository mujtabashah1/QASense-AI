# VisionQA AtomBlock YOLO11n Model Report

## Outcome

VisionQA trained YOLO11n for 30 completed epochs on the fixed AtomBlock split. Epoch 28 was selected by validation mAP50-95. The selected checkpoint was evaluated once on the held-out test split and promoted to `C:\QASense-AI\models\visionqa\atomblock_yolo11n\best.pt`.

## Dataset and integrity

- Dataset: `C:\QASense-AI\datasets\processed\visionqa_atomblock`
- Configuration: `C:\QASense-AI\datasets\processed\visionqa_atomblock\data.yaml`
- Train: 5,422 images / 5,422 labels
- Validation: 1,388 images / 1,388 labels
- Test: 1,389 images / 1,389 labels
- Classes: 14; label audit: PASS
- Training boxes: 760,188; empty training labels: 8
- Loader-usable training images: 5,420; corrupt exclusions: 2

The two excluded PNGs are physically truncated in both the original and prepared datasets. They were preserved unchanged. Ultralytics also removes exact duplicate annotation rows in memory; source labels and split membership remain unchanged.

### Training class distribution

| Class | Boxes | Share |
|---|---:|---:|
| button | 65,302 | 8.59% |
| link | 353,642 | 46.52% |
| input | 10,634 | 1.40% |
| image | 113,039 | 14.87% |
| icon | 102,599 | 13.50% |
| checkbox | 29,823 | 3.92% |
| radio | 2,609 | 0.34% |
| select | 12,207 | 1.61% |
| block-nav | 6,387 | 0.84% |
| block-sidebar | 2,029 | 0.27% |
| block-footer | 5,087 | 0.67% |
| block-form | 1,256 | 0.17% |
| block-table | 416 | 0.05% |
| block | 55,158 | 7.26% |

## Environment

- Python: 3.14.8
- PyTorch: 2.14.1+cu126
- TorchVision: 0.29.1+cu126
- PyTorch CUDA runtime: 12.6
- Ultralytics: 8.4.176
- GPU: NVIDIA T1200 Laptop GPU (4,096 MiB)

CUDA tensor execution and YOLO11n inference passed. AMP was requested, but Ultralytics disabled it after its runtime safety check on this GPU; valid training used FP32.

## Smoke test and throughput benchmark

| Run | Epochs | Batch | Precision | Recall | mAP50 | mAP50-95 | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Smoke | 1 | 2 | 0.4292 | 0.2553 | 0.2122 | 0.1382 | 1345 MiB |
| Benchmark | 3 | 4 | 0.4248 | 0.3156 | 0.2769 | 0.1907 | 2501 MiB |

The three-epoch benchmark averaged 7.20 minutes per epoch (6.98, 7.31, 7.31 minutes). Local training was selected because this is below the 15–20 minute guideline. At the benchmark rate, 30 epochs are about 3.60 hours and 50 epochs about 6.00 hours. Batch 2 was retained for the main run because its measured smoke epoch was faster than batch 4; VRAM size alone did not determine the choice.

## Main training

Configuration: pretrained YOLO11n, 30/30 completed epochs, batch 2, image size 640, seed 42, patience 10, max_det 1500, checkpoint every epoch.

| Epoch | Train box | Train cls | Train DFL | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.7670 | 1.8977 | 1.2201 | 0.4352 | 0.2528 | 0.2214 | 0.1466 |
| 2 | 1.5109 | 1.4027 | 1.0839 | 0.4977 | 0.2923 | 0.2643 | 0.1791 |
| 3 | 1.4373 | 1.3273 | 1.0484 | 0.4583 | 0.3122 | 0.2793 | 0.1896 |
| 4 | 1.3820 | 1.2736 | 1.0242 | 0.4544 | 0.3149 | 0.2937 | 0.1992 |
| 5 | 1.3549 | 1.2575 | 1.0142 | 0.4474 | 0.3400 | 0.3080 | 0.2119 |
| 6 | 1.3284 | 1.2248 | 1.0016 | 0.5156 | 0.3513 | 0.3154 | 0.2198 |
| 7 | 1.2975 | 1.2065 | 0.9955 | 0.4651 | 0.3592 | 0.3170 | 0.2198 |
| 8 | 1.2720 | 1.1861 | 0.9853 | 0.4074 | 0.3632 | 0.3337 | 0.2379 |
| 9 | 1.2617 | 1.1649 | 0.9800 | 0.4109 | 0.3772 | 0.3403 | 0.2432 |
| 10 | 1.2521 | 1.1584 | 0.9732 | 0.4869 | 0.3828 | 0.3488 | 0.2489 |
| 11 | 1.2271 | 1.1398 | 0.9653 | 0.4312 | 0.3759 | 0.3534 | 0.2533 |
| 12 | 1.2134 | 1.1299 | 0.9634 | 0.5143 | 0.3713 | 0.3556 | 0.2542 |
| 13 | 1.2068 | 1.1165 | 0.9595 | 0.4529 | 0.3803 | 0.3589 | 0.2603 |
| 14 | 1.1941 | 1.1033 | 0.9558 | 0.5256 | 0.3809 | 0.3592 | 0.2595 |
| 15 | 1.1857 | 1.0953 | 0.9519 | 0.4414 | 0.4021 | 0.3739 | 0.2731 |
| 16 | 1.1770 | 1.0846 | 0.9488 | 0.4573 | 0.3903 | 0.3704 | 0.2703 |
| 17 | 1.1692 | 1.0751 | 0.9440 | 0.4572 | 0.4035 | 0.3742 | 0.2737 |
| 18 | 1.1598 | 1.0716 | 0.9430 | 0.4394 | 0.4078 | 0.3750 | 0.2724 |
| 19 | 1.1570 | 1.0636 | 0.9398 | 0.4715 | 0.3959 | 0.3747 | 0.2741 |
| 20 | 1.1522 | 1.0660 | 0.9361 | 0.4599 | 0.4009 | 0.3793 | 0.2782 |
| 21 | 1.1434 | 1.0664 | 0.9313 | 0.4662 | 0.4034 | 0.3750 | 0.2750 |
| 22 | 1.1311 | 1.0424 | 0.9250 | 0.4614 | 0.3995 | 0.3786 | 0.2807 |
| 23 | 1.1245 | 1.0357 | 0.9216 | 0.4794 | 0.3913 | 0.3782 | 0.2778 |
| 24 | 1.1131 | 1.0287 | 0.9205 | 0.4814 | 0.4082 | 0.3891 | 0.2875 |
| 25 | 1.1011 | 1.0164 | 0.9163 | 0.4717 | 0.4103 | 0.3881 | 0.2853 |
| 26 | 1.1013 | 1.0126 | 0.9145 | 0.4795 | 0.4008 | 0.3875 | 0.2866 |
| 27 | 1.0993 | 1.0094 | 0.9120 | 0.4698 | 0.4115 | 0.3871 | 0.2857 |
| 28 **(selected)** | 1.0833 | 0.9940 | 0.9084 | 0.4844 | 0.4111 | 0.3933 | 0.2902 |
| 29 | 1.0812 | 0.9887 | 0.9064 | 0.4763 | 0.4092 | 0.3923 | 0.2900 |
| 30 | 1.0747 | 0.9804 | 0.9063 | 0.4872 | 0.4026 | 0.3912 | 0.2899 |

Training was resumable from full optimizer-bearing epoch checkpoints. `last.pt`, `best.pt`, training arguments, results, and plots remain in the original run directory.

## Held-out test evaluation

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.4738 | 0.4013 | 0.3775 | 0.2783 |

### Per-class test metrics

| Class | Precision | Recall | AP50 | AP50-95 |
|---|---:|---:|---:|---:|
| button | 0.5809 | 0.3688 | 0.3874 | 0.2589 |
| link | 0.4429 | 0.3241 | 0.2805 | 0.1279 |
| input | 0.5016 | 0.3266 | 0.3060 | 0.1841 |
| image | 0.8235 | 0.7525 | 0.8018 | 0.6534 |
| icon | 0.4770 | 0.0449 | 0.0588 | 0.0206 |
| checkbox | 0.2046 | 0.0789 | 0.0393 | 0.0100 |
| **radio** | 0.1935 | 0.0014 | 0.0129 | 0.0042 |
| select | 0.4211 | 0.1179 | 0.1477 | 0.0951 |
| **block-nav** | 0.5265 | 0.7545 | 0.6683 | 0.4546 |
| **block-sidebar** | 0.5460 | 0.7336 | 0.6427 | 0.5147 |
| **block-footer** | 0.7406 | 0.8405 | 0.8794 | 0.7805 |
| **block-form** | 0.2933 | 0.2234 | 0.1927 | 0.1302 |
| **block-table** | 0.3564 | 0.3571 | 0.2433 | 0.1717 |
| block | 0.5257 | 0.6943 | 0.6248 | 0.4897 |

### Rare-class behavior

The rare classes must be judged individually because `link` dominates the training boxes. Rare-class AP50-95 from weakest to strongest is: radio 0.0042, block-form 0.1302, block-table 0.1717, block-nav 0.4546, block-sidebar 0.5147, block-footer 0.7805. These values, rather than overall mAP alone, determine which UI categories need more data.

- Raw confusion matrix: `C:\QASense-AI\runs\visionqa\atomblock_yolo11n_test\confusion_matrix.png`
- Normalized confusion matrix: `C:\QASense-AI\runs\visionqa\atomblock_yolo11n_test\confusion_matrix_normalized.png`

## Qualitative evidence and inference

Annotated examples were generated for correct_detections, dense_page, false_positives, forms, missed_detections, navigation, rare_classes, tables. Green boxes are matched predictions, orange boxes are false positives, and red boxes are missed ground truth.

`scripts/visionqa_inference.py` loads the promoted checkpoint and returns image metadata plus class ID, class name, confidence, and pixel bounding box for each detection.

## Limitations and next step

AtomBlock is heavily imbalanced and contains very dense pages. Two corrupted training images are unusable, and exact duplicate boxes are ignored in memory. The detector has not yet been validated on WebUI, Google Screen Annotation, live Playwright screenshots, responsive failures, or controlled CSS mutations. Detection scores identify UI elements; they do not by themselves prove a visual defect.

The next project step is to integrate the stable screenshot-to-JSON detector with Playwright evidence capture, then design a separate visual-defect dataset/evaluation stage before multimodal fusion.

## Artifacts

- Stable checkpoint: `C:\QASense-AI\models\visionqa\atomblock_yolo11n\best.pt`
- Checkpoint SHA-256: `7f527e18271067e2c3ed1c85b557b8b794646b996fed2128bd5a06f8c7234f57`
- Original selected checkpoint: `C:\QASense-AI\runs\visionqa\atomblock_yolo11n_main_b2\weights\best.pt`
- Evaluation JSON: `C:\QASense-AI\reports\visionqa\metrics\evaluation.json`
- Prediction examples: `C:\QASense-AI\reports\visionqa\predictions`
- Training run: `C:\QASense-AI\runs\visionqa\atomblock_yolo11n_main_b2`
