"""Build the final AtomBlock report from measured VisionQA artifacts only."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "visionqa" / "metrics"
RUN = ROOT / "runs" / "visionqa" / "atomblock_yolo11n_main_b2"
REPORT = ROOT / "reports" / "visionqa" / "VISIONQA_ATOMBLOCK_REPORT.md"
RARE = {"block-table", "block-form", "block-sidebar", "radio", "block-footer", "block-nav"}


def load_json(name: str) -> dict[str, Any]:
    path = METRICS / name
    if not path.is_file():
        raise FileNotFoundError(f"Required measured artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key.strip(): float(value.strip()) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> int:
    environment = load_json("environment.json")
    audit = load_json("dataset_audit.json")
    loader = load_json("loader_audit.json")
    smoke = load_json("smoke_test.json")
    benchmark = load_json("benchmark.json")
    evaluation = load_json("evaluation.json")
    manifest = json.loads(
        (ROOT / "models" / "visionqa" / "atomblock_yolo11n" / "model_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    examples = json.loads(
        (ROOT / "reports" / "visionqa" / "predictions" / "visual_examples.json").read_text(
            encoding="utf-8"
        )
    )
    main_rows = rows(RUN / "results.csv")
    train_args = yaml.safe_load((RUN / "args.yaml").read_text(encoding="utf-8"))
    best = max(main_rows, key=lambda row: row["metrics/mAP50-95(B)"])
    last = main_rows[-1]
    benchmark_times = [row["time"] for row in benchmark["epoch_metrics"]]
    benchmark_epoch_seconds = [
        benchmark_times[0],
        *[b - a for a, b in zip(benchmark_times, benchmark_times[1:])],
    ]
    mean_benchmark_epoch = sum(benchmark_epoch_seconds) / len(benchmark_epoch_seconds)
    train_split = audit["splits"]["train"]
    lines = [
        "# VisionQA AtomBlock YOLO11n Model Report",
        "",
        "## Outcome",
        "",
        f"VisionQA trained YOLO11n for {len(main_rows)} completed epochs on the fixed AtomBlock split. "
        f"Epoch {int(best['epoch'])} was selected by validation mAP50-95. The selected checkpoint "
        f"was evaluated once on the held-out test split and promoted to `{manifest['stable_checkpoint']}`.",
        "",
        "## Dataset and integrity",
        "",
        f"- Dataset: `{audit['dataset_path']}`",
        f"- Configuration: `{audit['data_yaml']}`",
        f"- Train: {audit['splits']['train']['images']:,} images / {audit['splits']['train']['labels']:,} labels",
        f"- Validation: {audit['splits']['val']['images']:,} images / {audit['splits']['val']['labels']:,} labels",
        f"- Test: {audit['splits']['test']['images']:,} images / {audit['splits']['test']['labels']:,} labels",
        f"- Classes: {audit['nc']}; label audit: {audit['status']}",
        f"- Training boxes: {train_split['total_boxes']:,}; empty training labels: {train_split['empty_labels']}",
        f"- Loader-usable training images: {int(loader['splits']['train']['loader_results'][0]):,}; "
        f"corrupt exclusions: {len(loader['splits']['train']['corrupt_files_excluded'])}",
        "",
        "The two excluded PNGs are physically truncated in both the original and prepared datasets. "
        "They were preserved unchanged. Ultralytics also removes exact duplicate annotation rows in "
        "memory; source labels and split membership remain unchanged.",
        "",
        "### Training class distribution",
        "",
        "| Class | Boxes | Share |",
        "|---|---:|---:|",
    ]
    for name, count in train_split["class_counts"].items():
        lines.append(f"| {name} | {count:,} | {100 * count / train_split['total_boxes']:.2f}% |")

    lines += [
        "",
        "## Environment",
        "",
        f"- Python: {environment['python_version']}",
        f"- PyTorch: {environment['torch_version']}",
        f"- TorchVision: {environment['torchvision_version']}",
        f"- PyTorch CUDA runtime: {environment['torch_cuda_version']}",
        f"- Ultralytics: {environment['ultralytics_version']}",
        f"- GPU: {environment['pytorch_gpu']['name']} ({environment['nvidia_smi']['vram_mib']:,} MiB)",
        "",
        "CUDA tensor execution and YOLO11n inference passed. AMP was requested, but Ultralytics "
        "disabled it after its runtime safety check on this GPU; valid training used FP32.",
        "",
        "## Smoke test and throughput benchmark",
        "",
        "| Run | Epochs | Batch | Precision | Recall | mAP50 | mAP50-95 | Peak VRAM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Smoke | 1 | 2 | {smoke['result_metrics']['metrics/precision(B)']:.4f} | "
        f"{smoke['result_metrics']['metrics/recall(B)']:.4f} | "
        f"{smoke['result_metrics']['metrics/mAP50(B)']:.4f} | "
        f"{smoke['result_metrics']['metrics/mAP50-95(B)']:.4f} | {smoke['gpu']['peak_memory_used_mib']:.0f} MiB |",
        f"| Benchmark | 3 | 4 | {benchmark['result_metrics']['metrics/precision(B)']:.4f} | "
        f"{benchmark['result_metrics']['metrics/recall(B)']:.4f} | "
        f"{benchmark['result_metrics']['metrics/mAP50(B)']:.4f} | "
        f"{benchmark['result_metrics']['metrics/mAP50-95(B)']:.4f} | {benchmark['gpu']['peak_memory_used_mib']:.0f} MiB |",
        "",
        f"The three-epoch benchmark averaged {mean_benchmark_epoch / 60:.2f} minutes per epoch "
        f"({', '.join(f'{value / 60:.2f}' for value in benchmark_epoch_seconds)} minutes). "
        "Local training was selected because this is below the 15–20 minute guideline. "
        f"At the benchmark rate, 30 epochs are about {30 * mean_benchmark_epoch / 3600:.2f} hours "
        f"and 50 epochs about {50 * mean_benchmark_epoch / 3600:.2f} hours. Batch 2 was retained "
        "for the main run because its measured smoke epoch was faster than batch 4; VRAM size alone "
        "did not determine the choice.",
        "",
        "## Main training",
        "",
        f"Configuration: pretrained YOLO11n, {len(main_rows)}/{train_args['epochs']} completed epochs, "
        f"batch {train_args['batch']}, image size {train_args['imgsz']}, seed {train_args['seed']}, "
        f"patience {train_args['patience']}, max_det {train_args['max_det']}, checkpoint every epoch.",
        "",
        "| Epoch | Train box | Train cls | Train DFL | Precision | Recall | mAP50 | mAP50-95 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        marker = " **(selected)**" if int(row["epoch"]) == int(best["epoch"]) else ""
        lines.append(
            f"| {int(row['epoch'])}{marker} | {row['train/box_loss']:.4f} | "
            f"{row['train/cls_loss']:.4f} | {row['train/dfl_loss']:.4f} | "
            f"{row['metrics/precision(B)']:.4f} | {row['metrics/recall(B)']:.4f} | "
            f"{row['metrics/mAP50(B)']:.4f} | {row['metrics/mAP50-95(B)']:.4f} |"
        )

    overall = evaluation["overall"]
    lines += [
        "",
        "Training was resumable from full optimizer-bearing epoch checkpoints. `last.pt`, "
        "`best.pt`, training arguments, results, and plots remain in the original run directory.",
        "",
        "## Held-out test evaluation",
        "",
        "| Precision | Recall | mAP50 | mAP50-95 |",
        "|---:|---:|---:|---:|",
        f"| {overall['precision']:.4f} | {overall['recall']:.4f} | {overall['map50']:.4f} | {overall['map50_95']:.4f} |",
        "",
        "### Per-class test metrics",
        "",
        "| Class | Precision | Recall | AP50 | AP50-95 |",
        "|---|---:|---:|---:|---:|",
    ]
    by_class = {row["class_name"]: row for row in evaluation["per_class"]}
    for row in evaluation["per_class"]:
        label = f"**{row['class_name']}**" if row["class_name"] in RARE else row["class_name"]
        lines.append(
            f"| {label} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['ap50']:.4f} | {row['ap50_95']:.4f} |"
        )
    rare_ranked = sorted((by_class[name] for name in RARE), key=lambda row: row["ap50_95"])
    lines += [
        "",
        "### Rare-class behavior",
        "",
        "The rare classes must be judged individually because `link` dominates the training boxes. "
        "Rare-class AP50-95 from weakest to strongest is: "
        + ", ".join(f"{row['class_name']} {row['ap50_95']:.4f}" for row in rare_ranked)
        + ". These values, rather than overall mAP alone, determine which UI categories need more data.",
        "",
        f"- Raw confusion matrix: `{evaluation['confusion_matrix']}`",
        f"- Normalized confusion matrix: `{evaluation['confusion_matrix_normalized']}`",
        "",
        "## Qualitative evidence and inference",
        "",
        f"Annotated examples were generated for {', '.join(sorted(examples['examples']))}. "
        "Green boxes are matched predictions, orange boxes are false positives, and red boxes are missed ground truth.",
        "",
        "`scripts/visionqa_inference.py` loads the promoted checkpoint and returns image metadata plus "
        "class ID, class name, confidence, and pixel bounding box for each detection.",
        "",
        "## Limitations and next step",
        "",
        "AtomBlock is heavily imbalanced and contains very dense pages. Two corrupted training images "
        "are unusable, and exact duplicate boxes are ignored in memory. The detector has not yet been "
        "validated on WebUI, Google Screen Annotation, live Playwright screenshots, responsive failures, "
        "or controlled CSS mutations. Detection scores identify UI elements; they do not by themselves "
        "prove a visual defect.",
        "",
        "The next project step is to integrate the stable screenshot-to-JSON detector with Playwright "
        "evidence capture, then design a separate visual-defect dataset/evaluation stage before multimodal fusion.",
        "",
        "## Artifacts",
        "",
        f"- Stable checkpoint: `{manifest['stable_checkpoint']}`",
        f"- Checkpoint SHA-256: `{manifest['sha256']}`",
        f"- Original selected checkpoint: `{manifest['source_checkpoint']}`",
        f"- Evaluation JSON: `{METRICS / 'evaluation.json'}`",
        f"- Prediction examples: `{ROOT / 'reports' / 'visionqa' / 'predictions'}`",
        f"- Training run: `{RUN}`",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
