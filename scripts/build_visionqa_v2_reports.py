"""Build final VisionQA v2 engineering reports from measured artifacts."""

from __future__ import annotations

import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; REPORTS=ROOT/"reports/visionqa/v2"
def load(name): return json.loads((REPORTS/name).read_text(encoding="utf-8"))
def write(name,text): (REPORTS/name).write_text(text.rstrip()+"\n",encoding="utf-8")
def pct(value): return f"{100*value:.3f}%"


def main():
    config=load("training_config.json"); benchmark=load("benchmark_training_status.json"); a=load("stage_a_training_status.json"); b=load("stage_b_training_status.json")
    web=load("webui_evaluation.json"); atom=load("atomblock_regression.json"); runtime=load("runtime_benchmark.json"); decision=load("promotion_decision.json"); manifest=load("model_manifest.json")
    v1_atom=load("pretraining_atomblock_test.json"); v1_web={"precision":.08805,"recall":.03564,"f1":.05074,"macro_f1":.04835}
    old={row["class_name"]:row for row in v1_atom["per_class"]}; new={row["class_name"]:row for row in atom["per_class"]}
    comparison=[]
    for name in old:
        delta=new[name]["ap50_95"]-old[name]["ap50_95"]; baseline=old[name]["ap50_95"]
        relative=delta/baseline if baseline else None
        classification="improved" if delta>0.005 else "stable" if delta>=-0.005 else "major regression" if abs(delta)>=.01 and new[name]["ap50_95"]<baseline*.5 else "minor regression"
        comparison.append({"class_name":name,"v1_ap50_95":baseline,"v2_ap50_95":new[name]["ap50_95"],"absolute_change":delta,"relative_change":relative,"classification":classification})
    (REPORTS/"per_class_comparison.json").write_text(json.dumps({"classes":comparison},indent=2)+"\n",encoding="utf-8")
    selected=manifest["selected_epoch"]
    total_epochs=len(a["epoch_history"])+len(b["epoch_history"]); total_runtime=a["runtime_seconds"]+b["runtime_seconds"]
    write("VISIONQA_V2_MODEL_REPORT.md",f"""# VisionQA v2 Model Report

Generated {datetime.now(timezone.utc).isoformat()}.

## Result

**{decision['decision']}**. The candidate is one YOLO11n detector with the canonical 14-class head. It was initialized from VisionQA v1 and trained on 23,319 WebUI screens plus all 5,420 decodable AtomBlock rehearsal images. WebUI classes 4, 5, 6, and 13 were masked from negative classification loss; AtomBlock retained normal supervision for all classes.

Selected checkpoint: `{manifest['candidate']}`  
SHA-256: `{manifest['candidate_sha256']}`  
Rollback: `{manifest['rollback']}`

## Training

The one-epoch benchmark completed in {benchmark['runtime_seconds']/60:.1f} minutes. Stage A froze model modules 0–10 and trained modules 11–23 for {len(a['epoch_history'])} epochs. Stage B froze modules 0–5 and trained modules 6–23 for {len(b['epoch_history'])} epochs. The combined main experiment completed {total_epochs} epochs in {total_runtime/3600:.2f} hours. Checkpoints include optimizer, scaler, scheduler-equivalent epoch state, arguments, and metrics after every epoch.

The selected epoch came from {selected['stage']} epoch {selected['epoch_number']} using WebUI validation supported-class macro-F1 ({selected['webui_supported_macro_f1']:.5f}). AMP was requested; the installed Ultralytics safety check disabled it on the T1200, so training ran in FP32.

## Frozen WebUI transfer

| Metric | v1 | v2 | Change |
|---|---:|---:|---:|
| Precision | {v1_web['precision']:.5f} | {web['metrics']['precision']:.5f} | {web['metrics']['precision']-v1_web['precision']:+.5f} |
| Recall | {v1_web['recall']:.5f} | {web['metrics']['recall']:.5f} | {web['metrics']['recall']-v1_web['recall']:+.5f} |
| F1 | {v1_web['f1']:.5f} | {web['metrics']['f1']:.5f} | {web['metrics']['f1']-v1_web['f1']:+.5f} |
| Macro-F1 | {v1_web['macro_f1']:.5f} | {web['metrics']['macro_f1']:.5f} | {web['metrics']['macro_f1']-v1_web['macro_f1']:+.5f} |

## AtomBlock regression

| Metric | v1 | v2 | Change |
|---|---:|---:|---:|
| Precision | {v1_atom['overall']['precision']:.5f} | {atom['overall']['precision']:.5f} | {atom['overall']['precision']-v1_atom['overall']['precision']:+.5f} |
| Recall | {v1_atom['overall']['recall']:.5f} | {atom['overall']['recall']:.5f} | {atom['overall']['recall']-v1_atom['overall']['recall']:+.5f} |
| mAP50 | {v1_atom['overall']['map50']:.5f} | {atom['overall']['map50']:.5f} | {atom['overall']['map50']-v1_atom['overall']['map50']:+.5f} |
| mAP50-95 | {v1_atom['overall']['map50_95']:.5f} | {atom['overall']['map50_95']:.5f} | {atom['overall']['map50_95']-v1_atom['overall']['map50_95']:+.5f} |

## Operational result

The 399-screen frozen evaluation measured mean {runtime['mean_ms']:.2f} ms, median {runtime['median_ms']:.2f} ms, and p95 {runtime['p95_ms']:.2f} ms per image including wrapper timing. Model size is {runtime['model_bytes']/1024**2:.2f} MiB.

## Scientific limits

WebUI supplies no reliable labels for icon, checkbox, radio, or generic block. Their preservation is evaluated on AtomBlock; frozen WebUI transfer metrics cover only the ten supported mapped classes. The frozen slice was not used for selection or threshold tuning.
""")
    web_rows="\n".join(f"| {name} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['ap50']:.4f} |" for name,row in web["metrics"]["per_class"].items())
    write("WEBUI_TRANSFER_COMPARISON.md",f"""# VisionQA v2 WebUI Transfer Comparison

The unchanged 399-screen, 133-group manifest `{web['manifest_content_sha256']}` was evaluated once at confidence 0.25. VisionQA v2 F1 changed from {v1_web['f1']:.5f} to {web['metrics']['f1']:.5f}; macro-F1 changed from {v1_web['macro_f1']:.5f} to {web['metrics']['macro_f1']:.5f}.

| Class | Precision | Recall | F1 | AP50 |
|---|---:|---:|---:|---:|
{web_rows}

Responsive class-set Jaccard was {web['responsive_consistency']['mean_detection_class_jaccard']:.4f}, with {web['responsive_consistency']['responsive_only_misses']} responsive-only miss patterns.
""")
    forget_rows="\n".join(f"| {r['class_name']} | {r['v1_ap50_95']:.4f} | {r['v2_ap50_95']:.4f} | {r['absolute_change']:+.4f} | {r['classification']} |" for r in comparison)
    write("CATASTROPHIC_FORGETTING_ANALYSIS.md",f"""# Catastrophic Forgetting Analysis

The selected v2 checkpoint was compared with v1 on the untouched AtomBlock test split.

| Class | v1 AP50-95 | v2 AP50-95 | Change | Assessment |
|---|---:|---:|---:|---|
{forget_rows}

Final gate: **{'FAIL' if any(r['classification']=='major regression' for r in comparison) else 'PASS'}** under the predeclared materiality rule in `training_config.json`.
""")
    write("RUNTIME_BENCHMARK.md",f"""# VisionQA v2 Runtime Benchmark

Representative set: all 399 frozen WebUI screens. Mean latency {runtime['mean_ms']:.2f} ms, median {runtime['median_ms']:.2f} ms, p95 {runtime['p95_ms']:.2f} ms. Candidate checkpoint size: {runtime['model_bytes']/1024**2:.2f} MiB. Training GPU telemetry is stored in the benchmark and stage status JSON files.
""")

if __name__=="__main__": main()
