"""Run the single authorized staged VisionQA v2 experiment and final evaluations."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "visionqa" / "v2"
MODEL_DIR = ROOT / "models" / "visionqa" / "visionqa_v2_mixed14"
RUN_DIR = ROOT / "runs" / "visionqa" / "visionqa_v2_mixed14"
PIPELINE_STATUS = REPORTS / "training_status.json"
BASELINE_AP95 = {"icon": 0.0206, "checkbox": 0.0100, "radio": 0.00421, "block": 0.4897}


def now() -> str: return datetime.now(timezone.utc).isoformat()
def load(path: Path) -> dict: return json.loads(path.read_text(encoding="utf-8"))
def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8"); temporary.replace(path)
def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        completed=subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if completed.returncode: raise RuntimeError(f"Command failed ({completed.returncode}); see {log}")


def selected_checkpoint(status_path: Path) -> Path:
    selected=load(status_path).get("selected_epoch")
    if not selected: raise RuntimeError(f"No validation-selected epoch in {status_path}")
    checkpoint=Path(selected["checkpoint"])
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    return checkpoint


def resume_argument(status_path: Path) -> list[str]:
    """Return the latest full-state epoch checkpoint for an interrupted stage."""
    if not status_path.is_file():
        return []
    status = load(status_path)
    if status.get("status") == "completed":
        return []
    for epoch in reversed(status.get("epoch_history", [])):
        checkpoint = Path(epoch["checkpoint"])
        if checkpoint.is_file():
            return ["--resume", str(checkpoint)]
    return []


def major_forgetting(evaluation: dict) -> tuple[bool,list[str]]:
    reasons=[]
    if evaluation["overall"]["map50_95"] < 0.2782520238296197*0.75: reasons.append("overall AtomBlock mAP50-95 retention below 75%")
    by_name={row["class_name"]:row for row in evaluation["per_class"]}
    for name,baseline in BASELINE_AP95.items():
        value=by_name[name]["ap50_95"]
        if baseline-value >= 0.01 and value < baseline*0.5: reasons.append(f"{name} AP50-95 major regression")
    return bool(reasons),reasons


def run_playwright_regression(python: str, checkpoint: Path) -> dict:
    integration_python = ROOT / ".venv-integration" / "Scripts" / "python.exe"
    fixture = ROOT / "tests" / "fixtures" / "runtime_fixture.html"
    run_id = "visionqa-v2-playwright-regression"
    if not integration_python.is_file():
        return {"passed": False, "reason": f"missing integration Python: {integration_python}"}
    if not fixture.is_file():
        return {"passed": False, "reason": f"missing fixture: {fixture}"}
    run([
        str(integration_python), "scripts/run_playwright_visionqa.py", fixture.resolve().as_uri(),
        "--run-id", run_id, "--checkpoint", str(checkpoint),
    ], REPORTS / "playwright_regression.log")
    artifact_dir = ROOT / "runs" / "playwright" / run_id
    required = ["viewport.png", "full_page.png", "dom.json", "visionqa.json"]
    missing = [name for name in required if not (artifact_dir / name).is_file()]
    if missing:
        return {"passed": False, "reason": "missing artifacts", "missing": missing, "run_dir": str(artifact_dir)}
    vision = load(artifact_dir / "visionqa.json")
    ids = [int(row["class_id"]) for row in vision.get("detections", [])]
    passed = all(0 <= value < 14 for value in ids) and vision.get("model", {}).get("class_count") == 14
    return {
        "passed": passed, "run_dir": str(artifact_dir), "detection_count": vision.get("detection_count"),
        "class_ids_valid": all(0 <= value < 14 for value in ids),
        "model_class_count": vision.get("model", {}).get("class_count"),
    }


def main() -> int:
    benchmark=load(REPORTS/"benchmark_training_status.json")
    if benchmark.get("status") != "completed": raise RuntimeError("Benchmark must complete before the main experiment")
    status={"schema_version":"1.0","status":"running","started_at":now(),"completed_steps":["benchmark"],"current_step":"stage_a"}; write(PIPELINE_STATUS,status)
    python=sys.executable
    stage_a_status=REPORTS/"stage_a_training_status.json"
    if not stage_a_status.exists() or load(stage_a_status).get("status") != "completed":
        run([python,"scripts/train_visionqa_v2.py","--stage","stage_a","--epochs","3","--freeze","11","--lr0","0.0005","--batch","4","--name","visionqa_v2_mixed14_stage_a","--status",str(stage_a_status),*resume_argument(stage_a_status)],REPORTS/"stage_a.log")
    stage_a=selected_checkpoint(stage_a_status); status["completed_steps"].append("stage_a"); status["current_step"]="stage_a_atomblock_regression"; write(PIPELINE_STATUS,status)
    stage_a_regression=REPORTS/"stage_a_atomblock_regression.json"
    if not stage_a_regression.exists():
        run([python,"scripts/evaluate_visionqa_v2_domain.py","--checkpoint",str(stage_a),"--domain","atomblock_test","--output",str(stage_a_regression),"--name","v2_stage_a_atomblock_regression","--batch","2"],REPORTS/"stage_a_atomblock_regression.log")
    catastrophic,reasons=major_forgetting(load(stage_a_regression))
    status["stage_a_forgetting_gate"]={"catastrophic":catastrophic,"reasons":reasons}; status["completed_steps"].append("stage_a_atomblock_regression")
    if catastrophic:
        status.update({"status":"stopped","finished_at":now(),"stop_reason":"Stage A catastrophic-forgetting gate","current_step":None}); write(PIPELINE_STATUS,status); return 2

    status["current_step"]="stage_b"; write(PIPELINE_STATUS,status)
    stage_b_status=REPORTS/"stage_b_training_status.json"
    if not stage_b_status.exists() or load(stage_b_status).get("status") != "completed":
        run([python,"scripts/train_visionqa_v2.py","--stage","stage_b","--model",str(stage_a),"--epochs","9","--freeze","6","--lr0","0.0001","--batch","4","--name","visionqa_v2_mixed14_stage_b","--status",str(stage_b_status),*resume_argument(stage_b_status)],REPORTS/"stage_b.log")
    status["completed_steps"].append("stage_b"); status["current_step"]="candidate_selection"; write(PIPELINE_STATUS,status)
    candidates=[]
    for stage,path in (("stage_a",stage_a_status),("stage_b",stage_b_status)):
        selected=load(path)["selected_epoch"]; candidates.append({"stage":stage,**selected})
    chosen=max(candidates,key=lambda row:(row["webui_supported_macro_f1"],row["stage"]=="stage_b"))
    selected_source=Path(chosen["checkpoint"]); MODEL_DIR.mkdir(parents=True,exist_ok=True)
    candidate=MODEL_DIR/"candidate.pt"; shutil.copy2(selected_source,candidate)
    status["selected_candidate"]={**chosen,"path":str(candidate),"sha256":sha256(candidate)}; status["completed_steps"].append("candidate_selection"); status["current_step"]="final_domain_evaluations"; write(PIPELINE_STATUS,status)

    webui_val=REPORTS/"selected_webui_validation.json"; atom=REPORTS/"atomblock_regression.json"
    if not webui_val.exists(): run([python,"scripts/evaluate_visionqa_v2_domain.py","--checkpoint",str(candidate),"--domain","webui_val","--output",str(webui_val),"--name","v2_selected_webui_validation","--batch","4"],REPORTS/"selected_webui_validation.log")
    if sha256(candidate)==sha256(stage_a) and stage_a_regression.exists(): shutil.copy2(stage_a_regression,atom)
    elif not atom.exists(): run([python,"scripts/evaluate_visionqa_v2_domain.py","--checkpoint",str(candidate),"--domain","atomblock_test","--output",str(atom),"--name","v2_selected_atomblock_test","--batch","2"],REPORTS/"selected_atomblock_test.log")
    catastrophic,reasons=major_forgetting(load(atom)); status["final_forgetting_gate"]={"catastrophic":catastrophic,"reasons":reasons}; status["completed_steps"].append("final_domain_evaluations"); status["current_step"]="frozen_webui_evaluation"; write(PIPELINE_STATUS,status)

    frozen_output=REPORTS/"webui_evaluation.json"; predictions=RUN_DIR/"frozen_webui_predictions"
    if not frozen_output.exists():
        manifest=load(ROOT/"reports/visionqa/webui_eval_manifest.json")
        items=[{"image":row["image_identifier"]["extracted_path"],"output":str(predictions/f"{row['screen_id']}.json")} for row in manifest["screens"]]
        inference_manifest=RUN_DIR/"frozen_inference_manifest.json"; write(inference_manifest,{"items":items})
        run([python,"scripts/batch_visionqa_paths.py",str(inference_manifest),"--checkpoint",str(candidate),"--device","0","--confidence","0.25"],REPORTS/"frozen_webui_inference.log")
        run([python,"scripts/evaluate_frozen_webui_checkpoint.py","--predictions",str(predictions),"--output",str(frozen_output),"--errors",str(REPORTS/"webui_error_analysis.json")],REPORTS/"frozen_webui_evaluation.log")
    frozen=load(frozen_output); atom_eval=load(atom)
    metrics_pass=(frozen["metrics"]["f1"]>=0.1 and frozen["metrics"]["macro_f1"]>=0.1 and atom_eval["overall"]["map50_95"]>=0.2782520238296197*0.8 and not catastrophic)
    playwright = run_playwright_regression(python, candidate) if metrics_pass else {"passed": None, "reason": "metric gates did not pass"}
    write(REPORTS/"playwright_regression.json", playwright)
    promote=metrics_pass and playwright["passed"]
    promoted=MODEL_DIR/"best.pt" if promote else None
    if promote: shutil.copy2(candidate,promoted)
    decision={"decision":"PROMOTE AS UNIFIED VISIONQA" if promote else "KEEP V2 AS EXPERIMENTAL","automatic_promotion":False,"candidate":str(candidate),"promoted_model":str(promoted) if promoted else None,"current_stable_model":str(ROOT/"models/visionqa/atomblock_yolo11n/best.pt"),"rollback_model":str(ROOT/"models/visionqa/atomblock_yolo11n/best.pt"),"gates":{"webui_f1":frozen["metrics"]["f1"],"webui_macro_f1":frozen["metrics"]["macro_f1"],"atomblock_map50_95":atom_eval["overall"]["map50_95"],"catastrophic_forgetting":catastrophic,"playwright_regression":playwright}}
    write(REPORTS/"promotion_decision.json",decision)
    write(REPORTS/"runtime_benchmark.json",{**frozen["runtime"],"model_bytes":candidate.stat().st_size,"checkpoint":str(candidate)})
    write(REPORTS/"model_manifest.json",{"architecture":"YOLO11n","classes":14,"candidate":str(candidate),"candidate_sha256":sha256(candidate),"promoted":str(promoted) if promoted else None,"rollback":str(ROOT/"models/visionqa/atomblock_yolo11n/best.pt"),"selection":"WebUI validation supported-class macro-F1","selected_epoch":chosen})
    status.update({"status":"completed","finished_at":now(),"current_step":None,"promotion_decision":decision["decision"]}); status["completed_steps"].extend(["frozen_webui_evaluation","promotion_decision"]); write(PIPELINE_STATUS,status)
    return 0


if __name__ == "__main__": raise SystemExit(main())
