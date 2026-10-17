import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_trained_detector_returns_runtime_contract(tmp_path):
    sample = next((ROOT / "datasets" / "processed" / "visionqa_atomblock" / "test" / "images").iterdir())
    output = tmp_path / "visionqa.json"
    completed = subprocess.run(
        [str(ROOT / ".venv-visionqa" / "Scripts" / "python.exe"), str(ROOT / "scripts" / "visionqa_inference.py"), str(sample), "--device", "0", "--output", str(output)],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["image"]["width"] > 0 and payload["image"]["height"] > 0
    assert payload["model"]["name"] == "atomblock_yolo11n"
    assert payload["model"]["device"] == "cuda:0"
    assert payload["detection_count"] == len(payload["detections"])
    assert sum(payload["counts_by_class"].values()) == payload["detection_count"]
    assert all({"class_id", "class_name", "confidence", "bbox_xyxy"} <= row.keys() for row in payload["detections"])
