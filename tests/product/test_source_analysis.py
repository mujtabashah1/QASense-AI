import time

import pytest

from src.product.api import RunRegistry
from src.product.contracts import empty_analysis
from src.product.fusion import build_analysis
from src.product.source_analysis import RISKNET_FEATURES, build_source_manifest, discover_source_files


def test_java_manifest_has_exact_risknet_feature_contract(tmp_path):
    source = tmp_path / "Checkout.java"
    source.write_text(
        """// checkout fixture
class Checkout {
  boolean pay(int amount) {
    if (amount <= 0) return false;
    return amount < 1000;
  }
}
""",
        encoding="utf-8",
    )

    manifest = build_source_manifest(source, max_files=1)

    assert len(manifest["files"]) == 1
    metrics = manifest["files"][0]["metrics"]
    assert tuple(metrics) == RISKNET_FEATURES
    assert all(isinstance(value, float) and value >= 0 for value in metrics.values())
    assert metrics["loc"] == 7
    assert metrics["branchCount"] == 1


def test_source_discovery_is_grouped_sorted_and_excludes_build_output(tmp_path):
    (tmp_path / "src").mkdir(); (tmp_path / "target").mkdir(); (tmp_path / ".venv-local").mkdir()
    (tmp_path / "src" / "B.java").write_text("class B {}", encoding="utf-8")
    (tmp_path / "src" / "A.java").write_text("class A {}", encoding="utf-8")
    (tmp_path / "target" / "Generated.java").write_text("class Generated {}", encoding="utf-8")
    (tmp_path / ".venv-local" / "Dependency.java").write_text("class Dependency {}", encoding="utf-8")

    files = discover_source_files(tmp_path, max_files=20)

    assert [path.name for path in files] == ["A.java", "B.java"]


def test_source_discovery_rejects_unsupported_file(tmp_path):
    source = tmp_path / "service.py"
    source.write_text("print('x')", encoding="utf-8")
    with pytest.raises(ValueError, match="Java"):
        discover_source_files(source)


@pytest.mark.parametrize(
    ("analysis_type", "available"),
    [("web", ("runtime", "visual")), ("code", ("risk", "code")),
     ("full", ("runtime", "visual", "risk", "code"))],
)
def test_requested_modalities_define_completion(analysis_type, available):
    analysis = empty_analysis("mode-test", {"analysis_type": analysis_type})
    for modality in available:
        analysis[modality] = {"status": "available", "items": []}
    assert build_analysis(analysis)["status"] == "completed"


def test_registry_passes_code_workflow_to_orchestrator(tmp_path):
    source = tmp_path / "Checkout.java"
    source.write_text("class Checkout {}", encoding="utf-8")

    class FakeOrchestrator:
        calls = []

        def run(self, url=None, **kwargs):
            self.calls.append({"url": url, **kwargs})
            value = empty_analysis(kwargs["run_id"], {
                "analysis_type": kwargs["analysis_type"], "source_path": kwargs["source_path"],
            })
            value["risk"] = {"status": "available", "items": []}
            value["code"] = {"status": "available", "items": []}
            return build_analysis(value)

    fake = FakeOrchestrator(); registry = RunRegistry(fake)
    run_id = registry.start({"analysis_type": "code", "source_path": str(source), "run_id": "unit-code-flow"})
    for _ in range(100):
        result = registry.get(run_id)
        if result and result.get("status") not in {"queued", "running"}:
            break
        time.sleep(.01)

    assert result["status"] == "completed"
    assert fake.calls[0]["analysis_type"] == "code"
    assert fake.calls[0]["url"] is None
    assert fake.calls[0]["source_path"] == str(source)
