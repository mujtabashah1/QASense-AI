from src.adapters.codelocator_adapter import CodeLocatorAdapter
from src.adapters.risknet_adapter import RiskNetAdapter


class FakeRiskNet:
    def predict_risk(self, metrics):
        return {"defect_probability": 0.81, "risk_level": "HIGH", "qa_priority": "P2", "prediction": "Defect-Prone", "top_risk_factors": []}


class FakeCodeLocator:
    def locate(self, code, top_k=5):
        return {"suspicious_lines": [{"line_number": 3, "score": 0.84, "code": "return total;"}], "limitation": "ranking only"}


def test_risknet_adapter_contract():
    result = RiskNetAdapter(FakeRiskNet()).predict("src/checkout.ts", {"loc": 2})
    assert result["file"] == "src/checkout.ts"
    assert result["defect_probability"] == 0.81
    assert result["priority"] == "high"


def test_codelocator_adapter_contract():
    result = CodeLocatorAdapter(FakeCodeLocator()).locate("Checkout.java", "class Checkout {}", top_k=1)
    assert result["suspicious_lines"] == [{"line": 3, "score": 0.84, "rank": 1, "code": "return total;"}]
