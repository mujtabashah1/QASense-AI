import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from src.product.api import QASenseHandler, RunRegistry
from src.product.contracts import empty_analysis, validate_analysis
from src.product.fusion import build_analysis
from src.product.model_status import get_model_status
from src.visual_defects.runtime import inspect_runtime_visuals


def example_analysis():
    value=empty_analysis("unit-run",{"url":"file:///fixture.html"})
    value["risk"]={"status":"available","items":[{"file":"Checkout.java","defect_probability":.81,"model_path":"risknet"}]}
    value["code"]={"status":"available","items":[{"file":"Checkout.java","suspicious_lines":[{"line":8,"score":.84}],"model_path":"codelocator"}]}
    value["runtime"]={"status":"available","page_errors":[{"message":"checkout failed"}],"http_error_responses":[{"status":500,"url":"/api/checkout","resource_type":"fetch"}],"failed_requests":[{"url":"/api/checkout","failure":"timeout"}],"console":[{"type":"error","text":"checkout failed"}]}
    value["visual"]={"status":"available","defects":[{"defect_type":"overlap","selector":"#submit","severity":"high","confidence":.91}],"matching":{"ambiguous_candidates":[]}}
    return build_analysis(value)


def test_contract_fusion_recommendations_and_priorities():
    value=example_analysis(); validate_analysis(value)
    assert value["status"]=="completed"
    assert all(item["evidence"] for item in value["findings"])
    assert any(item["category"]=="code_risk" and item["confidence"]>.8 for item in value["findings"])
    network=[item for item in value["findings"] if item["entity"]=="endpoint:/api/checkout"]
    assert len(network)==1 and len(network[0]["evidence"])==2
    assert len(value["recommendations"])==len(value["findings"])
    assert any(item["type"]=="API test" for item in value["test_suggestions"])
    assert any(item["runnable_skeleton"] for item in value["test_suggestions"] if item["type"]=="responsive test")
    assert value["regression_priority"][0]["priority"] in {"P0","P1"}


def test_partial_results_when_modalities_unavailable():
    value=empty_analysis("partial",{"url":"x"}); value["runtime"]={"status":"available","page_errors":[],"http_error_responses":[],"failed_requests":[],"console":[]}
    assert build_analysis(value)["status"]=="partial"


def test_runtime_visual_rules_are_conservative():
    dom={"elements":[{"selector":"#tiny","visible":True,"role":"button","bbox_xyxy":[0,0,12,12],"geometry":{}}]}
    issues=inspect_runtime_visuals(dom,{"width":100,"height":100},{"unmatched_dom_indices":[0]})
    assert {x["defect_type"] for x in issues}=={"undersized_click_target","unmatched_interactive_control"}
    assert next(x for x in issues if x["defect_type"]=="unmatched_interactive_control")["confidence"]<.5


def test_model_status_contract_without_loading_models():
    status=get_model_status(False)["models"]
    assert status["visionqa_v1"]["status"]=="production"
    assert status["visionqa_v2"]["status"]=="experimental"
    assert all(status[name]["available"] for name in status)


def test_api_health_models_and_run_contract(tmp_path):
    class FakeRegistry:
        def list(self): return [{"run_id":"unit-run","status":"completed","target":{},"started_at":None,"finding_count":1}]
        def get(self,run_id): return example_analysis() if run_id=="unit-run" else None
        def start(self,payload): return "unit-run"
    original=QASenseHandler.registry; QASenseHandler.registry=FakeRegistry()
    server=ThreadingHTTPServer(("127.0.0.1",0),QASenseHandler); thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start(); base=f"http://127.0.0.1:{server.server_port}"
    try:
        assert json.load(urllib.request.urlopen(base+"/api/health"))["status"]=="ok"
        assert json.load(urllib.request.urlopen(base+"/api/models/status"))["models"]["visionqa_v1"]["status"]=="production"
        assert json.load(urllib.request.urlopen(base+"/api/runs/unit-run"))["findings"]
        assert json.load(urllib.request.urlopen(base+"/api/runs/unit-run/evidence"))["runtime"]["status"]=="available"
        assert urllib.request.urlopen(base+"/").status==200
    finally:
        server.shutdown(); server.server_close(); QASenseHandler.registry=original
