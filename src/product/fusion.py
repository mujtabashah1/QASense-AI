"""Deterministic multimodal fusion, QA actions, and regression priority."""

from __future__ import annotations

import hashlib
import math
import time
from collections import defaultdict
from typing import Any

from .contracts import evidence_ref, validate_analysis

SEVERITY_WEIGHT = {"critical": 1.0, "high": .82, "medium": .58, "low": .32, "info": .12}


def _finding_id(category: str, entity: str, title: str) -> str:
    return "finding-" + hashlib.sha256(f"{category}|{entity}|{title}".encode()).hexdigest()[:10]


def _finding(title: str, category: str, severity: str, confidence: float, entity: str,
             evidence: list[dict[str, Any]], **modalities: Any) -> dict[str, Any]:
    return {
        "id": _finding_id(category, entity, title), "title": title, "category": category,
        "entity": entity, "severity": severity,
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "risk": modalities.get("risk"), "code": modalities.get("code"),
        "runtime": modalities.get("runtime"), "visual": modalities.get("visual"),
        "evidence": evidence, "fusion_method": "deterministic_rules_v1",
    }


def _runtime_findings(runtime: dict[str, Any]) -> list[dict[str, Any]]:
    if runtime.get("status") != "available":
        return []
    findings = []
    for index, error in enumerate(runtime.get("page_errors", [])):
        message = error.get("message", "Unhandled page error")
        findings.append(_finding(f"Unhandled page error: {message[:90]}", "runtime", "high", .95,
            f"page-error:{message}", [evidence_ref("page_error", "errors.json", message, index=index)], runtime={"page_error": error}))
    for response in runtime.get("http_error_responses", []):
        status = int(response.get("status", 0)); url = response.get("url", "unknown endpoint")
        severity = "critical" if status >= 500 and response.get("resource_type") == "document" else "high" if status >= 500 else "medium"
        findings.append(_finding(f"HTTP {status} from {url}", "network", severity, .95, f"endpoint:{url}",
            [evidence_ref("http_response", "network.json", f"HTTP {status}", url=url, status=status)], runtime={"http_error_response": response}))
    for request in runtime.get("failed_requests", []):
        url = request.get("url", "unknown endpoint"); failure = request.get("failure") or "request failed"
        findings.append(_finding(f"Network request failed: {url}", "network", "medium", .85, f"endpoint:{url}",
            [evidence_ref("failed_request", "network.json", str(failure), url=url)], runtime={"failed_request": request}))
    for index, message in enumerate(row for row in runtime.get("console", []) if row.get("type") == "error"):
        text = message.get("text", "Console error")
        findings.append(_finding(f"Browser console error: {text[:90]}", "runtime", "medium", .75, f"console:{text}",
            [evidence_ref("console", "console.json", text, index=index)], runtime={"console_error": message}))
    return findings


def _visual_findings(visual: dict[str, Any]) -> list[dict[str, Any]]:
    if visual.get("status") != "available":
        return []
    findings = []
    for issue in visual.get("defects", []):
        defect = issue.get("defect_type", "visual issue"); selector = issue.get("selector") or "page"
        findings.append(_finding(f"Visual issue: {defect} at {selector}", "visual", issue.get("severity", "medium"),
            float(issue.get("confidence", .6)), f"selector:{selector}",
            [evidence_ref("visual_rule", issue.get("source", "visual_rules.json"), defect, selector=selector)], visual=issue))
    conflicts = visual.get("matching", {}).get("ambiguous_candidates", [])
    if conflicts:
        findings.append(_finding(f"{len(conflicts)} DOM/visual semantic conflicts require review", "visual", "low", .55,
            "page:semantic-conflicts", [evidence_ref("dom_vision_match", "dom_vision_matches.json", f"{len(conflicts)} conflicts")],
            visual={"semantic_conflicts": conflicts}))
    return findings


def _code_findings(risk: dict[str, Any], code: dict[str, Any]) -> list[dict[str, Any]]:
    findings = []
    risk_by_file = {item.get("file"): item for item in risk.get("items", [])}
    code_by_file = {item.get("file"): item for item in code.get("items", [])}
    for file in sorted(set(risk_by_file) | set(code_by_file)):
        risk_item, code_item = risk_by_file.get(file), code_by_file.get(file)
        probability = float(risk_item.get("defect_probability", 0)) if risk_item else 0
        lines = code_item.get("suspicious_lines", []) if code_item else []
        localizer = max((float(row.get("score", 0)) for row in lines), default=0)
        if probability < .5 and localizer < .5:
            continue
        severity = "high" if probability >= .8 and localizer >= .7 else "medium" if max(probability, localizer) >= .65 else "low"
        # RiskNet probability and CodeLocator ranking score remain distinct. This
        # conservative support score is not represented as a joint probability.
        confidence = max(probability, localizer) if min(probability, localizer) == 0 else .6 * max(probability, localizer) + .4 * min(probability, localizer)
        refs = []
        if risk_item:
            refs.append(evidence_ref("risknet", risk_item.get("model_path", "risknet"), f"risk={probability:.3f}", file=file))
        if code_item:
            refs.append(evidence_ref("codelocator", code_item.get("model_path", "codelocator_v2"), f"top suspiciousness={localizer:.3f}", file=file, lines=lines[:5]))
        findings.append(_finding(f"Prioritize QA around {file}", "code_risk", severity, confidence,
            f"file:{file}", refs, risk=risk_item, code=code_item))
    return findings


def _deduplicate(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in findings:
        grouped[(item["category"], item["entity"])].append(item)
    output = []
    for items in grouped.values():
        primary = max(items, key=lambda x: (SEVERITY_WEIGHT[x["severity"]], x["confidence"]))
        combined = dict(primary)
        if len(items) > 1:
            combined["confidence"] = round(1 - math.prod(1 - min(.95, item["confidence"]) for item in items), 4)
            combined["evidence"] = [e for item in items for e in item["evidence"]]
            combined["deduplicated_finding_ids"] = [item["id"] for item in items]
        output.append(combined)
    return sorted(output, key=lambda x: (-SEVERITY_WEIGHT[x["severity"]], -x["confidence"], x["id"]))


def _recommendations(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for item in findings:
        if item["category"] == "network":
            action = f"Add a deterministic failure-path test for {item['entity'].removeprefix('endpoint:')} and verify a safe user-visible fallback."
        elif item["category"] == "runtime":
            action = "Reproduce the captured browser error, assert graceful handling, and prevent the exception from escaping the page flow."
        elif item["category"] == "visual":
            action = "Review the linked screenshot and DOM geometry, then add a responsive assertion around the affected element."
        else:
            action = f"Run focused regression tests for {item['entity'].removeprefix('file:')} starting with the ranked suspicious lines."
        output.append({"id": f"rec-{item['id'][8:]}", "finding_id": item["id"], "priority": item["severity"], "action": action, "evidence": item["evidence"]})
    return output


def _test_suggestions(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    suggestions = []
    for item in findings:
        category = item["category"]
        if category == "network":
            kind = "API test"; steps = ["Reproduce the captured request with a safe non-destructive fixture.", "Inject or simulate the recorded failure status.", "Observe API and UI error handling."]; expected = "The failure is handled predictably without exposing an unhandled exception."
        elif category == "visual":
            kind = "responsive test"; steps = ["Open the linked route at desktop, tablet, and mobile viewports.", "Locate the affected selector from DOM evidence.", "Assert visibility and non-overlap geometry."]; expected = "The control remains visible, usable, and unobstructed across supported viewports."
        elif category == "code_risk":
            kind = "regression test"; steps = ["Exercise behavior owned by the ranked file.", "Cover the highest-ranked suspicious branches.", "Assert the expected functional outcome."]; expected = "The high-risk code path produces the expected result without regression."
        else:
            kind = "error-handling test"; steps = ["Open the linked page state.", "Trigger the recorded interaction.", "Capture page and console errors."]; expected = "No unhandled page or console error occurs."
        selector = item["entity"].removeprefix("selector:") if item["entity"].startswith("selector:") else None
        skeleton = None
        if category == "visual" and selector and selector != "page" and "'" not in selector:
            skeleton = "import { test, expect } from '@playwright/test';\n\ntest('affected control remains visible', async ({ page }) => {\n  await page.goto(process.env.QASENSE_TARGET_URL!);\n  await expect(page.locator('" + selector + "')).toBeVisible();\n});\n"
        suggestions.append({"id": f"test-{item['id'][8:]}", "finding_id": item["id"], "title": f"Verify: {item['title']}", "type": kind, "priority": item["severity"], "preconditions": ["Use an isolated deterministic test environment."], "steps": steps, "expected_result": expected, "evidence_source": item["evidence"], "runnable_skeleton": skeleton})
    return suggestions


def _regression_priorities(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for item in findings:
        grouped[item["entity"]].append(item)
    output = []
    for entity, items in grouped.items():
        score = min(100, round(max(SEVERITY_WEIGHT[i["severity"]] * 70 + i["confidence"] * 30 for i in items)))
        priority = "P0" if score >= 90 else "P1" if score >= 70 else "P2" if score >= 45 else "P3"
        output.append({"area": entity, "priority": priority, "priority_score": score, "reason": f"{len(items)} linked finding(s); highest severity {items[0]['severity']}", "related_findings": [i["id"] for i in items]})
    return sorted(output, key=lambda x: (x["priority_score"], x["area"]), reverse=True)


def build_analysis(base: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    findings = _deduplicate(_runtime_findings(base["runtime"]) + _visual_findings(base["visual"]) + _code_findings(base["risk"], base["code"]))
    fused_at = time.perf_counter()
    base["findings"] = findings
    base["recommendations"] = _recommendations(findings)
    base["test_suggestions"] = _test_suggestions(findings)
    base["regression_priority"] = _regression_priorities(findings)
    # A transparent presentation score: start at 100 and subtract up to 12
    # points per finding according to severity weight and evidence confidence.
    base["qa_score"] = max(0, round(100 - sum(SEVERITY_WEIGHT[item["severity"]] * item["confidence"] * 12 for item in findings)))
    completed = time.perf_counter()
    base["timing"]["fusion_ms"] = round((fused_at - started) * 1000, 3)
    base["timing"]["recommendation_test_priority_ms"] = round((completed - fused_at) * 1000, 3)
    base["status"] = "partial" if base["errors"] or any(base[x].get("status") != "available" for x in ("risk", "code", "runtime", "visual")) else "completed"
    validate_analysis(base)
    return base
