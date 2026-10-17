"""Temporary browser mutations with explicit reversible state."""

from __future__ import annotations

from typing import Any


MUTATION_TYPES = (
    "overlap", "clipping", "overflow", "hidden_element", "missing_element",
    "misalignment", "responsive_breakage", "text_truncation", "offscreen_element",
    "z_index_issue", "oversized_element", "undersized_click_target",
)


APPLY_MUTATION_SCRIPT = r"""
({defectType, targetSelector, affectedSelector}) => {
  const target = document.querySelector(targetSelector);
  if (!target) throw new Error(`Missing target: ${targetSelector}`);
  const affected = affectedSelector ? document.querySelector(affectedSelector) : null;
  const before = target.getBoundingClientRect();
  const state = {
    defectType, targetSelector, affectedSelector,
    style: target.getAttribute('style'), outerHTML: target.outerHTML, injectedId: null
  };
  window.__qasenseMutationState = state;
  const px = value => `${Math.round(value)}px`;
  if (defectType === 'overlap') {
    const box = (affected || target.previousElementSibling || target.parentElement).getBoundingClientRect();
    Object.assign(target.style, {transform:`translate(${px(box.left-before.left)}, ${px(box.top-before.top)})`, width:px(Math.max(box.width, 80)), height:px(Math.max(box.height, 32)), zIndex:'9999'});
  } else if (defectType === 'clipping') {
    Object.assign(target.style, {width:'34px', height:'16px', overflow:'hidden', clipPath:'inset(0)', whiteSpace:'nowrap'});
  } else if (defectType === 'overflow') {
    Object.assign(target.style, {position:'relative', width:'180vw', minWidth:'1200px'});
  } else if (defectType === 'hidden_element') {
    target.style.visibility = 'hidden';
  } else if (defectType === 'missing_element') {
    const marker = document.createComment('qasense-removed-element');
    target.parentNode.insertBefore(marker, target);
    window.__qasenseMutationMarker = marker; target.remove();
  } else if (defectType === 'misalignment') {
    target.style.transform = 'translate(180px, 80px)';
  } else if (defectType === 'responsive_breakage') {
    Object.assign(target.style, {minWidth:'1180px', width:'1180px'});
  } else if (defectType === 'text_truncation') {
    Object.assign(target.style, {width:'42px', maxWidth:'42px', overflow:'hidden', whiteSpace:'nowrap', textOverflow:'ellipsis'});
  } else if (defectType === 'offscreen_element') {
    Object.assign(target.style, {position:'fixed', left:'-1200px', top:'-800px'});
  } else if (defectType === 'z_index_issue') {
    const cover = document.createElement('div'); cover.id = `qasense-cover-${Date.now()}`;
    const box = target.getBoundingClientRect();
    Object.assign(cover.style, {position:'fixed', left:px(box.left), top:px(box.top), width:px(box.width), height:px(box.height), background:'rgba(255,0,0,.92)', zIndex:'10000'});
    document.body.appendChild(cover); state.injectedId = cover.id; target.style.zIndex = '-1';
  } else if (defectType === 'oversized_element') {
    Object.assign(target.style, {width:'90vw', height:'60vh', maxWidth:'none'});
  } else if (defectType === 'undersized_click_target') {
    Object.assign(target.style, {width:'10px', height:'10px', minWidth:'0', minHeight:'0', padding:'0', fontSize:'2px'});
  } else { throw new Error(`Unsupported defect type: ${defectType}`); }
  const current = document.querySelector(targetSelector);
  const after = current ? current.getBoundingClientRect() : null;
  return {...state,
    before_bbox:[before.left,before.top,before.right,before.bottom],
    after_bbox:after ? [after.left,after.top,after.right,after.bottom] : null,
    expected_region:after ? [after.left,after.top,after.right,after.bottom] : [before.left,before.top,before.right,before.bottom]};
}
"""


REVERT_MUTATION_SCRIPT = r"""
() => {
  const state = window.__qasenseMutationState;
  if (!state) return {reverted:false, reason:'no_state'};
  let target = document.querySelector(state.targetSelector);
  if (state.defectType === 'missing_element' && !target) {
    const template = document.createElement('template'); template.innerHTML = state.outerHTML.trim();
    target = template.content.firstElementChild;
    const marker = window.__qasenseMutationMarker;
    if (marker?.parentNode) marker.parentNode.replaceChild(target, marker); else document.body.appendChild(target);
  }
  if (target) { if (state.style === null) target.removeAttribute('style'); else target.setAttribute('style', state.style); }
  if (state.injectedId) document.getElementById(state.injectedId)?.remove();
  window.__qasenseMutationState = null; window.__qasenseMutationMarker = null;
  return {reverted:Boolean(target), target_exists:Boolean(document.querySelector(state.targetSelector))};
}
"""


async def apply_mutation(page: Any, defect_type: str, target_selector: str, affected_selector: str | None = None) -> dict[str, Any]:
    if defect_type not in MUTATION_TYPES:
        raise ValueError(f"Unsupported mutation {defect_type!r}")
    return await page.evaluate(APPLY_MUTATION_SCRIPT, {"defectType": defect_type, "targetSelector": target_selector, "affectedSelector": affected_selector})


async def revert_mutation(page: Any) -> dict[str, Any]:
    return await page.evaluate(REVERT_MUTATION_SCRIPT)


def apply_mutation_sync(page: Any, defect_type: str, target_selector: str, affected_selector: str | None = None) -> dict[str, Any]:
    if defect_type not in MUTATION_TYPES:
        raise ValueError(f"Unsupported mutation {defect_type!r}")
    return page.evaluate(APPLY_MUTATION_SCRIPT, {"defectType": defect_type, "targetSelector": target_selector, "affectedSelector": affected_selector})


def revert_mutation_sync(page: Any) -> dict[str, Any]:
    return page.evaluate(REVERT_MUTATION_SCRIPT)
