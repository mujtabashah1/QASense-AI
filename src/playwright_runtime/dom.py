"""Browser-side DOM extraction script."""

DOM_EXTRACTION_SCRIPT = r"""
() => {
  const selectors = ['button','a','input','img','select','textarea','nav','form','table',
    '[role="button"]','[role="link"]','[role="checkbox"]','[role="radio"]',
    '[role="navigation"]','[role="form"]','[role="table"]'];
  const candidates = Array.from(new Set(document.querySelectorAll(selectors.join(','))));
  const cssPath = (element) => {
    if (element.id) return `#${CSS.escape(element.id)}`;
    const parts = []; let node = element;
    while (node && node.nodeType === Node.ELEMENT_NODE && node !== document.documentElement) {
      let part = node.tagName.toLowerCase();
      const siblings = node.parentElement ? Array.from(node.parentElement.children).filter(x => x.tagName === node.tagName) : [];
      if (siblings.length > 1) part += `:nth-of-type(${siblings.indexOf(node) + 1})`;
      parts.unshift(part); node = node.parentElement;
    }
    return `html > ${parts.join(' > ')}`;
  };
  const implicitRole = (element) => {
    const tag = element.tagName.toLowerCase(); const type = (element.getAttribute('type') || '').toLowerCase();
    if (tag === 'a' && element.hasAttribute('href')) return 'link'; if (tag === 'button') return 'button';
    if (tag === 'select') return 'combobox'; if (tag === 'textarea') return 'textbox';
    if (tag === 'nav') return 'navigation'; if (tag === 'form') return 'form';
    if (tag === 'table') return 'table'; if (tag === 'img') return 'img';
    if (tag === 'input' && ['button','submit','reset'].includes(type)) return 'button';
    if (tag === 'input' && type === 'checkbox') return 'checkbox'; if (tag === 'input' && type === 'radio') return 'radio';
    if (tag === 'input') return 'textbox'; return '';
  };
  const accessibleName = (element) => {
    const labelledBy = element.getAttribute('aria-labelledby');
    if (labelledBy) { const text = labelledBy.split(/\s+/).map(id => document.getElementById(id)?.innerText || '').join(' ').trim(); if (text) return text; }
    if (element.labels?.length) return Array.from(element.labels).map(x => x.innerText).join(' ').trim();
    return (element.getAttribute('aria-label') || element.getAttribute('alt') || element.getAttribute('title') || element.innerText || element.value || '').trim().slice(0, 300);
  };
  return candidates.map((element, index) => {
    const rect = element.getBoundingClientRect(); const style = getComputedStyle(element);
    const visible = style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0' && rect.width > 0 && rect.height > 0 && element.getAttribute('aria-hidden') !== 'true';
    return {index, selector: cssPath(element), tag: element.tagName.toLowerCase(),
      role: (element.getAttribute('role') || implicitRole(element)).toLowerCase(),
      input_type: (element.getAttribute('type') || '').toLowerCase(),
      text: (element.innerText || element.value || '').trim().replace(/\s+/g, ' ').slice(0, 300),
      accessible_name: accessibleName(element), visible,
      enabled: !element.disabled && element.getAttribute('aria-disabled') !== 'true',
      bbox: {x: rect.x, y: rect.y, width: rect.width, height: rect.height},
      bbox_xyxy: [rect.left, rect.top, rect.right, rect.bottom],
      document_bbox: {x: rect.x + scrollX, y: rect.y + scrollY, width: rect.width, height: rect.height},
      css_classes: Array.from(element.classList), dom_id: element.id || '',
      checked: typeof element.checked === 'boolean' ? element.checked : null,
      geometry: {client_width: element.clientWidth, client_height: element.clientHeight,
        scroll_width: element.scrollWidth, scroll_height: element.scrollHeight,
        overflow_x: style.overflowX, overflow_y: style.overflowY,
        white_space: style.whiteSpace, text_overflow: style.textOverflow,
        clip_path: style.clipPath, position: style.position, z_index: style.zIndex}};
  });
}
"""
