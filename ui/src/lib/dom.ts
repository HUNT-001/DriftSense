/** Tiny DOM builder. Text is always set via text nodes (never innerHTML) so API data can't inject markup. */
type Child = Node | string | number | null | undefined | false;
type Attrs = Record<string, string | number | boolean | null | undefined | EventListener>;

export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Attrs | null = null,
  ...children: (Child | Child[])[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined || v === false) continue;
      if (k.startsWith('on') && typeof v === 'function') {
        el.addEventListener(k.slice(2).toLowerCase(), v as EventListener);
      } else if (k === 'class') el.className = String(v);
      else if (v === true) el.setAttribute(k, '');
      else el.setAttribute(k, String(v));
    }
  }
  append(el, children);
  return el;
}

export function append(el: Element, children: (Child | Child[])[]): void {
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

export function clear(el: Element): void {
  while (el.firstChild) el.removeChild(el.firstChild);
}

let idSeq = 0;
export function uid(prefix = 'id'): string {
  return `${prefix}-${++idSeq}`;
}

/** Percentage width via CSSOM (allowed under a strict style-src CSP, unlike style attributes). */
export function setWidthPct(el: HTMLElement, pct: number): HTMLElement {
  el.style.setProperty('width', `${Math.max(0, Math.min(100, pct)).toFixed(2)}%`);
  return el;
}

export function setLeftPct(el: HTMLElement, pct: number): HTMLElement {
  el.style.setProperty('left', `${Math.max(0, Math.min(100, pct)).toFixed(2)}%`);
  return el;
}
