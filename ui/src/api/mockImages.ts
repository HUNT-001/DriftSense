import type { Candidate } from '../types/contracts';
import { viridis } from '../lib/format';

/** Synthetic search-image size used by demo mode (px). */
export const MOCK_IMAGE_W = 400;
export const MOCK_IMAGE_H = 250;
const PERIOD = 32;

function svgDataUri(svg: string): string {
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

function frame(w: number, h: number, body: string, label: string): string {
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="${label}">` +
    body +
    `<text x="6" y="${h - 6}" font-family="system-ui,sans-serif" font-size="11" fill="#fff" stroke="#000" stroke-width="0.6" paint-order="stroke">SYNTHETIC</text>` +
    `</svg>`
  );
}

/** Viridis heatmap: weak periodic background (every copy looks alike) plus peaks at candidates. */
export function heatmapSvg(cands: Candidate[]): string {
  const cols = 25;
  const rows = 16;
  const cw = MOCK_IMAGE_W / cols;
  const ch = MOCK_IMAGE_H / rows;
  const top = Math.max(...cands.map((c) => c.score), 1e-6);
  const vals: number[] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = (c + 0.5) * cw;
      const y = (r + 0.5) * ch;
      let v = 0.18 * Math.cos((Math.PI * x) / PERIOD) ** 2;
      for (const k of cands) {
        const d2 = (x - k.x_px) ** 2 + (y - k.y_px) ** 2;
        v += (k.score / top) * Math.exp(-d2 / (2 * 14 * 14));
      }
      vals.push(v);
    }
  }
  const max = Math.max(...vals);
  let body = '';
  vals.forEach((v, i) => {
    const r = Math.floor(i / cols);
    const c = i % cols;
    body += `<rect x="${(c * cw).toFixed(1)}" y="${(r * ch).toFixed(1)}" width="${(cw + 0.5).toFixed(1)}" height="${(ch + 0.5).toFixed(1)}" fill="${viridis(v / max)}"/>`;
  });
  return svgDataUri(frame(MOCK_IMAGE_W, MOCK_IMAGE_H, body, 'Synthetic candidate score heatmap'));
}

function stripes(): string {
  let s = `<rect width="${MOCK_IMAGE_W}" height="${MOCK_IMAGE_H}" fill="#3a3a3a"/>`;
  for (let x = 4; x < MOCK_IMAGE_W; x += PERIOD) {
    s += `<rect x="${x}" y="0" width="14" height="${MOCK_IMAGE_H}" fill="#9a9a9a"/>`;
  }
  return s;
}

/** Overlay: gray line pattern with the rank-1 template box outlined. */
export function overlaySvg(cands: Candidate[]): string {
  const c = cands[0];
  let body = stripes();
  if (c) {
    body +=
      `<rect x="${c.x_px}" y="${c.y_px}" width="64" height="48" fill="none" stroke="#fde725" stroke-width="3"/>` +
      `<rect x="${c.x_px}" y="${c.y_px}" width="64" height="48" fill="none" stroke="#000" stroke-width="1" stroke-dasharray="4 3"/>`;
  }
  return svgDataUri(frame(MOCK_IMAGE_W, MOCK_IMAGE_H, body, 'Synthetic overlay of the matched location'));
}

/** Candidates: numbered markers (shape + number, not colour alone). */
export function candidatesSvg(cands: Candidate[]): string {
  let body = stripes();
  for (const c of cands.slice(0, 5)) {
    const cx = c.x_px + 32;
    const cy = c.y_px + 24;
    body +=
      `<circle cx="${cx}" cy="${cy}" r="13" fill="#fff" stroke="#000" stroke-width="2"/>` +
      `<text x="${cx}" y="${cy + 5}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="700" fill="#000">${c.rank}</text>`;
  }
  return svgDataUri(frame(MOCK_IMAGE_W, MOCK_IMAGE_H, body, 'Synthetic candidate markers by rank'));
}
