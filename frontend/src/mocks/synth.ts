// Synthetic demo images for the "Load sample data" button and seeded datasets.
// Detection scenes carry their ground-truth boxes so AI suggestions can land on real objects.
import type { BBox } from '@/api/data-types';

let seed = 7;
export const rand = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;
export const pick = <T,>(xs: T[]) => xs[Math.floor(rand() * xs.length)]!;

const svgUrl = (svg: string) => `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;

const PALETTES: Record<string, [string, string]> = {
  healthy: ['#2f855a', '#9ae6b4'],
  early_blight: ['#975a16', '#f6e05e'],
  late_blight: ['#4a5568', '#a0aec0'],
  defect: ['#c53030', '#feb2b2'],
  ok: ['#2b6cb0', '#bee3f8'],
};

/** A leaf-ish classification image whose colours depend on the class. */
export function classImage(label: string, i: number): { url: string; w: number; h: number } {
  const [dark, light] = PALETTES[label] ?? [`hsl(${(label.length * 47) % 360} 45% 35%)`, `hsl(${(label.length * 47) % 360} 60% 80%)`];
  const r = 70 + ((i * 13) % 40);
  const spots = Array.from({ length: 3 + (i % 4) }, (_, k) => `<circle cx="${80 + ((k * 37 + i * 11) % 100)}" cy="${70 + ((k * 23 + i * 7) % 100)}" r="${5 + (k % 3) * 3}" fill="${dark}" opacity=".7"/>`).join('');
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256"><rect width="256" height="256" fill="${light}"/><ellipse cx="128" cy="128" rx="${r + 30}" ry="${r}" transform="rotate(${(i * 29) % 180} 128 128)" fill="${dark}" opacity=".85"/>${spots}<text x="10" y="248" font-family="sans-serif" font-size="12" fill="#1a202c" opacity=".6">${label} #${i}</text></svg>`;
  return { url: svgUrl(svg), w: 256, h: 256 };
}

/** A site photo with workers; returns the true boxes per class. */
export function detectionScene(classes: string[], i: number): { url: string; w: number; h: number; boxes: { class_name: string; bbox: BBox }[] } {
  const W = 640;
  const H = 480;
  const people = 1 + (i % 4);
  const boxes: { class_name: string; bbox: BBox }[] = [];
  let shapes = '';
  for (let p = 0; p < people; p++) {
    const x = 40 + p * 150 + ((i * 17 + p * 31) % 40);
    const y = 140 + ((i * 7 + p * 13) % 60);
    const helmet = (i + p) % 3 !== 0;
    shapes += `<rect x="${x}" y="${y + 60}" width="70" height="170" rx="12" fill="#4a5568"/>`;
    shapes += `<circle cx="${x + 35}" cy="${y + 40}" r="24" fill="#f6ad55"/>`;
    if (helmet) {
      shapes += `<path d="M${x + 9} ${y + 34} a26 22 0 0 1 52 0 z" fill="#ecc94b" stroke="#975a16" stroke-width="2"/>`;
      if (classes[0]) boxes.push({ class_name: classes[0], bbox: { x: x + 7, y: y + 12, w: 56, h: 26 } });
    } else if (classes[2]) boxes.push({ class_name: classes[2], bbox: { x: x + 9, y: y + 14, w: 52, h: 52 } });
    if ((i + p) % 2 === 0) {
      shapes += `<rect x="${x + 4}" y="${y + 70}" width="62" height="70" fill="#ed8936" opacity=".9"/><rect x="${x + 4}" y="${y + 95}" width="62" height="8" fill="#edf2f7"/>`;
      if (classes[1]) boxes.push({ class_name: classes[1], bbox: { x: x + 4, y: y + 70, w: 62, h: 70 } });
    }
  }
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><rect width="${W}" height="${H}" fill="#cbd5e0"/><rect y="${H - 90}" width="${W}" height="90" fill="#a0aec0"/><rect x="${W - 160}" y="40" width="120" height="${H - 130}" fill="#718096" opacity=".5"/>${shapes}<text x="12" y="${H - 12}" font-family="sans-serif" font-size="14" fill="#1a202c" opacity=".55">site cam ${i}</text></svg>`;
  return { url: svgUrl(svg), w: W, h: H, boxes };
}

const TEXTS: Record<string, string[]> = {
  positive: ['Bahut accha product, delivery bhi fast thi!', 'Loved the battery life, totally worth it.', 'Customer support was super helpful.', 'Quality ekdum mast hai.'],
  negative: ['Packaging tooti hui thi, very disappointed.', 'Stopped working after two days.', 'Refund abhi tak nahi aaya.', 'Worst experience, never again.'],
  neutral: ['Delivered on Tuesday.', 'Size is as described.', 'Colour thoda alag hai photo se.', 'Average product for the price.'],
};
export function textSample(label: string, i: number): string {
  const pool = TEXTS[label] ?? [`Sample text about ${label}`, `Another ${label} example`, `${label} related message`];
  return `${pool[i % pool.length]}${i >= pool.length ? ` (${Math.floor(i / pool.length) + 1})` : ''}`;
}
