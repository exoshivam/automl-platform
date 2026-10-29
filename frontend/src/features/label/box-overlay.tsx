import type { BBox, Sample } from '@/api/data-types';

const COLORS = ['#6366f1', '#0ea5e9', '#f59e0b', '#10b981', '#ef4444', '#a855f7', '#14b8a6', '#f97316'];
export const classColor = (classes: string[], name: string) => COLORS[Math.max(0, classes.indexOf(name)) % COLORS.length]!;

/** Read-only boxes over an image, scaled with an SVG viewBox in original-image pixels. */
export function BoxOverlay({ sample, fit, showLabels }: { sample: Sample; fit: 'cover' | 'contain'; showLabels?: boolean }) {
  const boxes = sample.labels.filter(l => l.bbox);
  if (!boxes.length || !sample.width || !sample.height) return null;
  const classes = [...new Set(sample.labels.map(l => l.class_name))];
  return (
    <svg
      className="pointer-events-none absolute inset-0 size-full"
      viewBox={`0 0 ${sample.width} ${sample.height}`}
      preserveAspectRatio={fit === 'cover' ? 'xMidYMid slice' : 'xMidYMid meet'}
      aria-hidden
    >
      {boxes.map((l, i) => {
        const c = classColor(classes, l.class_name);
        const b = l.bbox!;
        return (
          <g key={l.id ?? i}>
            <rect x={b.x} y={b.y} width={b.w} height={b.h} fill={`${c}22`} stroke={c} strokeWidth={Math.max(2, sample.width! / 200)} />
            {showLabels && (
              <text x={b.x + 4} y={b.y + Math.max(14, sample.width! / 40)} fill={c} fontSize={Math.max(12, sample.width! / 40)} fontWeight="600">
                {l.class_name}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

/** Confirmed boxes solid, suggested (not yet accepted) boxes dashed with their score: the review grid. */
export function ReviewOverlay({ width, height, classes, confirmed, suggested, fit = 'contain' }: { width: number; height: number; classes: string[]; confirmed: { class_name: string; bbox: BBox }[]; suggested: { class_name: string; bbox: BBox; confidence: number }[]; fit?: 'cover' | 'contain' }) {
  const sw = Math.max(2, width / 200);
  const fs = Math.max(12, width / 28);
  return (
    <svg className="pointer-events-none absolute inset-0 size-full" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio={fit === 'cover' ? 'xMidYMid slice' : 'xMidYMid meet'} aria-hidden>
      {confirmed.map((l, i) => {
        const c = classColor(classes, l.class_name);
        return <rect key={`c${i}`} x={l.bbox.x} y={l.bbox.y} width={l.bbox.w} height={l.bbox.h} fill={`${c}22`} stroke={c} strokeWidth={sw} />;
      })}
      {suggested.map((l, i) => {
        const c = classColor(classes, l.class_name);
        const b = l.bbox;
        return (
          <g key={`s${i}`}>
            <rect x={b.x} y={b.y} width={b.w} height={b.h} fill="none" stroke={c} strokeWidth={sw} strokeDasharray={`${sw * 3} ${sw * 2}`} />
            <rect x={b.x} y={Math.max(0, b.y - fs * 1.25)} width={fs * (l.class_name.length * 0.62 + 2.6)} height={fs * 1.25} fill={c} opacity={0.85} />
            <text x={b.x + fs * 0.25} y={Math.max(fs, b.y - fs * 0.3)} fill="#fff" fontSize={fs} fontWeight="600">{l.class_name} {Math.round(l.confidence * 100)}%</text>
          </g>
        );
      })}
    </svg>
  );
}
