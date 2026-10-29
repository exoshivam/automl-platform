import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Image as KImage, Layer, Rect, Stage, Text as KText, Transformer } from 'react-konva';
import type Konva from 'konva';
import { Check, ChevronLeft, ChevronRight, Copy, Hand, Loader2, MousePointer2, Sparkles, SquareDashed, Trash2, X, ZoomIn, ZoomOut } from 'lucide-react';
import { toast } from 'sonner';
import { useBoxSuggestions, useSaveLabels } from '@/api/data-hooks';
import { useModelVersion } from '@/api/ml';
import type { BBox, Sample, SampleLabel } from '@/api/data-types';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Checkbox, Tooltip } from '@/components/ui/misc';
import { cn } from '@/lib/utils';
import { classColor } from './box-overlay';

type Box = SampleLabel & { bbox: BBox; key: string };
type Suggestion = { key: string; class_name: string; bbox: BBox; confidence: number };

const MIN = 4;
const clampBox = (b: BBox, w: number, h: number): BBox => {
  const x = Math.max(0, Math.min(b.x, w - MIN));
  const y = Math.max(0, Math.min(b.y, h - MIN));
  return { x, y, w: Math.max(MIN, Math.min(b.w, w - x)), h: Math.max(MIN, Math.min(b.h, h - y)) };
};
const keyOf = () => crypto.randomUUID().slice(0, 8);

function useImage(url: string | null) {
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    if (!url) return;
    const i = new window.Image();
    i.onload = () => setImg(i);
    i.src = url;
    return () => setImg(null);
  }, [url]);
  return img;
}

/**
 * LBL-2 box tool + LBL-4a AI suggestions. Boxes live in original-image pixels (COCO x,y,w,h);
 * the Konva layer carries the zoom/pan transform so drawing maps 1:1 to image coordinates.
 */
export function BoxTool({
  samples,
  index,
  setIndex,
  classes,
  versionId,
  onVersion,
  onSaved,
  autoPreview,
}: {
  samples: Sample[];
  index: number;
  setIndex: (i: number) => void;
  classes: string[];
  versionId: string;
  /** Called when a save forked a new draft version (sample ids are remapped by the server). */
  onVersion: (vid: string, idMap: Record<string, string>) => void;
  onSaved: (s: Sample) => void;
  /** Hybrid detector: read-only boxes the ready-made model finds (labelled automatically at training time, not saved). */
  autoPreview?: (s: Sample) => Promise<{ class_name: string; bbox: { x: number; y: number; w: number; h: number } }[]>;
}) {
  const sample = samples[index];
  const [auto, setAuto] = useState<Record<string, { class_name: string; bbox: { x: number; y: number; w: number; h: number } }[]>>({});
  useEffect(() => {
    if (!autoPreview || !sample || auto[sample.id]) return;
    let off = false;
    autoPreview(sample).then(b => !off && setAuto(a => ({ ...a, [sample.id]: b })), () => undefined);
    return () => void (off = true);
  }, [autoPreview, sample, auto]);
  const img = useImage(sample?.url ?? null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const layerRef = useRef<Konva.Layer>(null);
  const trRef = useRef<Konva.Transformer>(null);
  const [size, setSize] = useState({ w: 800, h: 560 });
  const [view, setView] = useState({ scale: 1, x: 0, y: 0 });
  const [tool, setTool] = useState<'draw' | 'select' | 'pan'>('draw');
  const [active, setActive] = useState(classes[0] ?? '');
  // Photos taken "of" an object (camera picker / upload selector) open with that class selected.
  const hinted = sample?.capture_target;
  useEffect(() => {
    if (hinted && classes.includes(hinted)) setActive(hinted);
  }, [sample?.id, hinted, classes]);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [noObjects, setNoObjects] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [drawing, setDrawing] = useState<{ x0: number; y0: number; box: BBox } | null>(null);
  const [space, setSpace] = useState(false);
  const [suggest, setSuggest] = useState<Suggestion[]>([]);
  const [minConf, setMinConf] = useState(0.5);
  const [dirty, setDirty] = useState(false);
  const save = useSaveLabels();
  const ai = useBoxSuggestions();
  const panStart = useRef<{ x: number; y: number; vx: number; vy: number } | null>(null);
  const W = sample?.width ?? img?.naturalWidth ?? 1;
  const H = sample?.height ?? img?.naturalHeight ?? 1;

  // Load the sample's boxes when switching images.
  useEffect(() => {
    if (!sample) return;
    setBoxes(sample.labels.filter(l => l.bbox).map(l => ({ ...l, bbox: l.bbox!, key: keyOf() })));
    setNoObjects(sample.no_objects);
    setSelected(null);
    setSuggest([]);
    setDirty(false);
  }, [sample?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  // Fit the image into the stage.
  const fit = useCallback(() => {
    const s = Math.min(size.w / W, size.h / H) * 0.96;
    setView({ scale: s, x: (size.w - W * s) / 2, y: (size.h - H * s) / 2 });
  }, [size, W, H]);
  useEffect(fit, [fit, sample?.id]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => {
      const w = Math.floor(e!.contentRect.width);
      setSize({ w, h: Math.max(320, Math.min(Math.round(window.innerHeight * 0.68), Math.round(w * 0.72))) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Autosave (LBL-2): replace-per-sample, debounced.
  const persist = useCallback(
    (bx: Box[], none: boolean) => {
      if (!sample) return;
      const labels: SampleLabel[] = bx.map(({ key: _k, ...l }) => (void _k, { class_name: l.class_name, bbox: l.bbox, source: l.source ?? 'user', confidence: l.confidence ?? null }));
      save.mutate(
        { vid: versionId, upserts: [{ sample_id: sample.id, labels, no_objects: none }] },
        {
          onSuccess: r => {
            setDirty(false);
            onSaved({ ...sample, labels, no_objects: none });
            if (r.forked || r.dataset_version_id !== versionId) onVersion(r.dataset_version_id, r.sample_id_map ?? {});
          },
          onError: e => toast.error(`Couldn't save: ${e.message}`),
        },
      );
    },
    [sample, versionId, save, onSaved, onVersion],
  );
  useEffect(() => {
    if (!dirty) return;
    const t = setTimeout(() => persist(boxes, noObjects), 700);
    return () => clearTimeout(t);
  }, [boxes, noObjects, dirty, persist]);
  const change = (next: Box[]) => {
    setBoxes(next);
    if (next.length) setNoObjects(false);
    setDirty(true);
  };

  // Attach the transformer to the selected box.
  useEffect(() => {
    const tr = trRef.current;
    if (!tr) return;
    const node = selected ? layerRef.current?.findOne(`#box-${selected}`) : null;
    tr.nodes(node && tool !== 'pan' ? [node] : []);
    tr.getLayer()?.batchDraw();
  }, [selected, boxes, tool]);

  const go = (d: number) => {
    const i = Math.max(0, Math.min(samples.length - 1, index + d));
    if (dirty) persist(boxes, noObjects);
    setIndex(i);
  };

  const copyToNext = () => {
    if (index >= samples.length - 1) return;
    const next = samples[index + 1]!;
    const labels: SampleLabel[] = boxes.map(b => ({ class_name: b.class_name, bbox: b.bbox, source: 'user' }));
    if (dirty) persist(boxes, noObjects);
    save.mutate(
      { vid: versionId, upserts: [{ sample_id: next.id, labels: [...next.labels.filter(l => l.bbox), ...labels], no_objects: false }] },
      {
        onSuccess: r => {
          onSaved({ ...next, labels: [...next.labels.filter(l => l.bbox), ...labels], no_objects: false });
          if (r.forked || r.dataset_version_id !== versionId) onVersion(r.dataset_version_id, r.sample_id_map ?? {});
          setIndex(index + 1);
          toast.success(`Copied ${labels.length} box${labels.length === 1 ? '' : 'es'} to the next image`);
        },
      },
    );
  };

  const askAi = () =>
    sample &&
    ai.mutate(
      { vid: versionId, sampleIds: [sample.id], classes },
      {
        onSuccess: r => {
          const list = (r.suggestions[sample.id] ?? []).map(s => ({ ...s, key: keyOf(), bbox: clampBox(s.bbox, W, H) }));
          setSuggest(list);
          if (!list.length) {
            const best = Object.entries(r.max_scores?.[sample.id] ?? {}).sort((a, b) => b[1] - a[1])[0];
            toast.info(r.model.startsWith('own:')
              ? `Your model found no ${classes.join(' or ')} here${best ? ` (best ${best[0]} match ${Math.round(best[1] * 100)}%)` : ''}. Draw it, and the next training will learn this kind of photo; more varied photos help.`
              : 'No objects suggested for this image');
          }
        },
        onError: e => toast.error(e.message),
      },
    );
  const visibleSuggest = suggest.filter(s => s.confidence >= minConf);
  // Ready-made previews covered by a box the user drew (e.g. "cell phone" on their charger) are hidden, with the
  // server's keep_pseudo() rule: IoU ≥ 0.3, or ≥ 70% of the auto box inside the user box.
  const visibleAuto = (sample ? auto[sample.id] ?? [] : []).filter(a => !boxes.some(b => covered(a.bbox, b.bbox)));
  const accept = (ss: Suggestion[]) => {
    change([...boxes, ...ss.map(s => ({ key: keyOf(), class_name: s.class_name, bbox: s.bbox, source: 'ai_prelabel' as const, confidence: s.confidence }))]);
    setSuggest(cur => cur.filter(c => !ss.includes(c)));
  };

  // Keyboard: 1-9 class, ←/→ images, Del delete, C copy, N no objects, S suggest, Enter accept all, Esc deselect, Space pan.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest('input,select,textarea')) return;
      const n = Number(e.key);
      if (n >= 1 && n <= classes.length) {
        const c = classes[n - 1]!;
        setActive(c);
        if (selected) change(boxes.map(b => (b.key === selected ? { ...b, class_name: c, source: 'user' } : b)));
      } else if (e.key === 'ArrowRight' || e.key === 'd') go(1);
      else if (e.key === 'ArrowLeft' || e.key === 'a') go(-1);
      else if ((e.key === 'Delete' || e.key === 'Backspace') && selected && !e.shiftKey) {
        change(boxes.filter(b => b.key !== selected));
        setSelected(null);
      } else if (e.key === 'c') copyToNext();
      else if (e.key === 'n') (setNoObjects(v => !v), setDirty(true));
      else if (e.key === 's') askAi();
      else if (e.key === 'Enter' && visibleSuggest.length) accept(visibleSuggest);
      else if (e.key === 'Escape') setSelected(null);
      else if (e.code === 'Space') (e.preventDefault(), setSpace(true));
      else return;
    };
    const up = (e: KeyboardEvent) => e.code === 'Space' && setSpace(false);
    window.addEventListener('keydown', onKey);
    window.addEventListener('keyup', up);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('keyup', up);
    };
  });

  const panning = tool === 'pan' || space;
  const imgPoint = () => layerRef.current?.getRelativePointerPosition() ?? { x: 0, y: 0 };

  const onDown = (e: Konva.KonvaEventObject<MouseEvent | TouchEvent>) => {
    const pos = stageRef.current!.getPointerPosition()!;
    if (panning || ('button' in e.evt && e.evt.button === 1)) {
      panStart.current = { x: pos.x, y: pos.y, vx: view.x, vy: view.y };
      return;
    }
    const onBox = e.target.name() === 'box';
    if (onBox) return;
    setSelected(null);
    if (tool !== 'draw' || !active) return;
    const p = imgPoint();
    if (p.x < 0 || p.y < 0 || p.x > W || p.y > H) return;
    setDrawing({ x0: p.x, y0: p.y, box: { x: p.x, y: p.y, w: 0, h: 0 } });
  };
  const onMove = () => {
    if (panStart.current) {
      const pos = stageRef.current!.getPointerPosition()!;
      setView(v => ({ ...v, x: panStart.current!.vx + pos.x - panStart.current!.x, y: panStart.current!.vy + pos.y - panStart.current!.y }));
      return;
    }
    if (!drawing) return;
    const p = imgPoint();
    const x = Math.max(0, Math.min(drawing.x0, p.x));
    const y = Math.max(0, Math.min(drawing.y0, p.y));
    setDrawing({ ...drawing, box: { x, y, w: Math.min(W, Math.max(drawing.x0, p.x)) - x, h: Math.min(H, Math.max(drawing.y0, p.y)) - y } });
  };
  const onUp = () => {
    panStart.current = null;
    if (!drawing) return;
    if (drawing.box.w >= MIN && drawing.box.h >= MIN) {
      const b: Box = { key: keyOf(), class_name: active, bbox: clampBox(drawing.box, W, H), source: 'user' };
      change([...boxes, b]);
      setSelected(b.key);
    }
    setDrawing(null);
  };
  const zoom = (factor: number, at?: { x: number; y: number }) => {
    const c = at ?? { x: size.w / 2, y: size.h / 2 };
    setView(v => {
      const s = Math.max(0.1, Math.min(8, v.scale * factor));
      return { scale: s, x: c.x - ((c.x - v.x) / v.scale) * s, y: c.y - ((c.y - v.y) / v.scale) * s };
    });
  };

  const counts = useMemo(() => {
    const m: Record<string, number> = {};
    for (const b of boxes) m[b.class_name] = (m[b.class_name] ?? 0) + 1;
    return m;
  }, [boxes]);

  if (!sample) return null;
  const strokeW = 2 / view.scale;

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_300px]">
      <Card className="overflow-hidden">
        <div className="flex flex-wrap items-center gap-2 border-b p-2">
          <div className="flex rounded-md bg-muted p-0.5" role="radiogroup" aria-label="Tool">
            {(
              [
                ['draw', SquareDashed, 'Draw box'],
                ['select', MousePointer2, 'Select / move'],
                ['pan', Hand, 'Pan (or hold Space)'],
              ] as const
            ).map(([t, Icon, label]) => (
              <Tooltip key={t} content={label}>
                <button type="button" role="radio" aria-checked={tool === t} aria-label={label} onClick={() => setTool(t)} className={cn('cursor-pointer rounded p-1.5', tool === t && 'bg-card shadow-sm')}>
                  <Icon className="size-4" />
                </button>
              </Tooltip>
            ))}
          </div>
          <Button variant="ghost" size="icon" aria-label="Zoom in" onClick={() => zoom(1.25)}>
            <ZoomIn />
          </Button>
          <Button variant="ghost" size="icon" aria-label="Zoom out" onClick={() => zoom(0.8)}>
            <ZoomOut />
          </Button>
          <Button variant="ghost" size="sm" onClick={fit}>
            Fit
          </Button>
          <span className="mx-1 h-5 w-px bg-border" />
          <div className="flex flex-wrap gap-1" role="radiogroup" aria-label="Class for new boxes">
            {classes.map((c, i) => (
              <button
                key={c}
                type="button"
                role="radio"
                aria-checked={active === c}
                onClick={() => {
                  setActive(c);
                  if (selected) change(boxes.map(b => (b.key === selected ? { ...b, class_name: c, source: 'user' } : b)));
                }}
                className={cn('flex cursor-pointer items-center gap-1.5 rounded-md border px-2 py-1 text-xs', active === c ? 'border-transparent text-white' : 'hover:bg-accent')}
                style={active === c ? { background: classColor(classes, c) } : undefined}
              >
                <kbd className="opacity-70">{i + 1}</kbd> {c}
              </button>
            ))}
          </div>
          <span className="ml-auto flex items-center gap-1 text-xs text-muted-foreground" aria-live="polite">
            {save.isPending || dirty ? (
              <>
                <Loader2 className="size-3 animate-spin" /> Saving…
              </>
            ) : (
              <>
                <Check className="size-3 text-success" /> Saved
              </>
            )}
          </span>
        </div>

        <div ref={wrapRef} className={cn('relative bg-muted/60', panning ? 'cursor-grab' : tool === 'draw' ? 'cursor-crosshair' : 'cursor-default')} style={{ touchAction: 'none' }}>
          <Stage
            ref={stageRef}
            width={size.w}
            height={size.h}
            onMouseDown={onDown}
            onTouchStart={onDown}
            onMouseMove={onMove}
            onTouchMove={onMove}
            onMouseUp={onUp}
            onTouchEnd={onUp}
            onWheel={e => {
              e.evt.preventDefault();
              zoom(e.evt.deltaY < 0 ? 1.12 : 0.89, stageRef.current!.getPointerPosition() ?? undefined);
            }}
          >
            <Layer ref={layerRef} scaleX={view.scale} scaleY={view.scale} x={view.x} y={view.y}>
              {img && <KImage image={img} width={W} height={H} />}
              {visibleAuto.map((b, i) => (
                <Rect key={`auto-${i}`} x={b.bbox.x} y={b.bbox.y} width={b.bbox.w} height={b.bbox.h} stroke="#9ca3af" strokeWidth={strokeW} dash={[3 / view.scale, 3 / view.scale]} opacity={0.85} listening={false} />
              ))}
              {visibleAuto.map((b, i) => (
                <KText key={`autot-${i}`} x={b.bbox.x + 2 / view.scale} y={b.bbox.y + 2 / view.scale} text={`auto · ${b.class_name} · not saved`} fontSize={11 / view.scale} fill="#e5e7eb" listening={false} />
              ))}
              {visibleSuggest.map(s => (
                <Rect key={s.key} x={s.bbox.x} y={s.bbox.y} width={s.bbox.w} height={s.bbox.h} stroke={classColor(classes, s.class_name)} strokeWidth={strokeW} dash={[6 / view.scale, 4 / view.scale]} listening={false} />
              ))}
              {visibleSuggest.map(s => (
                <KText key={`t${s.key}`} x={s.bbox.x + 2 / view.scale} y={s.bbox.y - 16 / view.scale} text={`${s.class_name} ${Math.round(s.confidence * 100)}%`} fontSize={12 / view.scale} fill={classColor(classes, s.class_name)} listening={false} />
              ))}
              {boxes.map(b => (
                <Rect
                  key={b.key}
                  id={`box-${b.key}`}
                  name="box"
                  x={b.bbox.x}
                  y={b.bbox.y}
                  width={b.bbox.w}
                  height={b.bbox.h}
                  stroke={classColor(classes, b.class_name)}
                  strokeWidth={(selected === b.key ? 3 : 2) / view.scale}
                  fill={`${classColor(classes, b.class_name)}26`}
                  draggable={!panning}
                  onMouseDown={() => !panning && setSelected(b.key)}
                  onTouchStart={() => !panning && setSelected(b.key)}
                  onDragEnd={e => change(boxes.map(x => (x.key === b.key ? { ...x, source: 'user', bbox: clampBox({ ...x.bbox, x: e.target.x(), y: e.target.y() }, W, H) } : x)))}
                  onTransformEnd={e => {
                    const n = e.target;
                    const bb = clampBox({ x: n.x(), y: n.y(), w: n.width() * n.scaleX(), h: n.height() * n.scaleY() }, W, H);
                    n.scaleX(1);
                    n.scaleY(1);
                    change(boxes.map(x => (x.key === b.key ? { ...x, source: 'user', bbox: bb } : x)));
                  }}
                />
              ))}
              {drawing && <Rect x={drawing.box.x} y={drawing.box.y} width={drawing.box.w} height={drawing.box.h} stroke={classColor(classes, active)} strokeWidth={strokeW} dash={[4 / view.scale, 3 / view.scale]} listening={false} />}
              <Transformer ref={trRef} rotateEnabled={false} keepRatio={false} ignoreStroke anchorSize={9} borderStroke="#6366f1" anchorStroke="#6366f1" boundBoxFunc={(o, n) => (n.width < MIN || n.height < MIN ? o : n)} />
            </Layer>
          </Stage>
          {!img && (
            <div className="absolute inset-0 grid place-items-center text-sm text-muted-foreground">
              <Loader2 className="size-5 animate-spin" />
            </div>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2 border-t p-2">
          <Button variant="outline" size="sm" onClick={() => go(-1)} disabled={index === 0} aria-label="Previous image (←)">
            <ChevronLeft /> Prev
          </Button>
          <span className="text-sm tabular-nums text-muted-foreground">
            {index + 1} / {samples.length}
          </span>
          <Button variant="outline" size="sm" onClick={() => go(1)} disabled={index >= samples.length - 1} aria-label="Next image (→)">
            Next <ChevronRight />
          </Button>
          <Button variant="ghost" size="sm" onClick={copyToNext} disabled={!boxes.length || index >= samples.length - 1}>
            <Copy /> Copy boxes to next
          </Button>
          <label className="flex items-center gap-2 text-sm">
            <Checkbox
              checked={noObjects}
              onCheckedChange={c => {
                setNoObjects(!!c);
                if (c) setBoxes([]);
                setDirty(true);
              }}
            />
            No objects in this image
          </label>
          <span className="ml-auto hidden text-xs text-muted-foreground lg:inline">Keys: 1-9 class · ←/→ image · Del delete · C copy · N none · S suggest · Enter accept · Space pan</span>
        </div>
      </Card>

      <div className="grid content-start gap-4">
        <Card className="grid gap-3 p-4">
          <div className="flex items-center justify-between">
            <h3 className="flex items-center gap-2 text-sm font-semibold">
              <Sparkles className="size-4 text-primary" /> AI box suggestions
            </h3>
            {ai.data && <SuggestSource model={ai.data.model} />}
          </div>
          <Button variant="outline" onClick={askAi} disabled={ai.isPending}>
            {ai.isPending ? <Loader2 className="animate-spin" /> : <Sparkles />} Suggest boxes <kbd className="text-[10px] opacity-60">S</kbd>
          </Button>
          {suggest.length > 0 && (
            <>
              <label className="grid gap-1 text-xs text-muted-foreground">
                Hide below {Math.round(minConf * 100)}% confidence
                <input type="range" min={0} max={0.95} step={0.05} value={minConf} onChange={e => setMinConf(Number(e.target.value))} className="accent-[var(--primary)]" />
              </label>
              <ul className="grid gap-1.5">
                {visibleSuggest.map(s => (
                  <li key={s.key} className="flex items-center gap-2 rounded-md border px-2 py-1 text-sm">
                    <span className="size-2.5 rounded-sm" style={{ background: classColor(classes, s.class_name) }} />
                    <span className="flex-1">{s.class_name}</span>
                    <span className="tabular-nums text-xs text-muted-foreground">{Math.round(s.confidence * 100)}%</span>
                    <Button variant="ghost" size="icon" className="size-7" aria-label={`Accept ${s.class_name}`} onClick={() => accept([s])}>
                      <Check className="text-success" />
                    </Button>
                    <Button variant="ghost" size="icon" className="size-7" aria-label={`Reject ${s.class_name}`} onClick={() => setSuggest(cur => cur.filter(c => c !== s))}>
                      <X className="text-destructive" />
                    </Button>
                  </li>
                ))}
              </ul>
              <div className="flex gap-2">
                <Button size="sm" className="flex-1" onClick={() => accept(visibleSuggest)} disabled={!visibleSuggest.length}>
                  Accept all ({visibleSuggest.length})
                </Button>
                <Button size="sm" variant="ghost" onClick={() => setSuggest([])}>
                  Discard
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">Accepted boxes become normal boxes you can move and resize.</p>
            </>
          )}
        </Card>

        <Card className="grid gap-2 p-4">
          <h3 className="text-sm font-semibold">Boxes on this image ({boxes.length})</h3>
          {!boxes.length && <p className="text-xs text-muted-foreground">{noObjects ? 'Marked as containing no objects.' : 'Drag on the image to draw a box.'}</p>}
          <ul className="grid gap-1.5">
            {boxes.map(b => (
              <li
                key={b.key}
                className={cn('flex items-center gap-2 rounded-md border px-2 py-1 text-sm', selected === b.key && 'border-primary bg-primary/6')}
                onMouseEnter={() => setSelected(b.key)}
              >
                <span className="size-2.5 rounded-sm" style={{ background: classColor(classes, b.class_name) }} />
                <select
                  aria-label="Box class"
                  className="h-7 flex-1 rounded border border-input bg-card px-1 text-sm"
                  value={b.class_name}
                  onChange={e => change(boxes.map(x => (x.key === b.key ? { ...x, class_name: e.target.value, source: 'user' } : x)))}
                >
                  {classes.map(c => (
                    <option key={c}>{c}</option>
                  ))}
                </select>
                {b.source === 'ai_prelabel' && <Badge variant="info" className="px-1.5 text-[10px]">AI</Badge>}
                <Button variant="ghost" size="icon" className="size-7" aria-label="Delete box" onClick={() => change(boxes.filter(x => x.key !== b.key))}>
                  <Trash2 />
                </Button>
              </li>
            ))}
          </ul>
          {Object.keys(counts).length > 0 && (
            <p className="text-xs text-muted-foreground">
              {Object.entries(counts)
                .map(([c, n]) => `${c} ×${n}`)
                .join(' · ')}
            </p>
          )}
        </Card>
      </div>
    </div>
  );
}

type Rect4 = { x: number; y: number; w: number; h: number };
/** Auto box `a` is covered by user box `b`: IoU ≥ 0.3, or ≥ 70% of a's area inside b (a small box in a big one). */
export function covered(a: Rect4, b: Rect4) {
  const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  const inter = ix * iy;
  const areaA = a.w * a.h;
  const union = areaA + b.w * b.h - inter;
  return (union > 0 && inter / union >= 0.3) || (areaA > 0 && inter / areaA >= 0.7);
}

/** Where suggestions come from: the project's own trained model, the ready-made COCO detector, or the demo stub. */
function SuggestSource({ model }: { model: string }) {
  const own = model.startsWith('own:') ? model.slice(4) : null;
  const mv = useModelVersion(own ?? undefined);
  if (own) return <Badge variant="success" className="text-[10px]">Your model{mv.data ? ` (v${mv.data.version_no})` : ''}</Badge>;
  if (model.includes('coco') || model.includes('ssdlite')) return <Badge variant="info" className="text-[10px]">Ready-made model</Badge>;
  return (
    <Tooltip content="A placeholder detector until you train your own model; suggestions are only roughly placed.">
      <Badge variant="warning" className="text-[10px]" tabIndex={0}>Demo</Badge>
    </Tooltip>
  );
}
