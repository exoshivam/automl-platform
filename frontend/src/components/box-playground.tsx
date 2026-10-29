import { Camera, CameraOff, ImageUp, Loader2, Radio, ScanSearch } from 'lucide-react';
import { useEffect, useRef, useState, type ReactNode } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Switch } from '@/components/ui/misc';
import { classColor } from '@/features/label/box-overlay';
import { grabFrame, useAbortOnUnmount, useCamera } from '@/lib/camera';
import { cn } from '@/lib/utils';

export interface BoxItem {
  box: { x: number; y: number; w: number; h: number };
  /** null = below threshold ("Other"). */
  label: string | null;
  score: number;
  top3?: { name: string; score: number }[];
  /** Hybrid detector: which model found it. Custom boxes are dashed and tagged. */
  source?: 'ready-made' | 'custom';
}
export interface BoxResult {
  image: { width: number; height: number };
  items: BoxItem[];
  threshold?: number;
  latency_ms?: number;
  model_load_ms?: number;
}

const LIVE_WIDTH = 640;

/**
 * Upload or live webcam → a box on every detection (label + score, or `unknownLabel` in grey), top-3 on hover.
 * Runs only while mounted and visible: the camera, the live loop and in-flight requests stop on leave / hide.
 */
export function BoxPlayground({ title, description, badge, names, detect, unknownLabel = 'Other', emptyText, icon: Icon = ScanSearch }: {
  title: ReactNode;
  description: ReactNode;
  badge?: ReactNode;
  /** Known labels, for stable colours. */
  names: string[];
  detect: (blob: Blob, signal: AbortSignal) => Promise<BoxResult>;
  unknownLabel?: string;
  emptyText: string;
  icon?: typeof ScanSearch;
}) {
  const [src, setSrc] = useState<string>();
  const [cam, setCam] = useState(false);
  const [live, setLive] = useState(false);
  const [result, setResult] = useState<BoxResult | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string>();
  const [camError, setCamError] = useState<string>();
  const { videoRef: video, paused, resume } = useCamera({ enabled: cam, facing: 'user', onError: e => (setCamError(e instanceof Error ? e.message : 'Camera unavailable'), setCam(false)) });
  const nextSignal = useAbortOnUnmount();
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true; // StrictMode remounts: set it again, not only clear it
    return () => void (mounted.current = false);
  }, []);
  useEffect(() => { if (!cam || paused) setLive(false); }, [cam, paused]);

  const run = async (blob: Blob) => {
    setPending(true);
    setError(undefined);
    try {
      const r = await detect(blob, nextSignal());
      if (mounted.current) setResult(r);
    } catch (e) {
      if (mounted.current && !(e instanceof DOMException && e.name === 'AbortError')) setError(e instanceof Error ? e.message : 'Failed');
    } finally {
      if (mounted.current) setPending(false);
    }
  };

  // Live: one request in flight at a time, ~640 px frames.
  useEffect(() => {
    if (!live || pending) return;
    const t = setTimeout(async () => { const b = await grabFrame(video.current, LIVE_WIDTH); if (b) await run(b); }, 120);
    return () => clearTimeout(t);
  }, [live, pending]); // eslint-disable-line react-hooks/exhaustive-deps

  const items = result?.items ?? [];
  const color = (l: string | null) => (l ? classColor(names, l) : 'var(--muted-foreground)');
  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_18rem]">
      <Card>
        <CardHeader className="flex-row flex-wrap items-center justify-between gap-2">
          <div>
            <CardTitle className="flex items-center gap-2 text-base">{title} {badge}</CardTitle>
            <CardDescription>{description}</CardDescription>
          </div>
          <div className="flex gap-2">
            <label className="inline-flex h-8 cursor-pointer items-center gap-2 rounded-md border px-3 text-xs font-medium hover:bg-accent">
              <ImageUp className="size-4" /> Choose photo
              <input type="file" accept="image/*" className="sr-only" onChange={e => {
                const f = e.target.files?.[0];
                if (!f) return;
                setCam(false);
                setResult(null);
                setSrc(URL.createObjectURL(f));
                void run(f);
                e.target.value = '';
              }} />
            </label>
            <Button variant="outline" size="sm" onClick={() => (setCamError(undefined), setResult(null), setCam(c => !c))}>{cam ? <CameraOff /> : <Camera />} {cam ? 'Stop camera' : 'Camera'}</Button>
          </div>
        </CardHeader>
        <CardContent className="grid gap-3">
          {camError && <p className="text-sm text-destructive" role="alert">{camError}</p>}
          {/* The wrapper shrinks to the media, so %-positioned boxes line up even when the height is capped. */}
          <div className="flex justify-center rounded-lg bg-muted">
            <div className="relative inline-block">
              {cam ? (
                <video ref={video} autoPlay playsInline muted className="block max-h-[65vh] max-w-full rounded-lg" />
              ) : src ? (
                <img src={src} alt="Input" className="block max-h-[65vh] max-w-full rounded-lg" />
              ) : (
                <div className="grid h-64 w-[min(36rem,80vw)] place-items-center text-sm text-muted-foreground">No image yet</div>
              )}
              {result && (cam || src) && items.map((f, i) => {
                const W = result.image.width, H = result.image.height;
                return (
                  <div
                    key={i}
                    tabIndex={0}
                    className={cn('group absolute rounded-sm border-2 outline-none', f.source === 'custom' && 'border-dashed')}
                    style={{ left: `${(f.box.x / W) * 100}%`, top: `${(f.box.y / H) * 100}%`, width: `${(f.box.w / W) * 100}%`, height: `${(f.box.h / H) * 100}%`, borderColor: color(f.label) }}
                    aria-label={`${f.label ?? unknownLabel} ${(f.score * 100).toFixed(0)}%`}
                  >
                    <span className="absolute -top-6 left-[-2px] whitespace-nowrap rounded px-1.5 py-0.5 text-xs font-semibold text-white" style={{ background: color(f.label) }}>
                      {f.label ?? unknownLabel} {(f.score * 100).toFixed(0)}%
                      {f.source === 'custom' && <span className="ml-1 rounded-sm bg-black/25 px-1 text-[10px] uppercase tracking-wide">custom</span>}
                    </span>
                    {!!f.top3?.length && (
                      <div className="pointer-events-none absolute left-full top-0 z-10 ml-2 hidden w-44 rounded-md border bg-popover p-2 text-xs shadow-md group-hover:block group-focus:block">
                        <div className="mb-1 font-medium">Closest matches</div>
                        {f.top3.map(c => (
                          <div key={c.name} className="flex justify-between gap-2">
                            <span className="truncate">{c.name}</span>
                            <span className={cn('tabular-nums', result.threshold != null && c.score < result.threshold && 'text-muted-foreground')}>{(c.score * 100).toFixed(0)}%</span>
                          </div>
                        ))}
                        {result.threshold != null && <div className="mt-1 border-t pt-1 text-muted-foreground">Threshold {(result.threshold * 100).toFixed(0)}%</div>}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
          {cam && paused && (
            <div className="flex items-center gap-3 rounded-lg border p-3 text-sm" role="status">
              Camera paused while this tab was in the background.
              <Button size="sm" onClick={resume}><Camera /> Resume camera</Button>
            </div>
          )}
          {cam && !paused && (
            <div className="flex flex-wrap items-center gap-4">
              <Button size="sm" onClick={async () => { const b = await grabFrame(video.current, LIVE_WIDTH); if (b) void run(b); }} disabled={pending || live}>Run on this frame</Button>
              <label className="flex items-center gap-2 text-sm"><Switch checked={live} onCheckedChange={setLive} /> <Radio className="size-4" /> Live</label>
            </div>
          )}
        </CardContent>
      </Card>
      <Card className="h-fit" aria-live="polite">
        <CardHeader className="flex-row items-center justify-between">
          <CardTitle className="text-base">Found</CardTitle>
          {pending ? <Loader2 className="size-4 animate-spin text-muted-foreground" /> : result?.latency_ms != null && (
            <span className="text-xs text-muted-foreground tabular-nums">{result.latency_ms} ms{result.model_load_ms ? ` · model loaded in ${(result.model_load_ms / 1000).toFixed(1)} s` : ''}</span>
          )}
        </CardHeader>
        <CardContent>
          {error ? <p className="text-sm text-destructive" role="alert">{error}</p> : !result ? (
            <p className="text-sm text-muted-foreground">{pending ? 'Loading the model… the first image after a few idle minutes takes a moment.' : 'Give it a photo or turn on the camera.'}</p>
          ) : !items.length ? (
            <p className="text-sm text-muted-foreground">{emptyText}</p>
          ) : (
            <ul className="grid gap-2 text-sm">
              {items.map((f, i) => (
                <li key={i} className="flex items-center gap-2">
                  <Icon className="size-4" style={{ color: f.label ? color(f.label) : undefined }} />
                  <span className={cn('font-medium', !f.label && 'text-muted-foreground')}>{f.label ?? unknownLabel}</span>
                  {f.source && <span className="text-[10px] uppercase tracking-wide text-muted-foreground">{f.source}</span>}
                  <span className="ml-auto tabular-nums text-muted-foreground">{(f.score * 100).toFixed(0)}%</span>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
