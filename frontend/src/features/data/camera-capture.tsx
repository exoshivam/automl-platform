import { isPretrained, trainableLabels } from '@/api/detector';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useCamera } from '@/lib/camera';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { AlertTriangle, Camera, CloudOff, Lightbulb, Loader2, RefreshCw, SwitchCamera, Zap } from 'lucide-react';
import { useUndoUpload, useValidation, useVersions, pickVersion } from '@/api/data-hooks';
import { toast } from 'sonner';
import type { Project } from '@/api/types';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Progress } from '@/components/ui/misc';
import { drain, queue, subscribe, undoPhotos, type QueuedPhoto } from '@/lib/camera-queue';
import { cn } from '@/lib/utils';

/** DAT-9 recommended minimum per class ("helmet: 34/50"). */
const RECOMMENDED = 50;
const TIPS = [
  'Change the angle: shoot from above, the side and up close.',
  'Vary the lighting: daylight, indoor light and shade.',
  'Use different backgrounds so the model learns the object, not the room.',
  'Fill the frame with the object for some shots, keep it small in others.',
  'Include the hard cases you expect in real use (blur, partial views).',
];

type CamError = { title: string; detail: string } | null;

const MIXED = '__mixed';

function describeError(e: unknown): CamError {
  const name = (e as DOMException)?.name;
  if (!window.isSecureContext)
    return { title: 'The camera needs a secure page', detail: 'Open the app on https:// (or http://localhost on this device). Phones on a LAN IP over plain http are blocked by the browser.' };
  if (name === 'NotAllowedError') return { title: 'Camera permission denied', detail: 'Allow camera access in the address bar (or phone settings) and try again.' };
  if (name === 'NotFoundError' || name === 'OverconstrainedError') return { title: 'No camera found', detail: 'Connect a camera, or use Upload instead.' };
  if (name === 'NotReadableError') return { title: 'The camera is busy', detail: 'Another app or tab is using it. Close it and retry.' };
  return { title: "Couldn't start the camera", detail: (e as Error)?.message ?? 'Unknown error' };
}

export function CameraCapture({ project }: { project: Project }) {
  const detection = project.task_type === 'object_detection';
  const classes = trainableLabels(project);
  const [facing, setFacing] = useState<'environment' | 'user'>('environment');
  const [cams, setCams] = useState(1);
  const [error, setError] = useState<CamError>(null);
  const [target, setTarget] = useState<string | null>(detection ? null : (classes[0] ?? null));
  // Detection: which object the photos are of (a capture hint stored with each photo). Required before the first shot.
  const [capTarget, setCapTarget] = useState<string | undefined>(classes.length === 1 && detection ? classes[0] : undefined);
  const hint = capTarget === MIXED ? null : (capTarget ?? null);
  const [mode, setMode] = useState<'single' | 'burst'>('single');
  const [burstCount, setBurstCount] = useState(10);
  const [burstGap, setBurstGap] = useState(300);
  // After a burst: near-identical frames teach the background, not the object. Nudge for variety.
  const [nudge, setNudge] = useState(false);
  const [shooting, setShooting] = useState(0);
  const [flash, setFlash] = useState(false);
  const [recent, setRecent] = useState<string[]>([]);
  const [items, setItems] = useState<QueuedPhoto[]>([]);
  const [online, setOnline] = useState(navigator.onLine);
  const [tip, setTip] = useState(0);
  const [captured, setCaptured] = useState<Record<string, number>>({});
  const qc = useQueryClient();
  const versions = useVersions(project.dataset_id);
  const current = pickVersion(versions.data?.items);
  // Per-object counts of photos already uploaded: capture_counts is on the single-version detail only.
  // (Its "__none" bucket also holds older photos without a hint, so "Mixed" counts this session only.)
  const detail = useQuery({
    queryKey: ['version-detail', current?.id],
    queryFn: () => api<{ capture_counts?: Record<string, number> }>(`/dataset-versions/${current!.id}`),
    enabled: detection && !!current,
    refetchInterval: detection ? 5000 : false,
  });
  const captureCounts: Record<string, number> = { ...(detail.data?.capture_counts ?? {}), [MIXED]: 0 };
  // Sessions = distinct capture/upload batches per class (validation diversity stat, last sealed version) + this visit.
  const lastSealed = versions.data?.items.find(v => !v.is_draft && v.validation_status !== 'pending');
  const diversity = useValidation(lastSealed?.id, false).data?.diversity;
  const sessions = (cls: string) => (diversity?.sessions_per_class[cls] ?? 0) + (captured[cls] ? 1 : 0);

  // Runs only while this tab is open and visible: leaving / hiding stops every track (camera light off).
  const [camOn, setCamOn] = useState(true);
  const { videoRef, ready, paused, resume } = useCamera({
    enabled: camOn,
    facing,
    constraints: { width: { ideal: 1280 }, height: { ideal: 960 } },
    onError: e => setError(describeError(e)),
  });
  useEffect(() => {
    if (ready) void navigator.mediaDevices.enumerateDevices().then(d => setCams(d.filter(x => x.kind === 'videoinput').length));
  }, [ready]);
  const start = () => {
    setError(null);
    if (paused) return resume();
    setCamOn(false);
    requestAnimationFrame(() => setCamOn(true));
  };
  // A burst in progress must stop when the camera does (unmount / tab hidden).
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true; // StrictMode remounts: set it again, not only clear it
    return () => void (alive.current = false);
  }, []);

  // Queue: subscribe, and drain when online / visible again (DAT-9: survives tab sleep and reconnects).
  const kick = useCallback(
    () => void drain(project.id, project.dataset_id!, () => qc.invalidateQueries({ queryKey: ['versions'] })).then(() => qc.invalidateQueries({ queryKey: ['project'] })),
    [project.id, project.dataset_id, qc],
  );
  useEffect(() => {
    const unsub = subscribe(all => setItems(all.filter(p => p.datasetId === project.dataset_id)));
    const on = () => (setOnline(true), kick());
    const off = () => setOnline(false);
    const vis = () => document.visibilityState === 'visible' && kick();
    window.addEventListener('online', on);
    window.addEventListener('offline', off);
    document.addEventListener('visibilitychange', vis);
    kick();
    const t = setInterval(() => setTip(i => (i + 1) % TIPS.length), 6000);
    return () => {
      unsub();
      window.removeEventListener('online', on);
      window.removeEventListener('offline', off);
      document.removeEventListener('visibilitychange', vis);
      clearInterval(t);
    };
  }, [project.dataset_id, kick]);

  const grab = async (burst?: string): Promise<string | null> => {
    const v = videoRef.current;
    if (!v || !v.videoWidth) return null;
    const c = document.createElement('canvas');
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext('2d')!.drawImage(v, 0, 0);
    const blob = await new Promise<Blob | null>(res => c.toBlob(res, 'image/jpeg', 0.9));
    if (!blob) return null;
    const id = crypto.randomUUID();
    await queue.add({ id, burst, projectId: project.id, datasetId: project.dataset_id!, label: detection ? null : target, target: detection ? hint : undefined, blob, createdAt: Date.now(), status: 'queued' });
    setRecent(r => [URL.createObjectURL(blob), ...r].slice(0, 8));
    const k = detection ? (hint ?? MIXED) : (target ?? '__unlabelled');
    setCaptured(m => ({ ...m, [k]: (m[k] ?? 0) + 1 }));
    setFlash(true);
    setTimeout(() => setFlash(false), 120);
    return id;
  };
  const undoUpload = useUndoUpload();
  /** "Captured 10 · Undo": drops the burst whether it's still queued or already uploaded (its own upload session). */
  const offerUndo = (ids: string[], key: string) =>
    toast(`Captured ${ids.length}`, {
      id: 'burst-undo',
      duration: 10_000,
      action: {
        label: 'Undo',
        onClick: () =>
          void undoPhotos(ids, u => undoUpload.mutateAsync(u))
            .then(() => (setCaptured(m => ({ ...m, [key]: Math.max(0, (m[key] ?? 0) - ids.length) })), toast.success(`Removed ${ids.length} photos`)))
            .catch(e => toast.error(`Couldn't undo: ${(e as Error).message}`)),
      },
    });

  const shoot = async () => {
    const n = mode === 'burst' ? burstCount : 1;
    const burst = n > 1 ? crypto.randomUUID() : undefined;
    const key = detection ? (hint ?? MIXED) : (target ?? '__unlabelled');
    const ids: string[] = [];
    setNudge(false);
    for (let i = 0; i < n; i++) {
      if (!alive.current || document.hidden) break;
      setShooting(n - i);
      const id = await grab(burst);
      if (id) ids.push(id);
      if (i < n - 1) await new Promise(r => setTimeout(r, burstGap));
    }
    setShooting(0);
    if (n > 1) setNudge(true);
    if (n > 1 && ids.length) offerUndo(ids, key);
    kick();
  };

  // Keyboard: 1-9 picks a class, Space takes the shot.
  const shootRef = useRef(shoot);
  shootRef.current = shoot;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest('input,select,textarea')) return;
      const n = Number(e.key);
      if (n >= 1 && n <= classes.length) (detection ? setCapTarget : setTarget)(classes[n - 1]!);
      if (e.code === 'Space') {
        e.preventDefault();
        void shootRef.current();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [classes]);

  const pending = items.length;
  const failed = items.filter(p => p.status === 'failed');
  const have = (cls: string) => (current?.label_stats[cls]?.images ?? 0) + (captured[cls] ?? 0);

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
      <Card className="overflow-hidden">
        {detection && (
          <div className="flex flex-wrap items-center gap-2 border-b p-3" role="radiogroup" aria-label="What are you capturing?">
            <span className="text-sm font-medium">What are you capturing?</span>
            {[...classes, MIXED].map((c, i) => (
              <button
                key={c}
                type="button"
                role="radio"
                aria-checked={capTarget === c}
                onClick={() => setCapTarget(c)}
                className={cn('inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-sm transition-colors', capTarget === c ? 'border-primary bg-primary text-primary-foreground' : 'hover:bg-accent')}
              >
                {c !== MIXED && <kbd className="rounded border border-current/30 px-1 text-[10px] opacity-70">{i + 1}</kbd>}
                {c === MIXED ? 'Mixed / several objects' : c}
              </button>
            ))}
            {!capTarget && <span className="text-xs text-warning">Pick one to start</span>}
          </div>
        )}
        <div className="relative aspect-[4/3] w-full bg-black">
          <video ref={videoRef} className="size-full object-contain" playsInline muted autoPlay aria-label="Camera preview" />
          {detection && capTarget && (
            <Badge className="absolute bottom-3 left-3 bg-black/70 text-sm text-white">Capturing: {capTarget === MIXED ? 'several objects' : capTarget}</Badge>
          )}
          {flash && <div className="absolute inset-0 bg-white/70" aria-hidden />}
          {paused && !error ? (
            <div className="absolute inset-0 grid place-items-center gap-2 text-sm text-white/80">
              <div className="grid justify-items-center gap-2">
                Camera paused while this tab was in the background.
                <Button variant="secondary" size="sm" onClick={resume}><Camera /> Resume camera</Button>
              </div>
            </div>
          ) : !ready && !error && (
            <div className="absolute inset-0 grid place-items-center text-sm text-white/80">
              <span className="flex items-center gap-2">
                <Loader2 className="size-4 animate-spin" /> Starting camera…
              </span>
            </div>
          )}
          {error && (
            <div className="absolute inset-0 grid place-items-center p-6 text-center text-white">
              <div className="grid max-w-sm gap-2">
                <AlertTriangle className="mx-auto size-8 text-warning" />
                <p className="font-semibold">{error.title}</p>
                <p className="text-sm text-white/80">{error.detail}</p>
                <Button variant="secondary" size="sm" className="mx-auto" onClick={start}>
                  <RefreshCw /> Try again
                </Button>
              </div>
            </div>
          )}
          {shooting > 0 && <Badge className="absolute left-3 top-3 bg-black/60 text-white">Burst: {shooting} left</Badge>}
          {nudge && shooting === 0 && (
            <div role="status" className="absolute inset-x-3 bottom-3 flex items-center gap-2 rounded-lg bg-black/70 px-3 py-2 text-sm text-white">
              <RefreshCw className="size-4 shrink-0" />
              <span className="flex-1">Change the place, lighting or angle before the next burst. Variety beats more of the same.</span>
              <button type="button" className="text-white/70 hover:text-white" onClick={() => setNudge(false)} aria-label="Dismiss tip">✕</button>
            </div>
          )}
          {!online && (
            <Badge variant="warning" className="absolute right-3 top-3">
              <CloudOff className="size-3" /> Offline, photos are queued
            </Badge>
          )}
        </div>
        <div className="flex flex-wrap items-center justify-between gap-3 p-4">
          <div className="flex rounded-lg bg-muted p-1 text-sm" role="radiogroup" aria-label="Capture mode">
            {(['single', 'burst'] as const).map(m => (
              <button key={m} type="button" role="radio" aria-checked={mode === m} onClick={() => setMode(m)} className={cn('cursor-pointer rounded-md px-3 py-1 capitalize', mode === m && 'bg-card font-medium shadow-sm')}>
                {m === 'burst' ? (
                  <span className="inline-flex items-center gap-1">
                    <Zap className="size-3.5" /> Burst
                  </span>
                ) : (
                  'Single'
                )}
              </button>
            ))}
          </div>
          {mode === 'burst' && (
            <div className="flex items-center gap-2 text-sm">
              <label className="sr-only" htmlFor="burst-n">
                Photos per burst
              </label>
              <select id="burst-n" value={burstCount} onChange={e => setBurstCount(Number(e.target.value))} className="h-8 rounded-md border border-input bg-card px-2">
                {[5, 10, 20, 30].map(n => (
                  <option key={n} value={n}>
                    {n} photos
                  </option>
                ))}
              </select>
              <label className="sr-only" htmlFor="burst-gap">
                Interval
              </label>
              <select id="burst-gap" value={burstGap} onChange={e => setBurstGap(Number(e.target.value))} className="h-8 rounded-md border border-input bg-card px-2">
                {[150, 300, 500, 1000].map(n => (
                  <option key={n} value={n}>
                    every {n} ms
                  </option>
                ))}
              </select>
            </div>
          )}
          <div className="flex items-center gap-2">
            {cams > 1 && (
              <Button variant="outline" size="icon" aria-label="Switch camera" onClick={() => setFacing(f => (f === 'environment' ? 'user' : 'environment'))}>
                <SwitchCamera />
              </Button>
            )}
            <button
              type="button"
              onClick={shoot}
              disabled={!ready || shooting > 0 || (!detection && !target) || (detection && !capTarget)}
              aria-label={mode === 'burst' ? `Take ${burstCount} photos` : 'Take photo'}
              className="grid size-16 cursor-pointer place-items-center rounded-full border-4 border-primary bg-primary/15 text-primary transition-transform active:scale-95 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <Camera className="size-7" />
            </button>
          </div>
        </div>
        {recent.length > 0 && (
          <div className="flex gap-2 overflow-x-auto px-4 pb-4" aria-label="Recent captures">
            {recent.map(u => (
              <img key={u} src={u} alt="" className="h-14 w-auto shrink-0 rounded-md border object-cover" />
            ))}
          </div>
        )}
      </Card>

      <div className="grid content-start gap-4">
        <Card className="grid gap-3 p-4">
          <h3 className="text-sm font-semibold">{detection ? 'Photos go to the box queue' : 'Capture into class'}</h3>
          {detection ? (
            <div className="grid gap-1.5 text-sm">
              <ul className="grid gap-2" aria-label="Photos per object">
                {[...classes, MIXED].map(c => {
                  // Uploaded (server) + still in the offline queue; "Mixed" is this session's count.
                  const n = c === MIXED ? (captured[MIXED] ?? 0) : (captureCounts[c] ?? 0) + items.filter(p => p.target === c).length;
                  if (c === MIXED && !n) return null;
                  return (
                    <li key={c} className={cn('grid gap-1 rounded-lg border p-2', capTarget === c && 'border-primary bg-primary/6')}>
                      <span className="flex justify-between">
                        <span className="font-medium">{c === MIXED ? 'Mixed' : c}</span>
                        <span className={cn('tabular-nums', c !== MIXED && n >= RECOMMENDED ? 'text-success' : 'text-muted-foreground')}>{c === MIXED ? n : `${n}/${RECOMMENDED}`}</span>
                      </span>
                      {c !== MIXED && <Progress value={Math.min(100, (n / RECOMMENDED) * 100)} className="h-1.5" indicatorClassName={n >= RECOMMENDED ? 'bg-success' : undefined} />}
                      {c !== MIXED && capTarget === c && n > 0 && n < RECOMMENDED && (
                        <span className="text-xs text-muted-foreground">Next {c} shots: change the angle, distance or background.</span>
                      )}
                    </li>
                  );
                })}
              </ul>
              <p className="text-xs text-muted-foreground">
                Draw the boxes on the Label tab: photos keep the object you picked, so the right class is preselected there.
                {isPretrained(project) && ' Ready-made objects need no photos; they are detected automatically.'}
              </p>
            </div>
          ) : (
            <ul className="grid gap-2" role="radiogroup" aria-label="Class">
              {classes.map((c, i) => {
                const n = have(c);
                const ses = sessions(c);
                return (
                  <li key={c}>
                    <button
                      type="button"
                      role="radio"
                      aria-checked={target === c}
                      onClick={() => setTarget(c)}
                      className={cn('grid w-full cursor-pointer gap-1.5 rounded-lg border p-2.5 text-left', target === c ? 'border-primary bg-primary/6' : 'hover:bg-accent/60')}
                    >
                      <span className="flex items-center justify-between text-sm">
                        <span className="font-medium">
                          <kbd className="mr-1.5 rounded border px-1 text-[10px] text-muted-foreground">{i + 1}</kbd>
                          {c}
                        </span>
                        <span className="tabular-nums text-muted-foreground">
                          <span className={cn(n >= RECOMMENDED && 'text-success')}>{n}/{RECOMMENDED}</span>
                          {' · '}
                          {ses} {ses === 1 ? 'session' : 'sessions'}
                        </span>
                      </span>
                      <Progress value={Math.min(100, (n / RECOMMENDED) * 100)} className="h-1.5" indicatorClassName={n >= RECOMMENDED ? 'bg-success' : undefined} />
                      {n > 0 && ses < 3 && (
                        <span className="flex items-center gap-1 text-xs text-warning">
                          <AlertTriangle className="size-3" /> Fewer than 3 sessions: capture again somewhere else or at another time
                        </span>
                      )}
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </Card>

        <Card className="flex gap-3 p-4 text-sm" aria-live="polite">
          <Lightbulb className="mt-0.5 size-4 shrink-0 text-warning" />
          <p>{TIPS[tip]}</p>
        </Card>

        <Card className="grid gap-2 p-4 text-sm" role="status" aria-live="polite">
          <div className="flex items-center justify-between">
            <span className="font-medium">Upload queue</span>
            {pending > 0 && online && <Loader2 className="size-4 animate-spin text-primary" />}
          </div>
          <p className="text-muted-foreground">{pending === 0 ? 'All photos uploaded.' : `${pending} photo${pending === 1 ? '' : 's'} waiting${online ? ', uploading…' : ' for the network'}`}</p>
          {failed.length > 0 && (
            <div className="grid gap-2">
              <p className="text-xs text-destructive">
                {failed.length} failed: {failed[0]?.error}
              </p>
              <Button size="sm" variant="outline" onClick={async () => (await Promise.all(failed.map(p => queue.put({ ...p, status: 'queued' }))), kick())}>
                <RefreshCw /> Retry now
              </Button>
            </div>
          )}
          <p className="text-xs text-muted-foreground">Photos are kept on this device until they upload, even if you close the tab.</p>
        </Card>
      </div>
    </div>
  );
}
