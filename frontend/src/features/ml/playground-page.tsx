import { SpeakerPlayground } from '@/features/speaker/speaker-pages';
import { hasSentiment, isQuickText } from '@/api/text';
import { TextPlayground } from '@/features/text/text-pages';
import { isPretrained } from '@/api/detector';
import { ReadyPlayground } from '@/features/projects/ready-detector';
import { useProject } from '@/api/hooks';
import { FacePlayground } from '@/features/face/face-playground';
import { Camera, CameraOff, ImageUp, Loader2, Play, Radio } from 'lucide-react';
import { useEffect, useState } from 'react';
import { grabFrame, useAbortOnUnmount, useCamera } from '@/lib/camera';
import { useModelVersion, usePredict, type Prediction } from '@/api/ml';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input, Textarea } from '@/components/ui/input';
import { Skeleton, Switch } from '@/components/ui/misc';
import { classColor } from '@/features/label/box-overlay';
import { ApiError } from '@/lib/api';
import { cn } from '@/lib/utils';
import { NoModel, VersionSelect, useVersionPick } from './evaluate-page';

interface Feature { name: string; type: 'numeric' | 'categorical' | 'boolean'; categories?: string[] }
type Boxes = Extract<Prediction, { boxes: unknown }>['boxes'];

export function PlaygroundPage({ projectId }: { projectId: string }) {
  const { data: project } = useProject(projectId);
  if (!project) return <Skeleton className="h-96 rounded-xl" />;
  if (project.task_type === 'face_recognition') return <FacePlayground projectId={projectId} />;
  if (project.task_type === 'speaker_recognition') return <SpeakerPlayground projectId={projectId} />;
  if (isQuickText(project) || hasSentiment(project)) return <TextPlayground projectId={projectId} />;
  return isPretrained(project) ? <ReadyPlayground projectId={projectId} /> : <ModelPlayground projectId={projectId} />;
}

function ModelPlayground({ projectId }: { projectId: string }) {
  const { loading, ready, mv, setPicked } = useVersionPick(projectId);
  const full = useModelVersion(mv?.id);
  if (loading) return <Skeleton className="h-96 rounded-xl" />;
  if (!mv) return <NoModel projectId={projectId} />;
  const meta = full.data?.model_json as { input?: { modality?: string; features?: Feature[]; target?: string }; task_type?: string } | null | undefined;
  const modality = meta?.input?.modality;
  return (
    <div className="grid gap-5">
      <VersionSelect ready={ready} mv={mv} onPick={setPicked} />
      {full.isLoading ? <Skeleton className="h-80 rounded-xl" /> : modality === 'image' ? (
        <ImagePlay key={mv.id} mvId={mv.id} labels={mv.labels} />
      ) : modality === 'text' ? (
        <TextPlay key={mv.id} mvId={mv.id} labels={mv.labels} />
      ) : modality === 'tabular' ? (
        <TabularPlay key={mv.id} mvId={mv.id} labels={mv.labels} features={meta!.input!.features ?? []} target={meta!.input!.target} />
      ) : (
        <Card className="p-8 text-sm text-muted-foreground">This model's metadata isn't readable, so the playground can't build an input form.</Card>
      )}
    </div>
  );
}

function usePlay(mvId: string) {
  const predict = usePredict(mvId);
  const err = predict.error;
  const unavailable = err instanceof ApiError && err.status === 503;
  return { predict, unavailable, errorText: err && !unavailable ? err.message : null };
}

function ResultPanel({ result, labels, pending, unavailable, errorText }: { result?: Prediction; labels: string[]; pending: boolean; unavailable: boolean; errorText: string | null }) {
  return (
    <Card className="h-fit" aria-live="polite">
      <CardHeader className="flex-row items-center justify-between">
        <CardTitle className="text-base">Prediction</CardTitle>
        {pending ? <Loader2 className="size-4 animate-spin text-muted-foreground" /> : result && <span className="text-xs text-muted-foreground tabular-nums">{result.latency_ms} ms{result.model_load_ms ? ` · model loaded in ${(result.model_load_ms / 1000).toFixed(1)} s` : ''}</span>}
      </CardHeader>
      <CardContent>
        {unavailable ? (
          <p className="text-sm text-muted-foreground">Live predictions aren't available on this server yet (the model loader isn't running). Exports still work.</p>
        ) : errorText ? (
          <p className="text-sm text-destructive" role="alert">{errorText}</p>
        ) : !result && pending ? (
          <p className="flex items-center gap-2 text-sm text-muted-foreground"><Loader2 className="size-4 animate-spin" /> Loading the model… the first prediction after a few idle minutes takes a moment.</p>
        ) : !result ? (
          <p className="text-sm text-muted-foreground">Give the model an input to see what it predicts.</p>
        ) : 'value' in result ? (
          <div className="text-4xl font-semibold tabular-nums">{+result.value.toPrecision(6)}</div>
        ) : 'boxes' in result ? (
          <ul className="grid gap-1 text-sm">
            {result.boxes.length ? result.boxes.map((b, i) => (
              <li key={i} className="flex items-center gap-2"><span className="size-3 rounded-sm" style={{ background: classColor(labels, b.class_name) }} />{b.class_name}<span className="ml-auto tabular-nums text-muted-foreground">{(b.confidence * 100).toFixed(0)}%</span></li>
            )) : <li className="text-muted-foreground">No objects found.</li>}
          </ul>
        ) : (
          <Probs result={result} labels={labels} />
        )}
      </CardContent>
    </Card>
  );
}

function Probs({ result, labels }: { result: Extract<Prediction, { label: string }>; labels: string[] }) {
  const probs = Array.isArray(result.probs) ? labels.map((l, i) => [l, (result.probs as number[])[i] ?? 0] as const) : Object.entries(result.probs);
  const top = [...probs].sort((a, b) => b[1] - a[1]).slice(0, 8);
  return (
    <div className="grid gap-4">
      <div>
        <div className="text-3xl font-semibold">{result.label}</div>
        <div className="text-sm text-muted-foreground">{(result.confidence * 100).toFixed(1)}% confident</div>
      </div>
      <ul className="grid gap-2">
        {top.map(([l, p]) => (
          <li key={l} className="grid gap-1">
            <div className="flex justify-between text-xs"><span className={cn(l === result.label && 'font-medium')}>{l}</span><span className="tabular-nums">{(p * 100).toFixed(1)}%</span></div>
            <div className="h-2 overflow-hidden rounded-full bg-muted"><div className="h-full rounded-full bg-primary transition-[width]" style={{ width: `${p * 100}%` }} /></div>
          </li>
        ))}
      </ul>
    </div>
  );
}

function ImagePlay({ mvId, labels }: { mvId: string; labels: string[] }) {
  const { predict, unavailable, errorText } = usePlay(mvId);
  const [src, setSrc] = useState<string>();
  const [size, setSize] = useState<{ w: number; h: number }>();
  const [cam, setCam] = useState(false);
  const [live, setLive] = useState(false);
  const [camError, setCamError] = useState<string>();
  // Runs only while this page is open and visible: leaving / hiding stops the camera, the loop and the request.
  const { videoRef: video, paused, resume } = useCamera({ enabled: cam, facing: 'environment', onError: e => (setCamError(e instanceof Error ? e.message : 'Camera unavailable'), setCam(false)) });
  const nextSignal = useAbortOnUnmount();
  const run = (blob: Blob) => predict.mutate({ file: blob, signal: nextSignal() });
  useEffect(() => { if (!cam || paused) setLive(false); }, [cam, paused]);

  const snap = async () => {
    const v = video.current;
    if (v?.videoWidth) setSize({ w: v.videoWidth, h: v.videoHeight });
    return grabFrame(v);
  };

  // Live mode: one request in flight at a time, as fast as the server answers.
  const pending = predict.isPending;
  useEffect(() => {
    if (!live || pending || unavailable) return;
    const t = setTimeout(async () => { const b = await snap(); if (b) run(b); }, 150);
    return () => clearTimeout(t);
  }, [live, pending, unavailable]); // eslint-disable-line react-hooks/exhaustive-deps

  const result = predict.data;
  const boxes: Boxes = result && 'boxes' in result ? result.boxes : [];

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_20rem]">
      <Card>
        <CardHeader className="flex-row flex-wrap items-center justify-between gap-2">
          <div><CardTitle className="text-base">Try an image</CardTitle><CardDescription>Upload a photo or use your camera.</CardDescription></div>
          <div className="flex gap-2">
            <label className="inline-flex h-8 cursor-pointer items-center gap-2 rounded-md border px-3 text-xs font-medium hover:bg-accent">
              <ImageUp className="size-4" /> Choose image
              <input type="file" accept="image/*" className="sr-only" onChange={e => {
                const f = e.target.files?.[0];
                if (!f) return;
                setCam(false);
                setSrc(URL.createObjectURL(f));
                run(f);
              }} />
            </label>
            <Button variant="outline" size="sm" onClick={() => (setCamError(undefined), setCam(c => !c))}>{cam ? <CameraOff /> : <Camera />} {cam ? 'Stop camera' : 'Camera'}</Button>
          </div>
        </CardHeader>
        <CardContent>
          {camError && <p className="mb-3 text-sm text-destructive" role="alert">{camError}</p>}
          <div className="relative grid min-h-64 place-items-center overflow-hidden rounded-lg bg-muted">
            {cam ? (
              <video ref={video} autoPlay playsInline muted className="max-h-[60vh] w-full object-contain" onLoadedMetadata={e => setSize({ w: e.currentTarget.videoWidth, h: e.currentTarget.videoHeight })} />
            ) : src ? (
              <img src={src} alt="Input" className="max-h-[60vh] w-full object-contain" onLoad={e => setSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })} />
            ) : (
              <p className="p-8 text-sm text-muted-foreground">No image yet</p>
            )}
            {size && !!boxes.length && (
              <svg viewBox={`0 0 ${size.w} ${size.h}`} className="pointer-events-none absolute inset-0 size-full" preserveAspectRatio="xMidYMid meet" aria-hidden>
                {boxes.map((b, i) => (
                  <g key={i}>
                    <rect x={b.bbox.x} y={b.bbox.y} width={b.bbox.w} height={b.bbox.h} fill="none" stroke={classColor(labels, b.class_name)} strokeWidth={Math.max(2, size.w / 300)} />
                    <text x={b.bbox.x + 4} y={b.bbox.y + size.w / 40} fill={classColor(labels, b.class_name)} fontSize={size.w / 40} fontWeight={600}>{b.class_name} {(b.confidence * 100).toFixed(0)}%</text>
                  </g>
                ))}
              </svg>
            )}
          </div>
          {cam && (
            <div className="mt-3 flex flex-wrap items-center gap-4">
              {paused ? <Button size="sm" onClick={resume}><Camera /> Resume camera</Button> : <Button size="sm" onClick={async () => { const b = await snap(); if (b) run(b); }} disabled={pending || live}><Play /> Predict this frame</Button>}
              <label className="flex items-center gap-2 text-sm"><Switch checked={live} onCheckedChange={setLive} disabled={unavailable} /> <Radio className="size-4" /> Live</label>
            </div>
          )}
        </CardContent>
      </Card>
      <ResultPanel result={result} labels={labels} pending={pending} unavailable={unavailable} errorText={errorText} />
    </div>
  );
}

function TextPlay({ mvId, labels }: { mvId: string; labels: string[] }) {
  const { predict, unavailable, errorText } = usePlay(mvId);
  const [text, setText] = useState('');
  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_20rem]">
      <Card>
        <CardHeader><CardTitle className="text-base">Try some text</CardTitle><CardDescription>Ctrl/⌘ + Enter to predict.</CardDescription></CardHeader>
        <CardContent className="grid gap-3">
          <Textarea value={text} onChange={e => setText(e.target.value)} rows={7} placeholder="Type or paste text…" aria-label="Text input"
            onKeyDown={e => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && text.trim()) predict.mutate({ text }); }} />
          <Button className="justify-self-start" disabled={!text.trim() || predict.isPending} onClick={() => predict.mutate({ text })}><Play /> Predict</Button>
        </CardContent>
      </Card>
      <ResultPanel result={predict.data} labels={labels} pending={predict.isPending} unavailable={unavailable} errorText={errorText} />
    </div>
  );
}

function TabularPlay({ mvId, labels, features, target }: { mvId: string; labels: string[]; features: Feature[]; target?: string }) {
  const { predict, unavailable, errorText } = usePlay(mvId);
  const [row, setRow] = useState<Record<string, string | boolean>>(() =>
    Object.fromEntries(features.map(f => [f.name, f.type === 'boolean' ? false : f.type === 'categorical' ? f.categories?.[0] ?? '' : ''])));
  const submit = () =>
    predict.mutate({ row: Object.fromEntries(features.map(f => [f.name, f.type === 'numeric' ? (row[f.name] === '' ? null : Number(row[f.name])) : row[f.name]])) });
  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_20rem]">
      <Card>
        <CardHeader><CardTitle className="text-base">Try a row</CardTitle><CardDescription>Predicting <span className="font-medium">{target ?? 'the target'}</span>. Leave a number empty if unknown.</CardDescription></CardHeader>
        <CardContent>
          <form className="grid gap-3 sm:grid-cols-2" onSubmit={e => (e.preventDefault(), submit())}>
            {features.map(f => (
              <label key={f.name} className="grid gap-1 text-sm">
                <span className="truncate font-medium" title={f.name}>{f.name}</span>
                {f.type === 'categorical' ? (
                  <select className="h-9 rounded-md border border-input bg-card px-2" value={String(row[f.name])} onChange={e => setRow(r => ({ ...r, [f.name]: e.target.value }))}>
                    {(f.categories ?? []).map(c => <option key={c}>{c}</option>)}
                  </select>
                ) : f.type === 'boolean' ? (
                  <Switch checked={!!row[f.name]} onCheckedChange={v => setRow(r => ({ ...r, [f.name]: v }))} aria-label={f.name} />
                ) : (
                  <Input type="number" step="any" value={String(row[f.name])} onChange={e => setRow(r => ({ ...r, [f.name]: e.target.value }))} />
                )}
              </label>
            ))}
            <Button type="submit" className="justify-self-start sm:col-span-2" disabled={predict.isPending}><Play /> Predict</Button>
          </form>
        </CardContent>
      </Card>
      <ResultPanel result={predict.data} labels={labels} pending={predict.isPending} unavailable={unavailable} errorText={errorText} />
    </div>
  );
}
