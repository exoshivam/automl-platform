import { useNavigate } from '@tanstack/react-router';
import { Link } from '@tanstack/react-router';
import { Box, CheckCircle2, GraduationCap, Info, Loader2, Package, Plus, Save, Trash2 } from 'lucide-react';
import { useEffect, useState } from 'react';
import { toast } from 'sonner';
import { customLabels, customStatus, detect, usePretrainedClasses, useAddCustomObject, useDetectorSettings, useRemoveCustomObject, useSaveDetectorSettings, type CustomStatus } from '@/api/detector';
import { useProject } from '@/api/hooks';
import { Input } from '@/components/ui/input';
import { ApiError } from '@/lib/api';
import { cn } from '@/lib/utils';
import { BoxPlayground } from '@/components/box-playground';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Skeleton } from '@/components/ui/misc';

/** Ready-made (COCO) detector: same live box overlay, only the project's chosen classes above the threshold. */
export function ReadyPlayground({ projectId }: { projectId: string }) {
  const s = useDetectorSettings(projectId).data;
  const project = useProject(projectId).data;
  const custom = customLabels(project);
  const [status, setStatus] = useState<CustomStatus | undefined>();
  const st = status ?? customStatus(project);
  return (
    <div className="grid gap-3">
      {custom.length > 0 && st === 'outdated' && (
        <div className="flex gap-2 rounded-lg border border-warning/40 bg-warning/8 p-3 text-sm" role="note">
          <Info className="mt-0.5 size-4 shrink-0 text-warning" />
          <span>Your objects changed since the last training: this model only finds what it was trained on. Retrain on the Train tab to include the changes.</span>
        </div>
      )}
      {custom.length > 0 && st === 'needs_attention' && (
        <div className="flex gap-2 rounded-lg border border-warning/40 bg-warning/8 p-3 text-sm" role="note">
          <Info className="mt-0.5 size-4 shrink-0 text-warning" />
          <span>Training couldn't confirm the ready-made objects were kept, so they are found by the original ready-made model and your custom objects by your trained one. Details on the Evaluate tab.</span>
        </div>
      )}
      {custom.length > 0 && st !== 'ready' && st !== 'outdated' && st !== 'needs_attention' && (
        <div className="flex gap-2 rounded-lg border border-info/40 bg-info/8 p-3 text-sm" role="note">
          <Info className="mt-0.5 size-4 shrink-0 text-info" />
          <span>Custom objects ({custom.join(', ')}) appear here after training{st === 'training' ? ': training is running now' : ''}. Ready-made objects work already.</span>
        </div>
      )}
      <BoxPlayground
        title="Try the detector"
        description={<>Looks for: {s?.classes.join(', ') ?? '…'}{custom.length ? <> · custom: {custom.join(', ')}</> : null}. Boxes marked <SourceTag source="custom" /> come from your trained objects.</>}
        names={[...(s?.classes ?? []), ...custom]}
        emptyText="None of the chosen objects found in this image."
        icon={Box}
        detect={async (blob, signal) => {
          const r = await detect(projectId, blob, signal);
          if (r.custom_status) setStatus(r.custom_status);
          return { ...r, threshold: s?.score_threshold, items: r.boxes.map(b => ({ box: b.bbox, label: b.class_name, score: b.confidence, source: b.source })) };
        }}
      />
    </div>
  );
}

export function SourceTag({ source }: { source?: string }) {
  if (source !== 'custom') return null;
  return <span className="ml-1 inline-flex rounded-sm bg-black/25 px-1 text-[10px] font-medium uppercase tracking-wide">custom</span>;
}

export function ReadySettingsPage({ projectId }: { projectId: string }) {
  const q = useDetectorSettings(projectId);
  const save = useSaveDetectorSettings(projectId);
  const [t, setT] = useState<number | null>(null);
  useEffect(() => { if (q.data && t === null) setT(q.data.score_threshold); }, [q.data, t]);
  if (q.isLoading || t === null) return <Skeleton className="h-80 rounded-xl" />;
  if (!q.data) return <Card className="p-6 text-sm text-destructive">{q.error?.message}</Card>;
  const s = q.data;
  const dirty = Math.abs(t - s.score_threshold) > 1e-6;
  return (
    <div className="grid gap-5 lg:grid-cols-[22rem_1fr]">
      <Card className="h-fit">
        <CardHeader>
          <CardTitle>Confidence threshold</CardTitle>
          <CardDescription>Boxes below this confidence are hidden. <b>Higher</b> = fewer false alarms but may miss small or partly hidden objects. <b>Lower</b> = finds more, with more mistakes.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4">
          <div className="flex items-baseline justify-between">
            <span className="text-3xl font-semibold tabular-nums">{Math.round(t * 100)}%</span>
            <span className="text-xs text-muted-foreground">default {Math.round(s.default_score_threshold * 100)}%</span>
          </div>
          <input type="range" min={0.05} max={0.95} step={0.01} value={t} onChange={e => setT(Number(e.target.value))} aria-label="Confidence threshold" className="w-full accent-[var(--primary)]" />
          <div className="flex justify-between text-xs text-muted-foreground"><span>Finds more</span><span>Fewer false alarms</span></div>
          <div className="flex gap-2">
            <Button size="sm" variant="ghost" onClick={() => setT(s.default_score_threshold)}>Reset to default</Button>
          </div>
          <Button disabled={!dirty || save.isPending} onClick={() => save.mutate({ score_threshold: +t.toFixed(2) }, { onSuccess: () => toast.success(`Threshold set to ${Math.round(t * 100)}%`), onError: e => toast.error(e.message) })}>
            {save.isPending ? <Loader2 className="animate-spin" /> : <Save />} {dirty ? 'Save threshold' : 'Saved'}
          </Button>
        </CardContent>
      </Card>
      <ReadyClassesCard projectId={projectId} chosen={s.classes} model={s.model} />
      <CustomObjectsCard projectId={projectId} />
    </div>
  );
}

/** Editable ready-made class list (PATCH detector-settings {classes}). Changing it after training marks the model outdated. */
function ReadyClassesCard({ projectId, chosen, model }: { projectId: string; chosen: string[]; model: { name: string; licence: string; commercial_use: boolean } }) {
  const all = usePretrainedClasses().data?.classes.map(c => c.name) ?? [];
  const save = useSaveDetectorSettings(projectId);
  const project = useProject(projectId).data;
  const [picked, setPicked] = useState(chosen);
  const [q, setQ] = useState('');
  useEffect(() => setPicked(chosen), [chosen]);
  const term = q.trim().toLowerCase();
  const shown = term ? all.filter(c => c.toLowerCase().includes(term)) : all;
  const dirty = picked.length !== chosen.length || picked.some(c => !chosen.includes(c));
  const trained = customStatus(project) === 'ready' || customStatus(project) === 'outdated';
  return (
    <Card className="h-fit">
      <CardHeader>
        <CardTitle>Ready-made objects</CardTitle>
        <CardDescription>Tick what it should find ({picked.length} of {all.length || 80}). {trained && 'Changing this after training means retraining to include the change.'}</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-3">
        <Input value={q} onChange={e => setQ(e.target.value)} placeholder="Search ready-made objects…" aria-label="Search ready-made objects" />
        <div className="flex max-h-56 flex-wrap gap-1.5 overflow-y-auto rounded-lg border p-2" role="group" aria-label="Ready-made objects">
          {shown.map(c => {
            const on = picked.includes(c);
            return (
              <button key={c} type="button" aria-pressed={on} onClick={() => setPicked(p => (on ? p.filter(x => x !== c) : [...p, c]))}
                className={cn('rounded-full border px-2.5 py-1 text-xs transition-colors', on ? 'border-primary bg-primary/12 text-primary' : 'hover:bg-accent')}>
                {c}
              </button>
            );
          })}
          {term && !shown.length && <span className="p-1 text-xs text-muted-foreground">Not in the ready-made list: add it as a custom object instead.</span>}
        </div>
        <Button disabled={!dirty || !picked.length || save.isPending} onClick={() => save.mutate({ classes: picked }, { onSuccess: () => toast.success('Ready-made objects updated'), onError: e => toast.error(e.message) })}>
          {save.isPending ? <Loader2 className="animate-spin" /> : <Save />} {dirty ? 'Save objects' : 'Saved'}
        </Button>
        <p className="flex gap-2 text-xs text-muted-foreground"><Info className="mt-0.5 size-3.5 shrink-0" /> {model.name} · {model.licence}{model.commercial_use ? ' · commercial use allowed' : ''}</p>
      </CardContent>
    </Card>
  );
}

function CustomObjectsCard({ projectId }: { projectId: string }) {
  const project = useProject(projectId).data;
  const custom = customLabels(project);
  const remove = useRemoveCustomObject(projectId);
  const [adding, setAdding] = useState(false);
  const [confirm, setConfirm] = useState<string | null>(null);
  return (
    <Card className="h-fit lg:col-start-2">
      <CardHeader className="flex-row flex-wrap items-center justify-between gap-3">
        <div>
          <CardTitle>Custom objects</CardTitle>
          <CardDescription>Your own objects, trained on your photos ({STATUS_TEXT[customStatus(project)] || 'none yet'}).</CardDescription>
        </div>
        <Button size="sm" variant="outline" onClick={() => setAdding(true)}><Plus /> Add your custom object</Button>
      </CardHeader>
      <CardContent>
        {custom.length ? (
          <ul className="divide-y text-sm">
            {custom.map(c => (
              <li key={c} className="flex items-center justify-between py-2">
                <span className="font-medium">{c}</span>
                <Button size="sm" variant="ghost" onClick={() => setConfirm(c)} aria-label={`Remove ${c}`}><Trash2 /> Remove</Button>
              </li>
            ))}
          </ul>
        ) : <p className="text-sm text-muted-foreground">None yet.</p>}
      </CardContent>
      {adding && <AddCustomObjectDialog projectId={projectId} onClose={() => setAdding(false)} />}
      <Dialog open={!!confirm} onOpenChange={o => !o && setConfirm(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Remove “{confirm}”?</DialogTitle>
            <DialogDescription>It’s no longer trained or detected. Boxes you already drew stay in the dataset (shown as an unknown class) in case you add it back.</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setConfirm(null)}>Cancel</Button>
            <Button variant="destructive" disabled={remove.isPending} onClick={() => remove.mutate(confirm!, { onSuccess: () => (toast.success(`Removed “${confirm}”`), setConfirm(null)), onError: e => toast.error(e.message) })}>Remove</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
}

const STATUS_TEXT: Record<CustomStatus, string> = { none: '', not_trained: 'not trained yet', training: 'training…', ready: 'trained ✓', outdated: 'retrain needed', needs_attention: 'trained, check results' };

export function ReadyOverview({ projectId }: { projectId: string }) {
  const s = useDetectorSettings(projectId).data;
  const project = useProject(projectId).data;
  const custom = customLabels(project);
  const st = customStatus(project);
  const [adding, setAdding] = useState(false);
  const next = st === 'ready' ? { to: '/projects/$projectId/playground' as const, label: 'Try it' } : st === 'training' ? { to: '/projects/$projectId/train' as const, label: 'Watch training' } : st === 'outdated' ? { to: '/projects/$projectId/train' as const, label: 'Retrain to include your changes' } : st === 'needs_attention' ? { to: '/projects/$projectId/evaluate' as const, label: 'Check results' } : { to: '/projects/$projectId/data' as const, label: 'Add photos' };
  return (
    <div className="grid gap-6">
      <div className="grid gap-3 md:grid-cols-2">
        <Card className="border-success/40 p-4">
          <div className="flex items-center gap-2 font-medium"><CheckCircle2 className="size-4 text-success" /> Ready-made: {s?.classes.length ?? project?.current_spec.labels?.length ?? 0} objects ✓</div>
          <p className="mt-1 text-xs text-muted-foreground">{s?.classes.join(', ')}</p>
          <div className="mt-3 flex gap-3 text-xs font-medium">
            <Link to="/projects/$projectId/settings" params={{ projectId }} className="text-primary hover:underline">Settings →</Link>
            <Link to="/projects/$projectId/playground" params={{ projectId }} className="text-primary hover:underline">Try it →</Link>
          </div>
        </Card>
        <Card className={cn('p-4', st === 'ready' && 'border-success/40')}>
          <div className="flex items-center gap-2 font-medium">
            <GraduationCap className="size-4 text-muted-foreground" />
            {custom.length ? <>Custom: {custom.length} {custom.length === 1 ? 'object' : 'objects'}, {STATUS_TEXT[st]}</> : 'Custom objects'}
          </div>
          <p className="mt-1 text-xs text-muted-foreground">{custom.length ? custom.join(', ') : 'Anything the ready-made list doesn’t have: your products, parts, defects. Needs labelled photos and a short training.'}</p>
          <div className="mt-3 flex flex-wrap items-center gap-3 text-xs font-medium">
            {custom.length > 0 && <Link to={next.to} params={{ projectId }} className="text-primary hover:underline">{next.label} →</Link>}
            <Button size="sm" variant="outline" onClick={() => setAdding(true)}><Plus /> Add your custom object (requires training)</Button>
          </div>
        </Card>
      </div>
      <Card>
        <CardHeader className="flex-row items-center justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-2 text-base"><Package className="size-5" /> Export</CardTitle>
            <CardDescription>Downloading this detector to run offline comes in Phase 2. Use the Playground for now.</CardDescription>
          </div>
          <Badge variant="secondary">Phase 2</Badge>
        </CardHeader>
      </Card>
      {adding && <AddCustomObjectDialog projectId={projectId} onClose={() => setAdding(false)} />}
    </div>
  );
}

/** One name at a time; a ready-made name is redirected to the ready-made list instead. */
export function AddCustomObjectDialog({ projectId, onClose }: { projectId: string; onClose: () => void }) {
  const add = useAddCustomObject(projectId);
  const settings = useDetectorSettings(projectId).data;
  const saveClasses = useSaveDetectorSettings(projectId);
  const navigate = useNavigate();
  const [name, setName] = useState('');
  const [msg, setMsg] = useState<{ kind: 'ready' | 'error'; text: string; canonical?: string } | null>(null);
  const submit = () =>
    add.mutate(name.trim(), {
      onSuccess: r => {
        toast.success(`Added custom object "${name.trim()}": next, add photos that show it`);
        onClose();
        if (r.custom_labels.length === 1) navigate({ to: '/projects/$projectId/data', params: { projectId } });
      },
      onError: e => {
        const code = e instanceof ApiError ? e.problem.code : null;
        if (code === 'ready_made_class') {
          const canonical = ((e as ApiError).problem as { name?: string }).name;
          const already = !!canonical && settings?.classes.includes(canonical);
          setMsg({ kind: 'ready', canonical: already ? undefined : canonical, text: already ? `“${canonical}” is ready-made and already ticked: it's detected automatically.` : `“${name.trim()}” is already ready-made${canonical && canonical !== name.trim() ? ` (as “${canonical}”)` : ''}. Just tick it; no photos or training needed.` });
        }
        else if (code === 'duplicate_class') setMsg({ kind: 'error', text: `“${name.trim()}” is already one of your custom objects.` });
        else setMsg({ kind: 'error', text: e.message });
      },
    });
  return (
    <Dialog open onOpenChange={o => !o && onClose()}>
      <DialogContent>
        <form className="grid gap-4" onSubmit={e => (e.preventDefault(), name.trim() && submit())}>
          <DialogHeader>
            <DialogTitle>Add your custom object</DialogTitle>
            <DialogDescription>For something the ready-made detector doesn’t know. You’ll add photos, draw boxes around it and run a short training; ready-made objects keep working meanwhile.</DialogDescription>
          </DialogHeader>
          <label className="grid gap-1.5 text-sm">
            <span className="font-medium">Object name</span>
            <Input value={name} onChange={e => (setName(e.target.value), setMsg(null))} maxLength={100} autoFocus placeholder="e.g. company mug, bolt, label" />
          </label>
          {msg && (
            <div className={cn('flex flex-wrap items-center gap-3 rounded-lg border p-3 text-sm', msg.kind === 'ready' ? 'border-info/40 bg-info/8' : 'border-destructive/40 bg-destructive/8 text-destructive')} role="status">
              <span>{msg.text}</span>
              {msg.kind === 'ready' && msg.canonical && settings && (
                <Button type="button" size="sm" disabled={saveClasses.isPending} onClick={() => saveClasses.mutate({ classes: [...settings.classes, msg.canonical!] }, {
                  onSuccess: () => (toast.success(`“${msg.canonical}” added to the ready-made objects`), onClose()),
                  onError: e => toast.error(e.message),
                })}>Add “{msg.canonical}” to ready-made</Button>
              )}
            </div>
          )}
          <DialogFooter>
            <Button type="button" variant="ghost" onClick={onClose}>Cancel</Button>
            <Button type="submit" disabled={!name.trim() || add.isPending}>{add.isPending ? <Loader2 className="animate-spin" /> : <Plus />} Add object</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
