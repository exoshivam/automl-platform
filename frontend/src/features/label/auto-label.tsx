import { Check, Loader2, Pencil, Sparkles, X } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { toast } from 'sonner';
import {
  autoError,
  isActive,
  useAutoSources,
  useCancelAutolabel,
  useLatestJob,
  useReview,
  useStartAutolabel,
  useSuggestions,
  type AutoSource,
  type ReviewState,
  type SuggestedSample,
} from '@/api/autolabel';
import type { Sample, SampleLabel } from '@/api/data-types';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Progress, Skeleton } from '@/components/ui/misc';
import { cn } from '@/lib/utils';
import { ReviewOverlay } from './box-overlay';
import { BoxTool } from './box-tool';

const SOURCE_TEXT: Record<AutoSource, { label: string; desc: string }> = {
  auto: { label: 'Best per label', desc: 'Your trained model where it knows the label, else ready-made objects, else find by name' },
  text_prompt: { label: 'Find by name', desc: 'No examples needed: describe each object in English' },
  own_model: { label: 'Use my trained model', desc: 'Boxes from the model you trained on this project' },
  ready_made: { label: 'Ready-made objects', desc: 'Common objects (COCO): person, cup, bottle, phone…' },
};

/** Auto-label entry point on the detection Label page: start a job, follow it, switch to review. */
export function AutoLabelBar({ vid, classes, sealed, selected, reviewing, onReview }: { vid: string; classes: string[]; sealed: boolean; selected: string[]; reviewing: boolean; onReview: (v: boolean) => void }) {
  const job = useLatestJob(vid).data;
  const cancel = useCancelAutolabel();
  const counts = useSuggestions(vid, 'needs_review', isActive(job)).data?.counts;
  const [open, setOpen] = useState(false);
  const running = isActive(job);
  const pct = job && job.progress.total ? (job.progress.done / job.progress.total) * 100 : 0;
  return (
    <div className="grid gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="outline" size="sm" disabled={sealed || running} onClick={() => setOpen(true)} title={sealed ? 'Auto-label works on the draft' : undefined}>
          <Sparkles /> Auto-label…
        </Button>
        {counts && (counts.pending_boxes > 0 || counts.accepted > 0) && (
          <>
            <Button variant={reviewing ? 'secondary' : 'ghost'} size="sm" aria-pressed={reviewing} onClick={() => onReview(!reviewing)}>
              {reviewing ? 'Back to labelling' : `Review suggestions (${counts.needs_review})`}
            </Button>
            <span className="text-xs text-muted-foreground" aria-live="polite">
              <b className="text-foreground">{counts.confirmed_boxes}</b> confirmed boxes · <b className="text-foreground">{counts.pending_boxes}</b> suggested (not used for training until accepted)
            </span>
          </>
        )}
      </div>
      {job && running && (
        <Card className="flex flex-wrap items-center gap-3 p-3 text-sm" role="status">
          <Loader2 className="size-4 animate-spin text-primary" />
          <span className="font-medium">Auto-labelling {job.progress.done}/{job.progress.total || '…'} photos</span>
          <Progress value={pct} className="h-2 min-w-32 flex-1" aria-label="Auto-label progress" />
          <span className="w-full text-xs text-muted-foreground sm:w-auto">{job.status === 'queued' ? 'Waiting for the training queue…' : 'Runs in the background: you can leave this page and come back.'}</span>
          <Button size="sm" variant="ghost" disabled={cancel.isPending} onClick={() => cancel.mutate(job.id, { onError: e => toast.error(autoError(e)) })}><X /> Cancel</Button>
        </Card>
      )}
      {job && !running && job.status !== 'done' && job.finished_at && Date.now() - new Date(job.finished_at).getTime() < 10 * 60_000 && (
        <p className="text-xs text-muted-foreground" role="status">Last auto-label {job.status}{job.error ? `: ${job.error}` : ''}. Suggestions made before it stopped are kept for review.</p>
      )}
      {open && <AutoLabelDialog vid={vid} classes={classes} selected={selected} onClose={() => setOpen(false)} onStarted={() => setOpen(false)} />}
    </div>
  );
}

function AutoLabelDialog({ vid, classes, selected, onClose, onStarted }: { vid: string; classes: string[]; selected: string[]; onClose: () => void; onStarted: () => void }) {
  const sources = useAutoSources(vid).data;
  const start = useStartAutolabel(vid);
  const [source, setSource] = useState<AutoSource>('auto');
  const [pick, setPick] = useState<string[]>(classes);
  const [prompts, setPrompts] = useState<Record<string, string>>({});
  const [scope, setScope] = useState<'unlabelled' | 'sample_ids'>(selected.length ? 'sample_ids' : 'unlabelled');
  const [conf, setConf] = useState(0.3);
  const avail = (s: AutoSource) => s === 'auto' || !!sources?.[s].available;
  // Labels this source can't find (own model / COCO only know some of them).
  const known = source === 'own_model' ? sources?.own_model.classes : source === 'ready_made' ? sources?.ready_made.classes : undefined;
  const usable = pick.filter(c => !known || known.includes(c));
  const promptable = source === 'text_prompt' || (source === 'auto' && sources?.text_prompt.available);
  const go = () =>
    start.mutate(
      { source, classes: usable, prompts: promptable ? Object.fromEntries(Object.entries(prompts).filter(([c, p]) => usable.includes(c) && p.trim())) : undefined, min_confidence: conf, scope, sample_ids: scope === 'sample_ids' ? selected : undefined },
      { onSuccess: () => (toast.success('Auto-label started. It runs in the background.'), onStarted()), onError: e => toast.error(autoError(e)) },
    );
  return (
    <Dialog open onOpenChange={o => !o && onClose()}>
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle>Auto-label photos</DialogTitle>
          <DialogDescription>Boxes come back as suggestions. Nothing is used for training until you accept it.</DialogDescription>
        </DialogHeader>
        <div className="grid gap-2" role="radiogroup" aria-label="Source">
          {(Object.keys(SOURCE_TEXT) as AutoSource[]).map(s => (
            <button key={s} type="button" role="radio" aria-checked={source === s} disabled={!avail(s)} onClick={() => setSource(s)}
              className={cn('grid gap-0.5 rounded-lg border p-2.5 text-left text-sm', source === s ? 'border-primary bg-primary/6 ring-2 ring-primary/20' : 'hover:bg-accent/50', !avail(s) && 'cursor-not-allowed opacity-50')}>
              <span className="font-medium">{SOURCE_TEXT[s].label}{!avail(s) && <span className="ml-2 text-xs font-normal text-muted-foreground">{s === 'own_model' ? 'train a model first' : 'not available'}</span>}</span>
              <span className="text-xs text-muted-foreground">{SOURCE_TEXT[s].desc}</span>
            </button>
          ))}
        </div>
        <fieldset className="grid gap-2">
          <legend className="mb-1 text-sm font-medium">Labels{promptable ? ' and how to describe them (optional)' : ''}</legend>
          {classes.map(c => {
            const off = !!known && !known.includes(c);
            return (
              <div key={c} className="flex items-center gap-2">
                <label className={cn('flex w-36 shrink-0 items-center gap-2 text-sm', off && 'opacity-50')}>
                  <input type="checkbox" checked={pick.includes(c) && !off} disabled={off} onChange={e => setPick(p => (e.target.checked ? [...p, c] : p.filter(x => x !== c)))} />
                  <span className="truncate">{c}</span>
                </label>
                {promptable && !off ? (
                  <Input className="h-8" placeholder={`e.g. “a ${c}”`} value={prompts[c] ?? ''} onChange={e => setPrompts(p => ({ ...p, [c]: e.target.value }))} aria-label={`English description for ${c}`} maxLength={120} />
                ) : off ? <span className="text-xs text-muted-foreground">this source doesn't know “{c}”</span> : null}
              </div>
            );
          })}
        </fieldset>
        <div className="grid gap-3 sm:grid-cols-2">
          <fieldset className="grid gap-1 text-sm">
            <legend className="mb-1 font-medium">Photos</legend>
            <label className="flex items-center gap-2"><input type="radio" name="scope" checked={scope === 'unlabelled'} onChange={() => setScope('unlabelled')} /> All photos without boxes</label>
            <label className={cn('flex items-center gap-2', !selected.length && 'opacity-50')}><input type="radio" name="scope" disabled={!selected.length} checked={scope === 'sample_ids'} onChange={() => setScope('sample_ids')} /> The {selected.length} selected photos</label>
          </fieldset>
          <label className="grid gap-1 text-sm">
            <span className="font-medium">Minimum confidence: {Math.round(conf * 100)}%</span>
            <input type="range" min={0.05} max={0.95} step={0.05} value={conf} onChange={e => setConf(Number(e.target.value))} aria-label="Minimum confidence" />
            <span className="text-xs text-muted-foreground">Lower finds more, with more wrong boxes to reject.</span>
          </label>
        </div>
        {sources && promptable && <p className="text-xs text-muted-foreground">Find by name uses {sources.text_prompt.model_id} ({sources.text_prompt.licence}). {sources.text_prompt.note}</p>}
        <DialogFooter>
          <Button variant="ghost" onClick={onClose}>Cancel</Button>
          <Button onClick={go} disabled={!usable.length || start.isPending}>{start.isPending ? <Loader2 className="animate-spin" /> : <Sparkles />} Start auto-label</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

const FILTERS: { id: ReviewState; label: string }[] = [
  { id: 'needs_review', label: 'Needs review' },
  { id: 'accepted', label: 'Accepted' },
  { id: 'none_found', label: 'No boxes found' },
  { id: 'rejected', label: 'Rejected' },
];

/** Review grid: dashed = suggested (with score), solid = confirmed. A/R accept/reject the focused photo, E edits, arrows move. */
export function ReviewPanel({ vid, classes, onVersion }: { vid: string; classes: string[]; onVersion: (v: string) => void }) {
  const [state, setState] = useState<ReviewState>('needs_review');
  const job = useLatestJob(vid).data;
  const sug = useSuggestions(vid, state, isActive(job));
  const review = useReview(vid);
  const [focus, setFocus] = useState(0);
  const [above, setAbove] = useState(0.6);
  const [editing, setEditing] = useState<SuggestedSample | null>(null);
  const rows = useMemo(() => sug.data?.samples ?? [], [sug.data]);
  const counts = sug.data?.counts;
  const grid = useRef<HTMLUListElement>(null);
  useEffect(() => setFocus(0), [state]);

  const act = (body: Parameters<typeof review.mutate>[0], ok: (r: { accepted: number; rejected: number }) => string) =>
    review.mutate(body, {
      onSuccess: r => (toast.success(ok(r)), r.dataset_version_id !== vid && onVersion(r.dataset_version_id)),
      onError: e => toast.error(autoError(e)),
    });
  const acceptOne = (s: SuggestedSample) => act({ accept_samples: [s.sample_id] }, r => `Accepted ${r.accepted} box${r.accepted === 1 ? '' : 'es'}`);
  const rejectOne = (s: SuggestedSample) => act({ reject_samples: [s.sample_id] }, r => `Rejected ${r.rejected} box${r.rejected === 1 ? '' : 'es'}`);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (editing || document.querySelector('[role=dialog]') || (e.target as HTMLElement)?.closest('input,select,textarea')) return;
      const s = rows[focus];
      const cols = grid.current ? Math.max(1, Math.round(grid.current.clientWidth / ((grid.current.firstElementChild as HTMLElement | null)?.clientWidth || 1))) : 1;
      if (e.key === 'ArrowRight') setFocus(f => Math.min(rows.length - 1, f + 1));
      else if (e.key === 'ArrowLeft') setFocus(f => Math.max(0, f - 1));
      else if (e.key === 'ArrowDown') setFocus(f => Math.min(rows.length - 1, f + cols));
      else if (e.key === 'ArrowUp') setFocus(f => Math.max(0, f - cols));
      else if (!s || review.isPending) return;
      else if (e.key === 'a' && s.state === 'needs_review') acceptOne(s);
      else if (e.key === 'r' && s.state === 'needs_review') rejectOne(s);
      else if (e.key === 'e') setEditing(s);
      else return;
      e.preventDefault();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });
  useEffect(() => { grid.current?.children[focus]?.scrollIntoView({ block: 'nearest' }); }, [focus]);

  return (
    <div className="grid gap-3">
      <Card className="sticky top-16 z-20 flex flex-wrap items-center gap-2 p-3">
        <div className="flex rounded-md border p-0.5" role="tablist" aria-label="Review filter">
          {FILTERS.map(f => (
            <button key={f.id} type="button" role="tab" aria-selected={state === f.id} onClick={() => setState(f.id)} className={cn('rounded px-2.5 py-1 text-xs', state === f.id ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:bg-accent')}>
              {f.label}{counts ? ` (${counts[f.id]})` : ''}
            </button>
          ))}
        </div>
        <span className="mx-1 h-5 w-px bg-border" />
        <label className="flex items-center gap-2 text-xs">
          Accept all ≥ <input type="range" min={0.05} max={0.95} step={0.05} value={above} onChange={e => setAbove(Number(e.target.value))} aria-label="Accept threshold" className="w-24" /> <span className="w-8 tabular-nums">{Math.round(above * 100)}%</span>
        </label>
        <Button size="sm" variant="outline" disabled={!counts?.pending_boxes || review.isPending} onClick={() => act({ accept_above: above }, r => `Accepted ${r.accepted} boxes at ${Math.round(above * 100)}% or more`)}><Check /> Accept</Button>
        <Button size="sm" variant="ghost" className="text-destructive hover:text-destructive" disabled={!counts?.pending_boxes || review.isPending} onClick={() => act({ reject_remaining: true }, r => `Rejected ${r.rejected} remaining boxes`)}><X /> Reject remaining</Button>
        <span className="ml-auto text-xs text-muted-foreground">{counts ? <><b className="text-foreground">{counts.confirmed_boxes}</b> confirmed · <b className="text-foreground">{counts.pending_boxes}</b> suggested</> : null} · <kbd>A</kbd> accept <kbd>R</kbd> reject <kbd>E</kbd> edit <kbd>←→</kbd> move</span>
      </Card>

      {sug.isLoading ? <Skeleton className="h-64 rounded-xl" /> : !rows.length ? (
        <Card className="p-10 text-center text-sm text-muted-foreground">{state === 'needs_review' ? 'Nothing left to review.' : 'No photos here.'}</Card>
      ) : (
        <ul ref={grid} className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4" aria-label="Suggested boxes">
          {rows.map((s, i) => {
            const pending = s.boxes.filter(b => b.status === 'pending');
            const confirmed = (s.labels ?? []).filter((l): l is SampleLabel & { bbox: NonNullable<SampleLabel['bbox']> } => !!l.bbox);
            return (
              <li key={s.sample_id} aria-current={i === focus} onClick={() => setFocus(i)} className={cn('grid gap-1.5 rounded-lg border bg-card p-1.5', i === focus && 'border-primary ring-2 ring-primary/25')}>
                <div className="relative aspect-[4/3] overflow-hidden rounded-md bg-muted">
                  {s.thumb_url && <img src={s.thumb_url} alt={s.name ?? 'photo'} loading="lazy" className="size-full object-contain" />}
                  {s.width && s.height && <ReviewOverlay width={s.width} height={s.height} classes={classes} confirmed={confirmed} suggested={pending} />}
                </div>
                <div className="flex items-center gap-1">
                  <span className="min-w-0 flex-1 truncate text-xs text-muted-foreground">{pending.length ? `${pending.length} suggested` : s.state === 'none_found' ? 'no boxes found' : s.state}</span>
                  {pending.length > 0 && (
                    <>
                      <Button size="icon" variant="ghost" className="size-7 text-success" aria-label="Accept all boxes on this photo" disabled={review.isPending} onClick={() => acceptOne(s)}><Check className="size-4" /></Button>
                      <Button size="icon" variant="ghost" className="size-7 text-destructive" aria-label="Reject all boxes on this photo" disabled={review.isPending} onClick={() => rejectOne(s)}><X className="size-4" /></Button>
                    </>
                  )}
                  <Button size="icon" variant="ghost" className="size-7" aria-label="Edit boxes" disabled={!s.url} onClick={() => setEditing(s)}><Pencil className="size-3.5" /></Button>
                </div>
              </li>
            );
          })}
        </ul>
      )}
      {editing && <EditDialog vid={vid} classes={classes} s={editing} onVersion={onVersion} onClose={() => setEditing(null)} />}
    </div>
  );
}

/** The normal box editor, preloaded with the photo's confirmed + pending boxes. Saving writes labels; the photo's
 * leftover suggestions are then cleared (they're labels now, or the user deleted them). */
function EditDialog({ vid, classes, s, onVersion, onClose }: { vid: string; classes: string[]; s: SuggestedSample; onVersion: (v: string) => void; onClose: () => void }) {
  const review = useReview(vid);
  const saved = useRef(false);
  const sample = useMemo(
    () => ({
      id: s.sample_id,
      name: s.name,
      url: s.url,
      thumb_url: s.thumb_url,
      width: s.width,
      height: s.height,
      no_objects: false,
      labels: [...(s.labels ?? []).filter(l => l.bbox), ...s.boxes.filter(b => b.status === 'pending').map(b => ({ class_name: b.class_name, bbox: b.bbox, source: 'ai_prelabel' as const, confidence: b.confidence }))],
    }) as unknown as Sample,
    [s],
  );
  const close = () => {
    if (saved.current) review.mutate({ reject_samples: [s.sample_id] }, { onSettled: onClose });
    else onClose();
  };
  return (
    <Dialog open onOpenChange={o => !o && close()}>
      <DialogContent className="max-w-5xl">
        <DialogHeader>
          <DialogTitle>Edit boxes: {s.name ?? 'photo'}</DialogTitle>
          <DialogDescription>Suggested boxes are preloaded. Adjust, delete or add boxes; changes save automatically. <Badge variant="secondary">saved boxes count as confirmed</Badge></DialogDescription>
        </DialogHeader>
        <BoxTool samples={[sample]} index={0} setIndex={() => undefined} classes={classes} versionId={vid} onVersion={v => onVersion(v)} onSaved={() => void (saved.current = true)} />
        <DialogFooter><Button onClick={close} disabled={review.isPending}>{review.isPending ? <Loader2 className="animate-spin" /> : null} Done</Button></DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
