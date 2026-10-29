import { isPretrained, trainableLabels } from '@/api/detector';
import { api } from '@/lib/api';
import type { BoxSuggestions } from '@/api/data-types';
import { Info } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { Link, useNavigate } from '@tanstack/react-router';
import { CheckSquare, Plus, ShieldCheck, Square, Trash2 } from 'lucide-react';
import { toast } from 'sonner';
import { pickVersion, useLabelSet, useLabelSetChange, useSamples, useSaveLabels, useSeal, useVersions } from '@/api/data-hooks';
import type { Sample, SampleLabel } from '@/api/data-types';
import { useProject } from '@/api/hooks';
import type { Project } from '@/api/types';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Progress, Skeleton, Switch, Tooltip } from '@/components/ui/misc';
import { RemoveSamplesDialog, SEALED_HINT } from '@/features/data/remove-samples';
import { Label } from '@/components/ui/label';
import { cn } from '@/lib/utils';
import { BoxTool } from './box-tool';
import { AutoLabelBar, ReviewPanel } from './auto-label';
import { classColor } from './box-overlay';
import { UnknownClasses } from '@/features/data/dataset-page';

export function LabelPage({ projectId }: { projectId: string }) {
  const { data: project } = useProject(projectId);
  const versions = useVersions(project?.dataset_id);
  const labelSet = useLabelSet(project?.label_set_id);
  const [vid, setVid] = useState<string | undefined>();
  const current = pickVersion(versions.data?.items, vid);
  const seal = useSeal();
  const navigate = useNavigate();
  const classes = labelSet.data?.classes ?? trainableLabels(project);

  if (!project || versions.isLoading) return <Skeleton className="h-96 rounded-xl" />;
  if (!current)
    return (
      <Card className="flex flex-col items-center gap-3 border-dashed p-12 text-center">
        <p className="font-medium">Nothing to label yet</p>
        <Button asChild>
          <Link to="/projects/$projectId/data" params={{ projectId }}>
            Add data
          </Link>
        </Button>
      </Card>
    );

  const labelled = Object.values(current.label_stats).reduce((n, s) => n + s.images, 0);
  const total = current.sample_count || 1;
  const finish = () =>
    seal.mutate(current.id, {
      onSuccess: r => navigate({ to: '/projects/$projectId/dataset', params: { projectId }, search: { v: r.dataset_version_id, tab: 'validation' } }),
      onError: e => toast.error(e.message),
    });

  return (
    <div className="grid gap-4">
      {isPretrained(project) && (
        <div className="flex gap-2 rounded-lg border border-info/40 bg-info/8 p-3 text-sm" role="note">
          <Info className="mt-0.5 size-4 shrink-0 text-info" />
          <span>Ready-made objects ({project.current_spec.labels?.join(', ')}) are labelled automatically during training: grey “auto · not saved” boxes preview them. You only draw your custom objects ({classes.join(', ')}).</span>
        </div>
      )}
      <Card className="flex flex-wrap items-center gap-4 p-4">
        <div className="min-w-48 flex-1">
          <div className="flex items-center justify-between text-sm">
            <span className="font-medium">
              Version {current.version_no}
              {!current.sealed_at && <Badge variant="secondary" className="ml-2">draft</Badge>}
            </span>
            <span className="tabular-nums text-muted-foreground">
              ~{Math.min(labelled, total).toLocaleString()} / {current.sample_count.toLocaleString()} labelled
            </span>
          </div>
          <Progress className="mt-2" value={Math.min(100, (labelled / total) * 100)} aria-label="Labelling progress" />
        </div>
        <ClassManager project={project} classes={classes} />
        <Button onClick={finish} disabled={seal.isPending}>
          <ShieldCheck /> {seal.isPending ? 'Sealing…' : 'Finish: seal and validate'}
        </Button>
      </Card>

      <UnknownClasses project={project} version={current} />
      {project.task_type === 'object_detection' ? (
        <DetectionLabeling project={project} vid={current.id} sealed={!!current.sealed_at} classes={classes} onVersion={setVid} />
      ) : (
        <ClassifyLabeling project={project} vid={current.id} sealed={!!current.sealed_at} classes={classes} onVersion={setVid} />
      )}
    </div>
  );
}

/** LBL-1 label creation (rename/merge live behind the same endpoints). */
function ClassManager({ project, classes }: { project: Project; classes: string[] }) {
  const [adding, setAdding] = useState(false);
  const [name, setName] = useState('');
  const change = useLabelSetChange();
  if (!project.label_set_id) return null;
  return adding ? (
    <form
      className="flex items-center gap-2"
      onSubmit={e => {
        e.preventDefault();
        if (!name.trim()) return;
        change.mutate(
          { id: project.label_set_id!, op: 'classes', body: { add: [name.trim()] } },
          { onSuccess: () => (toast.success(`Class "${name.trim()}" added`), setName(''), setAdding(false)), onError: e => toast.error(e.message) },
        );
      }}
    >
      <Input value={name} onChange={e => setName(e.target.value)} placeholder="New class name" className="h-8 w-40" autoFocus aria-label="New class name" />
      <Button size="sm" type="submit" disabled={change.isPending}>
        Add
      </Button>
      <Button size="sm" variant="ghost" type="button" onClick={() => setAdding(false)}>
        Cancel
      </Button>
    </form>
  ) : (
    <Button variant="outline" size="sm" onClick={() => setAdding(true)} title={`Classes: ${classes.join(', ')}`}>
      <Plus /> Add class
    </Button>
  );
}

/** Exactly the auto ready-made boxes training will add for a sample (read-only preview; not stored). */
const autoPreviewFor = (vid: string) => async (s: Sample) => {
  const r = await api<BoxSuggestions>(`/dataset-versions/${vid}/ready-made-preview`, { method: 'POST', body: { sample_ids: [s.id] } });
  return r.suggestions[s.id] ?? [];
};

function DetectionLabeling({ project, vid, sealed, classes, onVersion }: { project: Project; vid: string; sealed: boolean; classes: string[]; onVersion: (v: string) => void }) {
  const [confirm, setConfirm] = useState<string[] | null>(null);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [reviewing, setReviewing] = useState(false);
  const autoPreview = useMemo(() => autoPreviewFor(vid), [vid]);
  const [onlyTodo, setOnlyTodo] = useState(false);
  const [photosFor, setPhotosFor] = useState('');
  const samples = useSamples(vid, { unlabeled: onlyTodo || undefined, capture_target: photosFor || undefined }, 100);
  const [index, setIndex] = useState(0);
  const [local, setLocal] = useState<Record<string, Sample>>({});
  // Queue grouped by the object each photo was taken for (in class order; photos without a hint last).
  const list = useMemo(() => {
    const rank = (s: Sample) => (s.capture_target && classes.includes(s.capture_target) ? classes.indexOf(s.capture_target) : classes.length);
    return (samples.data?.pages.flatMap(p => p.items) ?? []).map(s => local[s.id] ?? s).map((s, i) => [s, i] as const).sort((a, b) => rank(a[0]) - rank(b[0]) || a[1] - b[1]).map(([s]) => s);
  }, [samples.data, local, classes]);
  // Keep the queue topped up as the labeller nears the end of what's loaded.
  useEffect(() => {
    if (index > list.length - 10 && samples.hasNextPage && !samples.isFetchingNextPage) void samples.fetchNextPage();
  }, [index, list.length, samples]);
  useEffect(() => (setIndex(0), setPicked(new Set())), [onlyTodo, photosFor]);
  void project;
  const cur = list[Math.min(index, list.length - 1)];
  // Shift+Del removes the current photo (plain Del deletes the selected box).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest('input,select,textarea,[role=dialog]')) return;
      if (e.shiftKey && (e.key === 'Delete' || e.key === 'Backspace') && cur) {
        e.preventDefault();
        if (sealed) toast.info(SEALED_HINT);
        else setConfirm(picked.size ? [...picked] : [cur.id]);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [cur, sealed, picked]);
  /** Strip: click = go to photo; Cmd/Ctrl-click toggles; Shift-click picks the range from the current photo. */
  const stripClick = (i: number, e: React.MouseEvent) => {
    if (sealed || !(e.shiftKey || e.metaKey || e.ctrlKey)) return setIndex(i);
    setPicked(prev => {
      const next = new Set(prev);
      if (e.shiftKey) for (let k = Math.min(index, i); k <= Math.max(index, i); k++) next.add(list[k]!.id);
      else if (next.has(list[i]!.id)) next.delete(list[i]!.id);
      else next.add(list[i]!.id);
      return next;
    });
  };

  const auto = <AutoLabelBar vid={vid} classes={classes} sealed={sealed} selected={[...picked]} reviewing={reviewing} onReview={setReviewing} />;
  if (reviewing)
    return (
      <div className="grid gap-3">
        {auto}
        <ReviewPanel vid={vid} classes={classes} onVersion={v => (setLocal({}), onVersion(v))} />
      </div>
    );
  if (samples.isLoading) return <Skeleton className="h-[60vh] rounded-xl" />;
  if (!list.length)
    return (
      <div className="grid gap-3">
        {auto}
        <Card className="p-10 text-center text-sm text-muted-foreground">
          {onlyTodo ? 'Every image has boxes or is marked "no objects".' : 'No images in this version.'}
          {onlyTodo && (
            <Button variant="link" onClick={() => setOnlyTodo(false)}>
              Show all images
            </Button>
          )}
        </Card>
      </div>
    );

  return (
    <div className="grid gap-3">
      {auto}
      <div className="flex items-center gap-2">
        <Switch id="todo" checked={onlyTodo} onCheckedChange={setOnlyTodo} />
        <Label htmlFor="todo">Only images without boxes</Label>
        {classes.length > 1 && (
          <>
            <Label htmlFor="photos-for" className="ml-3">Photos for</Label>
            <select id="photos-for" className="h-8 rounded-md border border-input bg-card px-2 text-sm" value={photosFor} onChange={e => setPhotosFor(e.target.value)}>
              <option value="">All objects</option>
              {classes.map(c => <option key={c} value={c}>{c}</option>)}
              <option value="__none">Mixed / not specified</option>
            </select>
          </>
        )}
        <div className="ml-auto" />
        {sealed ? (
          <Tooltip content={SEALED_HINT}>
            <span tabIndex={0} aria-label={`Remove photo (unavailable): ${SEALED_HINT}`}>
              <Button variant="ghost" size="sm" className="pointer-events-none" disabled><Trash2 /> Remove photo</Button>
            </span>
          </Tooltip>
        ) : (
          <>
            <Link to="/projects/$projectId/dataset" params={{ projectId: project.id }} search={{ v: vid, select: true }} className="text-xs text-muted-foreground hover:text-foreground hover:underline">Select many photos →</Link>
            {picked.size > 0 && <Button variant="ghost" size="sm" onClick={() => setPicked(new Set())}>Clear {picked.size}</Button>}
            <Button variant="ghost" size="sm" className="text-destructive hover:text-destructive" onClick={() => setConfirm(picked.size ? [...picked] : cur ? [cur.id] : null)} title="Remove from the draft (Shift+Del). Cmd/Ctrl-click or Shift-click thumbnails to pick several.">
              <Trash2 /> {picked.size ? `Remove ${picked.size} photos` : 'Remove photo'} <kbd className="text-[10px] opacity-60">⇧Del</kbd>
            </Button>
          </>
        )}
        <div className="flex max-w-[55%] gap-1 overflow-x-auto" aria-label="Image strip">
          {list.slice(Math.max(0, index - 4), index + 8).map((s, k) => {
            const i = Math.max(0, index - 4) + k;
            const n = s.labels.filter(l => l.bbox).length;
            return (
              <button
                key={s.id}
                type="button"
                onClick={e => stripClick(i, e)}
                aria-label={`Image ${i + 1}${n ? `, ${n} boxes` : ''}${picked.has(s.id) ? ', picked for removal' : ''}`}
                aria-current={i === index}
                className={cn('relative h-10 w-14 shrink-0 cursor-pointer overflow-hidden rounded border-2', picked.has(s.id) ? 'border-destructive' : i === index ? 'border-primary' : 'border-transparent opacity-70 hover:opacity-100')}
              >
                {picked.has(s.id) && <CheckSquare className="absolute left-0 top-0 z-10 size-3.5 rounded-sm bg-destructive text-white" />}
                {s.thumb_url && <img src={s.thumb_url} alt="" className="size-full object-cover" />}
                <span className={cn('absolute bottom-0 right-0 rounded-tl px-1 text-[9px] font-semibold text-white', n ? 'bg-success' : s.no_objects ? 'bg-muted-foreground' : 'bg-black/50')}>{n || (s.no_objects ? '∅' : '·')}</span>
              </button>
            );
          })}
        </div>
      </div>
      <BoxTool
        autoPreview={isPretrained(project) ? autoPreview : undefined}
        samples={list}
        index={Math.min(index, list.length - 1)}
        setIndex={setIndex}
        classes={classes}
        versionId={vid}
        onVersion={v => (setLocal({}), onVersion(v))}
        onSaved={s => setLocal(m => ({ ...m, [s.id]: s }))}
      />
      <RemoveSamplesDialog vid={vid} ids={confirm} onClose={() => setConfirm(null)} onDone={() => setPicked(new Set())} />
    </div>
  );
}

/** LBL-1: multi-select samples, then a class button or number key labels them all; autosaved. */
function ClassifyLabeling({ project, vid, sealed, classes, onVersion }: { project: Project; vid: string; sealed: boolean; classes: string[]; onVersion: (v: string) => void }) {
  const [onlyTodo, setOnlyTodo] = useState(true);
  const samples = useSamples(vid, { unlabeled: onlyTodo || undefined }, 120);
  const save = useSaveLabels();
  const [local, setLocal] = useState<Record<string, SampleLabel[]>>({});
  const [sel, setSel] = useState<Set<string>>(new Set());
  const anchor = useRef<number | null>(null);
  const text = project.task_type === 'text_classification';
  const [confirm, setConfirm] = useState<string[] | null>(null);
  const labelsOf = (s: Sample) => local[s.id] ?? s.labels;
  // In "unlabelled only", items labelled this session drop out right away.
  const list = (samples.data?.pages.flatMap(p => p.items) ?? []).filter(s => !onlyTodo || !local[s.id]);

  const apply = (cls: string) => {
    const ids = [...sel];
    if (!ids.length) return toast.info('Select samples first (click, Shift-click, Ctrl/⌘-click)');
    const labels: SampleLabel[] = [{ class_name: cls, bbox: null, source: 'user' }];
    setLocal(m => ({ ...m, ...Object.fromEntries(ids.map(id => [id, labels])) }));
    setSel(new Set());
    save.mutate(
      { vid, upserts: ids.map(id => ({ sample_id: id, labels })) },
      {
        onSuccess: r => {
          toast.success(`${ids.length} sample${ids.length === 1 ? '' : 's'} → ${cls}`);
          if (r.forked || r.dataset_version_id !== vid) (setLocal({}), onVersion(r.dataset_version_id));
        },
        onError: e => (toast.error(`Couldn't save: ${e.message}`), setLocal(m => Object.fromEntries(Object.entries(m).filter(([k]) => !ids.includes(k))))),
      },
    );
  };

  const click = (i: number, e: React.MouseEvent) => {
    const id = list[i]!.id;
    setSel(prev => {
      const next = new Set(e.metaKey || e.ctrlKey ? prev : []);
      if (e.shiftKey && anchor.current !== null) {
        const [a, b] = [Math.min(anchor.current, i), Math.max(anchor.current, i)];
        for (let k = a; k <= b; k++) next.add(list[k]!.id);
      } else if ((e.metaKey || e.ctrlKey) && prev.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
    if (!e.shiftKey) anchor.current = i;
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest('input,select,textarea')) return;
      const n = Number(e.key);
      if (n >= 1 && n <= classes.length) apply(classes[n - 1]!);
      else if ((e.metaKey || e.ctrlKey) && e.key === 'a') (e.preventDefault(), setSel(new Set(list.map(s => s.id))));
      else if (e.key === 'Escape') setSel(new Set());
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  return (
    <div className="grid gap-3">
      <Card className="sticky top-16 z-20 flex flex-wrap items-center gap-2 p-3">
        <span className="text-sm font-medium tabular-nums">{sel.size} selected</span>
        <Button variant="ghost" size="sm" onClick={() => setSel(sel.size ? new Set() : new Set(list.map(s => s.id)))}>
          {sel.size ? <Square /> : <CheckSquare />} {sel.size ? 'Clear' : 'Select all shown'}
        </Button>
        <span className="mx-1 h-5 w-px bg-border" />
        <span className="text-xs text-muted-foreground">Label as:</span>
        {classes.map((c, i) => (
          <Button key={c} size="sm" variant="outline" onClick={() => apply(c)} disabled={!sel.size}>
            <span className="size-2.5 rounded-sm" style={{ background: classColor(classes, c) }} />
            <kbd className="text-[10px] opacity-60">{i + 1}</kbd> {c}
          </Button>
        ))}
        {!sealed && (
          <Button size="sm" variant="ghost" className="text-destructive hover:text-destructive" disabled={!sel.size} onClick={() => setConfirm([...sel])}>
            <Trash2 /> Remove{sel.size ? ` ${sel.size}` : ''} from draft
          </Button>
        )}
        <div className="ml-auto flex items-center gap-2">
          <Switch id="todo-c" checked={onlyTodo} onCheckedChange={v => (setOnlyTodo(v), setSel(new Set()))} />
          <Label htmlFor="todo-c">Unlabelled only</Label>
        </div>
      </Card>

      {samples.isLoading ? (
        <Skeleton className="h-96 rounded-xl" />
      ) : !list.length ? (
        <Card className="p-10 text-center text-sm text-muted-foreground">{onlyTodo ? 'Everything here is labelled. 🎉' : 'No samples in this version.'}</Card>
      ) : text ? (
        <ul className="grid gap-1.5" role="listbox" aria-multiselectable aria-label="Texts to label">
          {list.map((s, i) => {
            const l = labelsOf(s)[0]?.class_name;
            return (
              <li key={s.id} role="option" aria-selected={sel.has(s.id)}>
                <button type="button" onClick={e => click(i, e)} className={cn('flex w-full cursor-pointer items-start gap-3 rounded-lg border bg-card p-3 text-left text-sm', sel.has(s.id) ? 'border-primary ring-2 ring-primary/25' : 'hover:bg-accent/40')}>
                  <span className="flex-1">{s.text}</span>
                  {l ? <Badge style={{ background: `${classColor(classes, l)}22`, color: classColor(classes, l) }}>{l}</Badge> : <Badge variant="outline">unlabelled</Badge>}
                </button>
              </li>
            );
          })}
        </ul>
      ) : (
        <ul className="grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8" role="listbox" aria-multiselectable aria-label="Images to label">
          {list.map((s, i) => {
            const l = labelsOf(s)[0]?.class_name;
            return (
              <li key={s.id} role="option" aria-selected={sel.has(s.id)}>
                <button type="button" onClick={e => click(i, e)} className={cn('relative block aspect-square w-full cursor-pointer overflow-hidden rounded-lg border-2 bg-muted', sel.has(s.id) ? 'border-primary' : 'border-transparent hover:border-primary/40')} aria-label={`${s.name ?? 'image'}${l ? `, labelled ${l}` : ''}`}>
                  {s.thumb_url && <img src={s.thumb_url} alt="" loading="lazy" className="size-full object-cover" />}
                  {sel.has(s.id) && <CheckSquare className="absolute right-1 top-1 size-5 rounded bg-primary text-primary-foreground" />}
                  {l && (
                    <span className="absolute inset-x-0 bottom-0 truncate px-1 py-0.5 text-[10px] font-medium text-white" style={{ background: classColor(classes, l) }}>
                      {l}
                    </span>
                  )}
                </button>
              </li>
            );
          })}
        </ul>
      )}
      {samples.hasNextPage && (
        <Button variant="outline" className="mx-auto" onClick={() => samples.fetchNextPage()}>
          Load more
        </Button>
      )}
      <RemoveSamplesDialog vid={vid} ids={confirm} unit={text ? 'text' : 'photo'} onClose={() => setConfirm(null)} onDone={() => setSel(new Set())} />
    </div>
  );
}
