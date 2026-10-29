import { trainableLabels } from '@/api/detector';
import { useEffect, useRef, useState } from 'react';
import { Link, useNavigate } from '@tanstack/react-router';
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip as RTooltip, XAxis, YAxis } from 'recharts';
import { AlertCircle, AlertTriangle, CheckCircle2, CheckSquare, Info, ListChecks, Loader2, ShieldCheck, Sparkles, Square, Trash2, Wrench, X } from 'lucide-react';
import { toast } from 'sonner';
import { VersionsPanel } from './versions-panel';
import { allSampleIds, pickVersion, useAcknowledge, useApplyFixes, useLabelSetChange, useRows, useSamples, useSeal, useValidation, useVersions, type SampleFilters } from '@/api/data-hooks';
import type { DatasetVersion, Sample, Severity, ValidationItem } from '@/api/data-types';
import { useProject } from '@/api/hooks';
import type { Project } from '@/api/types';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { DemoBadge, Skeleton, Switch, Tooltip } from '@/components/ui/misc';
import { RemoveSamplesDialog, SEALED_HINT } from './remove-samples';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { cn, timeAgo } from '@/lib/utils';
import { BoxOverlay } from '@/features/label/box-overlay';

export interface DatasetSearch extends SampleFilters {
  v?: string;
  tab?: 'samples' | 'validation' | 'stats' | 'versions';
  /** Open the Samples tab in Select mode (linked from Label). */
  select?: boolean;
}

const VAL_STATUS: Record<DatasetVersion['validation_status'], { label: string; variant: 'secondary' | 'info' | 'success' | 'warning' | 'destructive' }> = {
  pending: { label: 'Not validated', variant: 'secondary' },
  running: { label: 'Validating…', variant: 'info' },
  passed: { label: 'Validation passed', variant: 'success' },
  warnings: { label: 'Warnings to review', variant: 'warning' },
  failed: { label: 'Blocking errors', variant: 'destructive' },
};

export function DatasetPage({ projectId, search }: { projectId: string; search: DatasetSearch }) {
  const { data: project } = useProject(projectId);
  const versions = useVersions(project?.dataset_id);
  const navigate = useNavigate();
  const v = pickVersion(versions.data?.items, search.v);
  const setSearch = (patch: Partial<DatasetSearch>) =>
    navigate({ to: '/projects/$projectId/dataset', params: { projectId }, search: s => ({ ...s, ...patch }), replace: true });

  if (!project || versions.isLoading) return <Skeleton className="h-96 rounded-xl" />;
  if (!v)
    return (
      <Card className="flex flex-col items-center gap-3 border-dashed p-12 text-center">
        <p className="font-medium">No data yet</p>
        <p className="max-w-sm text-sm text-muted-foreground">Upload files or capture photos. New data collects in a draft here; seal it to create a version you can train on.</p>
        <Button asChild>
          <Link to="/projects/$projectId/data" params={{ projectId }}>
            Add data
          </Link>
        </Button>
      </Card>
    );

  const tabular = project.task_type.startsWith('tabular');
  const vs = VAL_STATUS[v.validation_status];
  const unit = tabular ? 'rows' : 'samples';
  const draft = versions.data!.items.find(x => x.is_draft);
  const sealed = versions.data!.items.filter(x => !x.is_draft);
  return (
    <div className="grid gap-4">
      {draft && <DraftBar draft={draft} unit={unit} selected={draft.id === v.id} onSelect={() => setSearch({ v: draft.id, flag: undefined })} />}
      <div className="flex flex-wrap items-center gap-3">
        <label htmlFor="version" className="sr-only">
          Dataset version
        </label>
        <select
          id="version"
          className="h-9 max-w-full rounded-md border border-input bg-card px-3 text-sm"
          value={v.id}
          onChange={e => setSearch({ v: e.target.value, flag: undefined })}
        >
          {draft && (
            <optgroup label="Draft">
              <option value={draft.id}>Draft (unsealed) · {draft.sample_count.toLocaleString()} {unit}</option>
            </optgroup>
          )}
          <optgroup label="Sealed versions">
            {sealed.map(x => (
              <option key={x.id} value={x.id}>
                v{x.version_no} · {x.sample_count.toLocaleString()} {unit} · {ORIGIN[x.origin] ?? x.origin} · {VAL_STATUS[x.validation_status].label.toLowerCase()} · {timeAgo(x.sealed_at ?? x.created_at)}
              </option>
            ))}
          </optgroup>
        </select>
        {v.ingest_status !== 'ready' ? (
          <Badge variant="info">
            <Loader2 className="size-3 animate-spin" /> Ingest {v.ingest_status}
          </Badge>
        ) : (
          <Badge variant={vs.variant}>{v.validation_status === 'running' ? <Loader2 className="size-3 animate-spin" /> : null}{vs.label}</Badge>
        )}
        {v.split_counts && (
          <span className="text-xs text-muted-foreground tabular-nums">
            Split {v.split_counts.train}/{v.split_counts.val}/{v.split_counts.test}
          </span>
        )}
      </div>

      <UnknownClasses project={project} version={v} />
      <Tabs value={search.tab ?? 'samples'} onValueChange={t => setSearch({ tab: t as DatasetSearch['tab'] })}>
        <TabsList>
          <TabsTrigger value="samples">{tabular ? 'Rows' : 'Samples'}</TabsTrigger>
          <TabsTrigger value="validation">
            <ShieldCheck /> Validation
          </TabsTrigger>
          <TabsTrigger value="stats">Stats</TabsTrigger>
          <TabsTrigger value="versions">Versions &amp; storage</TabsTrigger>
        </TabsList>
        <TabsContent value="samples">{tabular ? <RowsTable vid={v.id} /> : <SampleGrid project={project} version={v} search={search} setSearch={setSearch} />}</TabsContent>
        <TabsContent value="validation">
          <ValidationPanel project={project} version={v} onVersion={id => setSearch({ v: id })} onFlag={code => setSearch({ tab: 'samples', flag: code })} />
        </TabsContent>
        <TabsContent value="stats">
          <Stats project={project} version={v} />
        </TabsContent>
        <TabsContent value="versions">
          <VersionsPanel project={project} />
        </TabsContent>
      </Tabs>
    </div>
  );
}

const ORIGIN: Record<string, string> = { upload: 'from uploads', import: 'imported', labels: 'labelled', fix: 'auto-fix' };

function DraftBar({ draft, unit, selected, onSelect }: { draft: DatasetVersion; unit: string; selected: boolean; onSelect: () => void }) {
  const seal = useSeal();
  const busy = draft.ingest_status !== 'ready';
  const [clearIds, setClearIds] = useState<string[] | null>(null);
  const [loading, setLoading] = useState(false);
  const clear = () => {
    setLoading(true);
    allSampleIds(draft.id).then(setClearIds, e => toast.error((e as Error).message)).finally(() => setLoading(false));
  };
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-xl border border-primary/30 bg-primary/5 p-3">
      <div className="min-w-0 flex-1">
        <div className="text-sm font-medium">
          Draft (unsealed) · {draft.sample_count.toLocaleString()} {unit}
          {busy && <span className="ml-2 text-xs text-muted-foreground"><Loader2 className="inline size-3 animate-spin" /> adding files…</span>}
        </div>
        <p className="text-xs text-muted-foreground">New uploads and captures add here. Seal it to freeze the data, split it and run the checks before training.</p>
      </div>
      {!selected && <Button size="sm" variant="ghost" onClick={onSelect}>View draft</Button>}
      <Button size="sm" variant="ghost" className="text-destructive hover:text-destructive" disabled={busy || loading || !draft.sample_count} onClick={clear}>
        {loading ? <Loader2 className="animate-spin" /> : <Trash2 />} Clear draft…
      </Button>
      <RemoveSamplesDialog vid={draft.id} ids={clearIds} unit={unit.replace(/s$/, '')} typed onClose={() => setClearIds(null)} />
      <Button size="sm" disabled={busy || seal.isPending || !draft.sample_count} onClick={() => seal.mutate(draft.id, { onError: e => toast.error(e.message) })}>
        {seal.isPending ? <Loader2 className="animate-spin" /> : <ShieldCheck />} Seal &amp; validate
      </Button>
    </div>
  );
}

function SampleGrid({ project, version, search, setSearch }: { project: Project; version: DatasetVersion; search: DatasetSearch; setSearch: (p: Partial<DatasetSearch>) => void }) {
  const filters: SampleFilters = { label: search.label, split: search.split, flag: search.flag, unlabeled: search.unlabeled };
  const samples = useSamples(version.id, filters);
  const [open, setOpen] = useState<Sample | null>(null);
  const items = samples.data?.pages.flatMap(p => p.items) ?? [];
  const text = project.task_type === 'text_classification';
  const classes = [...new Set([...trainableLabels(project), ...Object.keys(version.label_stats)])];
  const draft = !version.sealed_at;
  const [selecting, setSelecting] = useState(!!search.select && draft);
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [confirm, setConfirm] = useState<string[] | null>(null);
  const anchor = useRef<number | null>(null);
  // A different version or filter means a different list: start the selection over.
  useEffect(() => (setSel(new Set()), setSelecting(s => s && draft), void (anchor.current = null)), [draft, version.id, search.label, search.split, search.flag, search.unlabeled]);
  /** Click toggles; Shift-click selects the range from the last clicked card. */
  const pick = (i: number, e: React.MouseEvent) => {
    const id = items[i]!.id;
    setSel(prev => {
      const next = new Set(prev);
      if (e.shiftKey && anchor.current !== null) for (let k = Math.min(anchor.current, i); k <= Math.max(anchor.current, i); k++) next.add(items[k]!.id);
      else if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
    if (!e.shiftKey) anchor.current = i;
  };
  const onCard = (s: Sample, i: number) => (e: React.MouseEvent) => (selecting ? pick(i, e) : setOpen(s));
  const unit = text ? 'text' : 'photo';
  const filtered = !!(search.label || search.split || search.flag || search.unlabeled);
  // Whole-version count when unfiltered; with filters, the loaded count until "select all" pages through.
  const [total, setTotal] = useState(0);
  useEffect(() => setTotal(filtered ? 0 : version.sample_count), [filtered, version.sample_count]);
  const [allBusy, setAllBusy] = useState(false);
  const selectAll = () => {
    setAllBusy(true);
    allSampleIds(version.id, filters).then(ids => (setSel(new Set(ids)), setTotal(ids.length)), e => toast.error((e as Error).message)).finally(() => setAllBusy(false));
  };

  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-end gap-3">
        <div className="grid gap-1">
          <Label htmlFor="f-label" className="text-xs text-muted-foreground">
            Label
          </Label>
          <select id="f-label" className="h-9 rounded-md border border-input bg-card px-3 text-sm" value={search.label ?? ''} onChange={e => setSearch({ label: e.target.value || undefined })}>
            <option value="">All labels</option>
            {classes.map(c => (
              <option key={c} value={c}>
                {c} ({version.label_stats[c]?.images ?? 0})
              </option>
            ))}
          </select>
        </div>
        <div className="grid gap-1">
          <Label htmlFor="f-split" className="text-xs text-muted-foreground">
            Split
          </Label>
          <select id="f-split" className="h-9 rounded-md border border-input bg-card px-3 text-sm" value={search.split ?? ''} onChange={e => setSearch({ split: (e.target.value || undefined) as DatasetSearch['split'] })}>
            <option value="">All splits</option>
            <option value="train">Train</option>
            <option value="val">Validation</option>
            <option value="test">Test</option>
          </select>
        </div>
        {search.flag && (
          <Badge variant="warning" className="h-9 px-3">
            Flag {search.flag}
            <button type="button" className="ml-1 cursor-pointer" onClick={() => setSearch({ flag: undefined })} aria-label="Clear flag filter">
              ×
            </button>
          </Badge>
        )}
        <div className="flex h-9 items-center gap-2">
          <Switch id="f-unlabeled" checked={!!search.unlabeled} onCheckedChange={c => setSearch({ unlabeled: c || undefined })} />
          <Label htmlFor="f-unlabeled">Unlabelled only</Label>
        </div>
        {draft ? (
          <Button variant={selecting ? 'secondary' : 'outline'} size="sm" className="h-9" aria-pressed={selecting} onClick={() => (setSelecting(v => !v), setSel(new Set()))}>
            <ListChecks /> {selecting ? 'Done selecting' : 'Select'}
          </Button>
        ) : (
          <Tooltip content={SEALED_HINT}>
            <span tabIndex={0} aria-label={`Select (unavailable): ${SEALED_HINT}`}>
              <Button variant="outline" size="sm" className="pointer-events-none h-9" disabled>
                <ListChecks /> Select
              </Button>
            </span>
          </Tooltip>
        )}
        <span className="ml-auto text-sm text-muted-foreground" aria-live="polite">
          {samples.isFetching ? 'Loading…' : `${items.length}${samples.hasNextPage ? '+' : ''} shown`}
        </span>
      </div>

      {samples.isLoading ? (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
          {Array.from({ length: 12 }, (_, i) => (
            <Skeleton key={i} className="aspect-square rounded-lg" />
          ))}
        </div>
      ) : items.length === 0 ? (
        <Card className="p-10 text-center text-sm text-muted-foreground">No samples match these filters.</Card>
      ) : text ? (
        <ul className="grid gap-2">
          {items.map(s => (
            <li key={s.id}>
              <button type="button" onClick={onCard(s, items.indexOf(s))} aria-pressed={selecting ? sel.has(s.id) : undefined} className={cn('flex w-full cursor-pointer items-start gap-3 rounded-lg border bg-card p-3 text-left hover:bg-accent/50', sel.has(s.id) && 'border-primary ring-2 ring-primary/25')}>
                {selecting && (sel.has(s.id) ? <CheckSquare className="size-4 shrink-0 text-primary" /> : <Square className="size-4 shrink-0 text-muted-foreground" />)}
                <span className="flex-1 text-sm">{s.text}</span>
                <SampleChips sample={s} />
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <ul className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
          {items.map((s, i) => (
            <li key={s.id}>
              <button type="button" onClick={onCard(s, i)} aria-pressed={selecting ? sel.has(s.id) : undefined} className={cn('group grid w-full cursor-pointer gap-1.5 rounded-lg border bg-card p-1.5 text-left hover:border-primary/50', sel.has(s.id) && 'border-primary ring-2 ring-primary/25')} aria-label={`${selecting ? 'Select' : 'Open'} ${s.name ?? 'sample'}`}>
                <div className="relative aspect-square overflow-hidden rounded-md bg-muted">
                  {selecting && (sel.has(s.id) ? <CheckSquare className="absolute right-1 top-1 z-10 size-5 rounded bg-primary text-primary-foreground" /> : <Square className="absolute right-1 top-1 z-10 size-5 rounded bg-black/40 text-white" />)}
                  {s.thumb_url && <img src={s.thumb_url} alt="" loading="lazy" className="size-full object-cover" />}
                  {project.task_type === 'object_detection' && s.width && s.height && <BoxOverlay sample={s} fit="cover" />}
                  {s.split && <span className="absolute left-1 top-1 rounded bg-black/55 px-1.5 text-[10px] font-medium uppercase text-white">{s.split}</span>}
                </div>
                <SampleChips sample={s} compact />
              </button>
            </li>
          ))}
        </ul>
      )}
      {samples.hasNextPage && (
        <Button variant="outline" className="mx-auto" onClick={() => samples.fetchNextPage()} disabled={samples.isFetchingNextPage}>
          {samples.isFetchingNextPage ? 'Loading…' : 'Load more'}
        </Button>
      )}

      {selecting && (
        <Card className="sticky bottom-4 z-20 flex flex-wrap items-center gap-2 p-3 shadow-lg" role="toolbar" aria-label="Selection">
          <span className="text-sm font-medium tabular-nums" aria-live="polite">{sel.size} selected</span>
          <Button variant="ghost" size="sm" onClick={() => setSel(new Set(items.map(s => s.id)))} disabled={sel.size >= items.length && items.every(s => sel.has(s.id))}>
            <CheckSquare /> Select all shown ({items.length})
          </Button>
          {(samples.hasNextPage || total > items.length) && (
            <Button variant="ghost" size="sm" disabled={allBusy || (total > 0 && sel.size === total)} onClick={selectAll}>
              {allBusy ? <Loader2 className="animate-spin" /> : <CheckSquare />} Select all {total || ''} {filtered ? 'matching' : 'in draft'}
            </Button>
          )}
          {sel.size > 0 && <Button variant="ghost" size="sm" onClick={() => setSel(new Set())}><X /> Clear</Button>}
          <span className="hidden text-xs text-muted-foreground sm:inline">Shift-click selects a range</span>
          <Button variant="destructive" size="sm" className="ml-auto" disabled={!sel.size} onClick={() => setConfirm([...sel])}>
            <Trash2 /> Remove {sel.size || ''} from draft
          </Button>
        </Card>
      )}
      <RemoveSamplesDialog vid={version.id} ids={confirm} unit={unit} onClose={() => setConfirm(null)} onDone={() => setSel(new Set())} />

      <Dialog open={!!open} onOpenChange={o => !o && setOpen(null)}>
        <DialogContent className="max-w-3xl">
          {open && (
            <>
              <DialogHeader>
                <DialogTitle>{open.name ?? 'Sample'}</DialogTitle>
                <DialogDescription>
                  {open.width && open.height ? `${open.width}×${open.height} · ` : ''}
                  {open.split ?? 'unsplit'} · {open.labels.length ? `${open.labels.length} label(s)` : 'no labels'}
                </DialogDescription>
              </DialogHeader>
              {open.text ? (
                <p className="rounded-md bg-muted p-4 text-sm">{open.text}</p>
              ) : (
                open.url && (
                  <div className="relative mx-auto w-fit max-w-full">
                    <img src={open.url} alt={open.name ?? 'sample'} className="max-h-[60vh] max-w-full rounded-md" />
                    {open.width && open.height && <BoxOverlay sample={open} fit="contain" showLabels />}
                  </div>
                )
              )}
              <SampleChips sample={open} />
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}

function SampleChips({ sample, compact }: { sample: Sample; compact?: boolean }) {
  const counts = new Map<string, number>();
  for (const l of sample.labels) counts.set(l.class_name, (counts.get(l.class_name) ?? 0) + 1);
  return (
    <div className="flex flex-wrap gap-1">
      {[...counts].map(([c, n]) => (
        <Badge key={c} variant="secondary" className={compact ? 'px-1.5 text-[10px]' : ''}>
          {c}
          {n > 1 ? ` ×${n}` : ''}
        </Badge>
      ))}
      {sample.no_objects && <Badge variant="outline">no objects</Badge>}
      {!sample.labels.length && !sample.no_objects && <Badge variant="outline" className={compact ? 'px-1.5 text-[10px]' : ''}>unlabelled</Badge>}
      {sample.flags.filter(f => f.startsWith('VAL-') || f === 'corrupt').map(f => (
        <Badge key={f} variant="warning" className={compact ? 'px-1.5 text-[10px]' : ''}>
          {f}
        </Badge>
      ))}
    </div>
  );
}

function RowsTable({ vid }: { vid: string }) {
  const rows = useRows(vid);
  const page = rows.data?.pages;
  const columns = page?.[0]?.columns ?? [];
  const items = page?.flatMap(p => p.items) ?? [];
  if (rows.isLoading) return <Skeleton className="h-64 rounded-xl" />;
  return (
    <div className="grid gap-3">
      <div className="overflow-x-auto rounded-lg border">
        <table className="w-full min-w-max text-sm">
          <thead className="bg-muted/60 text-left text-xs text-muted-foreground">
            <tr>
              <th className="px-3 py-2 font-medium">#</th>
              {columns.map(c => (
                <th key={c.name} className={cn('px-3 py-2 font-medium', c.role === 'target' && 'text-primary')}>
                  {c.name}
                  <span className="ml-1 font-normal opacity-70">{c.role === 'target' ? '(target)' : c.role !== 'feature' ? `(${c.role})` : ''}</span>
                </th>
              ))}
              <th className="px-3 py-2 font-medium">Split</th>
            </tr>
          </thead>
          <tbody>
            {items.map(r => (
              <tr key={r.row} className="border-t">
                <td className="px-3 py-1.5 tabular-nums text-muted-foreground">{r.row}</td>
                {columns.map(c => (
                  <td key={c.name} className={cn('px-3 py-1.5', c.type === 'numeric' && 'text-right tabular-nums', c.role === 'target' && 'font-medium')}>
                    {String(r.values[c.name] ?? '')}
                  </td>
                ))}
                <td className="px-3 py-1.5 text-xs uppercase text-muted-foreground">{r.split ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.hasNextPage && (
        <Button variant="outline" className="mx-auto" onClick={() => rows.fetchNextPage()}>
          Load more rows
        </Button>
      )}
    </div>
  );
}

const SEV: Record<Severity, { icon: typeof AlertCircle; cls: string; label: string }> = {
  error: { icon: AlertCircle, cls: 'text-destructive', label: 'Errors: block training' },
  warning: { icon: AlertTriangle, cls: 'text-warning', label: 'Warnings: acknowledge or fix' },
  info: { icon: Info, cls: 'text-info', label: 'Info' },
};

/** VAL report (PRD §6.5): Errors block training, Warnings must be acknowledged, Info is advice. */
function ValidationPanel({ project, version, onVersion, onFlag }: { project: Project; version: DatasetVersion; onVersion: (id: string) => void; onFlag: (code: string) => void }) {
  const running = version.validation_status === 'running' || version.ingest_status !== 'ready';
  const report = useValidation(version.ingest_status === 'ready' ? version.id : undefined, running);
  // One source of truth with the version picker: when the version's status changes, re-read the report.
  const { refetch } = report;
  useEffect(() => {
    if (version.ingest_status === 'ready') void refetch();
  }, [version.validation_status, version.ingest_status, refetch]);
  const fixes = useApplyFixes();
  const ack = useAcknowledge();
  const seal = useSeal();
  const r = report.data;

  const applyFix = (ids: string[]) =>
    fixes.mutate(
      { vid: version.id, fixIds: ids },
      {
        onSuccess: res => {
          toast.success('Fix applied: a new version is sealed and re-validated');
          onVersion(res.dataset_version_id);
        },
        onError: e => toast.error(e.message),
      },
    );

  if (!version.sealed_at)
    return (
      <Card className="flex flex-col items-start gap-3 p-6">
        <p className="font-medium">This version is a draft</p>
        <p className="text-sm text-muted-foreground">Seal it to freeze the samples, split train/val/test and run the validation checks.</p>
        <Button onClick={() => seal.mutate(version.id, { onError: e => toast.error(e.message) })} disabled={seal.isPending}>
          <ShieldCheck /> {seal.isPending ? 'Sealing…' : 'Seal and validate'}
        </Button>
      </Card>
    );
  if (running || report.isLoading || !r || r.status === 'running' || r.status === 'pending')
    return (
      <Card className="flex items-center gap-3 p-6 text-sm" role="status">
        <Loader2 className="size-4 animate-spin text-primary" /> Running validation checks (corrupt files, duplicates, leakage, class balance)…
      </Card>
    );

  const autoFixable = r.items.filter(i => i.fix.auto_fix_id).map(i => i.fix.auto_fix_id!) as string[];
  const unacked = [...new Set(r.items.filter(i => i.severity === 'warning' && !i.acknowledged && i.acknowledgeable !== false).map(i => i.code))];
  return (
    <div className="grid gap-4">
      <Card className="flex flex-wrap items-center gap-4 p-5">
        {r.status === 'passed' ? <CheckCircle2 className="size-8 text-success" /> : r.status === 'failed' ? <AlertCircle className="size-8 text-destructive" /> : <AlertTriangle className="size-8 text-warning" />}
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2 font-semibold">
            {r.status === 'passed' ? 'Ready to train' : r.status === 'failed' ? 'Fix the errors before training' : 'Review the warnings'}
            <DemoBadge group="validation" />
          </div>
          <p className="text-sm text-muted-foreground">
            {r.summary.errors} error{r.summary.errors === 1 ? '' : 's'} · {r.summary.warnings} warning{r.summary.warnings === 1 ? '' : 's'} · {r.summary.info} info
          </p>
        </div>
        {autoFixable.length > 0 && (
          <Button variant="outline" onClick={() => applyFix([...new Set(autoFixable)])} disabled={fixes.isPending}>
            <Sparkles /> Apply {autoFixable.length} safe fix{autoFixable.length === 1 ? '' : 'es'}
          </Button>
        )}
        {unacked.length > 0 && (
          <Button onClick={() => ack.mutate({ vid: version.id, codes: unacked })} disabled={ack.isPending}>
            Acknowledge all warnings
          </Button>
        )}
      </Card>
      {(['error', 'warning', 'info'] as const).map(sev => {
        const list = r.items.filter(i => i.severity === sev);
        if (!list.length) return null;
        const S = SEV[sev];
        return (
          <section key={sev} className="grid gap-2" aria-label={S.label}>
            <h3 className={cn('flex items-center gap-2 text-sm font-semibold', S.cls)}>
              <S.icon className="size-4" /> {S.label}
            </h3>
            {list.map((it, i) => (
              <ValidationRow
                key={`${it.code}-${i}`}
                item={it}
                project={project}
                onFix={() => applyFix([it.fix.auto_fix_id!])}
                onAck={() => ack.mutate({ vid: version.id, codes: [it.code] })}
                onShow={() => onFlag(it.code)}
                busy={fixes.isPending || ack.isPending}
              />
            ))}
          </section>
        );
      })}
      {r.items.length === 0 && <Card className="p-6 text-sm text-muted-foreground">No issues found.</Card>}
    </div>
  );
}

function ValidationRow({ item, onFix, onAck, onShow, busy }: { item: ValidationItem; project: Project; onFix: () => void; onAck: () => void; onShow: () => void; busy: boolean }) {
  return (
    <Card className={cn('grid gap-2 p-4', item.acknowledged && 'opacity-70')}>
      <div className="flex flex-wrap items-start gap-2">
        <Badge variant="outline">{item.code}</Badge>
        <span className="font-medium">{item.title}</span>
        {item.count > 0 && <Badge variant="secondary">{item.count.toLocaleString()}</Badge>}
        {item.acknowledged && <Badge variant="secondary">Acknowledged</Badge>}
      </div>
      <p className="text-sm text-muted-foreground">{item.why}</p>
      <p className="flex items-start gap-2 text-sm">
        <Wrench className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
        {item.fix.text}
      </p>
      <div className="flex flex-wrap gap-2">
        {item.examples.length > 0 && (
          <Button size="sm" variant="ghost" onClick={onShow}>
            Show {item.examples.length} example{item.examples.length === 1 ? '' : 's'}
          </Button>
        )}
        {item.fix.auto_fix_id && (
          <Button size="sm" variant="outline" onClick={onFix} disabled={busy}>
            <Sparkles /> Auto-fix
          </Button>
        )}
        {item.severity !== 'info' && !item.acknowledged && item.acknowledgeable !== false && (
          <Button size="sm" variant="ghost" onClick={onAck} disabled={busy}>
            Acknowledge
          </Button>
        )}
      </div>
    </Card>
  );
}

/** LBL-3: label distribution (images per class; boxes per class for detection) and split sizes. */
function Stats({ project, version }: { project: Project; version: DatasetVersion }) {
  const detection = project.task_type === 'object_detection';
  const data = Object.entries(version.label_stats)
    .map(([name, st]) => ({ name, images: st.images, ...(detection ? { boxes: st.boxes ?? 0 } : {}) }))
    .sort((a, b) => b.images - a.images);
  if (!data.length) return <Card className="p-6 text-sm text-muted-foreground">No labels yet.</Card>;
  return (
    <Card className="p-4">
      <h3 className="mb-3 text-sm font-semibold">{detection ? 'Images and boxes per class' : project.task_type.startsWith('tabular') ? 'Rows per target value' : 'Samples per class'}</h3>
      <div className="h-72 w-full">
        <ResponsiveContainer>
          <BarChart data={data} margin={{ left: -10, right: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="name" tick={{ fill: 'var(--muted-foreground)', fontSize: 12 }} />
            <YAxis allowDecimals={false} tick={{ fill: 'var(--muted-foreground)', fontSize: 12 }} />
            <RTooltip contentStyle={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 8, color: 'var(--card-foreground)' }} />
            <Legend />
            <Bar dataKey="images" name={detection ? 'Images' : 'Samples'} fill="var(--primary)" radius={[4, 4, 0, 0]} />
            {detection && <Bar dataKey="boxes" name="Boxes" fill="var(--info)" radius={[4, 4, 0, 0]} />}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

/** VersionOut.unknown_classes: labels found in the data (new folder names, CSV values) that the label set lacks. */
export function UnknownClasses({ project, version }: { project: Project; version: DatasetVersion }) {
  const change = useLabelSetChange();
  const unknown = version.unknown_classes ?? [];
  if (!unknown.length || !project.label_set_id) return null;
  return (
    <Card className="flex flex-wrap items-center gap-3 border-warning/40 bg-warning/8 p-4 text-sm" role="status">
      <AlertTriangle className="size-4 text-warning" />
      <span className="flex-1">
        The data has {unknown.length === 1 ? 'a class' : 'classes'} the project doesn't know yet: <b>{unknown.join(', ')}</b>. Add {unknown.length === 1 ? 'it' : 'them'}?
      </span>
      <Button
        size="sm"
        disabled={change.isPending}
        onClick={() =>
          change.mutate(
            { id: project.label_set_id!, op: 'classes', body: { add: unknown } },
            { onSuccess: () => toast.success(`Added ${unknown.join(', ')}`), onError: e => toast.error(e.message) },
          )
        }
      >
        Add {unknown.length === 1 ? 'class' : `${unknown.length} classes`}
      </Button>
    </Card>
  );
}
