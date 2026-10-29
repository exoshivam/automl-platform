import { Link } from '@tanstack/react-router';
import { CheckCircle2, Database, ExternalLink, Library, Loader2, XCircle } from 'lucide-react';
import { useState } from 'react';
import { toast } from 'sonner';
import { useVersions } from '@/api/data-hooks';
import { useImportStarter, useStarters, type Starter, type StarterImport } from '@/api/starters';
import type { Project } from '@/api/types';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { DemoBadge, Progress, Skeleton } from '@/components/ui/misc';
import { ApiError } from '@/lib/api';
import { formatBytes } from '@/lib/files';
import { metaOf } from '@/features/projects/meta';
import { cn } from '@/lib/utils';

/** PRD DAT-13: pick a curated public dataset; it is imported into the project's draft like any upload. */
export function StarterLibrary({ project }: { project: Project }) {
  const starters = useStarters(project.task_type);
  const run = useImportStarter();
  const [started, setStarted] = useState<{ starter: Starter; job: StarterImport } | null>(null);

  if (starters.isLoading) return <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">{[0, 1, 2].map(i => <Skeleton key={i} className="h-56 rounded-xl" />)}</div>;
  if (starters.error) return <p className="text-sm text-destructive">{starters.error.message}</p>;

  const use = (s: Starter) =>
    run.mutate(
      { projectId: project.id, starterId: s.id },
      {
        onSuccess: job => setStarted({ starter: s, job }),
        onError: e =>
          toast.error(
            importError(e),
          ),
      },
    );

  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
        <Library className="size-4" /> Curated public datasets to learn with or to start from. They go into the draft, and you can add your own data on top.
        <DemoBadge group="starters" />
      </div>
      {started && <ImportProgress project={project} starter={started.starter} job={started.job} onDone={() => setStarted(null)} />}
      {!starters.data?.length ? (
        <Card className="p-8 text-center text-sm text-muted-foreground">No starter datasets for this task type yet.</Card>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {starters.data.map(s => (
            <StarterCard key={s.id} s={s} projectClasses={project.current_spec.labels ?? []} busy={run.isPending && run.variables?.starterId === s.id} disabled={!!started || run.isPending} onUse={() => use(s)} />
          ))}
        </div>
      )}
    </div>
  );
}

const IMPORT_ERRORS: Record<string, string> = {
  import_running: 'An upload or import is still being processed for this project. Try again when it finishes.',
  already_imported: 'This starter dataset is already in this project.',
  task_mismatch: 'That dataset is for a different task type.',
  project_deleting: 'This project is being deleted.',
};
export const importError = (e: Error) => (e instanceof ApiError && IMPORT_ERRORS[e.problem.code ?? '']) || e.message;

export function StarterCard({ s, projectClasses, busy, disabled, onUse, cta = 'Use this dataset' }: { s: Starter; projectClasses?: string[]; busy?: boolean; disabled?: boolean; onUse: () => void; cta?: string }) {
  const m = metaOf(s.task_type);
  const added = projectClasses?.length ? s.classes.filter(c => !projectClasses.includes(c)) : [];
  return (
    <Card className="flex flex-col overflow-hidden">
      {s.preview_urls.length ? (
        <div className="grid h-32 grid-cols-3 grid-rows-2 gap-0.5 bg-muted">
          {s.preview_urls.slice(0, 6).map(u => <img key={u} src={u} alt="" className="size-full object-cover" loading="lazy" />)}
        </div>
      ) : (
        <div className={cn('grid h-20 place-items-center', m.tint)}>
          <m.icon className="size-7" />
        </div>
      )}
      <div className="flex flex-1 flex-col gap-2 p-4">
        <div className="font-semibold">{s.name}</div>
        <p className="text-sm text-muted-foreground">{s.description}</p>
        <div className="flex flex-wrap gap-1">
          {s.classes.slice(0, 6).map(c => <Badge key={c} variant="secondary">{c}</Badge>)}
          {s.classes.length > 6 && <Badge variant="outline">+{s.classes.length - 6}</Badge>}
          {s.target && !s.classes.length && <Badge variant="secondary">predicts {s.target}</Badge>}
        </div>
        <div className="flex flex-wrap gap-x-3 text-xs text-muted-foreground tabular-nums">
          <span>{s.sample_count.toLocaleString()} {s.task_type.startsWith('tabular') ? 'rows' : 'samples'}</span>
          <span>{formatBytes(s.size_bytes)}</span>
        </div>
        {added.length > 0 && <p className="text-xs text-info">Adds classes: {added.join(', ')}</p>}
        <p className="mt-auto pt-1 text-[11px] leading-snug text-muted-foreground">
          Source: {s.source_url ? <a href={s.source_url} target="_blank" rel="noreferrer" className="underline-offset-2 hover:underline">{s.source} <ExternalLink className="inline size-3" /></a> : s.source} · Licence: {s.licence}
        </p>
        <Button size="sm" onClick={onUse} disabled={disabled || busy}>
          {busy ? <Loader2 className="animate-spin" /> : <Database />} {cta}
        </Button>
      </div>
    </Card>
  );
}

/** Follows the draft the import appends to: ingest status and the growing sample count. */
export function ImportProgress({ project, starter, job, onDone }: { project: Project; starter: Starter; job: StarterImport; onDone?: () => void }) {
  const versions = useVersions(project.dataset_id);
  const v = versions.data?.items.find(x => x.id === job.dataset_version_id);
  const progress = (v as { progress?: { done: number; total: number } } | undefined)?.progress;
  const done = v?.ingest_status === 'ready';
  const failed = v?.ingest_status === 'failed';
  const pct = progress ? (progress.done / Math.max(progress.total, 1)) * 100 : done ? 100 : undefined;
  return (
    <Card className={cn('grid gap-3 p-4', done && 'border-success/40', failed && 'border-destructive/40')} role="status" aria-live="polite">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        {done ? <CheckCircle2 className="size-4 text-success" /> : failed ? <XCircle className="size-4 text-destructive" /> : <Loader2 className="size-4 animate-spin" />}
        <span className="font-medium">
          {done ? `“${starter.name}” is in your draft` : failed ? `Importing “${starter.name}” failed` : `Importing “${starter.name}”…`}
        </span>
        <span className="text-muted-foreground tabular-nums">
          {progress ? `${progress.done.toLocaleString()} / ${progress.total.toLocaleString()}` : v ? `${v.sample_count.toLocaleString()} samples in draft` : `${job.sample_count.toLocaleString()} samples`}
        </span>
      </div>
      {!failed && <Progress value={pct} aria-label="Import progress" className={cn(pct === undefined && 'animate-pulse')} />}
      {failed && v?.ingest_errors?.[0] && <p className="text-xs text-destructive">{v.ingest_errors[0].message ?? v.ingest_errors[0].code}</p>}
      {(done || failed) && (
        <div className="flex gap-2">
          {done && (
            <Button size="sm" asChild>
              <Link to="/projects/$projectId/dataset" params={{ projectId: project.id }}>Review and seal</Link>
            </Button>
          )}
          {onDone && <Button size="sm" variant="ghost" onClick={onDone}>Close</Button>}
        </div>
      )}
    </Card>
  );
}
