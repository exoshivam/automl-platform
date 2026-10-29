import { useState } from 'react';
import { Link } from '@tanstack/react-router';
import { Archive, ArchiveRestore, FolderPlus, MoreHorizontal, Plus, Search, Trash2 } from 'lucide-react';
import { METRIC_NAMES, fmtMetric } from '@/api/ml';
import { PurgeProjectDialog } from './purge-dialog';
import { toast } from 'sonner';
import { useArchiveProject, useCurrentOrg, useProjects } from '@/api/hooks';
import type { Project, ProjectStatus } from '@/api/types';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { Input } from '@/components/ui/input';
import { Skeleton } from '@/components/ui/misc';
import { cn, timeAgo } from '@/lib/utils';
import { METRIC_LABEL } from './spec-schema';
import { metaOf, STATUSES, StatusBadge, TASK_LABEL } from './meta';

const FILTERS: (ProjectStatus | '')[] = ['', ...STATUSES];

export function ProjectsPage() {
  const { org } = useCurrentOrg();
  const [q, setQ] = useState('');
  const [status, setStatus] = useState<ProjectStatus | ''>('');
  const [archived, setArchived] = useState(false);
  const projects = useProjects(org?.id, { search: q, status, archived });
  const items = projects.data?.items ?? [];

  return (
    <div className="grid gap-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Projects</h1>
          <p className="text-sm text-muted-foreground">Every project goes from data to a deployed model.</p>
        </div>
        <Button asChild>
          <Link to="/projects/new">
            <Plus /> New project
          </Link>
        </Button>
      </div>

      <div className="flex flex-col gap-3 md:flex-row md:items-center">
        <div className="relative md:w-72">
          <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input value={q} onChange={e => setQ(e.target.value)} placeholder="Search projects" className="pl-9" aria-label="Search projects" type="search" />
        </div>
        <div className="-mx-1 flex gap-1 overflow-x-auto px-1 pb-1 md:pb-0" role="group" aria-label="Filter by status">
          {FILTERS.map(f => (
            <button
              key={f || 'all'}
              type="button"
              aria-pressed={status === f}
              onClick={() => setStatus(f)}
              className={cn(
                'shrink-0 cursor-pointer rounded-full border px-3 py-1 text-xs font-medium transition-colors',
                status === f ? 'border-primary bg-primary text-primary-foreground' : 'bg-card text-muted-foreground hover:bg-accent hover:text-accent-foreground',
              )}
            >
              {f || 'All'}
            </button>
          ))}
        </div>
        <Button variant="ghost" size="sm" className="md:ml-auto" aria-pressed={archived} onClick={() => setArchived(a => !a)}>
          <Archive /> {archived ? 'Showing archived' : 'Archived'}
        </Button>
      </div>

      {projects.isLoading ? (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 6 }, (_, i) => (
            <Skeleton key={i} className="h-40 rounded-xl" />
          ))}
        </div>
      ) : projects.isError ? (
        <Card className="p-8 text-center text-sm text-destructive">Couldn't load projects: {projects.error.message}</Card>
      ) : items.length === 0 ? (
        <EmptyState filtered={!!(q || status || archived)} />
      ) : (
        <ul className={cn('grid gap-4 sm:grid-cols-2 xl:grid-cols-3', projects.isFetching && 'opacity-70 transition-opacity')} aria-busy={projects.isFetching}>
          {items.map(p => (
            <li key={p.id}>
              <ProjectCard project={p} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function ProjectCard({ project: p }: { project: Project }) {
  const cat = metaOf(p.task_type);
  const archive = useArchiveProject();
  const [purge, setPurge] = useState(false);
  const Icon = cat.icon;
  return (
    <Card className="group relative flex h-full flex-col p-5 transition-shadow hover:shadow-md focus-within:ring-2 focus-within:ring-ring">
      <div className="flex items-start gap-3">
        <span className={cn('grid size-10 shrink-0 place-items-center rounded-lg', cat.tint)}>
          <Icon className="size-5" />
        </span>
        <div className="min-w-0 flex-1">
          <Link to="/projects/$projectId" params={{ projectId: p.id }} className="block truncate font-semibold outline-none after:absolute after:inset-0 hover:underline">
            {p.name}
          </Link>
          <p className="truncate text-xs text-muted-foreground">{TASK_LABEL[p.task_type]}</p>
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" className="relative z-10 -mr-2 -mt-1" aria-label={`Actions for ${p.name}`}>
              <MoreHorizontal />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem
              onSelect={() =>
                archive.mutate(
                  { id: p.id, archived: !p.archived_at },
                  { onSuccess: () => toast.success(p.archived_at ? `Restored "${p.name}"` : `Archived "${p.name}"`) },
                )
              }
            >
              {p.archived_at ? <ArchiveRestore /> : <Archive />}
              {p.archived_at ? 'Restore' : 'Archive'}
            </DropdownMenuItem>
            <DropdownMenuItem className="text-destructive focus:text-destructive" onSelect={() => setPurge(true)}>
              <Trash2 /> Delete project permanently
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
        {purge && <PurgeProjectDialog project={p} onClose={() => setPurge(false)} />}
      </div>
      <div className="mt-4 flex flex-wrap items-center gap-2">
        <StatusBadge status={p.status} />
        {p.archived_at && <span className="text-xs text-muted-foreground">Archived</span>}
      </div>
      <dl className="mt-auto grid grid-cols-3 gap-2 pt-5 text-xs">
        <div>
          <dt className="text-muted-foreground">Samples</dt>
          <dd className="font-medium tabular-nums">{p.sample_count.toLocaleString()}</dd>
        </div>
        <div>
          <dt className="truncate text-muted-foreground" title={p.best_metric_split ? `Best model, ${p.best_metric_split} set` : undefined}>
            {p.best_metric != null && p.best_metric_name ? `${METRIC_NAMES[p.best_metric_name] ?? METRIC_LABEL[p.best_metric_name] ?? p.best_metric_name}${p.best_metric_split ? ` (${p.best_metric_split})` : ''}` : 'Best metric'}
          </dt>
          <dd className="font-medium tabular-nums">{p.best_metric != null ? fmtMetric(p.best_metric_name ?? '', p.best_metric) : '—'}</dd>
        </div>
        <div>
          <dt className="text-muted-foreground">Updated</dt>
          <dd className="font-medium">{timeAgo(p.updated_at)}</dd>
        </div>
      </dl>
    </Card>
  );
}

function EmptyState({ filtered }: { filtered: boolean }) {
  return (
    <Card className="flex flex-col items-center gap-3 border-dashed p-12 text-center">
      <span className="grid size-12 place-items-center rounded-full bg-primary/10 text-primary">
        <FolderPlus className="size-6" />
      </span>
      <h2 className="font-semibold">{filtered ? 'No projects match' : 'No projects yet'}</h2>
      <p className="max-w-sm text-sm text-muted-foreground">
        {filtered ? 'Try a different search or filter.' : 'Pick a category, add your data, and get your first model in under 30 minutes.'}
      </p>
      {!filtered && (
        <Button asChild>
          <Link to="/projects/new">
            <Plus /> New project
          </Link>
        </Button>
      )}
    </Card>
  );
}
