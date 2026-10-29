import { Link } from '@tanstack/react-router';
import { Eraser, Loader2, Trash2 } from 'lucide-react';
import { useState } from 'react';
import { toast } from 'sonner';
import { useDeleteVersion, usePrune, useStorage, type PruneResult, type VersionRow } from '@/api/storage';
import type { Project } from '@/api/types';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Skeleton, Tooltip } from '@/components/ui/misc';
import { ApiError } from '@/lib/api';
import { formatBytes } from '@/lib/files';
import { timeAgo } from '@/lib/utils';

const blockedReason = (v: VersionRow) =>
  v.is_draft ? 'This is the open draft' : v.is_current ? 'This is the current version' : v.has_models ? 'Used by a trained model. Delete the model first (Evaluate tab).' : null;

/** Version history with per-version delete and "Clean up old versions" (keep latest N + current/draft/used by models). */
export function VersionsPanel({ project }: { project: Project }) {
  const storage = useStorage(project.id);
  const del = useDeleteVersion(project.id);
  const [confirm, setConfirm] = useState<VersionRow | null>(null);
  const [cleanup, setCleanup] = useState(false);

  if (storage.isLoading) return <Skeleton className="h-64 rounded-xl" />;
  if (!storage.data) return <p className="text-sm text-destructive">{storage.error?.message}</p>;
  const rows = storage.data.versions;

  return (
    <Card>
      <CardHeader className="flex-row flex-wrap items-start justify-between gap-3">
        <div>
          <CardTitle>Versions</CardTitle>
          <CardDescription>Data in this project: {formatBytes(storage.data.totals.data_bytes + storage.data.totals.manifests_bytes)}. Versions share most files, so deleting one frees only what it alone holds.</CardDescription>
        </div>
        <Button variant="outline" size="sm" onClick={() => setCleanup(true)} disabled={rows.filter(r => !blockedReason(r)).length === 0}>
          <Eraser /> Clean up old versions
        </Button>
      </CardHeader>
      <CardContent>
        <ul className="divide-y">
          {rows.map(v => {
            const why = blockedReason(v);
            return (
              <li key={v.id} className="flex flex-wrap items-center gap-3 py-2.5 text-sm">
                <span className="w-20 font-medium">{v.is_draft ? 'Draft' : `v${v.version_no}`}</span>
                <span className="flex-1 text-muted-foreground">
                  {v.sample_count.toLocaleString()} samples · {timeAgo(v.created_at)}
                </span>
                {v.is_current && <Badge variant="info">current</Badge>}
                {v.has_models && <Badge variant="secondary">used by model</Badge>}
                {why ? (
                  <Tooltip content={why}>
                    <span tabIndex={0}>
                      <Button variant="ghost" size="sm" disabled aria-label={`Can't delete: ${why}`}><Trash2 /> Delete</Button>
                    </span>
                  </Tooltip>
                ) : (
                  <Button variant="ghost" size="sm" onClick={() => setConfirm(v)}>
                    <Trash2 /> Delete · {v.reclaimable_bytes ? `frees ${formatBytes(v.reclaimable_bytes)}` : 'files shared with other versions'}
                  </Button>
                )}
              </li>
            );
          })}
        </ul>
      </CardContent>

      <Dialog open={!!confirm} onOpenChange={o => !o && setConfirm(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete version v{confirm?.version_no}?</DialogTitle>
            <DialogDescription>
              This permanently deletes its samples, labels and validation report, freeing {formatBytes(confirm?.reclaimable_bytes ?? 0)}. It can't be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setConfirm(null)}>Cancel</Button>
            <Button
              variant="destructive"
              disabled={del.isPending}
              onClick={() =>
                del.mutate(confirm!.id, {
                  onSuccess: () => (toast.success(`Version v${confirm!.version_no} deleted`), setConfirm(null)),
                  onError: e => toast.error(e instanceof ApiError && e.problem.code === 'version_has_models' ? 'A trained model uses this version. Delete the model first.' : e.message),
                })
              }
            >
              {del.isPending ? <Loader2 className="animate-spin" /> : <Trash2 />} Delete permanently
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {cleanup && project.dataset_id && <CleanupDialog project={project} datasetId={project.dataset_id} onClose={() => setCleanup(false)} />}
    </Card>
  );
}

const REASON: Record<string, string> = { current: 'current', draft: 'draft', used_by_models: 'used by a model', keep_latest: 'kept (latest)' };

function CleanupDialog({ project, datasetId, onClose }: { project: Project; datasetId: string; onClose: () => void }) {
  const prune = usePrune(project.id, datasetId);
  const [keep, setKeep] = useState(2);
  const [preview, setPreview] = useState<PruneResult | null>(null);
  const run = (dry_run: boolean) =>
    prune.mutate(
      { keep_latest: keep, dry_run },
      {
        onSuccess: r => (dry_run ? setPreview(r) : (toast.success(`Deleted ${r.deleted.length} versions, freed ${formatBytes(r.reclaimable_bytes)}`), onClose())),
        onError: e => toast.error(e.message),
      },
    );
  return (
    <Dialog open onOpenChange={o => !o && onClose()}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Clean up old versions</DialogTitle>
          <DialogDescription>The current version, the open draft and versions used by trained models are always kept.</DialogDescription>
        </DialogHeader>
        <form className="flex items-end gap-3" onSubmit={e => (e.preventDefault(), run(true))}>
          <label className="grid gap-1 text-sm">
            <span className="font-medium">Also keep the latest</span>
            <Input type="number" min={0} max={50} value={keep} onChange={e => (setKeep(Math.max(0, Number(e.target.value) || 0)), setPreview(null))} className="w-24" />
          </label>
          <span className="pb-2 text-sm text-muted-foreground">sealed versions</span>
          <Button type="submit" variant="outline" className="ml-auto" disabled={prune.isPending}>Preview</Button>
        </form>
        {preview && (
          <div className="grid gap-2 text-sm">
            {preview.deleted.length ? (
              <>
                <p className="font-medium">Will delete {preview.deleted.length} {preview.deleted.length === 1 ? 'version' : 'versions'}, freeing {formatBytes(preview.reclaimable_bytes)}:</p>
                <p className="text-muted-foreground">{preview.deleted.map(v => `v${v.version_no}`).join(', ')}</p>
              </>
            ) : (
              <p className="text-muted-foreground">Nothing to delete with these settings.</p>
            )}
            <p className="text-xs text-muted-foreground">Kept: {preview.kept.map(v => `${v.is_draft ? 'draft' : `v${v.version_no}`} (${REASON[v.reason ?? ''] ?? v.reason})`).join(', ')}</p>
          </div>
        )}
        <DialogFooter>
          <Button variant="ghost" onClick={onClose}>Cancel</Button>
          <Button variant="destructive" disabled={!preview?.deleted.length || prune.isPending} onClick={() => run(false)}>
            {prune.isPending ? <Loader2 className="animate-spin" /> : <Trash2 />} Delete {preview?.deleted.length ?? 0} permanently
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export function StorageSummary({ projectId }: { projectId: string }) {
  const s = useStorage(projectId).data?.totals;
  if (!s) return null;
  const parts = [
    ['Data', s.data_bytes + s.manifests_bytes],
    ['Models', s.models_bytes],
    ['Exports', s.exports_bytes],
  ] as const;
  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between">
        <div>
          <CardTitle className="text-base">Storage · {formatBytes(s.total_bytes)}</CardTitle>
          <CardDescription>Free space by deleting old dataset versions or models you no longer need.</CardDescription>
        </div>
        <Link to="/projects/$projectId/dataset" params={{ projectId }} search={{ tab: 'versions' }} className="text-sm font-medium text-primary hover:underline">Manage →</Link>
      </CardHeader>
      <CardContent className="grid grid-cols-3 gap-3">
        {parts.map(([k, v]) => (
          <div key={k} className="rounded-lg bg-muted/50 p-3">
            <div className="text-xs text-muted-foreground">{k}</div>
            <div className="font-semibold tabular-nums">{formatBytes(v)}</div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
