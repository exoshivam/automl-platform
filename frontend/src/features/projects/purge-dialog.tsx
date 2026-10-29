import { useNavigate } from '@tanstack/react-router';
import { Loader2, Trash2 } from 'lucide-react';
import { useState } from 'react';
import { toast } from 'sonner';
import { usePurgeProject, waitForPurge } from '@/api/hooks';
import { TERMINAL, allVersions, useJobs, useModels } from '@/api/ml';
import { useStorage } from '@/api/storage';
import type { Project } from '@/api/types';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Skeleton } from '@/components/ui/misc';
import { ApiError } from '@/lib/api';
import { formatBytes } from '@/lib/files';

/** Typed-name confirmation for the permanent, async project purge. Mount only while open (it fetches usage). */
export function PurgeProjectDialog({ project, onClose }: { project: Project; onClose: () => void }) {
  const storage = useStorage(project.id);
  const models = useModels(project.id);
  const jobs = useJobs(project.id);
  const purge = usePurgeProject();
  const navigate = useNavigate();
  const [name, setName] = useState('');
  const [error, setError] = useState<string>();

  const versions = storage.data?.versions.filter(v => !v.is_draft).length ?? 0;
  const modelCount = allVersions(models.data).length;
  const running = jobs.data?.some(j => !TERMINAL.includes(j.status as never));
  const matches = name === project.name;

  const confirm = () =>
    purge.mutate(
      { id: project.id, confirmName: name },
      {
        onSuccess: () => {
          onClose();
          navigate({ to: '/projects' });
          const t = toast.loading(`Deleting "${project.name}"…`);
          void waitForPurge(project.id).then(done =>
            done ? toast.success('Project deleted', { id: t }) : toast.info('Still deleting in the background; it will disappear when done.', { id: t }),
          );
        },
        onError: e => setError(e instanceof ApiError && e.problem.code === 'confirmation_mismatch' ? 'The name doesn’t match.' : e.message),
      },
    );

  return (
    <Dialog open onOpenChange={o => !o && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="text-destructive">Delete project permanently?</DialogTitle>
          <DialogDescription>This deletes everything in “{project.name}”. It can't be undone. Use Archive if you only want to hide it.</DialogDescription>
        </DialogHeader>
        {storage.isLoading || models.isLoading ? (
          <Skeleton className="h-24" />
        ) : (
          <ul className="grid gap-1 rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm">
            {project.task_type === 'face_recognition' || project.task_type === 'speaker_recognition' ? (
              <li>All enrolled people: their {project.task_type === 'speaker_recognition' ? 'voice clips, voice data' : 'face photos, face data'} and consent records</li>
            ) : (
              <>
                <li>{project.sample_count.toLocaleString()} samples, with their labels, in {versions} sealed {versions === 1 ? 'version' : 'versions'} plus any draft</li>
                <li>{modelCount} trained {modelCount === 1 ? 'model' : 'models'} and all their exports</li>
              </>
            )}
            {storage.data && <li>{formatBytes(storage.data.totals.total_bytes)} of stored files</li>}
            {running && <li className="font-medium text-destructive">The running training job will be cancelled</li>}
          </ul>
        )}
        <label className="grid gap-1.5 text-sm">
          <span>
            Type <span className="font-mono font-semibold">{project.name}</span> to confirm
          </span>
          <Input value={name} onChange={e => (setName(e.target.value), setError(undefined))} autoComplete="off" aria-invalid={!!error} onKeyDown={e => e.key === 'Enter' && matches && confirm()} />
          {error && <span className="text-xs text-destructive" role="alert">{error}</span>}
        </label>
        <DialogFooter>
          <Button variant="ghost" onClick={onClose}>Cancel</Button>
          <Button variant="destructive" disabled={!matches || purge.isPending} onClick={confirm}>
            {purge.isPending ? <Loader2 className="animate-spin" /> : <Trash2 />} Delete permanently
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
