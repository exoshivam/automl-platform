import { useRef, useState } from 'react';
import { Link } from '@tanstack/react-router';
import { useQueryClient } from '@tanstack/react-query';
import { AlertTriangle, CheckCircle2, CloudOff, Loader2, X } from 'lucide-react';
import { useVersions } from '@/api/data-hooks';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/misc';
import { formatBytes } from '@/lib/files';
import type { UploadProgress, UploadResult } from '@/lib/uploader';

/** Runs one upload job at a time and exposes its progress, result and error. */
export function useUploadRun() {
  const [progress, setProgress] = useState<UploadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<UploadResult | null>(null);
  const ctrl = useRef<AbortController | null>(null);
  const qc = useQueryClient();

  const run = async (job: (signal: AbortSignal, onProgress: (p: UploadProgress) => void) => Promise<UploadResult>) => {
    ctrl.current = new AbortController();
    setError(null);
    setResult(null);
    try {
      const r = await job(ctrl.current.signal, setProgress);
      setResult(r);
      qc.invalidateQueries({ queryKey: ['versions'] });
      qc.invalidateQueries({ queryKey: ['project'] });
      return r;
    } catch (e) {
      setError((e as Error).name === 'AbortError' ? 'Upload cancelled.' : (e as Error).message);
      setProgress(null);
      return null;
    }
  };
  const cancel = () => ctrl.current?.abort();
  const reset = () => (setProgress(null), setError(null), setResult(null));
  return { progress, error, result, run, cancel, reset, busy: !!progress && progress.phase !== 'done' && !error };
}

const PHASE: Record<UploadProgress['phase'], string> = {
  starting: 'Starting upload…',
  uploading: 'Uploading straight to storage',
  checksums: 'Computing checksums…',
  completing: 'Finishing up…',
  done: 'Uploaded',
  offline: 'Offline: waiting for the network, then resuming',
};

export function UploadProgressPanel({ progress, onCancel }: { progress: UploadProgress; onCancel?: () => void }) {
  const pct = progress.bytesTotal ? Math.round((progress.bytesUploaded / progress.bytesTotal) * 100) : 0;
  return (
    <div className="grid gap-2 rounded-lg border bg-card p-4" role="status" aria-live="polite">
      <div className="flex items-center gap-2 text-sm font-medium">
        {progress.phase === 'offline' ? <CloudOff className="size-4 text-warning" /> : progress.phase === 'done' ? <CheckCircle2 className="size-4 text-success" /> : <Loader2 className="size-4 animate-spin text-primary" />}
        {PHASE[progress.phase]}
        <span className="ml-auto tabular-nums text-muted-foreground">{pct}%</span>
        {onCancel && progress.phase === 'uploading' && (
          <Button variant="ghost" size="sm" onClick={onCancel} aria-label="Cancel upload">
            <X /> Cancel
          </Button>
        )}
      </div>
      <Progress value={pct} aria-label="Upload progress" />
      <p className="text-xs tabular-nums text-muted-foreground">
        {progress.filesDone.toLocaleString()} / {progress.filesTotal.toLocaleString()} files · {formatBytes(progress.bytesUploaded)} of {formatBytes(progress.bytesTotal)}
      </p>
    </div>
  );
}

/** After complete: poll the new version until ingest finishes (DAT-5), then point at the dataset. */
export function IngestStatus({ projectId, datasetId, versionId, onAgain }: { projectId: string; datasetId: string; versionId: string | null; onAgain: () => void }) {
  const versions = useVersions(datasetId);
  const v = versions.data?.items.find(x => x.id === versionId);
  const ready = v?.ingest_status === 'ready';
  const failed = v?.ingest_status === 'failed';
  return (
    <div className="grid gap-3 rounded-lg border bg-card p-4 text-sm" role="status" aria-live="polite">
      {!versionId ? (
        <p className="flex items-center gap-2">
          <CheckCircle2 className="size-4 text-success" /> Files stored.
        </p>
      ) : ready ? (
        <p className="flex items-center gap-2">
          <CheckCircle2 className="size-4 text-success" /> Version {v.version_no} is ready: {v.sample_count.toLocaleString()} samples.
        </p>
      ) : failed ? (
        <p className="flex items-center gap-2 text-destructive">
          <AlertTriangle className="size-4" /> Ingest failed{v?.ingest_errors.length ? `: ${v.ingest_errors.map(e => e.message ?? e.code).join(', ')}` : ''}.
        </p>
      ) : (
        <p className="flex items-center gap-2">
          <Loader2 className="size-4 animate-spin text-primary" /> Processing on the server (unpacking, checksums, thumbnails)…
        </p>
      )}
      {ready && v.ingest_errors.length > 0 && (
        <p className="text-xs text-warning">
          {v.ingest_errors.length} file(s) skipped: {v.ingest_errors.slice(0, 3).map(e => `${e.file ?? ''} (${e.code ?? e.message})`).join(', ')}
          {v.ingest_errors.length > 3 ? '…' : ''}
        </p>
      )}
      <div className="flex flex-wrap gap-2">
        <Button asChild size="sm" disabled={!ready}>
          <Link to="/projects/$projectId/dataset" params={{ projectId }} search={{ v: versionId ?? undefined }}>
            Open dataset
          </Link>
        </Button>
        <Button size="sm" variant="outline" onClick={onAgain}>
          Add more data
        </Button>
      </div>
    </div>
  );
}
