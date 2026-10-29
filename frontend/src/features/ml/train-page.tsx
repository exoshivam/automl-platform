import { isPretrained } from '@/api/detector';
import { Link } from '@tanstack/react-router';
import { useQueryClient } from '@tanstack/react-query';
import { CheckCircle2, Clock, Cpu, Loader2, Play, Square, XCircle } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip as RTooltip, XAxis, YAxis } from 'recharts';
import { toast } from 'sonner';
import { useProject } from '@/api/hooks';
import { useVersions } from '@/api/data-hooks';
import { METRIC_NAMES, TERMINAL, mk, useCancelJob, useJobEvents, useJobs, useStartTraining, useTrials, type BudgetMode, type Job, type Progress } from '@/api/ml';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress as Bar, Skeleton } from '@/components/ui/misc';
import { BUDGETS } from '@/features/projects/spec-editor';
import { ApiError } from '@/lib/api';
import { cn, timeAgo } from '@/lib/utils';

const JOB_BADGE: Record<Job['status'], 'secondary' | 'info' | 'success' | 'destructive' | 'warning'> = {
  queued: 'secondary', running: 'info', succeeded: 'success', failed: 'destructive', cancelled: 'warning',
};
const PHASE: Record<string, string> = {
  queued: 'Waiting for the trainer', preparing: 'Preparing data', searching: 'Searching models', finalizing: 'Final model on the test set',
  succeeded: 'Done', failed: 'Failed', cancelled: 'Cancelled',
};

export const fmtEta = (s: number | null | undefined) =>
  s == null ? '—' : s < 60 ? `${Math.max(1, Math.round(s))} s` : s < 3600 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`;

export function TrainPage({ projectId }: { projectId: string }) {
  const { data: p } = useProject(projectId);
  const versions = useVersions(p?.dataset_id);
  const jobs = useJobs(projectId);
  const start = useStartTraining(projectId);
  const [budget, setBudget] = useState<BudgetMode>('quick');
  const [problem, setProblem] = useState<ApiError | null>(null);
  // Hybrid gate (409 replay_unavailable): keep the button off while the note explains why; re-check after 30 s
  // (the reference set is being built server-side) without auto-starting anything.
  const gated = problem?.problem.code === 'replay_unavailable';
  useEffect(() => {
    if (!gated) return;
    const t = setTimeout(() => setProblem(null), 30_000);
    return () => clearTimeout(t);
  }, [gated, problem]);

  const ready = versions.data?.items.find(v => v.sealed_at && v.validation_status === 'passed');
  const latest = versions.data?.items[0];
  const active = jobs.data?.find(j => !TERMINAL.includes(j.status as never));
  const shown = active ?? jobs.data?.[0];
  const vNo = (id: string) => versions.data?.items.find(v => v.id === id)?.version_no;
  const lastDone = jobs.data?.find(j => j.status === 'succeeded');
  const stale = ready && lastDone && lastDone.dataset_version_id !== ready.id && ready.version_no > (vNo(lastDone.dataset_version_id) ?? 0);
  const startRun = () => {
    setProblem(null);
    start.mutate(
      { dataset_version_id: ready!.id, budget_mode: budget },
      {
        onSuccess: r => toast.success(`Training on v${ready!.version_no} started · about ${r.estimate.eta_minutes} min`),
        onError: e => (e instanceof ApiError ? setProblem(e) : toast.error(e.message)),
      },
    );
  };

  if (!p || versions.isLoading || jobs.isLoading) return <Skeleton className="h-96 rounded-xl" />;

  return (
    <div className="grid gap-5 lg:grid-cols-[22rem_1fr]">
      <Card className="h-fit">
        <CardHeader>
          <CardTitle>Train a model</CardTitle>
          <CardDescription>
            {isPretrained(p)
              ? 'Training updates your detector to also find your custom objects. Ready-made objects are kept by auto-labelling them in your photos.'
              : 'The search tries a few architectures and settings, then retrains the best on train + validation.'}
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-5">
          <div className="grid gap-2">
            <div className="text-sm font-medium">Dataset version</div>
            {ready ? (
              <div className="rounded-lg border p-3 text-sm">
                <div className="font-medium">v{ready.version_no} · {ready.sample_count.toLocaleString()} samples</div>
                <div className="text-xs text-muted-foreground">Sealed {timeAgo(ready.sealed_at!)} · validation passed</div>
              </div>
            ) : (
              <div className="rounded-lg border border-warning/40 bg-warning/8 p-3 text-sm">
                {latest ? 'No version has passed validation yet.' : 'No data yet.'}{' '}
                <Link to={latest ? '/projects/$projectId/dataset' : '/projects/$projectId/data'} params={{ projectId }} className="font-medium text-primary hover:underline">
                  {latest ? 'Review the dataset →' : 'Add data →'}
                </Link>
              </div>
            )}
          </div>

          <div className="grid gap-2" role="radiogroup" aria-label="Budget">
            <div className="text-sm font-medium">Budget</div>
            {BUDGETS.map(b => (
              <button
                key={b.id}
                type="button"
                role="radio"
                aria-checked={budget === b.id}
                disabled={!b.available}
                onClick={() => setBudget(b.id)}
                className={cn(
                  'flex items-center justify-between rounded-lg border p-3 text-left text-sm transition-colors',
                  budget === b.id ? 'border-primary bg-primary/6 ring-1 ring-primary' : 'hover:bg-accent',
                  !b.available && 'cursor-not-allowed opacity-60 hover:bg-transparent',
                )}
              >
                <span>
                  <span className="font-medium">{b.label}</span>
                  <span className="block text-xs text-muted-foreground">{b.time}</span>
                </span>
                {!b.available && <Badge variant="secondary">Coming soon</Badge>}
              </button>
            ))}
          </div>

          {problem && <ProblemNote e={problem} projectId={projectId} />}
          <Button disabled={!ready || !!active || start.isPending || gated} onClick={startRun} title={gated ? 'Training is paused: see the note above. This re-checks automatically.' : undefined}>
            {start.isPending ? <Loader2 className="animate-spin" /> : <Play />} {active ? 'Training in progress' : ready ? `Train on v${ready.version_no}` : 'Start training'}
          </Button>
        </CardContent>
      </Card>

      <div className="grid h-fit gap-5">
        {stale && !active && (
          <div className="flex flex-wrap items-center gap-3 rounded-xl border border-info/40 bg-info/8 p-4 text-sm" role="status">
            <span>
              Dataset <span className="font-medium">v{ready!.version_no}</span> is newer than the one your latest model was trained on
              (<span className="font-medium">v{vNo(lastDone!.dataset_version_id) ?? '?'}</span>).
            </span>
            <Button size="sm" onClick={startRun} disabled={start.isPending}><Play /> Retrain on latest version</Button>
          </div>
        )}
        {shown ? <RunCard key={shown.id} job={shown} projectId={projectId} versionNo={vNo(shown.dataset_version_id)} /> : (
          <Card className="grid place-items-center p-12 text-center text-sm text-muted-foreground">
            <Cpu className="mb-2 size-8" />
            No training runs yet. Pick a budget and start one; progress streams here live.
          </Card>
        )}
        {(jobs.data?.length ?? 0) > 1 && <History jobs={jobs.data!.filter(j => j.id !== shown?.id)} vNo={vNo} />}
      </div>
    </div>
  );
}

function ProblemNote({ e, projectId }: { e: ApiError; projectId: string }) {
  const code = e.problem.code;
  if (code === 'replay_unavailable') {
    const x = e.problem as { replay_images?: number; required?: number };
    return (
      <div className="grid gap-1 rounded-lg border border-warning/40 bg-warning/8 p-3 text-sm" role="alert">
        <span className="font-medium">Training your custom objects isn't available yet in this build.</span>
        <span>
          Training would make the detector forget its ready-made objects: it needs reference photos of them
          ({x.replay_images ?? 0} of {x.required ?? 50} available) to remember them while it learns yours. Your photos and boxes are saved; train once
          the reference set is added. Ready-made detection keeps working meanwhile.
        </span>
      </div>
    );
  }
  return (
    <div className="rounded-lg border border-destructive/40 bg-destructive/8 p-3 text-sm text-destructive" role="alert">
      {code === 'task_not_supported_yet' ? 'Training for this task type is not in this build yet.' : e.message}
      {(code === 'validation_not_passed' || code === 'version_not_sealed') && (
        <Link to="/projects/$projectId/dataset" params={{ projectId }} className="ml-1 font-medium underline">Open validation</Link>
      )}
    </div>
  );
}

function RunCard({ job, projectId, versionNo }: { job: Job; projectId: string; versionNo?: number }) {
  const qc = useQueryClient();
  const cancel = useCancelJob(projectId);
  const { events, connected } = useJobEvents(job, () => {
    qc.invalidateQueries({ queryKey: mk.jobs(projectId) });
    qc.invalidateQueries({ queryKey: mk.models(projectId) });
    qc.invalidateQueries({ queryKey: ['project', projectId] });
  });
  const live = !TERMINAL.includes(job.status as never);
  const trials = useTrials(job.id, live);
  const p: Progress = { ...(job.progress as Progress), ...events.at(-1) };
  const status = (p.status && TERMINAL.includes(p.status as never) ? p.status : job.status) as Job['status'];
  const metric = p.metric_name ? METRIC_NAMES[p.metric_name] ?? p.metric_name : 'Validation score';

  const frac = useMemo(() => {
    if (status === 'succeeded') return 1;
    if (!p.trials_total || !p.trial) return p.phase === 'finalizing' ? 0.95 : 0.02;
    const inTrial = p.epochs_total && p.epoch ? p.epoch / p.epochs_total : 1;
    return Math.min(0.95, (p.trial - 1 + inTrial) / p.trials_total);
  }, [p, status]);

  const series = events
    .filter(e => e.current_metric != null)
    .map((e, i) => ({ step: i + 1, label: `T${e.trial}${e.epoch ? `·E${e.epoch}` : ''}`, current: e.current_metric, best: e.best_metric, loss: e.loss }));

  return (
    <Card>
      <CardHeader className="flex-row items-start justify-between gap-3">
        <div>
          <CardTitle className="flex items-center gap-2">
            Run {new Date(job.created_at).toLocaleString()}
            <Badge variant={JOB_BADGE[status]}>{status}</Badge>
          </CardTitle>
          <CardDescription>
            Dataset v{versionNo ?? '?'} · {job.budget_mode} budget{p.device && ` · ${p.device}`}
            {live && <span className={cn('ml-2', connected ? 'text-success' : 'text-muted-foreground')}>● {connected ? 'live' : 'connecting'}</span>}
          </CardDescription>
        </div>
        {live && (
          <Button variant="outline" size="sm" disabled={job.cancel_requested || cancel.isPending} onClick={() => cancel.mutate(job.id)}>
            <Square /> {job.cancel_requested ? 'Cancelling…' : 'Cancel'}
          </Button>
        )}
      </CardHeader>
      <CardContent className="grid gap-5">
        <div className="grid gap-2">
          <div className="flex items-center justify-between text-sm">
            <span className="font-medium">{PHASE[status === 'running' ? p.phase ?? 'running' : status] ?? p.phase}</span>
            {live && <span className="flex items-center gap-1 text-muted-foreground"><Clock className="size-3.5" /> {fmtEta(p.eta_seconds)} left</span>}
          </div>
          <Bar value={frac * 100} aria-label="Training progress" indicatorClassName={status === 'failed' ? 'bg-destructive' : undefined} />
          {p.message && live && <p className="text-xs text-muted-foreground">{p.message}</p>}
        </div>

        <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat k="Trial" v={p.trial ? `${p.trial} / ${p.trials_total}` : '—'} />
          <Stat k="Epoch" v={p.epoch ? `${p.epoch} / ${p.epochs_total}` : '—'} />
          <Stat k={`Current ${metric}`} v={p.current_metric?.toFixed(3) ?? '—'} />
          <Stat k={`Best ${metric}`} v={p.best_metric?.toFixed(3) ?? '—'} />
        </dl>

        {series.length > 1 && (
          <div className="h-56" aria-label="Validation score over time">
            <ResponsiveContainer>
              <LineChart data={series} margin={{ left: -16, right: 8 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                <XAxis dataKey="label" tick={{ fontSize: 11 }} interval="preserveStartEnd" />
                <YAxis tick={{ fontSize: 11 }} domain={['auto', 'auto']} />
                <RTooltip contentStyle={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Line type="monotone" dataKey="current" name={metric} stroke="var(--chart-1, var(--primary))" dot={false} strokeWidth={2} isAnimationActive={false} />
                <Line type="stepAfter" dataKey="best" name="Best so far" stroke="var(--success)" dot={false} strokeWidth={2} isAnimationActive={false} />
                {series.some(s => s.loss != null) && <Line type="monotone" dataKey="loss" name="Train loss" stroke="var(--muted-foreground)" dot={false} strokeDasharray="4 3" isAnimationActive={false} />}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {status === 'failed' && (
          <div className="flex gap-2 rounded-lg border border-destructive/40 bg-destructive/8 p-3 text-sm text-destructive" role="alert">
            <XCircle className="mt-0.5 size-4 shrink-0" />
            <div><span className="font-medium">{job.error_code ?? 'Training failed'}</span>{job.error_message && `: ${job.error_message}`}</div>
          </div>
        )}
        {status === 'succeeded' && (
          <div className="flex flex-wrap items-center gap-3 rounded-lg border border-success/40 bg-success/8 p-3 text-sm">
            <CheckCircle2 className="size-4 text-success" /> Model trained. Its test-set evaluation runs next.
            <Link to="/projects/$projectId/evaluate" params={{ projectId }} className="font-medium text-primary hover:underline">View evaluation →</Link>
          </div>
        )}

        {!!trials.data?.length && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <caption className="mb-2 text-left text-sm font-medium">Trials</caption>
              <thead className="text-left text-xs text-muted-foreground">
                <tr><th className="py-1 pr-3">#</th><th className="pr-3">State</th><th className="pr-3">Validation</th><th>Settings</th></tr>
              </thead>
              <tbody>
                {trials.data.map(t => (
                  <tr key={t.trial_no} className="border-t">
                    <td className="py-1.5 pr-3 tabular-nums">{t.trial_no + 1}</td>
                    <td className="pr-3"><Badge variant={t.state === 'complete' ? 'success' : t.state === 'running' ? 'info' : 'secondary'}>{t.state}</Badge></td>
                    <td className="pr-3 tabular-nums">{t.val_metric?.toFixed(3) ?? '—'}</td>
                    <td className="max-w-md truncate font-mono text-xs text-muted-foreground" title={JSON.stringify(t.params)}>
                      {Object.entries(t.params).map(([k, v]) => `${k}=${typeof v === 'number' ? +v.toPrecision(3) : v}`).join(' ')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function Stat({ k, v }: { k: string; v: string }) {
  return (
    <div className="rounded-lg bg-muted/50 p-3">
      <dt className="truncate text-xs text-muted-foreground">{k}</dt>
      <dd className="text-lg font-semibold tabular-nums">{v}</dd>
    </div>
  );
}

function History({ jobs, vNo }: { jobs: Job[]; vNo: (id: string) => number | undefined }) {
  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Earlier runs</CardTitle></CardHeader>
      <CardContent className="grid gap-2">
        {jobs.map(j => (
          <div key={j.id} className="flex items-center justify-between gap-3 text-sm">
            <span>{new Date(j.created_at).toLocaleString()} · dataset v{vNo(j.dataset_version_id) ?? '?'} · {j.budget_mode}</span>
            <span className="flex items-center gap-2">
              {(j.progress as Progress).best_metric != null && <span className="tabular-nums text-muted-foreground">best {(j.progress as Progress).best_metric!.toFixed(3)}</span>}
              <Badge variant={JOB_BADGE[j.status]}>{j.status}</Badge>
            </span>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
