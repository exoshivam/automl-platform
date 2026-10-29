import { AlertTriangle, Copy, KeyRound, Loader2, Pause, Play, Plus, Rocket, Trash2 } from 'lucide-react';
import { useState } from 'react';
import { Bar, BarChart, ResponsiveContainer, Tooltip as RTooltip, XAxis } from 'recharts';
import { toast } from 'sonner';
import { isPretrained } from '@/api/detector';
import { useCurrentOrg, useProject } from '@/api/hooks';
import {
  deployError,
  useCreateKey,
  useDeleteDeployment,
  useDeploy,
  useDeployment,
  useDocs,
  useKeys,
  useRevokeKey,
  useUpdateDeployment,
  useUsage,
  type ApiKey,
  type Deployment,
  type NewKey,
  type Usage,
} from '@/api/deploy';
import { isQuickText } from '@/api/text';
import type { Project } from '@/api/types';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Skeleton } from '@/components/ui/misc';
import { NoModel, VersionSelect, useVersionPick } from '@/features/ml/evaluate-page';
import { Snippets } from '@/features/ml/export-page';
import { isBiometric } from '@/features/projects/spec-schema';
import { cn, timeAgo } from '@/lib/utils';
import { API_BASE, KEY_PLACEHOLDER, inputKind, snippets } from './snippets';

/** Trained-model projects pin a model version; ready-made, quick text, sentiment, face and speaker resolve like the playground. */
export const pinsVersion = (p: Project) => !isPretrained(p) && !isQuickText(p) && !isBiometric(p.task_type);
/** Face/speaker deployments are owner-only (also enforced server-side: 403 owner_only). */
export const canDeploy = (p: Project, role: string | undefined) => !isBiometric(p.task_type) || role === 'owner';

const copy = (text: string, what = 'Copied') => navigator.clipboard.writeText(text).then(() => toast.success(what), () => toast.error('Copy failed: select the text and copy it manually'));

export function DeployPage({ projectId }: { projectId: string }) {
  const { data: p } = useProject(projectId);
  const { org } = useCurrentOrg();
  const dep = useDeployment(projectId);
  if (!p || dep.isLoading) return <Skeleton className="h-96 rounded-xl" />;
  if (!canDeploy(p, org?.role))
    return <Card className="p-8 text-center text-sm text-muted-foreground">Only the workspace owner can deploy face and speaker recognition projects.</Card>;
  if (!dep.data) return <DeployHero project={p} />;
  const d = dep.data;
  return (
    <div className="grid gap-5">
      <StatusCard project={p} d={d} />
      <div className="grid gap-5 lg:grid-cols-2">
        <KeysCard projectId={projectId} d={d} />
        <UsageCard d={d} />
      </div>
      <DocsCard d={d} />
    </div>
  );
}

function DeployHero({ project }: { project: Project }) {
  const deploy = useDeploy(project.id);
  const pins = pinsVersion(project);
  const { loading, ready, mv, setPicked } = useVersionPick(project.id);
  if (pins && loading) return <Skeleton className="h-48 rounded-xl" />;
  if (pins && !ready.length) return <NoModel projectId={project.id} />;
  return (
    <Card className="max-w-2xl">
      <CardHeader>
        <CardTitle className="flex items-center gap-2"><Rocket className="size-5" /> Deploy as an API</CardTitle>
        <CardDescription>Runs on this computer at <code className="text-xs">{API_BASE}/v1/predict</code>. Each API key you create calls this project's model; nothing else is needed in the request.</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-wrap items-center gap-3">
        {pins && <VersionSelect ready={ready} mv={mv} onPick={setPicked} />}
        <Button disabled={deploy.isPending} onClick={() => deploy.mutate(pins ? mv?.id : undefined, { onSuccess: () => toast.success('Deployed. Create an API key to call it.'), onError: e => toast.error(deployError(e)) })}>
          {deploy.isPending ? <Loader2 className="animate-spin" /> : <Rocket />} Deploy
        </Button>
      </CardContent>
    </Card>
  );
}

function StatusCard({ project, d }: { project: Project; d: Deployment }) {
  const update = useUpdateDeployment(project.id);
  const del = useDeleteDeployment(project.id);
  const pins = pinsVersion(project);
  const { ready } = useVersionPick(project.id);
  const pinned = ready.find(v => v.id === d.model_version_id);
  const [confirm, setConfirm] = useState(false);
  const active = d.status === 'active';
  const patch = (body: { status?: Deployment['status']; model_version_id?: string }, ok: string) =>
    update.mutate({ id: d.id, ...body }, { onSuccess: () => toast.success(ok), onError: e => toast.error(deployError(e)) });
  return (
    <Card>
      <CardHeader className="flex-row flex-wrap items-start justify-between gap-3">
        <div className="grid gap-1">
          <CardTitle className="flex items-center gap-2 text-base">
            Deployment <Badge variant={active ? 'success' : 'warning'}>{active ? 'Active' : 'Paused'}</Badge>
          </CardTitle>
          <CardDescription>
            {pins ? `Serving model v${d.model_version_no ?? '?'}` : 'Serving the same model as the Playground'} · {d.rate_limit_per_min} requests/min per key · deployed {timeAgo(d.created_at)}
            {!active && ' · calls get 403 deployment_paused until you resume'}
          </CardDescription>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {pins && ready.length > 1 && (
            <label className="flex items-center gap-2 text-sm">
              <span className="text-muted-foreground">Version</span>
              <VersionSelect ready={ready} mv={pinned} onPick={id => id !== d.model_version_id && patch({ model_version_id: id }, `Now serving v${ready.find(v => v.id === id)?.version_no}. Keys are unchanged.`)} />
            </label>
          )}
          <Button variant="outline" size="sm" disabled={update.isPending} onClick={() => patch({ status: active ? 'paused' : 'active' }, active ? 'Paused' : 'Resumed')}>
            {active ? <><Pause /> Pause</> : <><Play /> Resume</>}
          </Button>
          <Button variant="ghost" size="sm" className="text-destructive hover:text-destructive" onClick={() => setConfirm(true)}><Trash2 /> Delete</Button>
        </div>
      </CardHeader>
      {pins && ready.length > 1 && <CardContent className="pt-0 text-xs text-muted-foreground">Switching or rolling back the version keeps the same endpoint and API keys.</CardContent>}
      <Dialog open={confirm} onOpenChange={setConfirm}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete this deployment?</DialogTitle>
            <DialogDescription>The API stops answering and all {d.key_count} API {d.key_count === 1 ? 'key is' : 'keys are'} revoked. The project and its models are kept; you can deploy again later (with new keys).</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setConfirm(false)}>Cancel</Button>
            <Button variant="destructive" disabled={del.isPending} onClick={() => del.mutate(d.id, { onSuccess: () => (setConfirm(false), toast.success('Deployment deleted and keys revoked')), onError: e => toast.error(deployError(e)) })}>
              {del.isPending ? <Loader2 className="animate-spin" /> : <Trash2 />} Delete deployment
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
}

function KeysCard({ projectId, d }: { projectId: string; d: Deployment }) {
  const keys = useKeys(d.id);
  const create = useCreateKey(projectId, d.id);
  const revoke = useRevokeKey(projectId, d.id);
  const [name, setName] = useState('');
  const [made, setMade] = useState<NewKey | null>(null);
  const [revoking, setRevoking] = useState<ApiKey | null>(null);
  const live = (keys.data ?? []).filter(k => !k.revoked_at);
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base"><KeyRound className="size-4" /> API keys</CardTitle>
        <CardDescription>Each key calls this deployment only. Treat it like a password.</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-3">
        <form className="flex gap-2" onSubmit={e => (e.preventDefault(), name.trim() && create.mutate(name.trim(), { onSuccess: k => (setMade(k), setName('')), onError: e => toast.error(deployError(e)) }))}>
          <Input value={name} onChange={e => setName(e.target.value)} maxLength={100} placeholder="Key name, e.g. “shop app”" aria-label="New key name" className="h-9" />
          <Button type="submit" size="sm" className="h-9" disabled={!name.trim() || create.isPending}>{create.isPending ? <Loader2 className="animate-spin" /> : <Plus />} Create key</Button>
        </form>
        {keys.isLoading ? <Skeleton className="h-16" /> : !live.length ? (
          <p className="text-sm text-muted-foreground">No keys yet. Create one to call the API.</p>
        ) : (
          <ul className="divide-y rounded-lg border text-sm">
            {live.map(k => (
              <li key={k.id} className="flex items-center gap-3 p-2.5">
                <div className="min-w-0 flex-1">
                  <div className="truncate font-medium">{k.name}</div>
                  <div className="text-xs text-muted-foreground"><code>{k.prefix}…</code> · created {timeAgo(k.created_at)} · {k.last_used_at ? `last used ${timeAgo(k.last_used_at)}` : 'never used'}</div>
                </div>
                <Button size="sm" variant="ghost" className="text-destructive hover:text-destructive" onClick={() => setRevoking(k)}>Revoke</Button>
              </li>
            ))}
          </ul>
        )}
      </CardContent>

      <Dialog open={!!made} onOpenChange={o => !o && setMade(null)}>
        <DialogContent onInteractOutside={e => e.preventDefault()}>
          <DialogHeader>
            <DialogTitle>Your new API key: “{made?.name}”</DialogTitle>
            <DialogDescription>Copy it now and store it somewhere safe.</DialogDescription>
          </DialogHeader>
          <div className="flex items-center gap-2 rounded-lg border bg-muted/50 p-3">
            <code className="flex-1 break-all font-mono text-sm" data-testid="new-key">{made?.key}</code>
            <Button size="sm" variant="outline" onClick={() => made && void copy(made.key, 'Key copied')}><Copy /> Copy</Button>
          </div>
          <p className="flex items-start gap-2 text-sm text-warning" role="alert"><AlertTriangle className="mt-0.5 size-4 shrink-0" /> This key won't be shown again. If you lose it, revoke it and create a new one.</p>
          <DialogFooter><Button onClick={() => setMade(null)}>I've saved it</Button></DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={!!revoking} onOpenChange={o => !o && setRevoking(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Revoke “{revoking?.name}”?</DialogTitle>
            <DialogDescription>Apps using <code>{revoking?.prefix}…</code> get 401 from now on. This can't be undone.</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setRevoking(null)}>Cancel</Button>
            <Button variant="destructive" disabled={revoke.isPending} onClick={() => revoking && revoke.mutate(revoking.id, { onSuccess: () => (setRevoking(null), toast.success('Key revoked')), onError: e => toast.error(deployError(e)) })}>
              {revoke.isPending ? <Loader2 className="animate-spin" /> : null} Revoke key
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
}

function UsageCard({ d }: { d: Deployment }) {
  const [w, setW] = useState<Usage['window']>('24h');
  const u = useUsage(d.id, w).data;
  const stat = (label: string, value: string) => (
    <div><dt className="text-xs text-muted-foreground">{label}</dt><dd className="text-lg font-semibold tabular-nums">{value}</dd></div>
  );
  return (
    <Card>
      <CardHeader className="flex-row items-start justify-between gap-3">
        <div><CardTitle className="text-base">Usage</CardTitle><CardDescription>Calls to /v1/predict with this deployment's keys (requests with an invalid key aren't counted).</CardDescription></div>
        <div className="flex rounded-md border p-0.5" role="radiogroup" aria-label="Usage window">
          {(['24h', '7d'] as const).map(x => (
            <button key={x} type="button" role="radio" aria-checked={w === x} onClick={() => setW(x)} className={cn('rounded px-2.5 py-1 text-xs', w === x ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:bg-accent')}>{x}</button>
          ))}
        </div>
      </CardHeader>
      <CardContent className="grid gap-4">
        {!u ? <Skeleton className="h-24" /> : (
          <>
            <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {stat('Requests', u.requests.toLocaleString())}
              {stat('Errors', `${u.errors.toLocaleString()}${u.requests ? ` (${(u.error_rate * 100).toFixed(1)}%)` : ''}`)}
              {stat('Avg latency', u.avg_latency_ms != null ? `${Math.round(u.avg_latency_ms)} ms` : '—')}
              {stat('p95 latency', u.p95_latency_ms != null ? `${Math.round(u.p95_latency_ms)} ms` : '—')}
            </dl>
            {u.requests > 0 && (
              <div className="h-28" aria-label={`Requests per ${w === '24h' ? 'hour' : 'day'}`} role="img">
                <ResponsiveContainer>
                  <BarChart data={u.series.map(s => ({ ...s, ok: s.requests - s.errors, label: w === '24h' ? new Date(s.t).getHours() + ':00' : new Date(s.t).toLocaleDateString(undefined, { weekday: 'short' }) }))}>
                    <XAxis dataKey="label" tick={{ fontSize: 10 }} interval="preserveStartEnd" tickLine={false} axisLine={false} />
                    <RTooltip cursor={{ fillOpacity: 0.1 }} contentStyle={{ fontSize: 12 }} />
                    <Bar dataKey="ok" name="OK" stackId="a" fill="var(--primary)" />
                    <Bar dataKey="errors" name="Errors" stackId="a" fill="var(--destructive)" radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

const ERROR_ROWS: [string, string][] = [
  ['401 unauthorized', 'Missing, wrong or revoked key'],
  ['403 deployment_paused', 'The deployment is paused'],
  ['409 model_not_ready', 'The pinned model version is gone; pick another version'],
  ['413 too_large', 'File too big'],
  ['422 …', 'Wrong input: wrong_input_type, file_required, input_required, bad_input, not_an_image, too_short…'],
  ['429 rate_limited', 'Over the per-key limit; wait for the Retry-After header'],
];

function DocsCard({ d }: { d: Deployment }) {
  const docs = useDocs(d.id).data;
  const kind = inputKind(d.task_type);
  const json = docs?.input.content_type === 'application/json';
  const s = snippets(d.task_type, KEY_PLACEHOLDER, json ? docs?.input.example : undefined);
  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_1fr]">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">How to call it</CardTitle>
          <CardDescription>Local only: the API runs on this computer.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 text-sm">
          <Row label="Endpoint"><code>POST {docs?.endpoint ?? `${API_BASE}/v1/predict`}</code></Row>
          <Row label="Auth header"><code>Authorization: Bearer {KEY_PLACEHOLDER}</code></Row>
          <Row label="Input">
            {json ? <>JSON body: <code>{JSON.stringify(docs?.input.example ?? {})}</code></> : <>multipart/form-data with the field <code>{docs?.input.field ?? 'file'}</code> ({kind === 'audio' ? 'a WAV clip' : 'a JPEG/PNG image'})</>}
          </Row>
          {!!docs?.labels.length && <Row label="Labels">{docs.labels.join(', ')}</Row>}
          <div>
            <div className="mb-1 text-xs text-muted-foreground">Example response</div>
            <pre className="max-h-56 overflow-auto rounded-lg bg-muted/60 p-3 font-mono text-xs">{docs ? JSON.stringify(docs.output_example, null, 2) : '…'}</pre>
          </div>
          <div>
            <div className="mb-1 text-xs text-muted-foreground">Errors (application/problem+json with a <code>code</code>)</div>
            <table className="w-full text-xs">
              <tbody>
                {ERROR_ROWS.map(([c, m]) => <tr key={c} className="border-t"><td className="whitespace-nowrap py-1 pr-3 font-mono">{c}</td><td className="py-1 text-muted-foreground">{m}</td></tr>)}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
      <Snippets s={s} title="Code" description={`Replace ${KEY_PLACEHOLDER} with one of your keys.`} />
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="grid gap-0.5 sm:grid-cols-[7rem_1fr]">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="break-all">{children}</span>
    </div>
  );
}
