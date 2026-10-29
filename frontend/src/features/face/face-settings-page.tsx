import { AlertTriangle, Info, Loader2, Save, Wand2 } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Bar, BarChart, CartesianGrid, Legend, ReferenceLine, ResponsiveContainer, Tooltip as RTooltip, XAxis, YAxis } from 'recharts';
import { toast } from 'sonner';
import { useFaceReport, useFaceSettings, useSaveFaceSettings, type Hist } from '@/api/face';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { DemoBadge, Skeleton } from '@/components/ui/misc';

/** Share of a score histogram at or above t (bins are [lo, hi)); TAR for genuine pairs, FAR for impostor pairs. */
const above = (h: Hist | null, t: number) => {
  if (!h?.count) return null;
  const n = h.hist.reduce((s, b) => s + (b.lo >= t ? b.count : b.hi > t ? (b.count * (b.hi - t)) / (b.hi - b.lo) : 0), 0);
  return n / h.count;
};
const pct = (v: number | null | undefined, d = 1) => (v == null ? '—' : `${(v * 100).toFixed(d)}%`);

/** FACE-5: pick the operating point on the user's own enrolled photos (leave-one-out genuine vs impostor scores). */
export function FaceSettingsPage({ projectId }: { projectId: string }) {
  const settings = useFaceSettings(projectId);
  const report = useFaceReport(projectId);
  const save = useSaveFaceSettings(projectId);
  const [t, setT] = useState<number | null>(null);
  useEffect(() => { if (settings.data && t === null) setT(settings.data.threshold); }, [settings.data, t]);

  const r = report.data;
  const chart = useMemo(
    () => (r?.genuine && r.impostor ? r.genuine.hist.map((b, i) => ({ x: +((b.lo + b.hi) / 2).toFixed(3), genuine: b.count / r.genuine!.count, impostor: (r.impostor!.hist[i]?.count ?? 0) / r.impostor!.count })) : []),
    [r],
  );
  if (settings.isLoading || t === null) return <Skeleton className="h-96 rounded-xl" />;
  if (!settings.data) return <Card className="p-6 text-sm text-destructive">{settings.error?.message}</Card>;
  const s = settings.data;
  const voice = s.kind === 'speaker';
  const unit = voice ? 'clips' : 'photos';
  const dirty = Math.abs(t - s.threshold) > 1e-6;
  const tar = above(r?.genuine ?? null, t);
  const far = above(r?.impostor ?? null, t);

  return (
    <div className="grid gap-5 lg:grid-cols-[22rem_1fr]">
      <Card className="h-fit">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">Match threshold {!voice && <DemoBadge group="face" />}</CardTitle>
          <CardDescription>
            How similar a {voice ? 'voice' : 'face'} must be to someone you added before it's named. <b>Stricter</b> (higher) = fewer strangers misidentified as someone, but a
            known person may sometimes show as “{voice ? 'Unknown' : 'Other'}”. <b>Looser</b> = recognises people more often, with more mistakes.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4">
          <div className="flex items-baseline justify-between">
            <span className="text-3xl font-semibold tabular-nums">{t.toFixed(2)}</span>
            {voice && s.threshold_source ? (
              <Badge variant={s.threshold_source === 'user' ? 'secondary' : 'info'} title={s.threshold_source === 'auto' ? 'Picked from your clips so different speakers stay below it' : undefined}>
                {s.threshold_source === 'auto' ? 'Automatic' : s.threshold_source === 'user' ? 'Set by you' : 'Default'}
              </Badge>
            ) : (
              <span className="text-xs text-muted-foreground">default {s.default_threshold.toFixed(2)}</span>
            )}
          </div>
          <input type="range" min={voice ? 0.5 : 0.05} max={voice ? 0.99 : 0.95} step={0.01} value={t} onChange={e => setT(Number(e.target.value))} aria-label="Match threshold" className="w-full accent-[var(--primary)]" />
          <div className="flex justify-between text-xs text-muted-foreground"><span>Looser</span><span>Stricter</span></div>
          {voice && s.score_hint && <p className="text-xs text-muted-foreground">{s.score_hint}</p>}
          {voice && s.threshold_source === 'auto' && <p className="text-xs text-muted-foreground">Automatic: adjusts as you add speakers and clips, until you set it yourself.</p>}
          {r?.ready && (
            <dl className="grid grid-cols-2 gap-3 text-sm">
              <div className="rounded-lg bg-muted/50 p-3"><dt className="text-xs text-muted-foreground">Known people recognised</dt><dd className="text-lg font-semibold tabular-nums">{pct(tar)}</dd></div>
              <div className="rounded-lg bg-muted/50 p-3"><dt className="text-xs text-muted-foreground">Strangers wrongly named</dt><dd className="text-lg font-semibold tabular-nums">{pct(far, 2)}</dd></div>
            </dl>
          )}
          <div className="flex flex-wrap gap-2">
            {r?.suggested_threshold != null && (
              <Button variant="outline" size="sm" onClick={() => setT(r.suggested_threshold!)}><Wand2 /> Use suggested ({r.suggested_threshold.toFixed(2)})</Button>
            )}
            {voice ? (
              s.threshold_source === 'user' && (
                <Button size="sm" variant="ghost" disabled={save.isPending} onClick={() => save.mutate({ threshold_auto: true }, { onSuccess: r => (setT(r.threshold), toast.success(`Back to automatic (${r.threshold.toFixed(2)})`)), onError: e => toast.error(e.message) })}>Back to automatic</Button>
              )
            ) : (
              <Button size="sm" variant="ghost" onClick={() => setT(s.default_threshold)}>Reset to default</Button>
            )}
          </div>
          <Button disabled={!dirty || save.isPending} onClick={() => save.mutate({ threshold: +t.toFixed(2) }, { onSuccess: () => toast.success(`Threshold set to ${t.toFixed(2)}`), onError: e => toast.error(e.message) })}>
            {save.isPending ? <Loader2 className="animate-spin" /> : <Save />} {dirty ? 'Save threshold' : 'Saved'}
          </Button>
          <div className="flex gap-2 rounded-lg border border-warning/40 bg-warning/8 p-3 text-xs">
            <Info className="mt-0.5 size-4 shrink-0 text-warning" />
            <span>
              {voice ? 'Voice' : 'Face'} model: {s.embedding_model.name ?? s.embedding_model.id}, licence <b>{s.embedding_model.licence}</b>
              {!s.embedding_model.commercial_use && ': fine for trying it out, not for commercial use'}. {voice ? 'Voice' : 'Face'} data stays in this workspace; there is no export.
            </span>
          </div>
        </CardContent>
      </Card>

      <div className="grid h-fit gap-5">
        <Card>
          <CardHeader>
            <CardTitle>How well it tells people apart</CardTitle>
            <CardDescription>Measured on your own {unit}: each one is matched against everyone else ({r?.people ?? 0} people, {r?.photos ?? 0} {unit}).</CardDescription>
          </CardHeader>
          <CardContent>
            {report.isLoading ? <Skeleton className="h-64" /> : !r?.ready ? (
              <div className="grid gap-2 text-sm text-muted-foreground">
                {(r?.warnings.length ? r.warnings : [{ code: 'not_ready', detail: `Add at least 2 people with 2 or more ${unit} each.` }]).map(w => <p key={w.code} className="flex gap-2"><Info className="mt-0.5 size-4 shrink-0" /> {w.detail}</p>)}
              </div>
            ) : (
              <div className="grid gap-4">
                <div className="h-64" aria-label="Same-person vs different-person similarity scores">
                  <ResponsiveContainer>
                    <BarChart data={chart} barGap={0} barCategoryGap={1} margin={{ left: -12, right: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                      <XAxis dataKey="x" type="number" domain={[0, 1]} tick={{ fontSize: 11 }} label={{ value: 'Similarity', position: 'insideBottom', offset: -2, fontSize: 11 }} />
                      <YAxis tickFormatter={v => `${Math.round(v * 100)}%`} tick={{ fontSize: 11 }} />
                      <RTooltip formatter={v => `${(Number(v) * 100).toFixed(1)}%`} contentStyle={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }} />
                      <Legend wrapperStyle={{ fontSize: 12 }} />
                      <Bar dataKey="impostor" name="Different people" fill="var(--muted-foreground)" fillOpacity={0.55} isAnimationActive={false} />
                      <Bar dataKey="genuine" name="Same person" fill="var(--primary)" isAnimationActive={false} />
                      <ReferenceLine x={t} stroke="var(--foreground)" strokeWidth={2} label={{ value: 'threshold', fontSize: 11, fill: 'var(--foreground)', position: 'top' }} />
                      {r.suggested_threshold != null && <ReferenceLine x={r.suggested_threshold} stroke="var(--success)" strokeDasharray="4 3" label={{ value: 'suggested', fontSize: 11, fill: 'var(--success)', position: 'insideTopRight' }} />}
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <p className="text-xs text-muted-foreground">Good separation means the blue (same person) sits well to the right of the grey (different people). Put the threshold in the gap.</p>
                <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <div className="rounded-lg bg-muted/50 p-3"><dt className="text-xs text-muted-foreground">Top-1 correct</dt><dd className="font-semibold tabular-nums">{pct(r.rank1_accuracy)}</dd></div>
                  {r.tar_at_far?.map(x => (
                    <div key={x.far} className="rounded-lg bg-muted/50 p-3">
                      <dt className="text-xs text-muted-foreground">Recognised at {x.far * 100}% strangers</dt>
                      <dd className="font-semibold tabular-nums">{pct(x.tar)} <span className="text-xs font-normal text-muted-foreground">@ {x.threshold.toFixed(2)}</span></dd>
                    </div>
                  ))}
                </dl>
              </div>
            )}
          </CardContent>
        </Card>
        {!!r?.per_person.some(p => p.warnings.length) && (
          <Card>
            <CardHeader><CardTitle className="text-base">People to look at</CardTitle></CardHeader>
            <CardContent className="grid gap-2">
              {r.per_person.filter(p => p.warnings.length).map(p => (
                <div key={p.person_id} className="flex flex-wrap items-center gap-2 text-sm">
                  <span className="font-medium">{p.name}</span>
                  <Badge variant="secondary">{p.rank1_hits}/{p.photos} matched</Badge>
                  {p.warnings.map(w => <span key={w.code} className="flex items-center gap-1 text-warning"><AlertTriangle className="size-3.5" /> {w.detail}</span>)}
                </div>
              ))}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
