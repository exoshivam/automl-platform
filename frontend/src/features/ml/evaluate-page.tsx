import { Link } from '@tanstack/react-router';
import { AlertTriangle, ArrowDown, ArrowUp, FileText, Lightbulb, Loader2, RefreshCw, Trash2 } from 'lucide-react';
import { toast } from 'sonner';
import { useDeleteModelVersion } from '@/api/storage';
import { useState } from 'react';
import { Bar, BarChart, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Scatter, ScatterChart, Tooltip as RTooltip, XAxis, YAxis, ZAxis } from 'recharts';
import { useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import {
  LOWER_IS_BETTER, METRIC_NAMES, allVersions, fmtMetric, useCell, useEvaluation, useModelCard, useModelVersion, useModels,
  type EvalReport, type Evaluation, type ModelVersion, type SampleRef,
} from '@/api/ml';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Skeleton } from '@/components/ui/misc';
import { cn } from '@/lib/utils';

/** Picks a ready model version (newest by default); shared by Evaluate, Playground and Export. */
export function useVersionPick(projectId: string) {
  const models = useModels(projectId);
  const ready = allVersions(models.data).filter(v => v.status === 'ready');
  const [picked, setPicked] = useState<string>();
  const mv = ready.find(v => v.id === picked) ?? ready[0];
  return { loading: models.isLoading, ready, mv, setPicked };
}

export function VersionSelect({ ready, mv, onPick }: { ready: ModelVersion[]; mv?: ModelVersion; onPick: (id: string) => void }) {
  if (ready.length < 2) return mv ? <span className="text-sm text-muted-foreground">Model v{mv.version_no} · {mv.architecture}</span> : null;
  return (
    <select
      aria-label="Model version"
      className="h-9 rounded-md border border-input bg-card px-2 text-sm"
      value={mv?.id}
      onChange={e => onPick(e.target.value)}
    >
      {ready.map(v => <option key={v.id} value={v.id}>v{v.version_no} · {v.architecture} · {new Date(v.created_at).toLocaleDateString()}</option>)}
    </select>
  );
}

export function NoModel({ projectId }: { projectId: string }) {
  return (
    <Card className="grid place-items-center gap-2 p-12 text-center text-sm text-muted-foreground">
      No trained model yet.
      <Link to="/projects/$projectId/train" params={{ projectId }} className="font-medium text-primary hover:underline">Train one →</Link>
    </Card>
  );
}

export function EvaluatePage({ projectId }: { projectId: string }) {
  const { loading, ready, mv, setPicked } = useVersionPick(projectId);
  const ev = useEvaluation(mv?.id);
  const full = useModelVersion(mv?.id);
  const [card, setCard] = useState(false);
  const [del, setDel] = useState(false);
  const remove = useDeleteModelVersion(projectId);
  const qc = useQueryClient();

  if (loading) return <Skeleton className="h-96 rounded-xl" />;
  if (!mv) return <NoModel projectId={projectId} />;
  const rep = ev.data?.report;
  const hybrid = ((full.data?.model_json as { output?: { hybrid?: Hybrid } } | null | undefined)?.output?.hybrid) ?? null;

  return (
    <div className="grid gap-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <VersionSelect ready={ready} mv={mv} onPick={setPicked} />
        <div className="flex gap-2">
          <Button variant="ghost" size="sm" onClick={() => setDel(true)} aria-label="Delete this model version"><Trash2 /></Button>
          {rep && <Button variant="outline" size="sm" onClick={() => setCard(true)}><FileText /> Model card</Button>}
          <Link to="/projects/$projectId/playground" params={{ projectId }} className="inline-flex h-8 items-center rounded-md border px-3 text-xs font-medium hover:bg-accent">Try it</Link>
          <Link to="/projects/$projectId/export" params={{ projectId }} className="inline-flex h-8 items-center rounded-md bg-primary px-3 text-xs font-medium text-primary-foreground hover:bg-primary/90">Export</Link>
        </div>
      </div>

      {hybrid && <HybridCard h={hybrid} />}
      {ev.isLoading || ev.data?.status === 'pending' ? (
        <Card className="flex items-center gap-3 p-8 text-sm text-muted-foreground"><Loader2 className="size-5 animate-spin" /> Evaluating on the held-out test set…</Card>
      ) : ev.data?.status === 'failed' ? (
        <Card className="flex flex-wrap items-center gap-3 border-destructive/40 p-6 text-sm text-destructive" role="alert">
          <AlertTriangle className="size-5" /> Evaluation failed: {ev.data.error}
          <Button size="sm" variant="outline" onClick={async () => { await api(`/model-versions/${mv.id}/evaluation/rerun`, { method: 'POST' }); qc.invalidateQueries({ queryKey: ['evaluation', mv.id] }); }}>
            <RefreshCw /> Retry
          </Button>
        </Card>
      ) : rep ? (
        <Report ev={ev.data!} rep={rep} mvId={mv.id} />
      ) : null}

      <Dialog open={del} onOpenChange={setDel}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete model v{mv.version_no}?</DialogTitle>
            <DialogDescription>Deletes the trained weights, evaluation and exports for this version. Its dataset version can then be deleted too. This can't be undone.</DialogDescription>
          </DialogHeader>
          <div className="flex justify-end gap-2">
            <Button variant="ghost" onClick={() => setDel(false)}>Cancel</Button>
            <Button variant="destructive" disabled={remove.isPending} onClick={() => remove.mutate(mv.id, {
              onSuccess: () => (toast.success(`Model v${mv.version_no} deleted`), setDel(false), setPicked(undefined)),
              onError: e => toast.error(e.message),
            })}>
              {remove.isPending ? <Loader2 className="animate-spin" /> : <Trash2 />} Delete permanently
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog open={card} onOpenChange={setCard}>
        <DialogContent className="max-h-[85vh] max-w-3xl overflow-y-auto">
          <DialogHeader><DialogTitle>Model card</DialogTitle><DialogDescription>Generated from the test-set evaluation. Included in every export.</DialogDescription></DialogHeader>
          <ModelCard mvId={mv.id} open={card} />
        </DialogContent>
      </Dialog>
    </div>
  );
}

/** Ready-made + custom in one fine-tuned model (the backend model_json.output.hybrid). */
interface Hybrid { ready_made: string[]; custom: string[]; custom_map50: number | null; coco_map50_before: number | null; coco_map50_after: number | null; coco_drop: number | null; replay_eval?: number; warning?: string }
function HybridCard({ h }: { h: Hybrid }) {
  const pct = (v: number | null | undefined) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`);
  const worse = h.coco_drop != null && h.coco_drop > 0.01; // under 1 mAP point is noise, not a regression
  return (
    <Card className={cn(h.warning && 'border-warning/50')}>
      <CardHeader>
        <CardTitle>One detector: ready-made + your objects</CardTitle>
        <CardDescription>Your custom objects were trained into the same model; ready-made objects were kept by auto-labelling them.</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-3">
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="rounded-lg bg-muted/50 p-3">
            <div className="text-xs text-muted-foreground">Your objects ({h.custom.join(', ')}) · mAP@0.5</div>
            <div className="text-2xl font-semibold tabular-nums">{pct(h.custom_map50)}</div>
          </div>
          <div className="rounded-lg bg-muted/50 p-3">
            <div className="text-xs text-muted-foreground">Ready-made objects ({h.ready_made.join(', ')}) · before → after</div>
            {h.coco_map50_before == null || h.coco_map50_after == null ? (
              <div className="text-lg font-semibold text-muted-foreground">Not measured</div>
            ) : (
              <div className="text-2xl font-semibold tabular-nums">
                {pct(h.coco_map50_before)} → <span className={cn(worse ? 'text-destructive' : 'text-success')}>{pct(h.coco_map50_after)}</span>
              </div>
            )}
            {!!h.replay_eval && <div className="text-xs text-muted-foreground">checked on {h.replay_eval} held-out reference images</div>}
          </div>
        </div>
        {h.warning && (
          <p className="flex gap-2 rounded-lg border border-warning/40 bg-warning/8 p-3 text-sm" role="alert">
            <AlertTriangle className="mt-0.5 size-4 shrink-0 text-warning" /> {h.warning}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function ModelCard({ mvId, open }: { mvId: string; open: boolean }) {
  const q = useModelCard(mvId, open);
  if (q.isLoading) return <Skeleton className="h-64" />;
  return <pre className="whitespace-pre-wrap break-words rounded-lg bg-muted/50 p-4 font-mono text-xs leading-relaxed">{q.data ?? q.error?.message}</pre>;
}

function Report({ ev, rep, mvId }: { ev: Evaluation; rep: EvalReport; mvId: string }) {
  const keys = Object.keys(rep.metrics).sort((a, b) => (a === rep.primary_metric ? -1 : b === rep.primary_metric ? 1 : 0));
  return (
    <>
      {rep.suspicious_perfect && (
        <div className="flex gap-3 rounded-xl border border-warning/50 bg-warning/10 p-4" role="alert">
          <AlertTriangle className="mt-0.5 size-5 shrink-0 text-warning" />
          <div className="grid gap-1 text-sm">
            <p className="font-semibold">Suspiciously perfect: test images are very similar to the training images.</p>
            <p className="text-muted-foreground">
              Real-world accuracy will be lower.
              {rep.data_diversity?.near_duplicate_share ? ` ${Math.round(rep.data_diversity.near_duplicate_share * 100)}% of the dataset are near-duplicates.` : ''}{' '}
              Collect varied photos (different places, lighting, angles and people, in separate sessions) and retrain.
            </p>
          </div>
        </div>
      )}
      <Card>
        <CardHeader>
          <CardTitle>Test results</CardTitle>
          <CardDescription>{rep.n_test.toLocaleString()} test samples the model never saw while training. {rep.baseline && `Baseline: ${rep.baseline.kind}.`}</CardDescription>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          {keys.map(k => <MetricTile key={k} k={k} v={rep.metrics[k] ?? null} base={rep.baseline?.metrics[k] ?? null} primary={k === rep.primary_metric} />)}
        </CardContent>
      </Card>

      {!!rep.recommendations.length && (
        <Card>
          <CardHeader><CardTitle className="flex items-center gap-2"><Lightbulb className="size-5 text-warning" /> How to improve it</CardTitle></CardHeader>
          <CardContent className="grid gap-3">
            {rep.recommendations.map((r, i) => (
              <div key={i} className="grid gap-1 rounded-lg border p-3">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <Badge variant={r.severity === 'high' ? 'destructive' : r.severity === 'medium' ? 'warning' : 'secondary'}>{r.severity}</Badge> {r.title}
                </div>
                <p className="text-sm text-muted-foreground">{r.detail}</p>
                <p className="text-sm"><span className="font-medium">Try: </span>{r.action}</p>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      <div className="grid gap-5 xl:grid-cols-2">
        {rep.confusion && <Confusion rep={rep} mvId={mvId} />}
        {!!rep.per_class.length && <PerClassTable rep={rep} />}
        {rep.regression && <RegressionCharts rep={rep} />}
        {rep.calibration && <Reliability cal={rep.calibration} />}
      </div>

      {!!rep.worst_mistakes.length && <Mistakes ev={ev} rep={rep} />}
    </>
  );
}

function Reliability({ cal }: { cal: NonNullable<EvalReport['calibration']> }) {
  const t = cal.training;
  const data = cal.bins.map(b => ({ x: +(((b.lo + b.hi) / 2) * 100).toFixed(0), accuracy: +(b.accuracy * 100).toFixed(1), confidence: +(b.confidence * 100).toFixed(1), ideal: +(((b.lo + b.hi) / 2) * 100).toFixed(0), count: b.count }));
  return (
    <Card>
      <CardHeader>
        <CardTitle>Can you trust the confidence?</CardTitle>
        <CardDescription>
          When the model says 80%, it should be right about 80% of the time. Bars = actual accuracy per confidence band; the line is perfect calibration. Test ECE {(cal.ece * 100).toFixed(1)}%.
          {t && (t.applied
            ? ` Calibrated with temperature ${t.temperature.toFixed(2)}: validation ECE ${(t.val_ece_before * 100).toFixed(1)}% → ${(t.val_ece_after * 100).toFixed(1)}%.`
            : t.reason === 'val_too_small'
              ? ` Not calibrated: validation set too small (n${t.val_samples != null ? ` = ${t.val_samples}` : ''} < 30).`
              : ' Calibration was tried and not needed.')}
        </CardDescription>
      </CardHeader>
      <CardContent className="h-64">
        <ResponsiveContainer>
          <ComposedChart data={data} margin={{ left: -16, right: 8 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
            <XAxis dataKey="x" unit="%" tick={{ fontSize: 11 }} label={{ value: 'Confidence', position: 'insideBottom', offset: -2, fontSize: 11 }} />
            <YAxis domain={[0, 100]} unit="%" tick={{ fontSize: 11 }} />
            <RTooltip contentStyle={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }} />
            <Bar dataKey="accuracy" name="Actual accuracy" fill="var(--primary)" isAnimationActive={false} />
            <Line dataKey="ideal" name="Perfect calibration" stroke="var(--muted-foreground)" strokeDasharray="4 3" dot={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

function MetricTile({ k, v, base, primary }: { k: string; v: number | null; base?: number | null; primary: boolean }) {
  const better = v != null && base != null ? (LOWER_IS_BETTER.has(k) ? v < base : v > base) : null;
  return (
    <div className={cn('rounded-lg bg-muted/50 p-3', primary && 'bg-primary/8 ring-1 ring-primary/40')}>
      <div className="text-xs text-muted-foreground">{METRIC_NAMES[k] ?? k}{primary && ' · primary'}</div>
      <div className="text-2xl font-semibold tabular-nums">{fmtMetric(k, v)}</div>
      {base != null && (
        <div className={cn('flex items-center gap-1 text-xs', better ? 'text-success' : 'text-destructive')}>
          {better ? <ArrowUp className="size-3" /> : <ArrowDown className="size-3" />} baseline {fmtMetric(k, base)}
        </div>
      )}
    </div>
  );
}

function Confusion({ rep, mvId }: { rep: EvalReport; mvId: string }) {
  const cm = rep.confusion!;
  const [cell, setCell] = useState<{ y_true: string; y_pred: string } | null>(null);
  const samples = useCell(mvId, cell);
  return (
    <Card>
      <CardHeader>
        <CardTitle>Confusion matrix</CardTitle>
        <CardDescription>Rows are the true class, columns the prediction. Click a cell to see its samples.</CardDescription>
      </CardHeader>
      <CardContent className="overflow-x-auto">
        <table className="border-separate border-spacing-0.5 text-xs">
          <thead>
            <tr>
              <th />
              {cm.labels.map(l => <th key={l} className="max-w-20 truncate px-1 pb-1 font-medium text-muted-foreground" title={l}>{l}</th>)}
            </tr>
          </thead>
          <tbody>
            {cm.labels.map((t, i) => (
              <tr key={t}>
                <th className="max-w-28 truncate pr-2 text-right font-medium text-muted-foreground" title={t}>{t}</th>
                {cm.labels.map((pr, j) => {
                  const n = cm.counts[i]![j]!;
                  const share = cm.normalized[i]![j]!;
                  const diag = i === j;
                  return (
                    <td key={pr} className="p-0">
                      <button
                        type="button"
                        disabled={!n}
                        onClick={() => setCell({ y_true: t, y_pred: pr })}
                        aria-label={`${n} ${t} predicted as ${pr}`}
                        className="grid size-12 place-items-center rounded tabular-nums transition-transform enabled:hover:scale-105 enabled:hover:ring-2 enabled:hover:ring-ring"
                        style={{ background: n ? `color-mix(in oklch, ${diag ? 'var(--success)' : 'var(--destructive)'} ${Math.round(12 + share * 78)}%, transparent)` : 'var(--muted)' }}
                      >
                        <span className={cn(share > 0.55 && 'font-semibold text-white')}>{n}</span>
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
      <Dialog open={!!cell} onOpenChange={o => !o && setCell(null)}>
        <DialogContent className="max-h-[85vh] max-w-3xl overflow-y-auto">
          <DialogHeader>
            <DialogTitle>True “{cell?.y_true}” · predicted “{cell?.y_pred}”</DialogTitle>
            <DialogDescription>Test samples in this cell, most confident first.</DialogDescription>
          </DialogHeader>
          {samples.isLoading ? <Skeleton className="h-40" /> : <SampleGrid items={samples.data ?? []} />}
        </DialogContent>
      </Dialog>
    </Card>
  );
}

function SampleGrid({ items, caption }: { items: SampleRef[]; caption?: (s: SampleRef) => React.ReactNode }) {
  if (!items.length) return <p className="text-sm text-muted-foreground">No previews for these samples.</p>;
  return (
    <div className="grid grid-cols-[repeat(auto-fill,minmax(9rem,1fr))] gap-3">
      {items.map(s => (
        <figure key={s.id} className="overflow-hidden rounded-lg border bg-card">
          {s.thumb_url || s.url ? (
            <a href={s.url ?? s.thumb_url!} target="_blank" rel="noreferrer"><img src={s.thumb_url ?? s.url!} alt={s.name ?? 'sample'} className="aspect-square w-full object-cover" loading="lazy" /></a>
          ) : (
            <p className="line-clamp-5 p-2 text-xs">{s.text ?? s.id}</p>
          )}
          {caption && <figcaption className="p-2 text-xs">{caption(s)}</figcaption>}
        </figure>
      ))}
    </div>
  );
}

function PerClassTable({ rep }: { rep: EvalReport }) {
  const det = rep.per_class[0]?.ap50 !== undefined;
  const cols = det ? (['ap50', 'ap50_95'] as const) : (['precision', 'recall', 'f1'] as const);
  const names = { ap50: 'AP@0.5', ap50_95: 'AP@0.5:0.95', precision: 'Precision', recall: 'Recall', f1: 'F1' };
  return (
    <Card>
      <CardHeader><CardTitle>Per class</CardTitle><CardDescription>Low values point at classes that need more or cleaner examples.</CardDescription></CardHeader>
      <CardContent className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-left text-xs text-muted-foreground">
            <tr><th className="py-1">Class</th>{cols.map(c => <th key={c} className="text-right">{names[c]}</th>)}<th className="text-right">{det ? 'Boxes' : 'Support'}</th></tr>
          </thead>
          <tbody>
            {rep.per_class.map(r => (
              <tr key={r.class} className="border-t">
                <td className="py-1.5 font-medium">{r.class}</td>
                {cols.map(c => {
                  const v = r[c];
                  return <td key={c} className={cn('text-right tabular-nums', v != null && v < 0.7 && 'text-destructive')}>{v == null ? '—' : v.toFixed(3)}</td>;
                })}
                <td className="text-right tabular-nums text-muted-foreground">{r.support}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}

function RegressionCharts({ rep }: { rep: EvalReport }) {
  const r = rep.regression!;
  const hist = r.residual_hist.counts.map((c, i) => ({ bin: ((r.residual_hist.edges[i]! + r.residual_hist.edges[i + 1]!) / 2).toPrecision(3), count: c }));
  const tip = { contentStyle: { background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 } };
  return (
    <>
      <Card>
        <CardHeader><CardTitle>Predicted vs actual</CardTitle><CardDescription>Points on the diagonal are perfect predictions.</CardDescription></CardHeader>
        <CardContent className="h-72">
          <ResponsiveContainer>
            <ScatterChart margin={{ left: -8, right: 8 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
              <XAxis type="number" dataKey="y_true" name="Actual" tick={{ fontSize: 11 }} domain={['auto', 'auto']} />
              <YAxis type="number" dataKey="y_pred" name="Predicted" tick={{ fontSize: 11 }} domain={['auto', 'auto']} />
              <ZAxis range={[18, 18]} />
              <RTooltip {...tip} />
              <Scatter data={r.scatter} fill="var(--primary)" fillOpacity={0.6} isAnimationActive={false} />
            </ScatterChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle>Errors</CardTitle><CardDescription>Predicted minus actual. A narrow peak around 0 is good.</CardDescription></CardHeader>
        <CardContent className="h-72">
          <ResponsiveContainer>
            <BarChart data={hist} margin={{ left: -16, right: 8 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
              <XAxis dataKey="bin" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
              <RTooltip {...tip} />
              <Bar dataKey="count" fill="var(--primary)" isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </>
  );
}

function Mistakes({ ev, rep }: { ev: Evaluation; rep: EvalReport }) {
  const regression = rep.task_type === 'tabular_regression';
  const detection = rep.task_type === 'object_detection';
  const byId = Object.fromEntries(rep.worst_mistakes.map(w => [w.sample_id, w]));
  const items = rep.worst_mistakes.map(w => ev.samples?.[w.sample_id] ?? { id: w.sample_id });
  const boxes = (s: string) => { try { return (JSON.parse(s) as unknown[]).length; } catch { return 0; } };
  return (
    <Card>
      <CardHeader>
        <CardTitle>Worst mistakes</CardTitle>
        <CardDescription>{regression ? 'Largest errors.' : detection ? 'Images with the most missed or extra boxes.' : 'Wrong predictions the model was most sure about. Often a mislabel.'}</CardDescription>
      </CardHeader>
      <CardContent>
        {regression ? (
          <table className="w-full text-sm">
            <thead className="text-left text-xs text-muted-foreground"><tr><th className="py-1">Row</th><th className="text-right">Actual</th><th className="text-right">Predicted</th><th className="text-right">Error</th></tr></thead>
            <tbody>
              {rep.worst_mistakes.map(w => (
                <tr key={w.sample_id} className="border-t tabular-nums">
                  <td className="py-1.5 font-mono text-xs">{w.sample_id}</td><td className="text-right">{(+w.y_true).toPrecision(4)}</td>
                  <td className="text-right">{(+w.y_pred).toPrecision(4)}</td><td className="text-right text-destructive">{w.error?.toPrecision(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <SampleGrid
            items={items}
            caption={s => {
              const w = byId[s.id]!;
              return detection ? (
                <span>{boxes(w.y_true)} true · {boxes(w.y_pred)} predicted boxes</span>
              ) : (
                <>
                  <span className="text-success">{w.y_true}</span> → <span className="font-medium text-destructive">{w.y_pred}</span>
                  {w.confidence != null && <span className="text-muted-foreground"> · {(w.confidence * 100).toFixed(0)}%</span>}
                </>
              );
            }}
          />
        )}
      </CardContent>
    </Card>
  );
}
