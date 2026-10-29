import { Link } from '@tanstack/react-router';
import { CheckCircle2, Circle, GraduationCap, Info, ListChecks, Loader2, MessageSquareText, Play, Save, Smile, Tags, Wand2 } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { toast } from 'sonner';
import { useProject } from '@/api/hooks';
import { LOO_OK, TRAIN_READY, isSentimentOnly, predictText, useExamples, useSaveTextSettings, useTextReport, useTextSettings, type Sentiment, type TextPrediction } from '@/api/text';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/input';
import { Skeleton, Switch } from '@/components/ui/misc';
import { cn } from '@/lib/utils';

/** THE honest number: leave-one-out accuracy on the user's own examples, with a nudge when it's low. */
export function AccuracyLine({ loo, projectId }: { loo: number; projectId: string }) {
  const low = loo < LOO_OK;
  return (
    <div className={cn('flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg border p-3 text-sm', low ? 'border-warning/40 bg-warning/8' : 'border-success/40 bg-success/6')} role="status">
      <span><span className="text-2xl font-semibold tabular-nums">{Math.round(loo * 100)}%</span> correct when each of your examples is checked against the others.</span>
      {low && (
        <span className="text-muted-foreground">
          Expect lower on new wording. Add more varied examples, or{' '}
          <Link to="/projects/$projectId/dataset" params={{ projectId }} className="font-medium text-primary hover:underline">train for better accuracy</Link>.
        </span>
      )}
    </div>
  );
}

const SENTI: Record<Sentiment['label'], string> = { positive: 'bg-success/14 text-success', neutral: 'bg-muted text-muted-foreground', negative: 'bg-destructive/12 text-destructive' };
export function SentimentChip({ s }: { s?: Sentiment }) {
  if (!s) return null;
  const strong = s.detail?.startsWith('very ');
  return (
    <span className={cn('inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium', SENTI[s.label])} title={s.detail ? `Model says: ${s.detail}` : undefined}>
      <Smile className="size-3" /> {strong ? s.detail : s.label} {Math.round(s.score * 100)}%
    </span>
  );
}

export function TextSettingsPage({ projectId }: { projectId: string }) {
  const q = useTextSettings(projectId);
  const report = useTextReport(projectId);
  const save = useSaveTextSettings(projectId);
  const [t, setT] = useState<number | null>(null);
  useEffect(() => { if (q.data && t === null) setT(q.data.threshold); }, [q.data, t]);
  if (q.isLoading || t === null) return <Skeleton className="h-80 rounded-xl" />;
  if (!q.data) return <Card className="p-6 text-sm text-destructive">{q.error?.message}</Card>;
  const s = q.data;
  const r = report.data;
  const dirty = Math.abs(t - s.threshold) > 1e-6;
  const sentiment = s.ready_made.includes('sentiment');
  return (
    <div className="grid gap-5 lg:grid-cols-[22rem_1fr]">
      {s.labels.length > 0 && (
        <Card className="h-fit">
          <CardHeader>
            <CardTitle>Match threshold</CardTitle>
            <CardDescription>How close a text must be to your examples before it gets a label. <b>Stricter</b> (higher) = more texts go to “Other”, fewer wrong labels. <b>Looser</b> = labels more texts, with more mistakes.</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4">
            <div className="flex items-baseline justify-between">
              <span className="text-3xl font-semibold tabular-nums">{Math.round(t * 100)}%</span>
              <span className="text-xs text-muted-foreground">default {Math.round(s.default_threshold * 100)}%</span>
            </div>
            <input type="range" min={0.05} max={0.95} step={0.01} value={t} onChange={e => setT(Number(e.target.value))} aria-label="Match threshold" className="w-full accent-[var(--primary)]" />
            <div className="flex justify-between text-xs text-muted-foreground"><span>Looser</span><span>Stricter (more “Other”)</span></div>
            <div className="flex flex-wrap gap-2">
              {r?.suggested_threshold != null && <Button size="sm" variant="outline" onClick={() => setT(r.suggested_threshold!)}><Wand2 /> Use suggested ({Math.round(r.suggested_threshold * 100)}%)</Button>}
              <Button size="sm" variant="ghost" onClick={() => setT(s.default_threshold)}>Reset to default</Button>
            </div>
            <Button disabled={!dirty || save.isPending} onClick={() => save.mutate({ threshold: +t.toFixed(2) }, { onSuccess: () => toast.success(`Threshold set to ${Math.round(t * 100)}%`), onError: e => toast.error(e.message) })}>
              {save.isPending ? <Loader2 className="animate-spin" /> : <Save />} {dirty ? 'Save threshold' : 'Saved'}
            </Button>
            {r?.ready && r.loo_accuracy != null && <AccuracyLine loo={r.loo_accuracy} projectId={projectId} />}
          </CardContent>
        </Card>
      )}
      <Card className="h-fit">
        <CardHeader>
          <CardTitle>Ready-made sentiment</CardTitle>
          <CardDescription>Adds positive / negative / neutral to every result, including Hindi and Hinglish.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3">
          <label className="flex items-center gap-2 text-sm">
            <Switch checked={sentiment} disabled={save.isPending || (sentiment && !s.labels.length)} onCheckedChange={v => save.mutate({ ready_made: v ? ['sentiment'] : [] }, { onError: e => toast.error(e.message) })} />
            Show sentiment
          </label>
          {sentiment && !s.labels.length && <p className="text-xs text-muted-foreground">This project is sentiment-only, so sentiment stays on.</p>}
          {sentiment && (
            <p className="flex gap-2 rounded-lg border border-warning/40 bg-warning/8 p-2.5 text-xs">
              <Info className="mt-0.5 size-3.5 shrink-0 text-warning" />
              <span>Strong for English and Hindi; weaker for Hinglish (roman-script Hindi), especially complaints. For Hinglish you'll get better results by making positive / negative / neutral your own labels with examples, or training your own on your data.</span>
            </p>
          )}
          <p className="flex gap-2 text-xs text-muted-foreground"><Info className="mt-0.5 size-3.5 shrink-0" /> Matching: {s.embedding_model.id} ({s.embedding_model.licence}){s.sentiment_model ? ` · Sentiment: ${s.sentiment_model.id} (${s.sentiment_model.licence})` : ''}{s.has_trained_model ? ' · using your trained model' : ''}</p>
        </CardContent>
      </Card>
    </div>
  );
}

/** "Other (best: refund 41%)": score is always the best user label's similarity, also when other=true. */
const otherText = (r: TextPrediction) => (r.top3[0] ? `Other (best: ${r.top3[0].label} ${Math.round(r.top3[0].score * 100)}%)` : 'Other');

function Prediction({ r }: { r: TextPrediction }) {
  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center gap-2">
        {r.source !== 'sentiment_only' && (r.other ? (
          <span className="text-2xl font-semibold text-muted-foreground" title={r.background_score != null ? `Everyday-text similarity ${Math.round(r.background_score * 100)}%` : undefined}>{otherText(r)}</span>
        ) : (
          <>
            <span className="text-2xl font-semibold">{r.label}</span>
            <span className="text-sm text-muted-foreground tabular-nums">{Math.round(r.score * 100)}%</span>
          </>
        ))}
        <SentimentChip s={r.sentiment} />
        <Badge variant="secondary" className="ml-auto text-[10px]">{r.source === 'model' ? 'your trained model' : r.source === 'examples' ? 'from your examples' : 'ready-made'}</Badge>
      </div>
      {r.other && <p className="text-xs text-muted-foreground">Not close enough to any label (or more like everyday chat). Add examples like this text if it should have a label.</p>}
      {r.top3.length > 0 && (
        <ul className="grid gap-1.5">
          {r.top3.map(c => (
            <li key={c.label} className="grid gap-0.5">
              <div className="flex justify-between text-xs"><span>{c.label}</span><span className="tabular-nums">{Math.round(c.score * 100)}%</span></div>
              <div className="h-2 overflow-hidden rounded-full bg-muted"><div className="h-full rounded-full bg-primary" style={{ width: `${Math.max(2, c.score * 100)}%` }} /></div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function TextPlayground({ projectId }: { projectId: string }) {
  const [mode, setMode] = useState<'single' | 'batch'>('single');
  const [text, setText] = useState('');
  const [one, setOne] = useState<TextPrediction | null>(null);
  const [rows, setRows] = useState<{ text: string; r: TextPrediction | null; err?: string }[]>([]);
  const [busy, setBusy] = useState(false);
  const ctl = useRef<AbortController | null>(null);
  useEffect(() => () => ctl.current?.abort(), []); // leaving the page cancels in-flight requests
  const run = async () => {
    if (!text.trim()) return;
    ctl.current?.abort();
    ctl.current = new AbortController();
    const signal = ctl.current.signal;
    setBusy(true);
    try {
      if (mode === 'single') setOne(await predictText(projectId, text.trim(), signal));
      else {
        const lines = text.split('\n').map(l => l.trim()).filter(Boolean).slice(0, 200);
        setRows(lines.map(l => ({ text: l, r: null })));
        for (let i = 0; i < lines.length && !signal.aborted; i++) {
          try {
            const r = await predictText(projectId, lines[i]!, signal);
            setRows(rs => rs.map((x, j) => (j === i ? { ...x, r } : x)));
          } catch (e) {
            if (!signal.aborted) setRows(rs => rs.map((x, j) => (j === i ? { ...x, err: (e as Error).message } : x)));
          }
        }
      }
    } catch (e) {
      if (!signal.aborted) toast.error((e as Error).message);
    } finally {
      if (!signal.aborted) setBusy(false);
    }
  };
  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_22rem]">
      <Card>
        <CardHeader className="flex-row flex-wrap items-center justify-between gap-2">
          <div><CardTitle className="text-base">Try it</CardTitle><CardDescription>{mode === 'single' ? 'Type a text and press Ctrl/⌘ + Enter.' : 'Paste one text per line (up to 200).'}</CardDescription></div>
          <div className="flex gap-1" role="tablist">
            {(['single', 'batch'] as const).map(m => (
              <button key={m} role="tab" aria-selected={mode === m} onClick={() => setMode(m)} className={cn('rounded-md px-3 py-1 text-xs font-medium', mode === m ? 'bg-primary/12 text-primary' : 'text-muted-foreground hover:bg-accent')}>
                {m === 'single' ? 'One text' : 'Try a batch'}
              </button>
            ))}
          </div>
        </CardHeader>
        <CardContent className="grid gap-3">
          <Textarea value={text} onChange={e => setText(e.target.value)} rows={mode === 'single' ? 4 : 8} aria-label="Text to classify"
            placeholder={mode === 'single' ? 'e.g. paisa wapas kab milega? order cancel kiya tha' : 'one text per line'}
            onKeyDown={e => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) void run(); }} />
          <Button className="justify-self-start" onClick={() => void run()} disabled={!text.trim() || busy}>{busy ? <Loader2 className="animate-spin" /> : <Play />} {mode === 'single' ? 'Classify' : 'Classify all'}</Button>
          {mode === 'batch' && rows.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="text-left text-xs text-muted-foreground"><tr><th className="py-1 pr-3">Text</th><th className="pr-3">Label</th><th className="pr-3 text-right">Score</th><th>Sentiment</th></tr></thead>
                <tbody>
                  {rows.map((x, i) => (
                    <tr key={i} className="border-t align-top">
                      <td className="max-w-md py-1.5 pr-3">{x.text}</td>
                      <td className={cn('pr-3 font-medium', x.r?.other && 'text-muted-foreground')}>{x.err ? <span className="text-destructive">{x.err}</span> : !x.r ? <Loader2 className="size-3.5 animate-spin" /> : x.r.source === 'sentiment_only' ? '—' : x.r.other ? otherText(x.r) : x.r.label}</td>
                      <td className="pr-3 text-right tabular-nums">{x.r && x.r.source !== 'sentiment_only' && !x.r.other ? `${Math.round(x.r.score * 100)}%` : ''}</td>
                      <td><SentimentChip s={x.r?.sentiment} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
      {mode === 'single' && (
        <Card className="h-fit" aria-live="polite">
          <CardHeader className="flex-row items-center justify-between">
            <CardTitle className="text-base">Result</CardTitle>
            {one && <span className="text-xs text-muted-foreground tabular-nums">{one.latency_ms} ms{one.model_load_ms ? ` · model loaded in ${(one.model_load_ms / 1000).toFixed(1)} s` : ''}</span>}
          </CardHeader>
          <CardContent>{one ? <Prediction r={one} /> : <p className="text-sm text-muted-foreground">{busy ? 'Loading the model… the first text after a few idle minutes takes a moment.' : 'The label, top-3 matches and sentiment appear here.'}</p>}</CardContent>
        </Card>
      )}
    </div>
  );
}

export function TextOverview({ projectId }: { projectId: string }) {
  const project = useProject(projectId).data;
  const s = useTextSettings(projectId).data;
  const ex = useExamples(projectId).data ?? [];
  const sentimentOnly = isSentimentOnly(project);
  const report = useTextReport(projectId, !isSentimentOnly(project));
  const labels = s?.labels ?? [];
  const per = labels.map(l => ex.filter(e => e.label === l).length);
  const enough = labels.length >= 2 && per.every(n => n >= TRAIN_READY);
  const trainable = labels.length >= 2 && per.every(n => n >= TRAIN_READY);
  const steps = sentimentOnly
    ? [{ label: 'Try it', desc: 'Positive / negative / neutral, any language', icon: Wand2, done: false, to: '/projects/$projectId/playground' as const }]
    : [
        { label: 'Add labels', desc: labels.length ? labels.join(', ') : 'Name what to sort into', icon: Tags, done: labels.length >= 2, to: '/projects/$projectId/examples' as const },
        { label: 'Add examples', desc: labels.length ? `${Math.min(...per) === Math.max(...per) ? Math.min(...per) : `${Math.min(...per)}–${Math.max(...per)}`} per label (aim for ${TRAIN_READY}+ varied)` : `${TRAIN_READY}+ varied per label`, icon: ListChecks, done: enough, to: '/projects/$projectId/examples' as const },
        { label: 'Try it', desc: 'One text or a batch', icon: Wand2, done: false, to: '/projects/$projectId/playground' as const },
      ];
  return (
    <div className="grid gap-6">
      {!sentimentOnly && report.data?.ready && report.data.loo_accuracy != null && <AccuracyLine loo={report.data.loo_accuracy} projectId={projectId} />}
      <ol className={cn('grid gap-3', sentimentOnly ? 'sm:grid-cols-1 max-w-sm' : 'sm:grid-cols-3')} aria-label="Steps">
        {steps.map(x => (
          <li key={x.label}>
            <Card className={cn('h-full p-4', x.done && 'border-success/40')}>
              <div className="flex items-center justify-between"><x.icon className="size-5 text-muted-foreground" />{x.done ? <CheckCircle2 className="size-4 text-success" /> : <Circle className="size-4 text-muted-foreground/50" />}</div>
              <div className="mt-3 font-medium">{x.label}</div>
              <p className="truncate text-xs text-muted-foreground">{x.desc}</p>
              <Link to={x.to} params={{ projectId }} className="mt-3 inline-block text-xs font-medium text-primary hover:underline">Open →</Link>
            </Card>
          </li>
        ))}
      </ol>
      {!sentimentOnly && (
        <Card className={cn(trainable && 'border-primary/40')}>
          <CardHeader className="flex-row flex-wrap items-center justify-between gap-3">
            <div>
              <CardTitle className="flex items-center gap-2 text-base"><GraduationCap className="size-5" /> Train for better accuracy</CardTitle>
              <CardDescription>
                {s?.has_trained_model
                  ? 'A trained model is in use. Add more examples and retrain any time.'
                  : trainable
                    ? 'Trains a model on your examples (a few minutes). It learns the pattern behind them, so it usually handles new wording better than matching examples does.'
                    : `Available once every label has about ${TRAIN_READY} examples (the fewest now: ${per.length ? Math.min(...per) : 0}). Until then, matching your examples works instantly.`}
              </CardDescription>
            </div>
            <Link to="/projects/$projectId/dataset" params={{ projectId }} className={cn('inline-flex h-9 items-center gap-2 rounded-md px-4 text-sm font-medium', trainable || s?.has_trained_model ? 'bg-primary text-primary-foreground hover:bg-primary/90' : 'pointer-events-none border opacity-50')} aria-disabled={!(trainable || s?.has_trained_model)}>
              <MessageSquareText className="size-4" /> Review & train
            </Link>
          </CardHeader>
        </Card>
      )}
    </div>
  );
}
