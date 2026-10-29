import { useQueryClient } from '@tanstack/react-query';
import { AlertTriangle, FileUp, Loader2, Plus, Trash2 } from 'lucide-react';
import { useState } from 'react';
import { toast } from 'sonner';
import { useLabelSetChange } from '@/api/data-hooks';
import { useProject } from '@/api/hooks';
import { AccuracyLine } from './text-pages';
import { TRAIN_READY, useAddExample, useDeleteExample, useExamples, useTextReport, useTextSettings, type Example, type TextReport } from '@/api/text';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input, Textarea } from '@/components/ui/input';
import { Progress, Skeleton } from '@/components/ui/misc';
import { ApiError } from '@/lib/api';
import { cn } from '@/lib/utils';

const GOOD = TRAIN_READY; // 10+ varied examples per label

/** Quick text labels: a few example texts per label, instantly used for matching (no training). */
export function ExamplesPage({ projectId }: { projectId: string }) {
  const project = useProject(projectId).data;
  const settings = useTextSettings(projectId);
  const examples = useExamples(projectId);
  const report = useTextReport(projectId);
  const addLabel = useLabelSetChange();
  const qc = useQueryClient();
  const [newLabel, setNewLabel] = useState('');
  if (settings.isLoading || examples.isLoading) return <Skeleton className="h-96 rounded-xl" />;
  const labels = settings.data?.labels ?? [];
  const byLabel = (l: string) => (examples.data ?? []).filter(e => e.label === l);

  const createLabel = () => {
    const name = newLabel.trim();
    if (!name || !project?.label_set_id) return;
    addLabel.mutate(
      { id: project.label_set_id, op: 'classes', body: { add: [name] } },
      { onSuccess: () => (setNewLabel(''), qc.invalidateQueries({ queryKey: ['text-settings', projectId] }), qc.invalidateQueries({ queryKey: ['text-report', projectId] }), toast.success(`Label “${name}” added`)), onError: e => toast.error(e.message) },
    );
  };

  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">Examples</h2>
          <p className="text-sm text-muted-foreground">Add <b>10 or more varied examples per label</b>: different words, Hinglish and Hindi script, short and long. Changes apply instantly; there’s no training step.</p>
        </div>
        <form className="flex gap-2" onSubmit={e => (e.preventDefault(), createLabel())}>
          <Input value={newLabel} onChange={e => setNewLabel(e.target.value)} placeholder="New label" className="h-9 w-44" aria-label="New label name" />
          <Button type="submit" variant="outline" disabled={!newLabel.trim() || addLabel.isPending}><Plus /> Add label</Button>
        </form>
      </div>
      {!report.isFetching && report.data?.ready && report.data.loo_accuracy != null && <AccuracyLine loo={report.data.loo_accuracy} projectId={projectId} />}
      {report.isFetching ? (
        <p className="flex items-center gap-2 text-sm text-muted-foreground" role="status"><Loader2 className="size-3.5 animate-spin" /> Checking your examples…</p>
      ) : report.data && !report.data.ready && report.data.warnings.map(w => <p key={w.code} className="text-sm text-muted-foreground">{w.detail}</p>)}
      <div className="grid gap-4 lg:grid-cols-2 xl:grid-cols-3">
        {labels.map(l => <LabelCard key={l} projectId={projectId} label={l} items={byLabel(l)} report={report.isFetching ? undefined : report.data} />)}
      </div>
    </div>
  );
}

function LabelCard({ projectId, label, items, report }: { projectId: string; label: string; items: Example[]; report?: TextReport }) {
  const add = useAddExample(projectId);
  const del = useDeleteExample(projectId);
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const warnings = report?.per_label.find(p => p.label === label)?.warnings ?? [];
  const n = items.length;

  /** One example per line; duplicates are skipped quietly. */
  const addMany = async (lines: string[]) => {
    const clean = [...new Set(lines.map(t => t.trim()).filter(Boolean))].slice(0, 500);
    if (!clean.length) return;
    setBusy(true);
    let ok = 0, dup = 0, failed = 0;
    for (const t of clean) {
      try {
        await add.mutateAsync({ text: t.slice(0, 5000), label });
        ok++;
      } catch (e) {
        if (e instanceof ApiError && e.problem.code === 'duplicate_example') dup++;
        else failed++;
      }
    }
    setBusy(false);
    setText('');
    toast[failed ? 'error' : 'success'](`${label}: added ${ok}${dup ? `, ${dup} already there` : ''}${failed ? `, ${failed} failed` : ''}`);
  };

  const importCsv = async (f: File) => {
    // One column = texts for this label; two columns = text,label (rows for this label only).
    const rows = (await f.text()).split(/\r?\n/).map(r => r.trim()).filter(Boolean);
    const parsed = rows.map(r => {
      const m = r.match(/^"((?:[^"]|"")*)"\s*,\s*(.+)$/) ?? r.match(/^([^,]*),\s*(.+)$/);
      return m ? { t: m[1]!.replace(/""/g, '"'), l: m[2]!.replace(/^"|"$/g, '').trim() } : { t: r, l: label };
    });
    const header = /^(text|example)s?$/i.test(parsed[0]?.t ?? '');
    await addMany(parsed.slice(header ? 1 : 0).filter(p => p.l.toLowerCase() === label.toLowerCase() || !rows[0]!.includes(',')).map(p => p.t));
  };

  return (
    <Card className="grid content-start gap-3 p-4">
      <div className="flex items-center justify-between gap-2">
        <span className="font-semibold">{label}</span>
        <Badge variant={n >= GOOD ? 'success' : n ? 'warning' : 'secondary'}>{n} {n === 1 ? 'example' : 'examples'}</Badge>
      </div>
      <Progress value={Math.min(100, (n / TRAIN_READY) * 100)} className="h-1.5" indicatorClassName={n >= GOOD ? 'bg-success' : undefined} aria-label={`${n} of ${TRAIN_READY} recommended examples`} />
      {n < GOOD && <p className="text-xs text-muted-foreground">{GOOD - n} more to reach {GOOD}. Vary the wording; don’t repeat the same sentence.</p>}
      {warnings.map((w, i) => (
        <p key={`${w.code}-${i}`} className="flex gap-1.5 text-xs text-warning"><AlertTriangle className="mt-0.5 size-3.5 shrink-0" /> {w.detail}</p>
      ))}
      <Textarea value={text} onChange={e => setText(e.target.value)} rows={3} placeholder={`Type or paste examples for “${label}”, one per line`} aria-label={`Examples for ${label}`}
        onKeyDown={e => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) void addMany(text.split('\n')); }} />
      <div className="flex flex-wrap gap-2">
        <Button size="sm" onClick={() => void addMany(text.split('\n'))} disabled={!text.trim() || busy}>{busy ? <Loader2 className="animate-spin" /> : <Plus />} Add {text.split('\n').filter(t => t.trim()).length > 1 ? `${text.split('\n').filter(t => t.trim()).length} examples` : 'example'}</Button>
        <label className="inline-flex h-8 cursor-pointer items-center gap-1.5 rounded-md border px-3 text-xs font-medium hover:bg-accent">
          <FileUp className="size-3.5" /> Import CSV
          <input type="file" accept=".csv,.txt,text/csv,text/plain" className="sr-only" onChange={e => { const f = e.target.files?.[0]; if (f) void importCsv(f); e.target.value = ''; }} />
        </label>
      </div>
      <ul className={cn('grid max-h-56 gap-1 overflow-y-auto text-sm', !n && 'hidden')}>
        {items.map(x => (
          <li key={x.id} className="group flex items-start gap-2 rounded px-1 py-0.5 hover:bg-muted/50">
            <span className="flex-1 break-words">{x.text}</span>
            <button type="button" aria-label="Delete example" onClick={() => del.mutate(x.id, { onError: e => toast.error(e.message) })} className="opacity-0 transition-opacity group-hover:opacity-100 focus:opacity-100">
              <Trash2 className="size-3.5 text-muted-foreground" />
            </button>
          </li>
        ))}
      </ul>
    </Card>
  );
}
