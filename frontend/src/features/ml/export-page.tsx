import { CheckCircle2, Copy, Download, Loader2, Package, ShieldAlert, XCircle } from 'lucide-react';
import { useState } from 'react';
import { toast } from 'sonner';
import { useCreateExport, useExportPreview, useExports, useSnippets, type BundleFile, type ExportOut } from '@/api/ml';
import { useStorage } from '@/api/storage';
import { BundleContents } from './bundle-contents';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/misc';
import { useProject } from '@/api/hooks';
import { FaceExportCard } from '@/features/face/face-export';
import { formatBytes } from '@/lib/files';
import { cn } from '@/lib/utils';
import { NoModel, VersionSelect, useVersionPick } from './evaluate-page';

const LATER = [
  { name: 'TensorFlow Lite', desc: 'Android, Raspberry Pi, microcontrollers' },
  { name: 'Core ML', desc: 'iPhone, iPad, Mac' },
  { name: 'Hosted API', desc: 'One-click deploy with an API key' },
];

/** Trained models: ONNX export + snippets. Face projects: the offline gallery export. */
export function ExportPage({ projectId }: { projectId: string }) {
  const p = useProject(projectId).data;
  if (!p) return <Skeleton className="h-96 rounded-xl" />;
  return p.task_type === 'face_recognition' ? <FaceExportCard project={p} /> : <ModelExport projectId={projectId} />;
}

function ModelExport({ projectId }: { projectId: string }) {
  const { loading, ready, mv, setPicked } = useVersionPick(projectId);
  const exports = useExports(mv?.id);
  const create = useCreateExport(mv?.id);
  const snippets = useSnippets(mv?.id);
  const onnx = exports.data?.find(e => e.format === 'onnx');
  const busy = onnx?.status === 'queued' || onnx?.status === 'running';
  const built = onnx?.status === 'ready' ? (onnx.verify_report?.files as BundleFile[] | undefined) : undefined;
  const preview = useExportPreview(mv?.id, !built && mv?.framework !== 'torchvision');
  const storage = useStorage(projectId).data;
  if (loading) return <Skeleton className="h-96 rounded-xl" />;
  if (!mv) return <NoModel projectId={projectId} />;
  const excluded = 'Training data is not included: no photos, texts or dataset rows. Only the trained model, its labels and docs.';

  return (
    <div className="grid gap-5">
      <VersionSelect ready={ready} mv={mv} onPick={setPicked} />
      <div className="grid gap-5 lg:grid-cols-[1fr_20rem]">
        <Card>
          <CardHeader className="flex-row items-start justify-between gap-3">
            <div>
              <CardTitle className="flex items-center gap-2"><Package className="size-5" /> ONNX</CardTitle>
              <CardDescription>Runs anywhere with ONNX Runtime: Python, Node, browsers, C++, mobile. The bundle has the model, labels, model card and sample code.</CardDescription>
            </div>
            {mv.framework === 'torchvision' ? (
              <Badge variant="secondary" className="shrink-0" title="Object-detection ONNX export comes in Phase 2; use the playground or local API">Phase 2 for detection</Badge>
            ) : onnx?.status === 'ready' && onnx.download_url ? (
              <a href={onnx.download_url} className="inline-flex h-9 shrink-0 items-center gap-2 rounded-md bg-primary px-4 text-sm font-medium text-primary-foreground hover:bg-primary/90" download>
                <Download className="size-4" /> Download{onnx.size_bytes ? ` · ${formatBytes(onnx.size_bytes)}` : ''}
              </a>
            ) : (
              <Button disabled={busy || create.isPending || exports.isLoading} onClick={() => create.mutate(undefined, { onError: e => toast.error(e.message) })}>
                {busy || create.isPending ? <Loader2 className="animate-spin" /> : <Package />} {busy ? 'Exporting…' : onnx?.status === 'failed' ? 'Retry export' : 'Export ONNX'}
              </Button>
            )}
          </CardHeader>
          <CardContent className="grid gap-4">
            {built && onnx?.size_bytes ? (
              <BundleContents files={built} total={onnx.size_bytes} excluded={excluded} dataBytes={storage?.totals.data_bytes} />
            ) : preview.data ? (
              // Exports built before per-file sizes were recorded: estimated parts, but the real zip size.
              <BundleContents files={preview.data.files} total={onnx?.status === 'ready' && onnx.size_bytes ? onnx.size_bytes : preview.data.total_bytes} estimated excluded={excluded} dataBytes={storage?.totals.data_bytes} />
            ) : null}
            {onnx && <Verification e={onnx} />}
          </CardContent>
        </Card>

        <Card className="h-fit">
          <CardHeader><CardTitle className="text-base">More targets</CardTitle></CardHeader>
          <CardContent className="grid gap-2">
            {LATER.map(t => (
              <div key={t.name} className="flex items-center justify-between gap-2 rounded-lg border p-3 opacity-70" aria-disabled="true">
                <div><div className="text-sm font-medium">{t.name}</div><div className="text-xs text-muted-foreground">{t.desc}</div></div>
                <Badge variant="secondary">Phase 2</Badge>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      {snippets.data && <Snippets s={snippets.data} />}
    </div>
  );
}

function Verification({ e }: { e: ExportOut }) {
  if (e.status === 'failed') return <p className="flex gap-2 text-sm text-destructive" role="alert"><XCircle className="mt-0.5 size-4 shrink-0" /> {e.error ?? 'Export failed'}</p>;
  if (e.status !== 'ready') return <p className="flex items-center gap-2 text-sm text-muted-foreground"><Loader2 className="size-4 animate-spin" /> Converting and checking the ONNX model against the original…</p>;
  const r = (e.verify_report ?? {}) as { n?: number; max_abs_diff?: number; top1_agreement?: number; native_accuracy?: number; onnx_accuracy?: number };
  return (
    <div className={cn('grid gap-3 rounded-lg border p-4', e.verified ? 'border-success/40 bg-success/6' : 'border-warning/40 bg-warning/8')}>
      <div className="flex items-center gap-2 text-sm font-medium">
        {e.verified ? <CheckCircle2 className="size-4 text-success" /> : <ShieldAlert className="size-4 text-warning" />}
        {e.verified ? 'Verified: the ONNX model gives the same answers as the original' : 'Check failed: the ONNX model differs from the original'}
      </div>
      <dl className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
        <div><dt className="text-xs text-muted-foreground">Test samples checked</dt><dd className="tabular-nums">{r.n ?? '—'}</dd></div>
        <div><dt className="text-xs text-muted-foreground">Same top answer</dt><dd className="tabular-nums">{r.top1_agreement != null ? `${(r.top1_agreement * 100).toFixed(1)}%` : '—'}</dd></div>
        <div><dt className="text-xs text-muted-foreground">Max output difference</dt><dd className="tabular-nums">{r.max_abs_diff != null ? r.max_abs_diff.toExponential(1) : '—'}</dd></div>
        <div><dt className="text-xs text-muted-foreground">Accuracy change</dt><dd className="tabular-nums">{e.accuracy_delta != null ? `${e.accuracy_delta >= 0 ? '+' : ''}${(e.accuracy_delta * 100).toFixed(1)} pts` : '—'}</dd></div>
      </dl>
    </div>
  );
}

export function Snippets({ s, title = 'Use it in code', description = "Generated from this model's input spec and labels." }: { s: Record<string, string>; title?: string; description?: string }) {
  const tabs = Object.keys(s);
  const [tab, setTab] = useState(tabs[0]!);
  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between gap-3">
        <div><CardTitle className="text-base">{title}</CardTitle><CardDescription>{description}</CardDescription></div>
        <div className="flex gap-1" role="tablist">
          {tabs.map(t => (
            <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)} className={cn('rounded-md px-3 py-1 text-xs font-medium', tab === t ? 'bg-primary/12 text-primary' : 'text-muted-foreground hover:bg-accent')}>
              {t === 'curl' ? 'Local API (curl)' : t === 'js' ? 'JavaScript' : t[0]!.toUpperCase() + t.slice(1)}
            </button>
          ))}
        </div>
      </CardHeader>
      <CardContent className="relative">
        <Button size="icon" variant="ghost" className="absolute right-8 top-2" aria-label="Copy code" onClick={() => navigator.clipboard.writeText(s[tab]!).then(() => toast.success('Copied'))}><Copy /></Button>
        <pre className="overflow-x-auto rounded-lg bg-muted/60 p-4 font-mono text-xs leading-relaxed"><code>{s[tab]}</code></pre>
      </CardContent>
    </Card>
  );
}
