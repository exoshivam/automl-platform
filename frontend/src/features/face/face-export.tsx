import { Download, Loader2, PackageOpen, ShieldAlert } from 'lucide-react';
import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import type { BundleFile } from '@/api/ml';
import { useStorage } from '@/api/storage';
import { BundleContents } from '@/features/ml/bundle-contents';
import { toast } from 'sonner';
import { useFaceSettings } from '@/api/face';
import { useCurrentOrg } from '@/api/hooks';
import type { Project } from '@/api/types';
import { Button } from '@/components/ui/button';
import { Card, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Checkbox } from '@/components/ui/misc';
import { cn } from '@/lib/utils';
import { ApiError, api } from '@/lib/api';
import { formatBytes } from '@/lib/files';

/** Local export (owner/admin): ONNX detector + embedder, gallery.json (names + mean templates), runnable README. */
/** Face export contains biometric data: workspace owners and admins only (server-side too: 403 forbidden). */
export const canExportFace = (role: string | undefined) => role === 'owner' || role === 'admin';

export function FaceExportCard({ project }: { project: Project }) {
  const { orgs } = useCurrentOrg();
  const role = orgs.find(o => o.id === project.organization_id)?.role;
  const [open, setOpen] = useState(false);
  if (!canExportFace(role)) return <Card className="p-6 text-sm text-muted-foreground">Only workspace owners and admins can export face data.</Card>;
  return (
    <Card>
      <CardHeader className="flex-row flex-wrap items-center justify-between gap-3">
        <div>
          <CardTitle className="flex items-center gap-2 text-base"><PackageOpen className="size-5" /> Run it on your own computer</CardTitle>
          <CardDescription>Download the face models and this gallery to recognise people offline with a short Python script. Contains biometric data.</CardDescription>
        </div>
        <Button variant="outline" size="sm" onClick={() => setOpen(true)}><Download /> Export…</Button>
      </CardHeader>
      {open && <FaceExportDialog project={project} onClose={() => setOpen(false)} />}
    </Card>
  );
}

type Pack = 'accurate' | 'small';
interface Quality { photos: number; recognised: number; rate: number; mean_gap: number; people_checked: number }
interface Preview {
  people: number;
  photos_excluded: number;
  files: BundleFile[];
  total_bytes: number;
  filename: string;
  pack: Pack;
  pack_label: string;
  available: boolean;
  threshold: number;
  threshold_source: 'gallery' | 'calibrated' | 'default';
  quality: Quality | null;
  /** Enrolled photos this pack's detector found no face in (their person may be missing a template). */
  not_detected?: number;
}
const PACKS: { id: Pack; label: string; desc: string }[] = [
  { id: 'accurate', label: 'Accurate', desc: 'More people, hard lighting' },
  { id: 'small', label: 'Small', desc: 'Phones and small devices, slightly less accurate' },
];
const qualityLine = (d: Preview) =>
  (d.quality ? `Recognised ${d.quality.recognised}/${d.quality.photos} enrolled photos · gap to nearest other person ${d.quality.mean_gap.toFixed(2)}` : 'Accuracy check needs 2+ people to compare') +
  (d.not_detected ? ` · no face found in ${d.not_detected} photo${d.not_detected === 1 ? '' : 's'}` : '');
interface ExportOut { download_url: string; filename: string; size_bytes: number; sha256: string; people: number; created_at: string }
const EXPORT_ERRORS: Record<string, string> = {
  gallery_empty: 'Add at least one person with photos before exporting.',
  acknowledgement_required: 'Please confirm the biometric data notice first.',
  forbidden: 'Only workspace owners and admins can export face data.',
  pack_unavailable: 'The small model pack isn’t installed on this server.',
};

function FaceExportDialog({ project, onClose }: { project: Project; onClose: () => void }) {
  const s = useFaceSettings(project.id).data;
  const [ack, setAck] = useState(false);
  const [busy, setBusy] = useState(false);

  const [done, setDone] = useState<ExportOut | null>(null);
  const [pack, setPack] = useState<Pack>('accurate');
  // Accurate is instant; Small re-embeds the gallery the first time (cached server-side), so it loads second.
  const opts = (p: Pack, enabled = true) => ({
    queryKey: ['face-export-preview', project.id, p],
    queryFn: () => api<Preview>(`/projects/${project.id}/face-export/preview`, { query: { pack: p } }),
    retry: false,
    enabled,
    staleTime: 60_000,
  });
  const accurate = useQuery(opts('accurate'));
  const small = useQuery(opts('small', accurate.isSuccess));
  const previews = { accurate, small };
  const preview = previews[pack];
  // Photos only (gallery totals also hold earlier exports, which would overstate the data).
  const dataBytes = useStorage(project.id).data?.totals.gallery_photos_bytes;
  const pv = preview.data;
  const size = pv ? formatBytes(pv.total_bytes) : null;
  const download = async () => {
    setBusy(true);
    try {
      const r = await api<ExportOut>(`/projects/${project.id}/face-export`, { method: 'POST', body: { acknowledge_biometric: true, pack } });
      Object.assign(document.createElement('a'), { href: r.download_url, download: r.filename }).click();
      setDone(r);
    } catch (e) {
      toast.error(e instanceof ApiError ? (EXPORT_ERRORS[e.problem.code ?? ''] ?? e.message) : 'Export failed');
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open onOpenChange={o => !o && onClose()}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Export face recognition</DialogTitle>
          <DialogDescription>A zip with the two face models, this gallery and a ready-to-run Python script.</DialogDescription>
        </DialogHeader>
        <div className="grid gap-2 sm:grid-cols-2" role="radiogroup" aria-label="Model pack">
          {PACKS.map(o => {
            const q = previews[o.id];
            const d = q.data;
            const off = !!d && !d.available;
            return (
              <button
                key={o.id}
                type="button"
                role="radio"
                aria-checked={pack === o.id}
                disabled={off || busy}
                onClick={() => (setPack(o.id), setDone(null))}
                className={cn('grid gap-1 rounded-lg border p-3 text-left text-sm transition-colors', pack === o.id ? 'border-primary bg-primary/6 ring-2 ring-primary/20' : 'hover:bg-accent/50', off && 'cursor-not-allowed opacity-60')}
              >
                <span className="flex items-baseline justify-between gap-2 font-medium">
                  {o.label}
                  <span className="text-xs font-normal tabular-nums text-muted-foreground">{d ? formatBytes(d.total_bytes) : o.id === 'small' ? '~16 MB' : '~182 MB'}</span>
                </span>
                <span className="text-xs text-muted-foreground">{o.desc}</span>
                <span className="text-xs tabular-nums" aria-live="polite">
                  {off ? 'Not installed on this server' : d ? qualityLine(d) : q.isFetching ? <><Loader2 className="inline size-3 animate-spin" /> {o.id === 'small' ? 'Checking with the small model…' : 'Checking…'}</> : q.error ? 'Couldn’t check' : ''}
                </span>
              </button>
            );
          })}
        </div>
        {pack === 'small' && (
          <p className="text-xs text-muted-foreground" role="note">Face templates are rebuilt from your enrolled photos with the small model during export (photos are not included).{pv && pv.threshold_source !== 'gallery' ? ` Match threshold ${pv.threshold.toFixed(2)} (${pv.threshold_source === 'calibrated' ? 'calibrated for the small model' : 'default: too few photos to calibrate'}).` : ''}</p>
        )}
        <div className="flex gap-3 rounded-lg border border-destructive/40 bg-destructive/6 p-3 text-sm" role="alert">
          <ShieldAlert className="mt-0.5 size-5 shrink-0 text-destructive" />
          <div className="grid gap-1">
            <p className="font-medium">This file contains face data (biometric) for {pv?.people ?? s?.people_count ?? 'all enrolled'} {(pv?.people ?? s?.people_count) === 1 ? 'person' : 'people'}.</p>
            <p>Anyone with the file can recognise them. Removing a person here won't remove them from files you already downloaded. Models are licensed for non-commercial research only.</p>
          </div>
        </div>
        {pv ? (
          <BundleContents
            files={pv.files}
            total={pv.total_bytes}
            excluded={`Training photos are not included: only the two models and ${pv.people} ${pv.people === 1 ? "person's" : "people's"} face templates (one per person). The ${pv.photos_excluded} enrolled photos stay on the server.`}
            dataBytes={dataBytes}
          />
        ) : preview.isLoading ? (
          <p className="text-xs text-muted-foreground"><Loader2 className="inline size-3 animate-spin" /> Working out the download size…</p>
        ) : preview.error ? (
          <p className="text-xs text-destructive">{preview.error instanceof ApiError ? (EXPORT_ERRORS[preview.error.problem.code ?? ''] ?? preview.error.message) : 'Couldn’t get the download size'}</p>
        ) : null}
        <label className="flex items-start gap-3 text-sm">
          <Checkbox checked={ack} onCheckedChange={v => setAck(v === true)} className="mt-0.5" />
          <span>I understand, and I'll store this file securely and use it only for the purpose these people consented to.</span>
        </label>
        <div className="grid gap-1.5 text-sm">
          <span className="font-medium">How to run it</span>
          <pre className="overflow-x-auto rounded-lg bg-muted/60 p-3 font-mono text-xs leading-relaxed">{`unzip ${done?.filename ?? 'face-gallery-*.zip'} -d face-export && cd face-export
pip install onnxruntime numpy pillow        # + opencv-python for --camera
python identify.py photo.jpg                # or: python identify.py --camera`}</pre>
          <span className="text-xs text-muted-foreground">The README inside has the details and the exact script.</span>
        </div>
        {done && (
          <p className="text-xs text-muted-foreground" role="status">
            Downloaded {done.filename} ({formatBytes(done.size_bytes)}, {done.people} people). SHA-256 <span className="break-all font-mono">{done.sha256}</span>. The link expires in 15 minutes.
          </p>
        )}
        <DialogFooter>
          <Button variant="ghost" onClick={onClose}>{done ? 'Close' : 'Cancel'}</Button>
          <Button disabled={!ack || busy} onClick={download}>
            {busy ? <Loader2 className="animate-spin" /> : <Download />} {busy ? `Preparing${size ? ` ${size}` : ''}…` : done ? 'Download again' : `Download${size ? ` · ${size}` : ''}`}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
