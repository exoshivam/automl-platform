import { AlertTriangle, Camera, CameraOff, CheckCircle2, ImageUp, Loader2, Plus, ShieldCheck, Trash2, UserRound, XCircle } from 'lucide-react';
import { useState } from 'react';
import { grabFrame, useCamera } from '@/lib/camera';
import { toast } from 'sonner';
import { REJECT_TEXT, useAddPerson, useDeletePerson, useDeletePhoto, usePeople, usePerson, useUploadPhotos, type Person, type PhotoUpload } from '@/api/face';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Checkbox, DemoBadge, Skeleton } from '@/components/ui/misc';
import { ApiError } from '@/lib/api';
import { timeAgo } from '@/lib/utils';

const MIN_PHOTOS = 3;
const MAX_PHOTOS = 10;

/** FACE-2/3/6/8: enrol people (consent first), add 3–10 photos, remove instantly. No training. */
export function PeoplePage({ projectId }: { projectId: string }) {
  const people = usePeople(projectId);
  const [adding, setAdding] = useState(false);
  const [open, setOpen] = useState<string | null>(null);

  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="flex items-center gap-2 text-lg font-semibold">People <DemoBadge group="face" /></h2>
          <p className="text-sm text-muted-foreground">Each person needs their consent and {MIN_PHOTOS}–{MAX_PHOTOS} clear photos of their face. Adding or removing someone takes effect immediately; there's nothing to retrain.</p>
        </div>
        <Button onClick={() => setAdding(true)}><Plus /> Add person</Button>
      </div>
      {people.isLoading ? (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{[0, 1, 2].map(i => <Skeleton key={i} className="h-28 rounded-xl" />)}</div>
      ) : !people.data?.length ? (
        <Card className="flex flex-col items-center gap-3 border-dashed p-12 text-center">
          <UserRound className="size-8 text-muted-foreground" />
          <p className="font-medium">No people yet</p>
          <p className="max-w-sm text-sm text-muted-foreground">Add the people this model should recognise. Anyone else will show as “Other”.</p>
          <Button onClick={() => setAdding(true)}><Plus /> Add person</Button>
        </Card>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {people.data.map(p => <PersonCard key={p.id} p={p} onOpen={() => setOpen(p.id)} />)}
        </div>
      )}
      {adding && <AddPersonDialog projectId={projectId} onClose={() => setAdding(false)} />}
      {open && <PersonDialog projectId={projectId} personId={open} onClose={() => setOpen(null)} />}
    </div>
  );
}

function PersonCard({ p, onOpen }: { p: Person; onOpen: () => void }) {
  const detail = usePerson(p.id);
  const cover = detail.data?.photos[0]?.thumb_url;
  return (
    <button type="button" onClick={onOpen} className="flex gap-3 rounded-xl border bg-card p-3 text-left transition-colors hover:bg-accent/50 focus-visible:ring-2 focus-visible:ring-ring">
      <span className="grid size-16 shrink-0 place-items-center overflow-hidden rounded-lg bg-muted">
        {cover ? <img src={cover} alt="" className="size-full object-cover" /> : <UserRound className="size-6 text-muted-foreground" />}
      </span>
      <span className="grid min-w-0 content-start gap-1">
        <span className="truncate font-medium">{p.name}</span>
        <span className="text-xs text-muted-foreground tabular-nums">{p.photo_count} / {MAX_PHOTOS} photos · consent {timeAgo(p.consent.granted_at)}</span>
        {p.warnings.map(w => (
          <span key={w.code} className="flex items-center gap-1 text-xs text-warning"><AlertTriangle className="size-3" /> {w.detail}</span>
        ))}
      </span>
    </button>
  );
}

function AddPersonDialog({ projectId, onClose }: { projectId: string; onClose: () => void }) {
  const add = useAddPerson(projectId);
  const [name, setName] = useState('');
  const [ref, setRef] = useState('');
  const [consent, setConsent] = useState(false);
  const [person, setPerson] = useState<Person | null>(null);
  const [err, setErr] = useState<string>();

  return (
    <Dialog open onOpenChange={o => !o && onClose()}>
      <DialogContent className="max-h-[90vh] max-w-2xl overflow-y-auto">
        {!person ? (
          <form
            className="grid gap-4"
            onSubmit={e => {
              e.preventDefault();
              add.mutate(
                { name: name.trim(), external_ref: ref.trim() || undefined, consent: { attested: true } },
                { onSuccess: setPerson, onError: e => setErr(e instanceof ApiError && e.problem.code === 'consent_required' ? 'Consent is required before adding someone.' : e.message) },
              );
            }}
          >
            <DialogHeader>
              <DialogTitle>Add person</DialogTitle>
              <DialogDescription>Step 1 of 2: who they are, and their consent.</DialogDescription>
            </DialogHeader>
            <label className="grid gap-1.5 text-sm">
              <span className="font-medium">Name</span>
              <Input value={name} onChange={e => setName(e.target.value)} maxLength={200} required autoFocus placeholder="e.g. Asha Verma" />
            </label>
            <label className="grid gap-1.5 text-sm">
              <span className="font-medium">Employee or member ID <span className="font-normal text-muted-foreground">(optional)</span></span>
              <Input value={ref} onChange={e => setRef(e.target.value)} maxLength={100} />
            </label>
            <label className="flex items-start gap-3 rounded-lg border border-primary/30 bg-primary/5 p-3 text-sm">
              <Checkbox checked={consent} onCheckedChange={v => setConsent(v === true)} className="mt-0.5" aria-required="true" />
              <span>
                <span className="font-medium">This person agreed to have their face photos used here.</span>
                <span className="block text-xs text-muted-foreground">Required. Your confirmation is recorded with your name and the time. Remove them at any time and all their photos and face data are deleted.</span>
              </span>
            </label>
            {err && <p className="text-sm text-destructive" role="alert">{err}</p>}
            <DialogFooter>
              <Button type="button" variant="ghost" onClick={onClose}>Cancel</Button>
              <Button type="submit" disabled={!name.trim() || !consent || add.isPending}>
                {add.isPending ? <Loader2 className="animate-spin" /> : <ShieldCheck />} Continue to photos
              </Button>
            </DialogFooter>
          </form>
        ) : (
          <div className="grid gap-4">
            <DialogHeader>
              <DialogTitle>Photos of {person.name}</DialogTitle>
              <DialogDescription>Step 2 of 2: {MIN_PHOTOS}–{MAX_PHOTOS} photos, one face each, with different angles and lighting.</DialogDescription>
            </DialogHeader>
            <PhotoAdder projectId={projectId} personId={person.id} />
            <DialogFooter>
              <Button onClick={onClose}>Done</Button>
            </DialogFooter>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

/** Upload or camera → POST photos → per-file accepted / rejected with reason. */
function PhotoAdder({ projectId, personId }: { projectId: string; personId: string }) {
  const upload = useUploadPhotos(projectId);
  const [results, setResults] = useState<PhotoUpload | null>(null);
  const [cam, setCam] = useState(false);
  const send = (files: { blob: Blob; name: string }[]) =>
    files.length && upload.mutate({ personId, files: files.slice(0, MAX_PHOTOS) }, { onSuccess: setResults, onError: e => toast.error(e.message) });
  const count = results?.person.photo_count;

  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap gap-2">
        <label className="inline-flex h-9 cursor-pointer items-center gap-2 rounded-md border px-3 text-sm font-medium hover:bg-accent">
          <ImageUp className="size-4" /> Choose photos
          <input type="file" accept="image/*" multiple className="sr-only" onChange={e => (send([...(e.target.files ?? [])].map(f => ({ blob: f, name: f.name }))), (e.target.value = ''))} />
        </label>
        <Button variant="outline" onClick={() => setCam(c => !c)}>{cam ? <CameraOff /> : <Camera />} {cam ? 'Close camera' : 'Use camera'}</Button>
        {upload.isPending && <span className="flex items-center gap-2 text-sm text-muted-foreground"><Loader2 className="size-4 animate-spin" /> Checking faces…</span>}
        {count != null && <Badge variant={count >= MIN_PHOTOS ? 'success' : 'warning'} className="ml-auto">{count} / {MAX_PHOTOS} photos</Badge>}
      </div>
      {cam && <FaceCamera onShot={b => send([{ blob: b, name: `camera-${Date.now()}.jpg` }])} busy={upload.isPending} />}
      {results && (
        <ul className="grid gap-1.5 text-sm" aria-live="polite">
          {results.accepted.map(a => (
            <li key={a.photo_id} className="flex items-center gap-2">
              {a.thumb_url ? <img src={a.thumb_url} alt="" className="size-8 rounded object-cover" /> : <CheckCircle2 className="size-4 text-success" />}
              <span className="truncate">{a.filename}</span>
              <Badge variant="success" className="ml-auto">accepted</Badge>
            </li>
          ))}
          {results.rejected.map((r, i) => (
            <li key={`${r.filename}-${i}`} className="flex items-center gap-2">
              <XCircle className="size-4 shrink-0 text-destructive" />
              <span className="truncate">{r.filename}</span>
              <span className="ml-auto text-right text-xs text-destructive">{REJECT_TEXT[r.reason] ?? r.detail}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** Small webcam: one photo per click (the person should vary angle/lighting between shots). */
export function FaceCamera({ onShot, busy }: { onShot: (b: Blob) => void; busy?: boolean }) {
  const [error, setError] = useState<string>();
  const { videoRef: video, paused, resume } = useCamera({ enabled: true, facing: 'user', onError: e => setError(e instanceof Error ? e.message : 'Camera unavailable') });
  if (error) return <p className="text-sm text-destructive" role="alert">{error}</p>;
  return (
    <div className="grid gap-2">
      <video ref={video} autoPlay playsInline muted className="max-h-72 w-full rounded-lg bg-black object-contain" />
      {paused ? (
        <Button onClick={resume}><Camera /> Resume camera</Button>
      ) : (
        <Button onClick={async () => { const b = await grabFrame(video.current, Infinity, 0.9); if (b) onShot(b); }} disabled={busy}><Camera /> Take photo</Button>
      )}
      <p className="text-xs text-muted-foreground">Turn a little, change the light between shots: variety makes recognition more reliable.</p>
    </div>
  );
}

function PersonDialog({ projectId, personId, onClose }: { projectId: string; personId: string; onClose: () => void }) {
  const p = usePerson(personId);
  const delPhoto = useDeletePhoto(projectId);
  const delPerson = useDeletePerson(projectId);
  const [confirm, setConfirm] = useState(false);
  const x = p.data;
  return (
    <Dialog open onOpenChange={o => !o && onClose()}>
      <DialogContent className="max-h-[90vh] max-w-2xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{x?.name ?? 'Person'}</DialogTitle>
          <DialogDescription>
            {x && <>Consent recorded {new Date(x.consent.granted_at).toLocaleString()} ({x.consent.method.replace('_', ' ')}).{x.external_ref && ` ID ${x.external_ref}.`}</>}
          </DialogDescription>
        </DialogHeader>
        {!x ? <Skeleton className="h-40" /> : (
          <>
            {x.warnings.map(w => <p key={w.code} className="flex items-center gap-1.5 text-sm text-warning"><AlertTriangle className="size-4" /> {w.detail}</p>)}
            <div className="grid grid-cols-[repeat(auto-fill,minmax(6.5rem,1fr))] gap-2">
              {x.photos.map(ph => (
                <figure key={ph.id} className="group relative overflow-hidden rounded-lg border">
                  {ph.thumb_url ? <img src={ph.thumb_url} alt="" className="aspect-square w-full object-cover" /> : <div className="grid aspect-square place-items-center bg-muted text-xs text-muted-foreground">No preview</div>}
                  <figcaption className="px-1.5 py-1 text-[11px] text-muted-foreground tabular-nums">quality {(ph.quality * 100).toFixed(0)}% · {ph.face_px}px</figcaption>
                  <button type="button" aria-label="Delete photo" onClick={() => delPhoto.mutate({ personId, photoId: ph.id })} className="absolute right-1 top-1 rounded bg-black/60 p-1 text-white opacity-0 transition-opacity group-hover:opacity-100 focus:opacity-100">
                    <Trash2 className="size-3.5" />
                  </button>
                </figure>
              ))}
            </div>
            {x.photo_count < MAX_PHOTOS && <PhotoAdder projectId={projectId} personId={personId} />}
            <div className="flex justify-between border-t pt-3">
              <Button variant="ghost" className="text-destructive hover:text-destructive" onClick={() => setConfirm(true)}><Trash2 /> Delete person</Button>
              <Button onClick={onClose}>Done</Button>
            </div>
          </>
        )}
        <Dialog open={confirm} onOpenChange={setConfirm}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Delete {x?.name}?</DialogTitle>
              <DialogDescription>This removes all their photos and face data permanently. They'll show as “Other” from now on. It can't be undone.</DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button variant="ghost" onClick={() => setConfirm(false)}>Cancel</Button>
              <Button variant="destructive" disabled={delPerson.isPending} onClick={() => delPerson.mutate(personId, { onSuccess: () => (toast.success(`${x?.name} removed`), onClose()), onError: e => toast.error(e.message) })}>
                {delPerson.isPending ? <Loader2 className="animate-spin" /> : <Trash2 />} Delete permanently
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </DialogContent>
    </Dialog>
  );
}
