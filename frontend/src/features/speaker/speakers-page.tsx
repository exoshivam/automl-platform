import { AlertTriangle, CheckCircle2, FileAudio, Loader2, Mic, Plus, ShieldCheck, Square, Trash2, UserRound, XCircle } from 'lucide-react';
import { useEffect, useState } from 'react';
import { toast } from 'sonner';
import { REJECT_TEXT, useAddPerson, useDeletePerson, useDeletePhoto, useFaceSettings, usePeople, usePerson, useUploadClips, type Person, type PhotoUpload } from '@/api/face';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Checkbox, Skeleton } from '@/components/ui/misc';
import { ApiError } from '@/lib/api';
import { timeAgo } from '@/lib/utils';
import { useWavRecorder } from '@/lib/wav-recorder';
import { usePrompt } from './prompts';
import { ReadAloud } from './read-aloud';

const MIN_CLIPS = 3;
const MAX_CLIPS = 10;
const MIN_SECONDS = 3;

/** Speaker recognition: enrol voices (consent first), 3–5 clips each; adding/removing takes effect at once. */
export function SpeakersPage({ projectId }: { projectId: string }) {
  const people = usePeople(projectId);
  const [adding, setAdding] = useState(false);
  const [open, setOpen] = useState<string | null>(null);
  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">Speakers</h2>
          <p className="text-sm text-muted-foreground">Each speaker needs their consent and {MIN_CLIPS}–5 short voice clips (3–10 s each). Changes apply immediately; there's nothing to train.</p>
        </div>
        <Button onClick={() => setAdding(true)}><Plus /> Add speaker</Button>
      </div>
      {people.isLoading ? (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{[0, 1, 2].map(i => <Skeleton key={i} className="h-24 rounded-xl" />)}</div>
      ) : !people.data?.length ? (
        <Card className="flex flex-col items-center gap-3 border-dashed p-12 text-center">
          <Mic className="size-8 text-muted-foreground" />
          <p className="font-medium">No speakers yet</p>
          <p className="max-w-sm text-sm text-muted-foreground">Add the people this should recognise. Anyone else shows as “Unknown”.</p>
          <Button onClick={() => setAdding(true)}><Plus /> Add speaker</Button>
        </Card>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {people.data.map(p => (
            <button key={p.id} type="button" onClick={() => setOpen(p.id)} className="flex gap-3 rounded-xl border bg-card p-3 text-left transition-colors hover:bg-accent/50 focus-visible:ring-2 focus-visible:ring-ring">
              <span className="grid size-12 shrink-0 place-items-center rounded-lg bg-muted"><UserRound className="size-5 text-muted-foreground" /></span>
              <span className="grid min-w-0 content-start gap-1">
                <span className="truncate font-medium">{p.name}</span>
                <span className="text-xs text-muted-foreground tabular-nums">{p.photo_count} {p.photo_count === 1 ? 'clip' : 'clips'} · consent {timeAgo(p.consent.granted_at)}</span>
                {p.warnings.map(w => <span key={w.code} className="flex items-center gap-1 text-xs text-warning"><AlertTriangle className="size-3" /> {w.detail.replace(/photos?/g, 'clips')}</span>)}
              </span>
            </button>
          ))}
        </div>
      )}
      {adding && <AddSpeakerDialog projectId={projectId} onClose={() => setAdding(false)} />}
      {open && <SpeakerDialog projectId={projectId} personId={open} onClose={() => setOpen(null)} />}
    </div>
  );
}

function AddSpeakerDialog({ projectId, onClose }: { projectId: string; onClose: () => void }) {
  const add = useAddPerson(projectId);
  const [name, setName] = useState('');
  const [consent, setConsent] = useState(false);
  const [person, setPerson] = useState<Person | null>(null);
  const [err, setErr] = useState<string>();
  return (
    <Dialog open onOpenChange={o => !o && onClose()}>
      <DialogContent className="max-h-[90vh] max-w-2xl overflow-y-auto">
        {!person ? (
          <form className="grid gap-4" onSubmit={e => (e.preventDefault(), add.mutate({ name: name.trim(), consent: { attested: true } }, { onSuccess: setPerson, onError: e => setErr(e instanceof ApiError && e.problem.code === 'consent_required' ? 'Consent is required before adding someone.' : e.message) }))}>
            <DialogHeader>
              <DialogTitle>Add speaker</DialogTitle>
              <DialogDescription>Step 1 of 2: who they are, and their consent.</DialogDescription>
            </DialogHeader>
            <label className="grid gap-1.5 text-sm">
              <span className="font-medium">Name</span>
              <Input value={name} onChange={e => setName(e.target.value)} maxLength={200} required autoFocus placeholder="e.g. Speaker 1" />
            </label>
            <label className="flex items-start gap-3 rounded-lg border border-primary/30 bg-primary/5 p-3 text-sm">
              <Checkbox checked={consent} onCheckedChange={v => setConsent(v === true)} className="mt-0.5" aria-required="true" />
              <span>
                <span className="font-medium">This person agreed to have their voice recordings used here.</span>
                <span className="block text-xs text-muted-foreground">Required: a voice is biometric data. Your confirmation is recorded with your name and the time. Removing them deletes all their clips and voice data.</span>
              </span>
            </label>
            {err && <p className="text-sm text-destructive" role="alert">{err}</p>}
            <DialogFooter>
              <Button type="button" variant="ghost" onClick={onClose}>Cancel</Button>
              <Button type="submit" disabled={!name.trim() || !consent || add.isPending}>{add.isPending ? <Loader2 className="animate-spin" /> : <ShieldCheck />} Continue to recordings</Button>
            </DialogFooter>
          </form>
        ) : (
          <div className="grid gap-4">
            <DialogHeader>
              <DialogTitle>Voice clips for {person.name}</DialogTitle>
              <DialogDescription>Step 2 of 2: record or upload {MIN_CLIPS}–5 clips of 3–10 seconds, in a quiet place.</DialogDescription>
            </DialogHeader>
            <ClipAdder projectId={projectId} personId={person.id} />
            <DialogFooter><Button onClick={onClose}>Done</Button></DialogFooter>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

/** Record in the browser (16 kHz mono WAV) or upload WAV/FLAC/OGG → per-clip accepted / rejected with reason. */
export function ClipAdder({ projectId, personId }: { projectId: string; personId: string }) {
  const upload = useUploadClips(projectId);
  const phrase = useFaceSettings(projectId).data?.phrase;
  const [results, setResults] = useState<PhotoUpload | null>(null);
  const send = (files: { blob: Blob; name: string }[]) =>
    files.length && upload.mutate({ personId, files: files.slice(0, MAX_CLIPS) }, { onSuccess: setResults, onError: e => toast.error(e.message) });
  const count = results?.person.photo_count;
  return (
    <div className="grid gap-3">
      <ClipRecorder projectId={projectId} phrase={phrase} minSeconds={MIN_SECONDS} busy={upload.isPending} onSave={b => send([{ blob: b, name: `recording-${Date.now()}.wav` }])} />
      <div className="flex flex-wrap items-center gap-2">
        <label className="inline-flex h-9 cursor-pointer items-center gap-2 rounded-md border px-3 text-sm font-medium hover:bg-accent">
          <FileAudio className="size-4" /> Upload audio files
          <input type="file" accept=".wav,.flac,.ogg,audio/wav,audio/flac,audio/ogg" multiple className="sr-only" onChange={e => (send([...(e.target.files ?? [])].map(f => ({ blob: f, name: f.name }))), (e.target.value = ''))} />
        </label>
        {upload.isPending && <span className="flex items-center gap-2 text-sm text-muted-foreground"><Loader2 className="size-4 animate-spin" /> Checking the clip…</span>}
        {count != null && <Badge variant={count >= MIN_CLIPS ? 'success' : 'warning'} className="ml-auto">{count} / {MAX_CLIPS} clips</Badge>}
      </div>
      {results && (
        <ul className="grid gap-1.5 text-sm" aria-live="polite">
          {results.accepted.map(a => (
            <li key={a.photo_id} className="flex items-center gap-2"><CheckCircle2 className="size-4 text-success" /><span className="truncate">{a.filename}</span><span className="ml-auto text-xs text-muted-foreground tabular-nums">{a.seconds != null && `${a.seconds.toFixed(1)} s`}</span><Badge variant="success">accepted</Badge></li>
          ))}
          {results.rejected.map((r, i) => (
            <li key={`${r.filename}-${i}`} className="flex items-center gap-2"><XCircle className="size-4 shrink-0 text-destructive" /><span className="truncate">{r.filename}</span><span className="ml-auto text-right text-xs text-destructive">{REJECT_TEXT[r.reason] ?? r.detail}</span></li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** Record → listen back → save or discard. Shows the phrase, a live level meter and a seconds counter (auto-stops at 10 s). */
export function ClipRecorder({ projectId, phrase, minSeconds, maxSeconds = 10, busy, onSave, saveLabel = 'Save clip' }: { projectId: string; phrase?: string | null; minSeconds: number; maxSeconds?: number; busy?: boolean; onSave: (b: Blob) => void; saveLabel?: string }) {
  const rec = useWavRecorder({ maxSeconds });
  const prompt = usePrompt(projectId, phrase);
  const [clip, setClip] = useState<{ blob: Blob; url: string; seconds: number } | null>(null);
  useEffect(() => () => { if (clip) URL.revokeObjectURL(clip.url); }, [clip]);
  const keep = (b: Blob | null, secs: number) => b && setClip({ blob: b, url: URL.createObjectURL(b), seconds: secs });
  const begin = async () => {
    setClip(null);
    await rec.start();
    rec.onAutoStop(b => keep(b, maxSeconds));
  };
  const tooShort = clip && clip.seconds < minSeconds;
  return (
    <div className="grid gap-3 rounded-lg border p-3">
      <ReadAloud p={prompt} active={rec.recording} />
      {rec.error && <p className="text-sm text-destructive" role="alert">{rec.error}</p>}
      <div className="flex items-center gap-3">
        {rec.recording ? (
          <Button variant="destructive" onClick={() => keep(rec.stop(), rec.seconds)}><Square /> Stop</Button>
        ) : (
          <Button variant="outline" onClick={() => void begin()} disabled={busy}><Mic /> {clip ? 'Record again' : 'Record'}</Button>
        )}
        <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted" aria-label="Microphone level" role="meter" aria-valuenow={Math.round(rec.level * 100)} aria-valuemin={0} aria-valuemax={100}>
          <div className="h-full rounded-full bg-success transition-[width] duration-75" style={{ width: `${rec.level * 100}%` }} />
        </div>
        <span className="w-16 text-right text-sm tabular-nums text-muted-foreground">{rec.recording ? `${rec.seconds.toFixed(1)} s` : clip ? `${clip.seconds.toFixed(1)} s` : `0–${maxSeconds} s`}</span>
      </div>
      {rec.recording && rec.seconds < minSeconds && <p className="text-xs text-muted-foreground">Keep talking: at least {minSeconds} s.</p>}
      {clip && (
        <div className="flex flex-wrap items-center gap-2">
          <audio src={clip.url} controls className="h-9 max-w-full" aria-label="Recorded clip" />
          <Button size="sm" disabled={!!tooShort || busy} onClick={() => (onSave(clip.blob), setClip(null), prompt.next())}>{busy ? <Loader2 className="animate-spin" /> : null} {saveLabel}</Button>
          <Button size="sm" variant="ghost" onClick={() => setClip(null)}>Discard</Button>
          {tooShort && <span className="text-xs text-warning">Too short ({clip.seconds.toFixed(1)} s): record at least {minSeconds} s.</span>}
        </div>
      )}
    </div>
  );
}

function SpeakerDialog({ projectId, personId, onClose }: { projectId: string; personId: string; onClose: () => void }) {
  const p = usePerson(personId);
  const delClip = useDeletePhoto(projectId);
  const delPerson = useDeletePerson(projectId);
  const [confirm, setConfirm] = useState(false);
  const x = p.data;
  return (
    <Dialog open onOpenChange={o => !o && onClose()}>
      <DialogContent className="max-h-[90vh] max-w-2xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{x?.name ?? 'Speaker'}</DialogTitle>
          <DialogDescription>{x && <>Consent recorded {new Date(x.consent.granted_at).toLocaleString()} ({x.consent.method.replace('_', ' ')}).</>}</DialogDescription>
        </DialogHeader>
        {!x ? <Skeleton className="h-40" /> : (
          <>
            <ul className="grid gap-2">
              {x.photos.map((c, i) => (
                <li key={c.id} className="flex flex-wrap items-center gap-2 rounded-lg border p-2 text-sm">
                  <span className="w-14 text-muted-foreground">Clip {i + 1}</span>
                  {c.audio_url && <audio src={c.audio_url} controls className="h-8 max-w-full" />}
                  <span className="text-xs text-muted-foreground tabular-nums">{c.seconds != null && `${c.seconds.toFixed(1)} s · `}quality {Math.round(c.quality * 100)}%</span>
                  <Button size="icon" variant="ghost" className="ml-auto" aria-label={`Delete clip ${i + 1}`} onClick={() => delClip.mutate({ personId, photoId: c.id })}><Trash2 /></Button>
                </li>
              ))}
            </ul>
            {x.photo_count < MAX_CLIPS && <ClipAdder projectId={projectId} personId={personId} />}
            <div className="flex justify-between border-t pt-3">
              <Button variant="ghost" className="text-destructive hover:text-destructive" onClick={() => setConfirm(true)}><Trash2 /> Delete speaker</Button>
              <Button onClick={onClose}>Done</Button>
            </div>
          </>
        )}
        <Dialog open={confirm} onOpenChange={setConfirm}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Delete {x?.name}?</DialogTitle>
              <DialogDescription>This removes all their clips and voice data permanently. They'll show as “Unknown” from now on. It can't be undone.</DialogDescription>
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
