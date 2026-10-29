import { Link } from '@tanstack/react-router';
import { CheckCircle2, Circle, Loader2, Mic, Radio, Save, ShieldCheck, SlidersHorizontal, Square, Users, Wand2 } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { toast } from 'sonner';
import { identifySpeaker, useFaceSettings, usePeople, useSaveFaceSettings, type SpeakerResult } from '@/api/face';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/misc';
import { ApiError } from '@/lib/api';
import { cn } from '@/lib/utils';
import { useWavRecorder, wavRms } from '@/lib/wav-recorder';
import { usePrompt } from './prompts';
import { ReadAloud } from './read-aloud';

/** The phrase everyone says when enrolling and when testing (optional; same words = more reliable matching). */
export function PhraseCard({ projectId }: { projectId: string }) {
  const s = useFaceSettings(projectId).data;
  const save = useSaveFaceSettings(projectId);
  const [phrase, setPhrase] = useState<string | null>(null);
  useEffect(() => { if (s && phrase === null) setPhrase(s.phrase ?? ''); }, [s, phrase]);
  if (!s || phrase === null) return null;
  const dirty = phrase.trim() !== (s.phrase ?? '');
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Phrase to say</CardTitle>
        <CardDescription>Optional. Shown while recording, both when adding a speaker and in the Playground. Saying the same words makes matching more reliable.</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-wrap gap-2">
        <Input value={phrase} onChange={e => setPhrase(e.target.value)} maxLength={300} placeholder="e.g. Good morning, my voice is my password" className="min-w-0 flex-1" aria-label="Phrase" />
        <Button disabled={!dirty || save.isPending} onClick={() => save.mutate({ phrase: phrase.trim() || null }, { onSuccess: () => toast.success('Phrase saved'), onError: e => toast.error(e.message) })}>
          {save.isPending ? <Loader2 className="animate-spin" /> : <Save />} Save
        </Button>
      </CardContent>
    </Card>
  );
}

function Result({ r }: { r: SpeakerResult }) {
  return (
    <div className="grid gap-3">
      <div className="flex items-baseline gap-2">
        <span className={cn('text-2xl font-semibold', !r.name && 'text-muted-foreground')}>{r.name ?? 'Unknown'}</span>
        <span className="text-sm tabular-nums text-muted-foreground">{Math.round(r.score * 100)}%</span>
      </div>
      {!r.name && <p className="text-xs text-muted-foreground">Below the match threshold ({Math.round(r.threshold * 100)}%) for every enrolled speaker.</p>}
      {r.top3.length > 0 && (
        <ul className="grid gap-1.5">
          {r.top3.map(c => (
            <li key={c.person_id} className="grid gap-0.5">
              <div className="flex justify-between text-xs"><span>{c.name}</span><span className="tabular-nums">{Math.round(c.score * 100)}%</span></div>
              <div className="h-2 overflow-hidden rounded-full bg-muted"><div className={cn('h-full rounded-full', c.score >= r.threshold ? 'bg-primary' : 'bg-muted-foreground/40')} style={{ width: `${Math.max(2, c.score * 100)}%` }} /></div>
            </li>
          ))}
        </ul>
      )}
      <p className="text-xs text-muted-foreground tabular-nums">{r.seconds.toFixed(1)} s of speech · {r.latency_ms} ms{r.model_load_ms ? ` · model loaded in ${(r.model_load_ms / 1000).toFixed(1)} s` : ''}</p>
    </div>
  );
}

const LIVE_EVERY = 3;
const SILENT_RMS = 0.01; // below this a 3 s window has no speech worth sending

/** Hold to talk (or click to start/stop) → who is speaking; optional Live mode identifies each ~3 s chunk. */
export function SpeakerPlayground({ projectId }: { projectId: string }) {
  const s = useFaceSettings(projectId).data;
  const minSec = s?.min_identify_seconds ?? 1.5;
  const [live, setLive] = useState(false);
  const [silent, setSilent] = useState(false);
  // Live listens until stopped, keeping only the recent audio; hold-to-talk stops at 30 s.
  const rec = useWavRecorder({ maxSeconds: live ? 10 : 30, rolling: live });
  const prompt = usePrompt(projectId, s?.phrase);
  const [result, setResult] = useState<SpeakerResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string>();
  const ctl = useRef<AbortController | null>(null);
  useEffect(() => () => ctl.current?.abort(), []); // leaving the page aborts the request (the recorder stops itself)

  const send = async (b: Blob | null, liveChunk = false) => {
    if (!b) return;
    ctl.current?.abort();
    ctl.current = new AbortController();
    setBusy(true);
    setErr(undefined);
    try {
      setResult(await identifySpeaker(projectId, b, ctl.current.signal));
      setSilent(false);
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') return;
      const code = e instanceof ApiError ? e.problem.code : undefined;
      // Live: a chunk without enough speech just means "nobody talking right now", not an error.
      if (liveChunk && (code === 'too_short' || code === 'too_quiet')) setSilent(true);
      else setErr(code === 'too_short' ? `Too short: speak a bit longer (at least ${minSec} s of speech).` : code === 'too_quiet' ? 'Too quiet: speak closer to the microphone.' : (e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  // Live: every ~3 s while the mic is on, identify the latest 3 s (one request in flight; silent windows skipped).
  // The interval reads the recorder through a ref: closing over `rec` would freeze rec.seconds at 0.
  const now = useRef({ rec, busy, send });
  now.current = { rec, busy, send };
  useEffect(() => {
    if (!live || !rec.recording) return;
    setSilent(false);
    const t = setInterval(async () => {
      const { rec: r, busy: b, send: go } = now.current;
      if (b || r.seconds < LIVE_EVERY) return;
      const w = r.window(LIVE_EVERY);
      if (!w) return;
      if ((await wavRms(w)) < SILENT_RMS) return setSilent(true);
      void go(w, true);
    }, LIVE_EVERY * 1000);
    return () => clearInterval(t);
  }, [live, rec.recording]);

  const toggleLive = () => (rec.recording ? (rec.stop(), setSilent(false)) : (setErr(undefined), void rec.start()));
  const stopAndSend = () => {
    const secs = rec.seconds;
    const b = rec.stop();
    if (!live) secs < minSec ? setErr(`Too short: speak for at least ${minSec} s.`) : void send(b);
    prompt.next();
  };

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_20rem]">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Who is speaking?</CardTitle>
          <CardDescription>{live ? 'Click Start listening and talk; the speaker is identified every 3 s until you stop.' : 'Hold the button and talk for 2–5 seconds, then let go.'} Anyone not enrolled shows as “Unknown”.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4">
          <ReadAloud p={prompt} active={rec.recording} note="Any words work for identification; the sentence just gives you something to say." />
          {rec.error && <p className="text-sm text-destructive" role="alert">{rec.error}</p>}
          <div className="flex flex-wrap items-center gap-4">
            <Button
              size="lg"
              variant={rec.recording ? 'destructive' : 'default'}
              // Hold-to-talk uses press/release; Live is a plain toggle (one click starts, one stops).
              onPointerDown={() => !live && !rec.recording && void rec.start()}
              onPointerUp={() => !live && rec.recording && stopAndSend()}
              onClick={() => live && toggleLive()}
              onKeyDown={e => { if (!live && (e.key === ' ' || e.key === 'Enter') && !e.repeat && !rec.recording) (e.preventDefault(), void rec.start()); }}
              onKeyUp={e => { if (!live && (e.key === ' ' || e.key === 'Enter') && rec.recording) stopAndSend(); }}
              aria-pressed={rec.recording}
              disabled={busy && !live}
            >
              {rec.recording ? <Square /> : <Mic />} {live ? (rec.recording ? 'Stop listening' : 'Start listening') : rec.recording ? 'Release to identify' : 'Hold to talk'}
            </Button>
            <div className="h-2 min-w-24 flex-1 overflow-hidden rounded-full bg-muted" role="meter" aria-label="Microphone level" aria-valuenow={Math.round(rec.level * 100)} aria-valuemin={0} aria-valuemax={100}>
              <div className="h-full rounded-full bg-success transition-[width] duration-75" style={{ width: `${rec.level * 100}%` }} />
            </div>
            <span className="w-14 text-right text-sm tabular-nums text-muted-foreground">{rec.recording ? `${rec.seconds.toFixed(1)} s` : ''}</span>
          </div>
          <label className="flex items-center gap-2 text-sm">
            <Switch checked={live} onCheckedChange={v => { if (rec.recording) rec.stop(); setLive(v); setSilent(false); setErr(undefined); }} /> <Radio className="size-4" /> Live: identify every {LIVE_EVERY} s while listening
          </label>
          {err && <p className="text-sm text-destructive" role="alert">{err}</p>}
        </CardContent>
      </Card>
      <Card className="h-fit" aria-live="polite">
        <CardHeader className="flex-row items-center justify-between"><CardTitle className="text-base">Result</CardTitle>{busy && <Loader2 className="size-4 animate-spin text-muted-foreground" />}</CardHeader>
        <CardContent className="grid gap-3">
          {live && rec.recording && (
            <p className="flex items-center gap-1.5 text-xs text-muted-foreground" role="status">
              <Radio className={cn('size-3.5', !silent && 'animate-pulse text-success')} /> {silent ? 'Listening… (no speech detected)' : rec.seconds < LIVE_EVERY ? 'Listening…' : 'Listening: identifying every 3 s'}
            </p>
          )}
          {result ? <div className={cn(live && rec.recording && silent && 'opacity-50')}><Result r={result} /></div> : <p className="text-sm text-muted-foreground">{busy ? 'Loading the voice model… the first clip after a few idle minutes takes a moment.' : 'The speaker and the top-3 matches appear here.'}</p>}
        </CardContent>
      </Card>
    </div>
  );
}

export function SpeakerOverview({ projectId }: { projectId: string }) {
  const s = useFaceSettings(projectId).data;
  const people = usePeople(projectId).data ?? [];
  const enrolled = people.filter(p => p.photo_count >= 3).length;
  const steps = [
    { label: 'Add speakers', desc: s ? `${s.people_count} speakers · ${s.photo_count} clips` : 'With consent, 3–5 clips each', icon: Users, done: enrolled > 0, to: '/projects/$projectId/speakers' as const },
    { label: 'Tune threshold', desc: s ? `Currently ${s.threshold.toFixed(2)}` : 'Strict vs lenient matching', icon: SlidersHorizontal, done: !!s && s.threshold !== s.default_threshold, to: '/projects/$projectId/settings' as const },
    { label: 'Try it', desc: 'Hold to talk, or Live', icon: Wand2, done: false, to: '/projects/$projectId/playground' as const },
  ];
  return (
    <div className="grid gap-6">
      <ol className="grid gap-3 sm:grid-cols-3" aria-label="Steps">
        {steps.map(x => (
          <li key={x.label}>
            <Card className={cn('h-full p-4', x.done && 'border-success/40')}>
              <div className="flex items-center justify-between"><x.icon className="size-5 text-muted-foreground" />{x.done ? <CheckCircle2 className="size-4 text-success" /> : <Circle className="size-4 text-muted-foreground/50" />}</div>
              <div className="mt-3 font-medium">{x.label}</div>
              <p className="text-xs text-muted-foreground">{x.desc}</p>
              <Link to={x.to} params={{ projectId }} className="mt-3 inline-block text-xs font-medium text-primary hover:underline">Open →</Link>
            </Card>
          </li>
        ))}
      </ol>
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base"><ShieldCheck className="size-5" /> Using speaker recognition responsibly</CardTitle>
          <CardDescription>A gallery of voices from people who agreed, not a trained classifier.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-1.5 text-sm text-muted-foreground">
          <p>• A voice is biometric data: only add people who consented; each enrolment records who confirmed it and when.</p>
          <p>• Removing a speaker deletes all their clips and voice data immediately.</p>
          <p>• Not for covert recording or as the only basis for decisions about someone; voices can be imitated or recorded.</p>
          {s && <p>• Voice model: {s.embedding_model.id}, licence <span className="font-medium text-foreground">{s.embedding_model.licence}</span>.</p>}
        </CardContent>
      </Card>
    </div>
  );
}
