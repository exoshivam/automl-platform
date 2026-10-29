import { useCallback, useEffect, useRef, useState } from 'react';

export const TARGET_RATE = 16_000;

// Runs on the audio thread: forwards each 128-frame block of channel 0 to the page.
const WORKLET = `class Tap extends AudioWorkletProcessor {
  process(inputs) { const ch = inputs[0] && inputs[0][0]; if (ch) this.port.postMessage(ch.slice(0)); return true; }
}
registerProcessor('tap', Tap);`;

/** Average-downsample mono float audio from `from` Hz to TARGET_RATE. */
export function downsample(input: Float32Array, from: number): Float32Array {
  if (from === TARGET_RATE) return input;
  const ratio = from / TARGET_RATE;
  const out = new Float32Array(Math.floor(input.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const a = Math.floor(i * ratio), b = Math.min(input.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = a; j < b; j++) sum += input[j]!;
    out[i] = sum / Math.max(1, b - a);
  }
  return out;
}

/** 16-bit PCM mono WAV (RIFF) from float samples in [-1, 1]. */
export function encodeWav(samples: Float32Array, rate = TARGET_RATE): Blob {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const str = (o: number, s: string) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); str(8, 'WAVE');
  str(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, 'data'); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]!));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}

/** RMS of a PCM16 WAV made by encodeWav (0 = silence, ~0.05+ = normal speech). */
export async function wavRms(b: Blob): Promise<number> {
  const pcm = new Int16Array((await b.arrayBuffer()).slice(44));
  let sq = 0;
  for (const x of pcm) sq += (x / 0x8000) ** 2;
  return pcm.length ? Math.sqrt(sq / pcm.length) : 0;
}

function concat(chunks: Float32Array[]) {
  const out = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
  let o = 0;
  for (const c of chunks) (out.set(c, o), (o += c.length));
  return out;
}

/**
 * Microphone → 16 kHz mono PCM WAV in the browser (no server-side ffmpeg).
 * Only runs while recording: stop(), unmount and a hidden tab all release every track (mic light off).
 * `rolling`: never auto-stop; keep only the last `maxSeconds` of audio (Live mode listens indefinitely).
 */
export function useWavRecorder({ maxSeconds = 10, rolling = false }: { maxSeconds?: number; rolling?: boolean } = {}) {
  const [recording, setRecording] = useState(false);
  const [level, setLevel] = useState(0);
  const [seconds, setSeconds] = useState(0);
  const [error, setError] = useState<string>();
  const res = useRef<{ stream: MediaStream; ctx: AudioContext; chunks: Float32Array[]; rate: number; resolve?: (b: Blob | null) => void } | null>(null);

  const release = useCallback(() => {
    const r = res.current;
    res.current = null;
    if (!r) return null;
    r.stream.getTracks().forEach(t => t.stop());
    void r.ctx.close().catch(() => undefined);
    setRecording(false);
    setLevel(0);
    return r;
  }, []);

  const stop = useCallback((): Blob | null => {
    const r = release();
    if (!r || !r.chunks.length) return null;
    return encodeWav(downsample(concat(r.chunks), r.rate));
  }, [release]);

  /** WAV of the last `secs` seconds without stopping (Live mode). */
  const window = useCallback((secs: number): Blob | null => {
    const r = res.current;
    if (!r) return null;
    const need = Math.round(secs * r.rate);
    const all = concat(r.chunks);
    return all.length ? encodeWav(downsample(all.subarray(Math.max(0, all.length - need)), r.rate)) : null;
  }, []);

  const start = useCallback(async () => {
    setError(undefined);
    release();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
      const ctx = new AudioContext();
      await ctx.resume().catch(() => undefined); // may start suspended (autoplay policy); the press is the gesture
      const url = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }));
      await ctx.audioWorklet.addModule(url);
      URL.revokeObjectURL(url);
      const src = ctx.createMediaStreamSource(stream);
      const tap = new AudioWorkletNode(ctx, 'tap');
      const r: { stream: MediaStream; ctx: AudioContext; chunks: Float32Array[]; rate: number; resolve?: (b: Blob | null) => void } = { stream, ctx, chunks: [], rate: ctx.sampleRate };
      res.current = r;
      let n = 0, sq = 0, count = 0, kept = 0;
      tap.port.onmessage = (e: MessageEvent<Float32Array>) => {
        if (res.current !== r) return;
        r.chunks.push(e.data);
        n += e.data.length;
        for (const x of e.data) sq += x * x;
        count += e.data.length;
        if (count >= r.rate / 20) { // ~20 level updates per second
          setLevel(Math.min(1, Math.sqrt(sq / count) * 4));
          setSeconds(n / r.rate);
          sq = 0; count = 0;
        }
        if (rolling) {
          kept += e.data.length;
          while (kept - r.chunks[0]!.length >= maxSeconds * r.rate) kept -= r.chunks.shift()!.length;
        } else if (n / r.rate >= maxSeconds) r.resolve?.(stop());
      };
      src.connect(tap);
      setSeconds(0);
      setRecording(true);
    } catch (e) {
      release();
      setError(e instanceof Error ? e.message : 'Microphone unavailable');
    }
  }, [maxSeconds, rolling, release, stop]);

  /** Resolves with the clip when recording stops (button, auto-stop at maxSeconds). */
  const onAutoStop = useCallback((cb: (b: Blob | null) => void) => {
    if (res.current) res.current.resolve = cb;
  }, []);

  useEffect(() => {
    const hide = () => document.hidden && release();
    document.addEventListener('visibilitychange', hide);
    return () => (document.removeEventListener('visibilitychange', hide), void release());
  }, [release]);

  return { recording, level, seconds, error, start, stop, window, onAutoStop };
}
