import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Webcam that only runs while it is wanted: `enabled` and the page visible and the component mounted.
 * - Every track is stopped on disable/unmount (the camera light goes off), including a getUserMedia that resolves
 *   after unmount (the stream is stopped instead of attached).
 * - When the tab is hidden the camera stops and stays `paused` until the user calls resume().
 */
export function useCamera({ enabled, facing = 'user', constraints, onError }: {
  enabled: boolean;
  facing?: 'user' | 'environment';
  constraints?: MediaTrackConstraints;
  onError?: (e: unknown) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [paused, setPaused] = useState(false);
  const [ready, setReady] = useState(false);
  const onErr = useRef(onError);
  onErr.current = onError;
  const key = JSON.stringify(constraints ?? null);

  useEffect(() => {
    const hide = () => document.hidden && setPaused(true);
    document.addEventListener('visibilitychange', hide);
    return () => document.removeEventListener('visibilitychange', hide);
  }, []);

  const active = enabled && !paused;
  useEffect(() => {
    setReady(false);
    if (!active) return;
    let stopped = false;
    let stream: MediaStream | null = null;
    const md = navigator.mediaDevices;
    if (!md?.getUserMedia) {
      onErr.current?.(new DOMException('No mediaDevices', 'NotFoundError'));
      return;
    }
    md.getUserMedia({ video: { facingMode: { ideal: facing }, ...(JSON.parse(key) ?? {}) }, audio: false }).then(
      s => {
        if (stopped) return s.getTracks().forEach(t => t.stop());
        stream = s;
        const v = videoRef.current;
        if (v) {
          v.srcObject = s;
          void v.play().catch(() => undefined);
        }
        setReady(true);
      },
      e => !stopped && onErr.current?.(e),
    );
    return () => {
      stopped = true;
      stream?.getTracks().forEach(t => t.stop());
      if (videoRef.current) videoRef.current.srcObject = null;
      setReady(false);
    };
  }, [active, facing, key]);

  return { videoRef, ready, paused: enabled && paused, resume: useCallback(() => setPaused(false), []) };
}

/** One AbortController per request; the previous one and the last one (on unmount) are aborted. */
export function useAbortOnUnmount() {
  const ctl = useRef<AbortController | null>(null);
  useEffect(() => () => ctl.current?.abort(), []);
  return useCallback(() => {
    ctl.current?.abort();
    ctl.current = new AbortController();
    return ctl.current.signal;
  }, []);
}

/** JPEG of the current video frame, scaled to at most maxWidth. */
export function grabFrame(v: HTMLVideoElement | null, maxWidth = Infinity, quality = 0.85): Promise<Blob | null> {
  return new Promise(resolve => {
    if (!v?.videoWidth) return resolve(null);
    const s = Math.min(1, maxWidth / v.videoWidth);
    const c = document.createElement('canvas');
    c.width = Math.round(v.videoWidth * s);
    c.height = Math.round(v.videoHeight * s);
    c.getContext('2d')!.drawImage(v, 0, 0, c.width, c.height);
    c.toBlob(resolve, 'image/jpeg', quality);
  });
}
