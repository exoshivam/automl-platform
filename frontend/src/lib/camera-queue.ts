// DAT-9 upload queue for camera captures. Photos go to IndexedDB first, so the queue survives
// reloads, tab sleep and dead zones; a worker loop uploads them in batches whenever we're online.
import type { SignResponse, UploadSession } from '@/api/data-types';
import { api } from './api';

export interface QueuedPhoto {
  id: string;
  projectId: string;
  datasetId: string;
  /** Class for classification; null = unlabelled (detection box queue). */
  label: string | null;
  /** Detection: which object the user said they're photographing (a hint, not a label). null = mixed / unknown. */
  target?: string | null;
  /** Photos of one burst share this, so each burst is its own upload session (undo = that upload). */
  burst?: string;
  blob: Blob;
  createdAt: number;
  status: 'queued' | 'uploading' | 'failed';
  error?: string;
}

const DB = 'automl-camera';
const STORE = 'queue';

function db(): Promise<IDBDatabase> {
  return new Promise((res, rej) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: 'id' });
    req.onsuccess = () => res(req.result);
    req.onerror = () => rej(req.error);
  });
}
async function tx<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const d = await db();
  return new Promise((res, rej) => {
    const r = fn(d.transaction(STORE, mode).objectStore(STORE));
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}

export const queue = {
  add: (p: QueuedPhoto) => tx('readwrite', s => s.put(p)),
  remove: (id: string) => tx('readwrite', s => s.delete(id)),
  all: () => tx<QueuedPhoto[]>('readonly', s => s.getAll() as IDBRequest<QueuedPhoto[]>),
  put: (p: QueuedPhoto) => tx('readwrite', s => s.put(p)),
};

type Listener = (items: QueuedPhoto[], uploaded: number) => void;
const listeners = new Set<Listener>();
let uploadedThisSession = 0;
let running = false;
const uploadOf = new Map<string, string>(); // photo id → upload id that ingested it (this tab only)
const inFlight = new Set<string>(); // photo ids in the batch being uploaded right now
const cancelled = new Set<string>(); // undone before upload: never start uploading these

export function subscribe(fn: Listener) {
  listeners.add(fn);
  void queue.all().then(items => fn(items, uploadedThisSession));
  return () => listeners.delete(fn);
}
async function emit() {
  const items = await queue.all();
  listeners.forEach(fn => fn(items, uploadedThisSession));
}

/** Upload everything queued for this dataset, 20 photos per upload session (one session per label + capture target). */
export async function drain(projectId: string, datasetId: string, onVersion?: (vid: string) => void) {
  if (running || !navigator.onLine) return;
  running = true;
  try {
    for (;;) {
      const pending = (await queue.all()).filter(p => p.projectId === projectId && p.datasetId === datasetId && p.status !== 'uploading' && !cancelled.has(p.id)).sort((a, b) => a.createdAt - b.createdAt);
      if (!pending.length) break;
      const label = pending[0]!.label;
      const target = pending[0]!.target ?? null;
      const burst = pending[0]!.burst;
      const batch = pending.filter(p => p.label === label && (p.target ?? null) === target && p.burst === burst).slice(0, 20);
      batch.forEach(p => inFlight.add(p.id));
      for (const p of batch) await queue.put({ ...p, status: 'uploading', error: undefined });
      await emit();
      try {
        const session = await api<UploadSession>(`/projects/${projectId}/datasets/${datasetId}/uploads`, {
          method: 'POST',
          body: { source: 'camera', label_from_folder: false, file_count: batch.length, total_bytes: batch.reduce((n, p) => n + p.blob.size, 0), label, capture_target: target },
        });
        const files = [];
        for (const p of batch) {
          if (cancelled.has(p.id)) continue; // undone while this batch was starting
          const key = `${session.key_prefix}${p.id}.jpg`;
          const signed = await api<SignResponse>(`/uploads/${session.upload_id}/sign`, { method: 'POST', body: { method: 'PUT', key, size: p.blob.size } });
          const put = await fetch(signed.url, { method: 'PUT', body: p.blob, headers: { 'Content-Type': 'image/jpeg', ...(signed.headers ?? {}) } });
          if (!put.ok) throw new Error(`Storage returned ${put.status}`);
          files.push({ key, name: `${p.id}.jpg`, relative_path: null, size: p.blob.size, mime: 'image/jpeg', sha256: null, label });
        }
        const res = files.length ? await api<{ dataset_version_id: string | null }>(`/uploads/${session.upload_id}/complete`, { method: 'POST', body: { files } as unknown as Record<string, unknown> }) : null;
        for (const p of batch) (await queue.remove(p.id), cancelled.has(p.id) || uploadOf.set(p.id, session.upload_id));
        uploadedThisSession += files.length;
        if (res?.dataset_version_id) onVersion?.(res.dataset_version_id);
      } catch (e) {
        // Leave them queued; the next online/visibility event retries.
        for (const p of batch) if (!cancelled.has(p.id)) await queue.put({ ...p, status: 'failed', error: (e as Error).message });
        await emit();
        break;
      } finally {
        batch.forEach(p => inFlight.delete(p.id));
      }
      await emit();
    }
  } finally {
    running = false;
    await emit();
  }
}

/**
 * Undo photos taken in this tab: queued ones are dropped locally, uploaded ones are undone server-side via their
 * upload ids (one upload per burst, so nothing else goes with them). Waits for an in-flight batch to finish first.
 */
export async function undoPhotos(ids: string[], undoUploads: (uploadIds: string[]) => Promise<unknown>) {
  ids.forEach(id => cancelled.add(id));
  for (let t = 0; t < 240 && ids.some(id => inFlight.has(id)); t++) await new Promise(r => setTimeout(r, 250));
  const want = new Set(ids);
  for (const p of await queue.all()) if (want.has(p.id) && !inFlight.has(p.id)) await queue.remove(p.id);
  await emit();
  const uploads = [...new Set(ids.map(id => uploadOf.get(id)).filter((u): u is string => !!u))];
  if (uploads.length) await undoUploads(uploads);
  ids.forEach(id => uploadOf.delete(id));
}
