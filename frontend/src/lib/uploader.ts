import Uppy from '@uppy/core';
import AwsS3 from '@uppy/aws-s3';
import type { CompleteFile, SignResponse, UploadSession, UploadSource } from '@/api/data-types';
import { api } from './api';
import { mimeOf, safeName, sha256, type PickedFile } from './files';

export interface UploadProgress {
  bytesUploaded: number;
  bytesTotal: number;
  filesDone: number;
  filesTotal: number;
  phase: 'starting' | 'uploading' | 'checksums' | 'completing' | 'done' | 'offline';
}

export interface UploadResult {
  uploadId: string;
  datasetVersionId: string | null;
  /** key per picked file, same order (for imports that point at one of them). */
  keys: string[];
}


/**
 * DAT-3: direct-to-storage upload.
 * 1. POST …/uploads (session), 2. Uppy pushes every file to S3/MinIO through presigned URLs from
 * POST /uploads/{id}/sign (multipart over 100 MB, resuming parts via ListParts), 3. POST /uploads/{id}/complete.
 * Network drops: Uppy retries; when the browser comes back online we call retryAll().
 */
export async function uploadFiles(opts: {
  projectId: string;
  datasetId: string;
  source: UploadSource;
  files: PickedFile[];
  labelFromFolder?: boolean;
  label?: string | null;
  /** Detection: the object these photos are of (a hint stored per photo, not a label). */
  captureTarget?: string | null;
  labelFor?: (f: PickedFile) => string | null;
  onProgress?: (p: UploadProgress) => void;
  signal?: AbortSignal;
}): Promise<UploadResult> {
  const { projectId, datasetId, source, files, onProgress, signal } = opts;
  const total = files.reduce((n, f) => n + f.file.size, 0);
  const report = (p: Partial<UploadProgress>) => onProgress?.({ bytesUploaded: 0, bytesTotal: total, filesDone: 0, filesTotal: files.length, phase: 'uploading', ...p });
  report({ phase: 'starting' });

  const session = await api<UploadSession>(`/projects/${projectId}/datasets/${datasetId}/uploads`, {
    method: 'POST',
    body: { source, label_from_folder: !!opts.labelFromFolder, file_count: files.length, total_bytes: total, label: opts.label ?? null, capture_target: opts.captureTarget ?? null },
  });
  const tooBig = files.find(f => f.file.size > session.max_file_bytes);
  if (tooBig) throw new Error(`${tooBig.file.name} is larger than the ${Math.round(session.max_file_bytes / 1024 ** 3)} GB limit`);

  // Keys = key_prefix + a unique relative path (no "..", no leading "/").
  const keys = files.map((f, i) => `${session.key_prefix}${String(i).padStart(5, '0')}/${safeName(f.relativePath ?? f.file.name)}`);
  const sizes = new Map(keys.map((k, i) => [k, files[i]!.file.size]));

  const uppy = new Uppy({ autoProceed: false, allowMultipleUploadBatches: false });
  uppy.use(AwsS3, {
    limit: 6,
    shouldUseMultipart: f => (f.size ?? 0) > 100 * 1024 * 1024,
    getChunkSize: () => session.part_size,
    generateObjectKey: f => String(f.meta.key),
    signRequest: async req => {
      const r = req as { method: 'PUT' | 'POST' | 'GET' | 'DELETE'; key: string; uploadId?: string; partNumber?: number };
      // the backend contract: send size with PutObject and CreateMultipartUpload.
      const needsSize = !r.uploadId && (r.method === 'PUT' || r.method === 'POST');
      const res = await api<SignResponse>(`/uploads/${session.upload_id}/sign`, {
        method: 'POST',
        body: { method: r.method, key: r.key, upload_id: r.uploadId ?? null, part_number: r.partNumber ?? null, size: needsSize ? sizes.get(r.key) : null },
      });
      return { url: res.url, headers: res.headers ?? {} };
    },
  });
  files.forEach((f, i) =>
    uppy.addFile({
      name: f.file.name,
      type: mimeOf(f.file),
      data: f.file,
      meta: { key: keys[i]!, relativePath: f.relativePath, label: opts.labelFor?.(f) ?? null },
    }),
  );

  let done = 0;
  uppy.on('upload-success', () => (done += 1));
  uppy.on('progress', () => {
    const st = uppy.getState();
    const sent = Object.values(st.files).reduce((n, f) => n + (f.progress.bytesUploaded || 0), 0);
    report({ bytesUploaded: sent, filesDone: done });
  });
  const online = () => uppy.retryAll();
  const offline = () => report({ phase: 'offline', filesDone: done });
  window.addEventListener('online', online);
  window.addEventListener('offline', offline);
  const abort = () => uppy.cancelAll();
  signal?.addEventListener('abort', abort);

  try {
    let result = await uppy.upload();
    // One automatic retry pass for transient failures (expired URL, flaky network).
    if (result?.failed?.length) result = await uppy.retryAll();
    if (signal?.aborted) throw new DOMException('Upload cancelled', 'AbortError');
    if (result?.failed?.length) throw new Error(`${result.failed.length} file(s) failed to upload: ${result.failed[0]?.error ?? 'network error'}`);

    report({ phase: 'checksums', bytesUploaded: total, filesDone: files.length });
    const completeFiles: CompleteFile[] = [];
    for (const [i, f] of files.entries())
      completeFiles.push({
        key: keys[i]!,
        name: f.file.name,
        relative_path: f.relativePath,
        size: f.file.size,
        mime: mimeOf(f.file),
        sha256: await sha256(f.file),
        label: opts.labelFor?.(f) ?? null,
      });

    report({ phase: 'completing', bytesUploaded: total, filesDone: files.length });
    const done2 = await api<{ upload_id: string; dataset_version_id: string | null; status: string }>(`/uploads/${session.upload_id}/complete`, {
      method: 'POST',
      body: { files: completeFiles } as unknown as Record<string, unknown>,
    });
    report({ phase: 'done', bytesUploaded: total, filesDone: files.length });
    return { uploadId: session.upload_id, datasetVersionId: done2.dataset_version_id ?? null, keys };
  } finally {
    window.removeEventListener('online', online);
    window.removeEventListener('offline', offline);
    signal?.removeEventListener('abort', abort);
    uppy.destroy();
  }
}
