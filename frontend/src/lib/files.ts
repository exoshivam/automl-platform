// File helpers for uploads: folder drag-and-drop with relative paths, checksums, MIME guessing.

export interface PickedFile {
  file: File;
  /** "class_a/img1.jpg" for folder picks/drops; null for loose files. */
  relativePath: string | null;
}

const MIME: Record<string, string> = {
  jpg: 'image/jpeg', jpeg: 'image/jpeg', png: 'image/png', webp: 'image/webp', bmp: 'image/bmp', tif: 'image/tiff', tiff: 'image/tiff', heic: 'image/heic',
  zip: 'application/zip', csv: 'text/csv', tsv: 'text/tab-separated-values', json: 'application/json', jsonl: 'application/jsonl', txt: 'text/plain', xml: 'application/xml',
  xlsx: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', parquet: 'application/vnd.apache.parquet',
};
export const ext = (name: string) => name.split('.').pop()?.toLowerCase() ?? '';
export const mimeOf = (f: File) => f.type || MIME[ext(f.name)] || 'application/octet-stream';
export const isImage = (f: File) => mimeOf(f).startsWith('image/');

export function fromInput(list: FileList | null): PickedFile[] {
  return [...(list ?? [])].map(file => ({ file, relativePath: file.webkitRelativePath || null }));
}

/** Walk dropped folders (webkitGetAsEntry) so subfolder names survive as labels. */
export async function fromDrop(dt: DataTransfer): Promise<PickedFile[]> {
  const entries = [...dt.items].map(i => i.webkitGetAsEntry?.()).filter((e): e is FileSystemEntry => !!e);
  if (!entries.length) return fromInput(dt.files);
  const out: PickedFile[] = [];
  const walk = async (entry: FileSystemEntry, path: string): Promise<void> => {
    if (entry.isFile) {
      const file = await new Promise<File>((res, rej) => (entry as FileSystemFileEntry).file(res, rej));
      out.push({ file, relativePath: path ? `${path}/${file.name}` : null });
    } else if (entry.isDirectory) {
      const reader = (entry as FileSystemDirectoryEntry).createReader();
      // readEntries returns batches (100 in Chrome) until empty.
      for (;;) {
        const batch = await new Promise<FileSystemEntry[]>((res, rej) => reader.readEntries(res, rej));
        if (!batch.length) break;
        for (const e of batch) await walk(e, path ? `${path}/${entry.name}` : entry.name);
      }
    }
  };
  for (const e of entries) await walk(e, '');
  return out;
}

/** Skip OS junk files that ride along with folder picks. */
export const isJunk = (p: PickedFile) => /(^|\/)(\.DS_Store|Thumbs\.db|desktop\.ini|__MACOSX)(\/|$)/i.test(p.relativePath ?? p.file.name) || p.file.name.startsWith('._');

/** The folder a file sits in (its class when "subfolders are labels"). */
export const parentFolder = (p: PickedFile) => {
  const parts = (p.relativePath ?? '').split('/');
  return parts.length >= 2 ? parts[parts.length - 2]! : null;
};

/** DAT-4: client checksum for files under 200 MB (the server recomputes anyway). */
export async function sha256(file: Blob): Promise<string | null> {
  if (file.size > 200 * 1024 * 1024 || !crypto.subtle) return null;
  const buf = await crypto.subtle.digest('SHA-256', await file.arrayBuffer());
  return [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2, '0')).join('');
}

export function formatBytes(n: number) {
  if (n < 1024) return `${n} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) (v /= 1024), i++;
  return `${v.toFixed(v < 10 ? 1 : 0)} ${units[i]}`;
}

/** S3 key segment: keep it readable, no "..", no leading "/". */
export const safeName = (s: string) => s.replace(/\.\.+/g, '.').replace(/[^\w.\-/]+/g, '_').replace(/^\/+/, '').slice(-180);
