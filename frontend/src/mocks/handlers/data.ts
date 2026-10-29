// "data" mock group: uploads (incl. a fake S3/MinIO), imports, dataset versions, samples, rows,
// validation, labels and AI box suggestions. Shapes follow the backend accepted contract.
// State is in memory only (uploaded images are blob: URLs), so a reload starts empty.
import { bypass, delay, http, HttpResponse, passthrough } from 'msw';
import type {
  BBox,
  ColumnMapping,
  ColumnPreview,
  ColumnType,
  CompleteFile,
  DatasetVersion,
  RowsPage,
  Sample,
  SampleLabel,
  Split,
  ValidationItem,
  ValidationReport,
} from '@/api/data-types';
import { uuid } from '../db';
import { lag, paginate, problem, u } from '../util';
import { classImage, detectionScene, rand, textSample } from '../synth';

type Kind = 'image' | 'detection' | 'text' | 'tabular';
interface VersionRec extends DatasetVersion {
  kind: Kind;
  samples: Sample[];
  rows: RowsPage['items'];
  columns: RowsPage['columns'];
  acknowledged: string[];
  /** Ground truth for synthetic detection scenes, used by box suggestions. */
  truth: Record<string, { class_name: string; bbox: BBox }[]>;
}

const S3 = `${location.origin}/__mock_s3`;
const uploads = new Map<string, { dataset_id: string; key_prefix: string; source: string; label_from_folder: boolean; label: string | null }>();
const objects = new Map<string, { size: number; mime: string; url: string; etag: string }>();
const multiparts = new Map<string, { key: string; parts: Map<number, { size: number; etag: string; blob: Blob }> }>();
const versions = new Map<string, VersionRec>();
const datasetKind = new Map<string, Kind>();

const now = () => new Date().toISOString();
const etag = () => `"${crypto.randomUUID().replace(/-/g, '')}"`;
const forDataset = (did: string) => [...versions.values()].filter(v => v.dataset_id === did).sort((a, b) => b.version_no - a.version_no);
const latestReady = (did: string) => forDataset(did).find(v => v.ingest_status === 'ready');

function publicVersion(v: VersionRec): DatasetVersion {
  const { kind: _k, samples: _s, rows: _r, columns: _c, acknowledged: _a, truth: _t, ...pub } = v;
  void [_k, _s, _r, _c, _a, _t];
  return pub;
}

function labelStats(v: VersionRec): DatasetVersion['label_stats'] {
  const stats: DatasetVersion['label_stats'] = {};
  for (const s of v.samples)
    for (const name of new Set(s.labels.map(l => l.class_name))) {
      const st = (stats[name] ??= { images: 0, ...(v.kind === 'detection' ? { boxes: 0 } : {}) });
      st.images++;
      if (v.kind === 'detection') st.boxes = (st.boxes ?? 0) + s.labels.filter(l => l.class_name === name).length;
    }
  if (v.kind === 'tabular') {
    const target = v.columns.find(c => c.role === 'target');
    if (target) for (const r of v.rows) (stats[String(r.values[target.name])] ??= { images: 0 }).images++;
  }
  return stats;
}

function newVersion(did: string, kind: Kind, origin: DatasetVersion['origin'], parent?: VersionRec): VersionRec {
  const v: VersionRec = {
    id: uuid(),
    dataset_id: did,
    kind,
    version_no: (forDataset(did)[0]?.version_no ?? 0) + 1,
    parent_version_id: parent?.id ?? null,
    origin,
    sample_count: 0,
    label_stats: {},
    split_counts: null,
    validation_status: 'pending',
    ingest_status: 'running',
    ingest_errors: [],
    sealed_at: null,
    is_draft: true,
    created_at: now(),
    samples: parent ? structuredClone(parent.samples) : [],
    rows: parent ? structuredClone(parent.rows) : [],
    columns: parent ? structuredClone(parent.columns) : [],
    acknowledged: [],
    truth: parent ? { ...parent.truth } : {},
  };
  versions.set(v.id, v);
  datasetKind.set(did, kind);
  return v;
}

/** Ingest then seal + validate on timers, like the real async pipeline. */
function finishIngest(v: VersionRec, ms = 1200) {
  setTimeout(() => {
    v.ingest_status = 'ready';
    v.sample_count = v.kind === 'tabular' ? v.rows.length : v.samples.length;
    v.label_stats = labelStats(v);
    seal(v);
  }, ms);
}

function seal(v: VersionRec) {
  v.sealed_at = now();
  v.is_draft = false;
  // Deterministic stratified-ish split 70/15/15.
  const n = v.kind === 'tabular' ? v.rows.length : v.samples.length;
  const split = (i: number): Split => (i % 20 < 14 ? 'train' : i % 20 < 17 ? 'val' : 'test');
  v.samples.forEach((s, i) => (s.split = split(i)));
  v.rows.forEach((r, i) => (r.split = split(i)));
  v.split_counts = { train: 0, val: 0, test: 0 };
  for (let i = 0; i < n; i++) v.split_counts[split(i)]++;
  v.validation_status = 'running';
  v.label_stats = labelStats(v);
  v.sample_count = n;
  setTimeout(() => {
    const r = report(v);
    v.validation_status = r.status;
    applyFlags(v, r);
  }, 1500);
}

function item(code: string, severity: ValidationItem['severity'], title: string, why: string, count: number, examples: string[], fix: string, autoFix?: string): ValidationItem {
  return { code, severity, title, why, count, examples: examples.slice(0, 8), fix: { text: fix, auto_fix_id: autoFix ?? null }, acknowledged: false };
}

/** VAL rules over the version, roughly per PRD §6.5. */
function report(v: VersionRec): ValidationReport {
  const items: ValidationItem[] = [];
  const stats = labelStats(v);
  const classes = Object.entries(stats);
  if (v.kind === 'tabular') {
    const missingCol = v.columns.find(c => v.rows.filter(r => r.values[c.name] === '' || r.values[c.name] == null).length > v.rows.length * 0.5);
    if (missingCol) items.push(item('VAL-9', 'warning', `Column "${missingCol.name}" is more than 50% empty`, 'Mostly-empty columns add noise and can hide leakage.', 1, [], 'Drop the column or impute missing values.', 'drop_sparse_columns'));
    const idCol = v.columns.find(c => c.role === 'feature' && new Set(v.rows.map(r => r.values[c.name])).size === v.rows.length);
    if (idCol) items.push(item('VAL-10', 'warning', `"${idCol.name}" looks like an ID column`, 'Unique per row, so the model would memorise it instead of learning.', 1, [], 'Exclude it from the features.', 'exclude_id_columns'));
    if (v.rows.length < 50) items.push(item('VAL-2', 'error', `Only ${v.rows.length} rows`, 'Tabular models need at least 50 rows to say anything reliable.', v.rows.length, [], 'Add more rows.'));
  } else {
    const unlabeled = v.samples.filter(s => !s.labels.length && !s.no_objects);
    if (unlabeled.length)
      items.push(
        v.kind === 'detection'
          ? item('VAL-16', 'warning', `${unlabeled.length} images have no boxes`, 'Images without boxes teach the model that nothing is there.', unlabeled.length, unlabeled.map(s => s.id), 'Draw boxes, or mark the image "no objects".')
          : item('VAL-2', 'error', `${unlabeled.length} samples have no label`, 'Every training sample needs a class.', unlabeled.length, unlabeled.map(s => s.id), 'Label them on the Label tab, or exclude them.', 'exclude_unlabeled'),
      );
    const corrupt = v.samples.filter(s => s.flags.includes('corrupt'));
    if (corrupt.length) items.push(item('VAL-1', 'error', `${corrupt.length} files could not be decoded`, 'Corrupt files crash training or silently drop.', corrupt.length, corrupt.map(s => s.id), 'Exclude the files.', 'exclude_corrupt'));
    for (const [name, st] of classes) {
      const n = v.kind === 'detection' ? (st.boxes ?? 0) : st.images;
      const unit = v.kind === 'detection' ? 'boxes' : 'samples';
      const errMin = v.kind === 'detection' ? 15 : 10;
      const warnMin = v.kind === 'detection' ? 50 : 30;
      if (n < errMin) items.push(item(v.kind === 'detection' ? 'VAL-16' : 'VAL-2', 'error', `"${name}" has only ${n} ${unit}`, `Below ${errMin} ${unit} a class can't be learned or tested.`, n, [], `Add at least ${errMin - n} more, or merge it with a similar class.`));
      else if (n < warnMin) items.push(item('VAL-3', 'warning', `"${name}" has ${n} ${unit}`, `Under ${warnMin} ${unit} accuracy for this class will be shaky (expected ±10–15%).`, n, [], `Add about ${warnMin - n} more varied examples.`));
    }
    const counts = classes.map(([, st]) => st.images).filter(Boolean);
    if (counts.length > 1 && Math.max(...counts) / Math.min(...counts) > 10)
      items.push(item('VAL-4', 'warning', 'Classes are heavily imbalanced', 'The largest class has over 10× the smallest; accuracy alone would look misleadingly good.', counts.length, [], 'Class weights are applied automatically; collecting more of the rare class helps most.'));
    const dupGroups = new Map<string, string[]>();
    v.samples.forEach(s => {
      const k = s.flags.find(f => f.startsWith('dup:'));
      if (k) dupGroups.set(k, [...(dupGroups.get(k) ?? []), s.id]);
    });
    const dups = [...dupGroups.values()].filter(g => g.length > 1).flatMap(g => g.slice(1));
    if (dups.length) items.push(item('VAL-5', 'warning', `${dups.length} exact duplicates`, 'Duplicates inflate scores and can leak between train and test.', dups.length, dups, 'Keep one copy of each.', 'dedupe_exact'));
    if (v.kind === 'text') items.push(item('VAL-14', 'info', 'Hinglish and English detected', 'About 40% of texts are romanised Hindi mixed with English.', v.samples.length, [], 'A multilingual encoder (MuRIL / XLM-R) will be preferred.'));
    if (v.kind !== 'text' && v.samples.some(s => (s.width ?? 999) < 32)) items.push(item('VAL-13', 'info', 'Some images are very small', 'Images under 32 px lose detail when resized.', 1, [], 'They will be upscaled; prefer larger photos.'));
  }
  for (const it of items) it.acknowledged = v.acknowledged.includes(it.code);
  const errors = items.filter(i => i.severity === 'error').length;
  const warnings = items.filter(i => i.severity === 'warning').length;
  const info = items.filter(i => i.severity === 'info').length;
  const status = errors ? 'failed' : items.some(i => i.severity === 'warning' && !i.acknowledged) ? 'warnings' : 'passed';
  return { status, summary: { errors, warnings, info }, items };
}

function applyFlags(v: VersionRec, r: ValidationReport) {
  for (const s of v.samples) s.flags = s.flags.filter(f => !f.startsWith('VAL-'));
  for (const it of r.items) for (const id of it.examples) v.samples.find(s => s.id === id)?.flags.push(it.code);
}

// ---------- demo seed ----------
function seedVersion(did: string, task: string, labels: string[]): VersionRec {
  const kind: Kind = task === 'object_detection' ? 'detection' : task === 'text_classification' ? 'text' : task.startsWith('tabular') ? 'tabular' : 'image';
  const v = newVersion(did, kind, 'upload', latestReady(did));
  if (kind === 'image') {
    labels.forEach((l, li) => {
      const n = [42, 28, 9][li % 3]!;
      for (let i = 0; i < n; i++) {
        const img = classImage(l, i);
        v.samples.push({ id: uuid(), thumb_url: img.url, url: img.url, width: img.w, height: img.h, mime: 'image/svg+xml', split: null, labels: [{ class_name: l, bbox: null, source: 'import' }], no_objects: false, flags: i === 3 ? ['dup:a'] : i === 4 ? ['dup:a'] : [] });
      }
    });
    for (let i = 0; i < 6; i++) {
      const img = classImage('unsorted', i);
      v.samples.push({ id: uuid(), thumb_url: img.url, url: img.url, width: img.w, height: img.h, mime: 'image/svg+xml', split: null, labels: [], no_objects: false, flags: [] });
    }
  } else if (kind === 'detection') {
    for (let i = 0; i < 24; i++) {
      const sc = detectionScene(labels, i);
      const id = uuid();
      v.truth[id] = sc.boxes;
      // The first 8 come pre-labelled ("imported"); the rest are for the box tool + AI suggestions.
      v.samples.push({ id, thumb_url: sc.url, url: sc.url, width: sc.w, height: sc.h, mime: 'image/svg+xml', split: null, labels: i < 8 ? sc.boxes.map(b => ({ ...b, source: 'import' as const })) : [], no_objects: false, flags: [] });
    }
  } else if (kind === 'text') {
    labels.forEach((l, li) => {
      for (let i = 0; i < [36, 24, 14][li % 3]!; i++) v.samples.push({ id: uuid(), thumb_url: '', url: '', width: null, height: null, mime: 'text/plain', split: null, labels: [{ class_name: l, bbox: null, source: 'import' }], no_objects: false, flags: [], text: textSample(l, i) });
    });
  } else {
    v.columns = [
      { name: 'customer_id', type: 'categorical', role: 'feature' },
      { name: 'tenure_months', type: 'numeric', role: 'feature' },
      { name: 'monthly_spend', type: 'numeric', role: 'feature' },
      { name: 'plan', type: 'categorical', role: 'feature' },
      { name: 'support_tickets', type: 'numeric', role: 'feature' },
      { name: task === 'tabular_regression' ? 'next_month_spend' : 'churned', type: task === 'tabular_regression' ? 'numeric' : 'categorical', role: 'target' },
    ];
    for (let i = 0; i < 240; i++) {
      const tenure = Math.floor(rand() * 60);
      const spend = Math.round(200 + rand() * 1800);
      v.rows.push({
        row: i + 1,
        split: null,
        values: {
          customer_id: `C${10000 + i}`,
          tenure_months: tenure,
          monthly_spend: spend,
          plan: ['basic', 'plus', 'pro'][i % 3],
          support_tickets: Math.floor(rand() * 6),
          [task === 'tabular_regression' ? 'next_month_spend' : 'churned']: task === 'tabular_regression' ? Math.round(spend * (0.8 + rand() * 0.4)) : tenure < 8 && rand() < 0.7 ? 'yes' : 'no',
        },
      });
    }
  }
  finishIngest(v, 600);
  return v;
}

// ---------- CSV / import helpers ----------
function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = '';
  let q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i]!;
    if (q) {
      if (c === '"' && text[i + 1] === '"') (cell += '"'), i++;
      else if (c === '"') q = false;
      else cell += c;
    } else if (c === '"') q = true;
    else if (c === ',') row.push(cell), (cell = '');
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(cell);
      rows.push(row);
      row = [];
      cell = '';
    } else cell += c;
  }
  if (cell || row.length) row.push(cell), rows.push(row);
  return rows.filter(r => r.some(x => x.trim()));
}

function inferType(values: string[]): ColumnType {
  const v = values.filter(x => x !== '');
  if (v.length && v.every(x => /^-?\d+(\.\d+)?$/.test(x))) return new Set(v).size <= 2 ? 'boolean' : 'numeric';
  if (v.length && v.every(x => !Number.isNaN(Date.parse(x)) && /\d{4}-\d{2}-\d{2}/.test(x))) return 'datetime';
  if (v.some(x => x.length > 40)) return 'text';
  return 'categorical';
}

async function tableFor(key: string): Promise<{ header: string[]; body: string[][] } | null> {
  const obj = objects.get(key);
  if (!obj) return null;
  const text = await (await fetch(obj.url)).text();
  const rows = parseCsv(text);
  return rows.length ? { header: rows[0]!, body: rows.slice(1) } : null;
}

export const dataHandlers = [
  // Demo-only: fill a dataset with synthetic samples ("Load sample data" button).
  http.post(u('/projects/:pid/datasets/:did/_demo-seed'), async ({ request, params }) => {
    const { task_type, labels } = (await request.json()) as { task_type: string; labels: string[] };
    await lag();
    const v = seedVersion(params.did as string, task_type, labels.length ? labels : ['class_a', 'class_b']);
    return HttpResponse.json({ dataset_version_id: v.id }, { status: 202 });
  }),

  // ----- uploads -----
  http.post(u('/projects/:pid/datasets/:did/uploads'), async ({ request, params }) => {
    const body = (await request.json()) as { source: string; label_from_folder: boolean; file_count: number; total_bytes: number; label?: string };
    if (body.file_count > 50000) return problem(413, 'Too many files', { code: 'too_many_files' });
    const upload_id = uuid();
    const key_prefix = `orgs/demo/datasets/${params.did}/uploads/${upload_id}/`;
    uploads.set(upload_id, { dataset_id: params.did as string, key_prefix, source: body.source, label_from_folder: body.label_from_folder, label: body.label ?? null });
    return HttpResponse.json({ upload_id, key_prefix, expires_at: new Date(Date.now() + 3600_000).toISOString(), max_file_bytes: 20 * 1024 ** 3, part_size: 8 * 1024 ** 2 }, { status: 201 });
  }),

  http.post(u('/uploads/:id/sign'), async ({ request, params }) => {
    const up = uploads.get(params.id as string);
    if (!up) return problem(404, 'Upload not found');
    const r = (await request.json()) as { method: string; key: string; upload_id?: string; part_number?: number; size?: number };
    if (!r.key.startsWith(up.key_prefix) || r.key.includes('..') || r.key.length > 1024) return problem(422, 'Invalid key', { code: 'invalid_key' });
    if ((r.size ?? 0) > 20 * 1024 ** 3) return problem(413, 'File too large', { code: 'file_too_large' });
    const qs = new URLSearchParams({ 'X-Amz-Signature': 'mock' });
    if (r.upload_id) qs.set('uploadId', r.upload_id);
    if (r.part_number) qs.set('partNumber', String(r.part_number));
    if (r.method === 'POST' && !r.upload_id) qs.set('uploads', '');
    return HttpResponse.json({ url: `${S3}/${r.key}?${qs}`, headers: {} });
  }),

  // Fake MinIO: simple PUT, multipart create / part / list / complete / abort.
  http.put(`${S3}/*`, async ({ request }) => {
    const url = new URL(request.url);
    const key = decodeURIComponent(url.pathname.replace('/__mock_s3/', ''));
    const blob = await request.blob();
    await delay(Math.min(1500, 80 + blob.size / 4000));
    const tag = etag();
    const mpId = url.searchParams.get('uploadId');
    if (mpId) {
      multiparts.get(mpId)?.parts.set(Number(url.searchParams.get('partNumber')), { size: blob.size, etag: tag, blob });
    } else objects.set(key, { size: blob.size, mime: blob.type || request.headers.get('Content-Type') || 'application/octet-stream', url: URL.createObjectURL(blob), etag: tag });
    return new HttpResponse(null, { status: 200, headers: { ETag: tag } });
  }),
  http.post(`${S3}/*`, async ({ request }) => {
    const url = new URL(request.url);
    const key = decodeURIComponent(url.pathname.replace('/__mock_s3/', ''));
    const xml = (s: string) => new HttpResponse(`<?xml version="1.0" encoding="UTF-8"?>${s}`, { headers: { 'Content-Type': 'application/xml' } });
    if (url.searchParams.has('uploads')) {
      const id = crypto.randomUUID();
      multiparts.set(id, { key, parts: new Map() });
      return xml(`<InitiateMultipartUploadResult><Bucket>automl</Bucket><Key>${key}</Key><UploadId>${id}</UploadId></InitiateMultipartUploadResult>`);
    }
    const mp = multiparts.get(url.searchParams.get('uploadId') ?? '');
    if (!mp) return new HttpResponse('<Error><Code>NoSuchUpload</Code></Error>', { status: 404, headers: { 'Content-Type': 'application/xml' } });
    const parts = [...mp.parts.entries()].sort(([a], [b]) => a - b).map(([, p]) => p.blob);
    const blob = new Blob(parts);
    const tag = etag();
    objects.set(key, { size: blob.size, mime: 'application/octet-stream', url: URL.createObjectURL(blob), etag: tag });
    multiparts.delete(url.searchParams.get('uploadId')!);
    return xml(`<CompleteMultipartUploadResult><Location>${S3}/${key}</Location><Bucket>automl</Bucket><Key>${key}</Key><ETag>${tag}</ETag></CompleteMultipartUploadResult>`);
  }),
  http.get(`${S3}/*`, ({ request }) => {
    const url = new URL(request.url);
    const mp = multiparts.get(url.searchParams.get('uploadId') ?? '');
    const parts = [...(mp?.parts.entries() ?? [])].map(([n, p]) => `<Part><PartNumber>${n}</PartNumber><ETag>${p.etag}</ETag><Size>${p.size}</Size></Part>`).join('');
    return new HttpResponse(`<?xml version="1.0" encoding="UTF-8"?><ListPartsResult>${parts}</ListPartsResult>`, { headers: { 'Content-Type': 'application/xml' } });
  }),
  http.delete(`${S3}/*`, ({ request }) => {
    multiparts.delete(new URL(request.url).searchParams.get('uploadId') ?? '');
    return new HttpResponse(null, { status: 204 });
  }),

  http.post(u('/uploads/:id/complete'), async ({ request, params }) => {
    const up = uploads.get(params.id as string);
    if (!up) return problem(404, 'Upload not found');
    const { files } = (await request.json()) as { files: CompleteFile[] };
    const missing = files.filter(f => objects.get(f.key)?.size !== f.size);
    if (missing.length) return problem(409, 'Upload incomplete', { code: 'upload_incomplete', detail: `${missing.length} file(s) missing or size mismatch` });
    await lag();
    if (up.source === 'annotations') return HttpResponse.json({ upload_id: params.id, dataset_version_id: null, status: 'stored' }, { status: 202 });
    const kind = datasetKind.get(up.dataset_id) ?? 'image';
    const v = newVersion(up.dataset_id, kind === 'tabular' || kind === 'text' ? 'image' : kind, 'upload', latestReady(up.dataset_id));
    for (const f of files) {
      const obj = objects.get(f.key)!;
      if (f.name.toLowerCase().endsWith('.zip')) {
        // The real server expands zips; the mock fakes a small expanded set.
        for (let i = 0; i < 18; i++) {
          const l = up.label_from_folder ? ['cats', 'dogs'][i % 2]! : null;
          const img = classImage(l ?? 'unsorted', i);
          v.samples.push({ id: uuid(), thumb_url: img.url, url: img.url, width: img.w, height: img.h, mime: 'image/svg+xml', split: null, labels: l ? [{ class_name: l, bbox: null, source: 'user' }] : [], no_objects: false, flags: [] });
        }
        continue;
      }
      const folder = f.relative_path?.split('/').slice(-2, -1)[0] ?? null;
      const label = f.label ?? up.label ?? (up.label_from_folder ? folder : null);
      const dims = await new Promise<{ w: number; h: number }>(res => {
        const im = new Image();
        im.onload = () => res({ w: im.naturalWidth, h: im.naturalHeight });
        im.onerror = () => res({ w: 0, h: 0 });
        im.src = obj.url;
      });
      v.samples.push({
        id: uuid(),
        thumb_url: obj.url,
        url: obj.url,
        width: dims.w || null,
        height: dims.h || null,
        mime: f.mime,
        split: null,
        labels: label ? [{ class_name: label, bbox: null, source: 'user' }] : [],
        no_objects: false,
        flags: f.size === 0 || (!dims.w && f.mime.startsWith('image/')) ? ['corrupt'] : [],
      });
    }
    finishIngest(v);
    return HttpResponse.json({ upload_id: params.id, dataset_version_id: v.id, status: 'ingesting' }, { status: 202 });
  }),

  // ----- imports (tables and detection annotations) -----
  http.post(u('/projects/:pid/datasets/:did/imports'), async ({ request, params }) => {
    const body = (await request.json()) as { upload_id: string; format: string; key: string; mapping?: ColumnMapping };
    await lag();
    const did = params.did as string;
    if (['csv', 'xlsx', 'parquet'].includes(body.format)) {
      const table = body.format === 'csv' ? await tableFor(body.key) : null;
      const header = table?.header ?? ['id', 'text', 'label', 'created_at'];
      const rows = table?.body ?? Array.from({ length: 60 }, (_, i) => [String(i + 1), textSample(['positive', 'negative', 'neutral'][i % 3]!, i), ['positive', 'negative', 'neutral'][i % 3]!, '2026-10-0' + ((i % 9) + 1)]);
      if (!body.mapping) {
        const columns: ColumnPreview[] = header.map((name, c) => {
          const vals = rows.map(r => r[c] ?? '');
          return { name, inferred_type: inferType(vals), sample_values: vals.slice(0, 5), unique_count: new Set(vals).size, missing_pct: Math.round((vals.filter(x => x === '').length / Math.max(1, vals.length)) * 100) };
        });
        return HttpResponse.json({ columns, row_count: rows.length });
      }
      const targets = body.mapping.columns.filter(c => c.role === 'target');
      if (targets.length !== 1) return problem(422, 'Mapping invalid', { errors: [{ loc: '/mapping', msg: 'Choose exactly one target column' }] });
      const textCol = body.mapping.columns.find(c => c.role === 'text');
      const target = targets[0]!;
      const idx = (n: string) => header.indexOf(n);
      if (textCol) {
        const v = newVersion(did, 'text', 'import', latestReady(did));
        for (const r of rows)
          v.samples.push({ id: uuid(), thumb_url: '', url: '', width: null, height: null, mime: 'text/plain', split: null, text: r[idx(textCol.name)] ?? '', labels: r[idx(target.name)] ? [{ class_name: r[idx(target.name)]!, bbox: null, source: 'import' }] : [], no_objects: false, flags: [] });
        finishIngest(v);
        return HttpResponse.json({ dataset_version_id: v.id }, { status: 202 });
      }
      const v = newVersion(did, 'tabular', 'import', latestReady(did));
      v.columns = body.mapping.columns.filter(c => c.role !== 'ignore');
      v.rows = rows.map((r, i) => ({ row: i + 1, split: null, values: Object.fromEntries(v.columns.map(c => [c.name, c.type === 'numeric' && r[idx(c.name)] !== '' ? Number(r[idx(c.name)]) : r[idx(c.name)] ?? ''])) }));
      finishIngest(v);
      return HttpResponse.json({ dataset_version_id: v.id }, { status: 202 });
    }
    // Detection: COCO json is parsed for real; YOLO / VOC are approximated.
    const v = newVersion(did, 'detection', 'import', latestReady(did));
    const imagesInUpload = [...objects.entries()].filter(([k, o]) => k.startsWith(uploads.get(body.upload_id)?.key_prefix ?? '#') && o.mime.startsWith('image/'));
    const byName = new Map(imagesInUpload.map(([k, o]) => [k.split('/').pop()!.replace(/^[0-9a-f]{8}-/, ''), o]));
    let coco: { images: { id: number; file_name: string; width: number; height: number }[]; annotations: { image_id: number; category_id: number; bbox: number[] }[]; categories: { id: number; name: string }[] } | null = null;
    if (body.format === 'coco') {
      try {
        coco = JSON.parse(await (await fetch(objects.get(body.key)!.url)).text());
      } catch {
        return problem(422, 'Invalid COCO file', { errors: [{ loc: '/key', msg: 'Not valid COCO JSON' }] });
      }
    }
    if (coco) {
      const cats = new Map(coco.categories.map(c => [c.id, c.name]));
      for (const im of coco.images) {
        const obj = byName.get(im.file_name.split('/').pop()!);
        const labels: SampleLabel[] = coco.annotations
          .filter(a => a.image_id === im.id)
          .map(a => ({ class_name: cats.get(a.category_id) ?? 'unknown', bbox: { x: a.bbox[0]!, y: a.bbox[1]!, w: a.bbox[2]!, h: a.bbox[3]! }, source: 'import' }));
        v.samples.push({ id: uuid(), thumb_url: obj?.url ?? '', url: obj?.url ?? '', width: im.width, height: im.height, mime: 'image/png', split: null, labels, no_objects: false, flags: obj ? [] : ['corrupt'] });
      }
    } else
      for (const [, o] of imagesInUpload)
        v.samples.push({ id: uuid(), thumb_url: o.url, url: o.url, width: 640, height: 480, mime: o.mime, split: null, labels: [], no_objects: false, flags: [] });
    finishIngest(v);
    return HttpResponse.json({ dataset_version_id: v.id }, { status: 202 });
  }),

  // ----- versions, samples, rows -----
  http.get(u('/datasets/:did/versions'), ({ request, params }) => HttpResponse.json(paginate(forDataset(params.did as string).map(publicVersion), new URL(request.url)))),
  http.get(u('/dataset-versions/:vid'), ({ params }) => {
    const v = versions.get(params.vid as string);
    return v ? HttpResponse.json(publicVersion(v)) : problem(404, 'Not found');
  }),
  http.get(u('/dataset-versions/:vid/samples'), async ({ request, params }) => {
    const v = versions.get(params.vid as string);
    if (!v) return problem(404, 'Not found');
    await delay(150);
    const url = new URL(request.url);
    const label = url.searchParams.get('label');
    const split = url.searchParams.get('split');
    const flag = url.searchParams.get('flag');
    const unlabeled = url.searchParams.get('unlabeled') === 'true';
    const items = v.samples.filter(
      s => (!label || s.labels.some(l => l.class_name === label)) && (!split || s.split === split) && (!flag || s.flags.includes(flag)) && (!unlabeled || (!s.labels.length && !s.no_objects)),
    );
    return HttpResponse.json(paginate(items, url));
  }),
  http.get(u('/dataset-versions/:vid/rows'), ({ request, params }) => {
    const v = versions.get(params.vid as string);
    if (!v) return problem(404, 'Not found');
    return HttpResponse.json({ columns: v.columns, ...paginate(v.rows, new URL(request.url)) });
  }),

  http.post(u('/dataset-versions/:vid/seal'), ({ params }) => {
    const v = versions.get(params.vid as string);
    if (!v) return problem(404, 'Not found');
    seal(v);
    return HttpResponse.json({ dataset_version_id: v.id, validation_status: 'running' }, { status: 202 });
  }),

];

// ---------- overlay mode: labels/validation mocked on top of the REAL data API ----------
// When a version id isn't in the mock store it came from the backend backend. Labels saved here live in
// a client-side overlay (merged into real /samples responses); validation is computed from real samples.
const overlay = new Map<string, { labels: SampleLabel[]; no_objects: boolean }>();
const OVERLAY_KEY = 'automl.labels-overlay.v1';
try {
  for (const [k, v] of Object.entries(JSON.parse(localStorage.getItem(OVERLAY_KEY) ?? '{}'))) overlay.set(k, v as { labels: SampleLabel[]; no_objects: boolean });
} catch {
  /* ignore */
}
const saveOverlay = () => {
  try {
    localStorage.setItem(OVERLAY_KEY, JSON.stringify(Object.fromEntries(overlay)));
  } catch {
    /* ignore */
  }
};
const seenSamples = new Map<string, Sample>();
const realAcks = new Map<string, string[]>();
const withOverlay = (s: Sample): Sample => {
  seenSamples.set(s.id, s);
  const o = overlay.get(s.id);
  return o ? { ...s, labels: o.labels, no_objects: o.no_objects } : s;
};

/** Build a throwaway VersionRec from the real API so the mock validator can run over it. */
async function realVersion(vid: string): Promise<VersionRec | null> {
  const meta = await fetch(bypass(u(`/dataset-versions/${vid}`)), { headers: authHeader() });
  if (!meta.ok) return null;
  const pub = (await meta.json()) as DatasetVersion;
  const samples: Sample[] = [];
  let cursor: string | null = null;
  do {
    const r = await fetch(bypass(u(`/dataset-versions/${vid}/samples?limit=500${cursor ? `&cursor=${cursor}` : ''}`)), { headers: authHeader() });
    if (!r.ok) break;
    const page = (await r.json()) as { items: Sample[]; next_cursor: string | null };
    samples.push(...page.items.map(withOverlay));
    cursor = page.next_cursor;
  } while (cursor && samples.length < 5000);
  let rows: RowsPage['items'] = [];
  let columns: RowsPage['columns'] = [];
  if (!samples.length) {
    const r = await fetch(bypass(u(`/dataset-versions/${vid}/rows?limit=500`)), { headers: authHeader() });
    if (r.ok) ({ items: rows, columns } = (await r.json()) as RowsPage);
  }
  const kind: Kind = rows.length ? 'tabular' : samples.some(x => x.text) ? 'text' : samples.some(x => x.labels.some(l => l.bbox)) || datasetKind.get(pub.dataset_id) === 'detection' ? 'detection' : 'image';
  return { ...pub, kind, samples, rows, columns, acknowledged: realAcks.get(vid) ?? [], truth: {} };
}

let lastAuth = '';
const authHeader = (): Record<string, string> => (lastAuth ? { Authorization: lastAuth } : {});
const remember = (request: Request) => {
  lastAuth = request.headers.get('Authorization') ?? lastAuth;
};

export const validationHandlers = [
  http.get(u('/dataset-versions/:vid/validation'), async ({ request, params }) => {
    remember(request);
    const v = versions.get(params.vid as string) ?? (await realVersion(params.vid as string));
    if (!v) return problem(404, 'Not found');
    if (v.validation_status === 'running' || v.validation_status === 'pending') return HttpResponse.json({ status: v.validation_status, summary: { errors: 0, warnings: 0, info: 0 }, items: [] });
    const r = report(v);
    applyFlags(v, r);
    return HttpResponse.json(r);
  }),
  http.post(u('/dataset-versions/:vid/validation/acknowledge'), async ({ request, params }) => {
    remember(request);
    const vid = params.vid as string;
    const { codes } = (await request.json()) as { codes: string[] };
    const local = versions.get(vid);
    if (local) local.acknowledged = [...new Set([...local.acknowledged, ...codes])];
    else realAcks.set(vid, [...new Set([...(realAcks.get(vid) ?? []), ...codes])]);
    const v = local ?? (await realVersion(vid));
    if (!v) return problem(404, 'Not found');
    const r = report(v);
    v.validation_status = r.status;
    return HttpResponse.json(r);
  }),
  http.post(u('/dataset-versions/:vid/validation/fixes'), async ({ request, params }) => {
    const v = versions.get(params.vid as string);
    if (!v) return problem(501, 'Auto-fix needs the validation backend', { code: 'not_implemented', detail: 'Auto-fixes arrive with the validation service (group 7). Fix it by hand for now.' });
    const { fix_ids } = (await request.json()) as { fix_ids: string[] };
    await lag();
    const nv = newVersion(v.dataset_id, v.kind, 'fix', v);
    if (fix_ids.includes('dedupe_exact')) {
      const seen = new Set<string>();
      nv.samples = nv.samples.filter(s => {
        const d = s.flags.find(f => f.startsWith('dup:'));
        if (!d) return true;
        if (seen.has(d)) return false;
        seen.add(d);
        return true;
      });
    }
    if (fix_ids.includes('exclude_corrupt')) nv.samples = nv.samples.filter(s => !s.flags.includes('corrupt'));
    if (fix_ids.includes('exclude_unlabeled')) nv.samples = nv.samples.filter(s => s.labels.length || s.no_objects);
    if (fix_ids.includes('exclude_id_columns') || fix_ids.includes('drop_sparse_columns')) {
      const drop = nv.columns.filter(c => c.role === 'feature' && new Set(nv.rows.map(r => r.values[c.name])).size === nv.rows.length).map(c => c.name);
      nv.columns = nv.columns.filter(c => !drop.includes(c.name));
    }
    finishIngest(nv, 500);
    return HttpResponse.json({ dataset_version_id: nv.id }, { status: 202 });
  }),
];

export const labelHandlers = [
  // Real /samples responses get overlay labels merged in (only for versions served by the real API).
  http.get(u('/dataset-versions/:vid/samples'), async ({ request, params }) => {
    if (versions.has(params.vid as string)) return passthrough();
    remember(request);
    const res = await fetch(bypass(request));
    if (!res.ok) return res;
    const page = (await res.json()) as { items: Sample[]; next_cursor: string | null };
    const url = new URL(request.url);
    let items = page.items.map(withOverlay);
    if (url.searchParams.get('unlabeled') === 'true') items = items.filter(x => !x.labels.length && !x.no_objects);
    return HttpResponse.json({ ...page, items });
  }),
  http.post(u('/dataset-versions/:vid/labels\\:batch'), async ({ request, params }) => {
    if (!versions.has(params.vid as string)) {
      const { upserts } = (await request.json()) as { upserts: { sample_id: string; labels: SampleLabel[]; no_objects?: boolean }[] };
      for (const up of upserts) overlay.set(up.sample_id, { labels: up.labels.map(l => ({ ...l, id: l.id ?? uuid(), source: l.source ?? 'user' })), no_objects: up.no_objects ?? false });
      saveOverlay();
      return HttpResponse.json({ updated: upserts.length, dataset_version_id: params.vid });
    }
    let v = versions.get(params.vid as string);
    if (!v) return problem(404, 'Not found');
    const { upserts } = (await request.json()) as { upserts: { sample_id: string; labels: SampleLabel[]; no_objects?: boolean }[] };
    if (upserts.some(x => !v!.samples.some(s => s.id === x.sample_id))) return problem(422, 'Unknown sample', { errors: [{ loc: '/upserts', msg: 'sample_id not in this version' }] });
    // Writing to a sealed version forks a draft; samples keep their ids in the mock.
    if (v.sealed_at) {
      const draft = newVersion(v.dataset_id, v.kind, 'labels', v);
      draft.samples = v.samples.map(s => ({ ...s, labels: [...s.labels] }));
      draft.ingest_status = 'ready';
      draft.sample_count = draft.samples.length;
      v = draft;
    }
    for (const up of upserts) {
      const s = v.samples.find(x => x.id === up.sample_id)!;
      s.labels = up.labels.map(l => ({ ...l, id: l.id ?? uuid(), source: l.source ?? 'user' }));
      if (up.no_objects !== undefined) s.no_objects = up.no_objects;
      if (s.labels.some(l => l.bbox)) v.kind = 'detection';
    }
    v.label_stats = labelStats(v);
    return HttpResponse.json({ updated: upserts.length, dataset_version_id: v.id });
  }),
  // ----- AI box suggestions (LBL-4a) -----
  http.post(u('/dataset-versions/:vid/box-suggestions'), async ({ request, params }) => {
    const v = versions.get(params.vid as string);
    const { sample_ids, classes } = (await request.json()) as { sample_ids: string[]; classes?: string[] };
    if (sample_ids.length > 50) return problem(422, 'Too many samples', { errors: [{ loc: '/sample_ids', msg: 'At most 50' }] });
    await delay(700);
    const jitter = (n: number) => n + (rand() - 0.5) * 8;
    const suggestions: Record<string, { class_name: string; bbox: BBox; confidence: number }[]> = {};
    for (const id of sample_ids) {
      const s = v?.samples.find(x => x.id === id) ?? seenSamples.get(id);
      const truth = v?.truth[id];
      if (truth)
        suggestions[id] = truth.map(t => ({ class_name: t.class_name, bbox: { x: jitter(t.bbox.x), y: jitter(t.bbox.y), w: jitter(t.bbox.w), h: jitter(t.bbox.h) }, confidence: Math.round((0.55 + rand() * 0.42) * 100) / 100 }));
      else if (s?.width && s.height)
        suggestions[id] = [{ class_name: (classes ?? ['object'])[0]!, bbox: { x: s.width * 0.3, y: s.height * 0.25, w: s.width * 0.35, h: s.height * 0.45 }, confidence: 0.41 }];
      else suggestions[id] = [];
    }
    return HttpResponse.json({ model: 'owlv2-base (demo)', suggestions });
  }),
];
