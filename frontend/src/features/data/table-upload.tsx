import { useState } from 'react';
import { AlertTriangle, FileSpreadsheet, Loader2 } from 'lucide-react';
import { useImport, useImportPreview } from '@/api/data-hooks';
import type { ColumnMapping, ColumnPreview, ColumnRole, ColumnType, ImportFormat, ImportPreview } from '@/api/data-types';
import type { Project } from '@/api/types';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { ApiError } from '@/lib/api';
import { ext, formatBytes, type PickedFile } from '@/lib/files';
import { uploadFiles } from '@/lib/uploader';
import { cn } from '@/lib/utils';
import { DropZone } from './drop-zone';
import { IngestStatus, UploadProgressPanel, useUploadRun } from './upload-status';

const FORMAT: Record<string, ImportFormat> = { csv: 'csv', tsv: 'csv', xlsx: 'xlsx', parquet: 'parquet', jsonl: 'jsonl' };
const ROLES: { id: ColumnRole; label: string }[] = [
  { id: 'target', label: 'Target (predict this)' },
  { id: 'feature', label: 'Feature' },
  { id: 'text', label: 'Text' },
  { id: 'id', label: 'ID' },
  { id: 'timestamp', label: 'Timestamp' },
  { id: 'group', label: 'Group (keep together in splits)' },
  { id: 'ignore', label: 'Ignore' },
];
const TYPES: ColumnType[] = ['categorical', 'numeric', 'text', 'datetime', 'boolean'];

/** First guess at roles (DAT-2); everything stays editable. */
function guessRoles(p: ImportPreview, text: boolean): ColumnMapping['columns'] {
  const targetName =
    p.columns.find(c => /^(label|class|target|sentiment|category|y|churn(ed)?|price|outcome)$/i.test(c.name))?.name ??
    (text ? p.columns.find(c => c.inferred_type === 'categorical' && c.unique_count <= 50)?.name : p.columns[p.columns.length - 1]?.name);
  const textName = text ? [...p.columns].filter(c => c.inferred_type === 'text' || c.name.toLowerCase().includes('text')).sort((a, b) => avgLen(b) - avgLen(a))[0]?.name : undefined;
  return p.columns.map(c => {
    let role: ColumnRole = text ? 'ignore' : 'feature';
    if (c.name === targetName) role = 'target';
    else if (c.name === textName) role = 'text';
    else if (/(^|_)(id|uuid)$/i.test(c.name) || (c.unique_count === p.row_count && c.inferred_type !== 'numeric' && p.row_count > 20)) role = 'id';
    else if (c.inferred_type === 'datetime') role = 'timestamp';
    return { name: c.name, role, type: role === 'text' ? 'text' : c.inferred_type };
  });
}
const avgLen = (c: ColumnPreview) => c.sample_values.reduce<number>((n, v) => n + String(v ?? '').length, 0) / Math.max(1, c.sample_values.length);

export function TableUpload({ project }: { project: Project }) {
  const text = project.task_type === 'text_classification';
  const [file, setFile] = useState<PickedFile | null>(null);
  const [staged, setStaged] = useState<{ uploadId: string; key: string; format: ImportFormat } | null>(null);
  const [preview, setPreview] = useState<ImportPreview | null>(null);
  const [mapping, setMapping] = useState<ColumnMapping['columns']>([]);
  const [versionId, setVersionId] = useState<string | null>(null);
  const up = useUploadRun();
  const previewer = useImportPreview();
  const importer = useImport();

  const fmt = file ? FORMAT[ext(file.file.name)] : undefined;
  const targets = mapping.filter(c => c.role === 'target').length;
  const texts = mapping.filter(c => c.role === 'text').length;
  const mappingError =
    targets !== 1 ? 'Choose exactly one target column.' : text && texts !== 1 ? 'Choose exactly one text column.' : !text && !mapping.some(c => c.role === 'feature') ? 'Pick at least one feature column.' : null;
  const serverError = importer.error instanceof ApiError ? importer.error : null;

  const uploadAndPreview = () =>
    up.run(async (signal, onProgress) => {
      const r = await uploadFiles({ projectId: project.id, datasetId: project.dataset_id!, source: 'annotations', files: [file!], signal, onProgress });
      const s = { uploadId: r.uploadId, key: r.keys[0]!, format: fmt! };
      setStaged(s);
      const p = await previewer.mutateAsync({ projectId: project.id, datasetId: project.dataset_id!, ...s });
      setPreview(p);
      setMapping(guessRoles(p, text));
      return r;
    });

  const doImport = () =>
    importer.mutate(
      { projectId: project.id, datasetId: project.dataset_id!, ...staged!, mapping: { columns: mapping } },
      { onSuccess: r => setVersionId(r.dataset_version_id) },
    );

  if (versionId) return <IngestStatus projectId={project.id} datasetId={project.dataset_id!} versionId={versionId} onAgain={() => (setFile(null), setStaged(null), setPreview(null), setVersionId(null), up.reset())} />;

  if (!file)
    return (
      <DropZone
        folder={false}
        multiple={false}
        accept=".csv,.tsv,.xlsx,.parquet,.jsonl"
        onFiles={f => f[0] && setFile(f[0])}
        title={text ? 'Drop a CSV / Excel / JSONL file with your texts and labels' : 'Drop a CSV, Excel or Parquet file'}
        hint={text ? 'One row per text: a column with the text and a column with its label. Hindi, Hinglish and other Indic scripts are fine.' : 'One row per example, one column per field. Next you pick the column to predict.'}
      />
    );

  return (
    <Card className="grid gap-4 p-5">
      <div className="flex flex-wrap items-center gap-3">
        <FileSpreadsheet className="size-5 text-muted-foreground" />
        <span className="font-medium">{file.file.name}</span>
        <span className="text-sm text-muted-foreground">{formatBytes(file.file.size)}</span>
        {preview && <span className="text-sm text-muted-foreground">· {preview.row_count.toLocaleString()} rows</span>}
        <Button variant="ghost" size="sm" className="ml-auto" disabled={up.busy || importer.isPending} onClick={() => (setFile(null), setPreview(null), setStaged(null), up.reset())}>
          Change file
        </Button>
      </div>
      {!fmt && <p className="text-sm text-destructive">Unsupported file type. Use CSV, TSV, XLSX, Parquet or JSONL.</p>}

      {!preview && fmt && (
        <>
          {up.progress && <UploadProgressPanel progress={up.progress} onCancel={up.cancel} />}
          {previewer.isPending && (
            <p className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="size-4 animate-spin" /> Reading columns…
            </p>
          )}
          {(up.error || previewer.error) && (
            <p role="alert" className="flex items-center gap-2 text-sm text-destructive">
              <AlertTriangle className="size-4" /> {up.error ?? previewer.error?.message}
            </p>
          )}
          <div className="flex justify-end">
            <Button onClick={uploadAndPreview} disabled={up.busy || previewer.isPending}>
              Upload and map columns
            </Button>
          </div>
        </>
      )}

      {preview && (
        <>
          <div>
            <h3 className="font-semibold">Map your columns</h3>
            <p className="text-sm text-muted-foreground">Tell us what each column is. Types were detected automatically; change them if they're wrong.</p>
          </div>
          <div className="overflow-x-auto rounded-lg border">
            <table className="w-full min-w-[720px] text-sm">
              <thead className="bg-muted/60 text-left text-xs text-muted-foreground">
                <tr>
                  <th className="px-3 py-2 font-medium">Column</th>
                  <th className="px-3 py-2 font-medium">Sample values</th>
                  <th className="px-3 py-2 text-right font-medium">Unique</th>
                  <th className="px-3 py-2 text-right font-medium">Missing</th>
                  <th className="px-3 py-2 font-medium">Role</th>
                  <th className="px-3 py-2 font-medium">Type</th>
                </tr>
              </thead>
              <tbody>
                {preview.columns.map((c, i) => {
                  const m = mapping[i]!;
                  const set = (patch: Partial<typeof m>) => setMapping(ms => ms.map((x, j) => (j === i ? { ...x, ...patch } : x)));
                  return (
                    <tr key={c.name} className={cn('border-t', m.role === 'ignore' && 'text-muted-foreground', m.role === 'target' && 'bg-primary/5')}>
                      <td className="px-3 py-2 font-medium">{c.name}</td>
                      <td className="max-w-64 truncate px-3 py-2 text-xs text-muted-foreground" title={c.sample_values.join(' · ')}>
                        {c.sample_values.slice(0, 3).map(String).join(' · ')}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums">{c.unique_count.toLocaleString()}</td>
                      <td className={cn('px-3 py-2 text-right tabular-nums', c.missing_pct > 50 && 'text-warning')}>{c.missing_pct}%</td>
                      <td className="px-3 py-2">
                        <select aria-label={`Role of ${c.name}`} className="h-8 rounded-md border border-input bg-card px-2 text-sm" value={m.role} onChange={e => set({ role: e.target.value as ColumnRole })}>
                          {ROLES.filter(r => text || r.id !== 'text').map(r => (
                            <option key={r.id} value={r.id}>
                              {r.label}
                            </option>
                          ))}
                        </select>
                      </td>
                      <td className="px-3 py-2">
                        <select aria-label={`Type of ${c.name}`} className="h-8 rounded-md border border-input bg-card px-2 text-sm" value={m.type} onChange={e => set({ type: e.target.value as ColumnType })}>
                          {TYPES.map(t => (
                            <option key={t}>{t}</option>
                          ))}
                        </select>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {(mappingError || serverError) && (
            <p role="alert" className="flex items-center gap-2 text-sm text-destructive">
              <AlertTriangle className="size-4" /> {mappingError ?? serverError?.problem.errors?.map(e => e.msg).join('; ') ?? serverError?.message}
            </p>
          )}
          <div className="flex justify-end">
            <Button onClick={doImport} disabled={!!mappingError || importer.isPending}>
              {importer.isPending ? 'Importing…' : `Import ${preview.row_count.toLocaleString()} rows`}
            </Button>
          </div>
        </>
      )}
    </Card>
  );
}
