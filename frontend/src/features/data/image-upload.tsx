import { trainableLabels } from '@/api/detector';
import { useMemo, useState } from 'react';
import { AlertTriangle, CheckCircle2, FileJson, Images } from 'lucide-react';
import { useImport } from '@/api/data-hooks';
import type { ImportFormat } from '@/api/data-types';
import type { Project } from '@/api/types';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/misc';
import { ext, formatBytes, isImage, parentFolder, type PickedFile } from '@/lib/files';
import { uploadFiles } from '@/lib/uploader';
import { DropZone } from './drop-zone';
import { IngestStatus, UploadProgressPanel, useUploadRun } from './upload-status';

/** Which annotation format (DAT-8) a set of dropped files carries, and the key file /imports wants. */
function detectAnnotations(files: PickedFile[]): { format: ImportFormat; file: PickedFile } | null {
  const coco = files.find(f => ext(f.file.name) === 'json');
  if (coco) return { format: 'coco', file: coco };
  const classes = files.find(f => f.file.name.toLowerCase() === 'classes.txt');
  if (classes) return { format: 'yolo', file: classes };
  const voc = files.find(f => ext(f.file.name) === 'xml');
  if (voc) return { format: 'voc', file: voc };
  return null;
}

export function ImageUpload({ project }: { project: Project }) {
  const detection = project.task_type === 'object_detection';
  const known = trainableLabels(project);
  const [files, setFiles] = useState<PickedFile[]>([]);
  const [fromFolders, setFromFolders] = useState(true);
  const [assign, setAssign] = useState<string>('');
  const [photosOf, setPhotosOf] = useState<string>('');
  const up = useUploadRun();
  const importer = useImport();
  const [versionId, setVersionId] = useState<string | null>(null);

  const images = files.filter(f => isImage(f.file));
  const zips = files.filter(f => ext(f.file.name) === 'zip');
  const annotations = detection ? detectAnnotations(files) : null;
  const ignored = files.length - images.length - zips.length - (detection ? files.filter(f => ['json', 'txt', 'xml'].includes(ext(f.file.name))).length : 0);
  const bytes = files.reduce((n, f) => n + f.file.size, 0);

  // Classification: subfolder names become labels (DAT-1).
  const classes = useMemo(() => {
    const m = new Map<string, number>();
    for (const f of images) {
      const c = parentFolder(f);
      if (c) m.set(c, (m.get(c) ?? 0) + 1);
    }
    return [...m.entries()].sort((a, b) => b[1] - a[1]);
  }, [images]);
  const hasFolders = classes.length > 0;
  const unknown = classes.filter(([c]) => known.length && !known.some(k => k.toLowerCase() === c.toLowerCase()));

  const start = () =>
    up.run(async (signal, onProgress) => {
      if (annotations) {
        // Annotations + their images go up together, then /imports parses them (not auto-ingested).
        const toSend = files.filter(f => isImage(f.file) || ['json', 'txt', 'xml'].includes(ext(f.file.name)));
        const r = await uploadFiles({ projectId: project.id, datasetId: project.dataset_id!, source: 'annotations', files: toSend, signal, onProgress });
        const key = r.keys[toSend.indexOf(annotations.file)]!;
        const imp = await importer.mutateAsync({ projectId: project.id, datasetId: project.dataset_id!, uploadId: r.uploadId, format: annotations.format, key });
        setVersionId(imp.dataset_version_id);
        return { ...r, datasetVersionId: imp.dataset_version_id };
      }
      if (zips.length === 1 && !images.length) {
        const r = await uploadFiles({ projectId: project.id, datasetId: project.dataset_id!, source: 'zip', files: zips, labelFromFolder: fromFolders && !detection, signal, onProgress });
        setVersionId(r.datasetVersionId);
        return r;
      }
      const useFolders = !detection && fromFolders && hasFolders;
      const r = await uploadFiles({
        projectId: project.id,
        datasetId: project.dataset_id!,
        source: useFolders ? 'folder' : 'files',
        files: [...images, ...zips],
        labelFromFolder: useFolders,
        label: !detection && !useFolders && assign ? assign : null,
        captureTarget: detection && photosOf ? photosOf : null,
        signal,
        onProgress,
      });
      setVersionId(r.datasetVersionId);
      return r;
    });

  if (up.result) return <IngestStatus projectId={project.id} datasetId={project.dataset_id!} versionId={versionId ?? up.result.datasetVersionId} onAgain={() => (up.reset(), setFiles([]), setVersionId(null))} />;

  return (
    <div className="grid gap-4">
      {!files.length ? (
        <DropZone
          onFiles={setFiles}
          accept={detection ? 'image/*,.zip,.json,.txt,.xml' : 'image/*,.zip'}
          title={detection ? 'Drop images, with COCO / YOLO / VOC labels if you have them' : 'Drop a folder of class folders, a zip, or images'}
          hint={
            detection
              ? 'Images alone go to the box tool. Add a COCO .json, a YOLO classes.txt with .txt label files, or VOC .xml files to import existing boxes.'
              : 'Folder layout: dataset/helmet/*.jpg, dataset/vest/*.jpg. Each subfolder name becomes a class. JPEG, PNG, WebP, BMP, TIFF, HEIC.'
          }
        />
      ) : (
        <Card className="grid gap-4 p-5">
          <div className="flex flex-wrap items-center gap-3">
            <Images className="size-5 text-muted-foreground" />
            <span className="font-medium">
              {images.length.toLocaleString()} images{zips.length ? ` + ${zips.length} zip` : ''} · {formatBytes(bytes)}
            </span>
            {ignored > 0 && <Badge variant="secondary">{ignored} unsupported files skipped</Badge>}
            <Button variant="ghost" size="sm" className="ml-auto" onClick={() => setFiles([])} disabled={up.busy}>
              Clear
            </Button>
          </div>

          {annotations && (
            <p className="flex items-center gap-2 rounded-md bg-info/8 p-3 text-sm">
              <FileJson className="size-4 text-info" /> Found {annotations.format.toUpperCase()} annotations in <b>{annotations.file.file.name}</b>. Boxes will be imported with the images.
            </p>
          )}
          {detection && !annotations && images.length > 0 && <p className="text-sm text-muted-foreground">No annotation file: images arrive unlabelled and wait for you on the Label tab (AI box suggestions included).</p>}

          {!detection && (hasFolders || zips.length > 0) && (
            <div className="flex items-center gap-3">
              <Switch id="from-folders" checked={fromFolders} onCheckedChange={setFromFolders} />
              <Label htmlFor="from-folders">Subfolder names are class names</Label>
            </div>
          )}

          {!detection && hasFolders && fromFolders && (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs text-muted-foreground">
                    <th className="py-1.5 font-medium">Class (folder)</th>
                    <th className="py-1.5 text-right font-medium">Images</th>
                    <th className="py-1.5 pl-4 font-medium">Matches project</th>
                  </tr>
                </thead>
                <tbody>
                  {classes.map(([c, n]) => (
                    <tr key={c} className="border-t">
                      <td className="py-1.5 font-medium">{c}</td>
                      <td className="py-1.5 text-right tabular-nums">{n}</td>
                      <td className="py-1.5 pl-4">
                        {!known.length || known.some(k => k.toLowerCase() === c.toLowerCase()) ? (
                          <span className="inline-flex items-center gap-1 text-success">
                            <CheckCircle2 className="size-3.5" /> yes
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-warning">
                            <AlertTriangle className="size-3.5" /> new class
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {unknown.length > 0 && (
                <p className="mt-2 text-xs text-warning">
                  {unknown.length} folder name(s) aren't in this project's classes ({known.join(', ')}). They'll be added as labels; you can merge or rename them later.
                </p>
              )}
            </div>
          )}

          {!detection && images.length > 0 && (!hasFolders || !fromFolders) && (
            <div className="grid gap-1.5 sm:max-w-sm">
              <Label htmlFor="assign">Put all these images in a class (optional)</Label>
              <select id="assign" className="h-9 rounded-md border border-input bg-card px-3 text-sm" value={assign} onChange={e => setAssign(e.target.value)}>
                <option value="">Leave unlabelled (label them later)</option>
                {known.map(k => (
                  <option key={k}>{k}</option>
                ))}
              </select>
            </div>
          )}

          {detection && !annotations && images.length > 0 && known.length > 0 && (
            <div className="grid gap-1.5 sm:max-w-sm">
              <Label htmlFor="photos-of">These photos are of (optional)</Label>
              <select id="photos-of" className="h-9 rounded-md border border-input bg-card px-3 text-sm" value={photosOf} onChange={e => setPhotosOf(e.target.value)}>
                <option value="">Mixed / several objects</option>
                {known.map(k => <option key={k}>{k}</option>)}
              </select>
              <span className="text-xs text-muted-foreground">Helps on the Label tab: that object is preselected and you can filter by it. No boxes are created.</span>
            </div>
          )}

          {up.progress && <UploadProgressPanel progress={up.progress} onCancel={up.cancel} />}
          {up.error && (
            <p role="alert" className="flex items-center gap-2 text-sm text-destructive">
              <AlertTriangle className="size-4" /> {up.error}
            </p>
          )}
          <div className="flex justify-end">
            <Button onClick={start} disabled={up.busy || (!images.length && !zips.length)}>
              {up.busy ? 'Uploading…' : up.error ? 'Retry upload' : `Upload ${images.length + zips.length} file${images.length + zips.length === 1 ? '' : 's'}`}
            </Button>
          </div>
        </Card>
      )}
    </div>
  );
}
