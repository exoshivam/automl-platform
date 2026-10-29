import { keepPreviousData, useInfiniteQuery, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { ApiError, api } from '@/lib/api';
import type { Page } from './types';
import type {
  BoxSuggestions,
  LabelSet,
  LabelSetChange,
  LabelsBatchOut,
  ColumnMapping,
  DatasetVersion,
  ImportFormat,
  ImportPreview,
  LabelUpsert,
  RowsPage,
  Sample,
  Split,
  ValidationReport,
} from './data-types';

export const dk = {
  versions: (datasetId: string) => ['versions', datasetId] as const,
  version: (vid: string) => ['version', vid] as const,
  samples: (vid: string, f: SampleFilters) => ['samples', vid, f] as const,
  rows: (vid: string) => ['rows', vid] as const,
  validation: (vid: string) => ['validation', vid] as const,
};

/** Versions, newest first. Polls while anything is still ingesting or validating. */
export const useVersions = (datasetId: string | null | undefined) =>
  useQuery({
    queryKey: dk.versions(datasetId ?? '-'),
    queryFn: () => api<Page<DatasetVersion>>(`/datasets/${datasetId}/versions`, { query: { limit: 50 } }),
    enabled: !!datasetId,
    // The API lists oldest first; every picker here wants the current (newest) version on top.
    select: page => ({ ...page, items: [...page.items].sort((a, b) => b.version_no - a.version_no) }),
    refetchInterval: q =>
      q.state.data?.items.some(v => v.ingest_status === 'pending' || v.ingest_status === 'running' || v.validation_status === 'running') ? 2000 : false,
  });

/** The version to show: the requested one, else the newest one that finished ingesting. */
export function pickVersion(versions: DatasetVersion[] | undefined, requested?: string) {
  if (!versions?.length) return undefined;
  return versions.find(v => v.id === requested) ?? versions.find(v => v.ingest_status === 'ready') ?? versions[0];
}

export interface SampleFilters {
  label?: string;
  split?: Split | '';
  flag?: string;
  unlabeled?: boolean;
  /** Detection capture hint: photos taken/uploaded "of" this object ('__none' = no hint). */
  capture_target?: string;
}

export const useSamples = (vid: string | undefined, f: SampleFilters, limit = 60) =>
  useInfiniteQuery({
    queryKey: dk.samples(vid ?? '-', f),
    queryFn: ({ pageParam }) =>
      api<Page<Sample>>(`/dataset-versions/${vid}/samples`, {
        query: { label: f.label, split: f.split, flag: f.flag, unlabeled: f.unlabeled || undefined, capture_target: f.capture_target, cursor: pageParam, limit },
      }),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: last => last.next_cursor ?? undefined,
    enabled: !!vid,
    placeholderData: keepPreviousData,
  });

export const useRows = (vid: string | undefined) =>
  useInfiniteQuery({
    queryKey: dk.rows(vid ?? '-'),
    queryFn: ({ pageParam }) => api<RowsPage>(`/dataset-versions/${vid}/rows`, { query: { cursor: pageParam, limit: 50 } }),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: last => last.next_cursor ?? undefined,
    enabled: !!vid,
  });

export const useValidation = (vid: string | undefined, running: boolean) =>
  useQuery({
    queryKey: dk.validation(vid ?? '-'),
    queryFn: () => api<ValidationReport>(`/dataset-versions/${vid}/validation`),
    enabled: !!vid,
    // Poll on the report's own status too: a GET right after seal can return pending/running even when the
    // versions list (polled separately) already shows the final result.
    refetchInterval: q => (running || q.state.data?.status === 'pending' || q.state.data?.status === 'running' ? 2000 : false),
  });

function useInvalidateData() {
  const qc = useQueryClient();
  return () => {
    qc.invalidateQueries({ queryKey: ['versions'] });
    qc.invalidateQueries({ queryKey: ['samples'] });
    qc.invalidateQueries({ queryKey: ['validation'] });
    qc.invalidateQueries({ queryKey: ['project'] });
  };
}

/** 409 stale_version: someone sealed/fixed a newer version; refetch so pickers show the current one. */
const isStale = (e: Error) => e instanceof ApiError && e.problem.code === 'stale_version';

export const useApplyFixes = () => {
  const refresh = useInvalidateData();
  return useMutation({
    mutationFn: ({ vid, fixIds }: { vid: string; fixIds: string[] }) =>
      api<{ dataset_version_id: string }>(`/dataset-versions/${vid}/validation/fixes`, { method: 'POST', body: { fix_ids: fixIds } }),
    onSuccess: refresh,
    onError: e => isStale(e) && refresh(),
  });
};

export const useAcknowledge = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ vid, codes }: { vid: string; codes: string[] }) =>
      api<ValidationReport>(`/dataset-versions/${vid}/validation/acknowledge`, { method: 'POST', body: { codes } }),
    onSuccess: (report, { vid }) => qc.setQueryData(dk.validation(vid), report),
  });
};

/** Replace-per-sample. If vid is sealed the server forks a draft and returns its id. */
export const useSaveLabels = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ vid, upserts }: { vid: string; upserts: LabelUpsert[] }) =>
      api<LabelsBatchOut>(`/dataset-versions/${vid}/labels:batch`, { method: 'POST', body: { upserts } as unknown as Record<string, unknown> }),
    onSuccess: r => {
      qc.invalidateQueries({ queryKey: ['versions'] });
      if (r.forked) qc.invalidateQueries({ queryKey: ['samples'] });
    },
    onError: e => isStale(e) && qc.invalidateQueries({ queryKey: ['versions'] }),
  });
};

export const useSeal = () => {
  const refresh = useInvalidateData();
  return useMutation({
    mutationFn: (vid: string) => api<{ dataset_version_id: string; validation_status: string }>(`/dataset-versions/${vid}/seal`, { method: 'POST' }),
    onSuccess: refresh,
    onError: e => isStale(e) && refresh(),
  });
};

export const useBoxSuggestions = () =>
  useMutation({
    mutationFn: ({ vid, sampleIds, classes }: { vid: string; sampleIds: string[]; classes?: string[] }) =>
      api<BoxSuggestions>(`/dataset-versions/${vid}/box-suggestions`, { method: 'POST', body: { sample_ids: sampleIds, classes } }),
  });

export const useImportPreview = () =>
  useMutation({
    mutationFn: ({ projectId, datasetId, uploadId, format, key }: { projectId: string; datasetId: string; uploadId: string; format: ImportFormat; key: string }) =>
      api<ImportPreview>(`/projects/${projectId}/datasets/${datasetId}/imports`, { method: 'POST', body: { upload_id: uploadId, format, key } }),
  });

export const useImport = () => {
  const refresh = useInvalidateData();
  return useMutation({
    mutationFn: ({
      projectId,
      datasetId,
      uploadId,
      format,
      key,
      mapping,
    }: {
      projectId: string;
      datasetId: string;
      uploadId: string;
      format: ImportFormat;
      key: string;
      mapping?: ColumnMapping;
    }) =>
      api<{ dataset_version_id: string }>(`/projects/${projectId}/datasets/${datasetId}/imports`, {
        method: 'POST',
        body: { upload_id: uploadId, format, key, ...(mapping ? { mapping } : {}) } as Record<string, unknown>,
      }),
    onSuccess: refresh,
  });
};

export { useInvalidateData };

export const useLabelSet = (id: string | null | undefined) =>
  useQuery({ queryKey: ['label-set', id], queryFn: () => api<LabelSet>(`/label-sets/${id}`), enabled: !!id });

/** LBL-1: add, rename or merge classes. Rename/merge relabel existing samples (may fork a version). */
export const useLabelSetChange = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ id, op, body }: { id: string; op: 'classes' | 'rename' | 'merge'; body: Record<string, unknown> }) => {
      try {
        return await api<LabelSetChange>(`/label-sets/${id}/${op}`, { method: 'POST', body });
      } catch (e) {
        // Stale label set (edited in another tab): refetch, then the user retries against the current one.
        if (e instanceof ApiError && e.status === 409) {
          await Promise.all([qc.invalidateQueries({ queryKey: ['project'] }), qc.invalidateQueries({ queryKey: ['label-set'] })]);
          throw new Error('Classes changed elsewhere, refreshed. Please try again.');
        }
        throw e;
      }
    },
    onSuccess: (_r, { id }) => {
      qc.invalidateQueries({ queryKey: ['project'] });
      qc.invalidateQueries({ queryKey: ['label-set', id] });
      qc.invalidateQueries({ queryKey: ['samples'] });
      qc.invalidateQueries({ queryKey: ['versions'] });
    },
  });
};

export interface RemoveOut { removed: number; sample_count: number; dataset_version_id: string }
/** Open draft only (409 version_sealed otherwise); all-or-nothing per call; boxes/labels go with the photos. Chunked ≤1000 ids. */
export const useRemoveSamples = () => {
  const refresh = useInvalidateData();
  return useMutation({
    mutationFn: async ({ vid, ids }: { vid: string; ids: string[] }) => {
      let out: RemoveOut = { removed: 0, sample_count: 0, dataset_version_id: vid };
      for (let i = 0; i < ids.length; i += 1000) {
        const r = await api<RemoveOut>(`/dataset-versions/${vid}/samples:remove`, { method: 'POST', body: { sample_ids: ids.slice(i, i + 1000) } });
        out = { ...r, removed: out.removed + r.removed };
      }
      return out;
    },
    onSuccess: refresh,
    onError: refresh, // a later chunk may fail after earlier ones landed
  });
};
/** Every sample id in a version matching the filters (pages through the 200-per-page list). */
export async function allSampleIds(vid: string, f: SampleFilters = {}) {
  const ids: string[] = [];
  let cursor: string | undefined;
  do {
    const p = await api<Page<Sample>>(`/dataset-versions/${vid}/samples`, { query: { label: f.label, split: f.split, flag: f.flag, unlabeled: f.unlabeled || undefined, capture_target: f.capture_target, cursor, limit: 200 } });
    ids.push(...p.items.map(s => s.id));
    cursor = p.next_cursor ?? undefined;
  } while (cursor);
  return ids;
}
/** Removes every sample one upload / camera batch added, while its draft is still open. Repeat = {removed: 0}. */
export const useUndoUpload = () => {
  const refresh = useInvalidateData();
  const undo = async (id: string): Promise<RemoveOut> => {
    for (let t = 0; ; t++) {
      try {
        return await api<RemoveOut>(`/uploads/${id}:undo`, { method: 'POST' });
      } catch (e) {
        // The batch is still ingesting right after /complete: wait for it (≤30 s).
        if (!(e instanceof ApiError && e.problem.code === 'upload_not_finished') || t >= 30) throw e;
        await new Promise(r => setTimeout(r, 1000));
      }
    }
  };
  return useMutation({ mutationFn: (uploadIds: string[]) => Promise.all(uploadIds.map(undo)), onSuccess: refresh });
};
