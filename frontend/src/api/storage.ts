// Dataset-version deletion, pruning and storage usage (backend routers/storage.py).
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';

export interface VersionRow {
  id: string;
  version_no: number;
  sample_count: number;
  is_draft: boolean;
  is_current: boolean;
  has_models: boolean;
  /** Bytes only this version holds (cumulative versions share most blobs). */
  reclaimable_bytes: number;
  created_at: string;
  reason?: 'current' | 'draft' | 'used_by_models' | 'keep_latest';
}
export interface Storage {
  totals: { data_bytes: number; manifests_bytes: number; models_bytes: number; exports_bytes: number; total_bytes: number; gallery_bytes?: number; gallery_photos_bytes?: number };
  versions: VersionRow[];
}
export interface PruneResult {
  dry_run: boolean;
  deleted: VersionRow[];
  kept: VersionRow[];
  reclaimable_bytes: number;
}

export const useStorage = (projectId: string) =>
  useQuery({ queryKey: ['storage', projectId], queryFn: () => api<Storage>(`/projects/${projectId}/storage`) });

function useRefresh(projectId: string) {
  const qc = useQueryClient();
  return () => {
    for (const key of [['storage', projectId], ['versions'], ['project', projectId], ['models', projectId]]) qc.invalidateQueries({ queryKey: key });
  };
}

export const useDeleteVersion = (projectId: string) => {
  const refresh = useRefresh(projectId);
  return useMutation({ mutationFn: (id: string) => api<void>(`/dataset-versions/${id}`, { method: 'DELETE' }), onSuccess: refresh });
};

export const usePrune = (projectId: string, datasetId: string) => {
  const refresh = useRefresh(projectId);
  return useMutation({
    mutationFn: (body: { keep_latest: number; dry_run: boolean }) =>
      api<PruneResult>(`/datasets/${datasetId}/versions:prune`, { method: 'POST', body }),
    onSuccess: r => !r.dry_run && refresh(),
  });
};

export const useDeleteModelVersion = (projectId: string) => {
  const refresh = useRefresh(projectId);
  return useMutation({ mutationFn: (id: string) => api<void>(`/model-versions/${id}`, { method: 'DELETE' }), onSuccess: refresh });
};
