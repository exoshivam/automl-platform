// PRD DAT-13 starter dataset library: browse curated public sets and import one into the project's draft.
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type { TaskType } from './types';

export interface Starter {
  id: string;
  name: string;
  description: string;
  task_type: TaskType;
  /** Empty for regression. */
  classes: string[];
  target?: string | null;
  sample_count: number;
  size_bytes: number;
  source: string;
  licence: string;
  source_url?: string | null;
  /** 6 presigned thumbnails (one per class first) for image tasks; [] for text/tabular. Expire after 15 min. */
  preview_urls: string[];
}

export interface StarterImport {
  dataset_version_id: string;
  upload_id: string;
  status: 'ingesting';
  sample_count: number;
}

/** All starters when taskType is undefined (wizard), else those for the task. */
export const useStarters = (taskType: string | undefined) =>
  useQuery({
    queryKey: ['starters', taskType],
    queryFn: () => api<Starter[]>('/starter-datasets', { query: { task_type: taskType } }),
    staleTime: 5 * 60_000, // preview_urls are presigned for 15 min: refetch, don't hold them longer
  });

export const useImportStarter = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ projectId, starterId }: { projectId: string; starterId: string }) =>
      api<StarterImport>(`/projects/${projectId}/starter-imports`, { method: 'POST', body: { starter_id: starterId } }),
    onSuccess: (_r, { projectId }) => {
      qc.invalidateQueries({ queryKey: ['versions'] });
      qc.invalidateQueries({ queryKey: ['project', projectId] });
      qc.invalidateQueries({ queryKey: ['label-set'] });
    },
  });
};
