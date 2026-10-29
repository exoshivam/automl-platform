// Bulk auto-label for detection (the backend contract): a train-queue job writes SUGGESTED boxes, kept apart from
// labels (ignored by training/validation/seal) until reviewed.
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { ApiError, api } from '@/lib/api';
import type { BBox, Sample } from './data-types';

export type AutoSource = 'auto' | 'text_prompt' | 'own_model' | 'ready_made';
export interface AutoSources {
  text_prompt: { available: boolean; model_id: string; licence: string; note?: string };
  own_model: { available: boolean; model_version_id: string | null; classes: string[] };
  ready_made: { available: boolean; model_id: string; licence: string; classes: string[] };
}
export interface AutoJob {
  id: string;
  dataset_version_id: string;
  status: 'queued' | 'running' | 'done' | 'failed' | 'cancelled';
  source: AutoSource;
  classes: string[];
  min_confidence: number;
  progress: { done: number; total: number };
  counts: { suggested: number; samples_with_boxes: number; none_found: number };
  error: string | null;
  created_at: string;
  finished_at: string | null;
}
export interface SuggestedBox { id: string; class_name: string; bbox: BBox; confidence: number; source: string; model_id: string; status: 'pending' | 'accepted' | 'rejected' }
export type ReviewState = 'needs_review' | 'accepted' | 'rejected' | 'none_found';
export interface SuggestedSample extends Partial<Pick<Sample, 'width' | 'height' | 'url' | 'labels'>> { sample_id: string; name: string | null; thumb_url: string | null; state: ReviewState; boxes: SuggestedBox[] }
export interface Suggestions { samples: SuggestedSample[]; counts: Record<ReviewState, number> & { pending_boxes: number; confirmed_boxes: number } }
export interface ReviewBody {
  accept?: string[];
  reject?: string[];
  edit?: { id: string; bbox?: BBox; class_name?: string }[];
  accept_samples?: string[];
  reject_samples?: string[];
  accept_above?: number;
  reject_remaining?: boolean;
}
export interface ReviewOut { accepted: number; rejected: number; edited: number; labels_created: number; dataset_version_id: string }

export const AUTO_ERRORS: Record<string, string> = {
  version_sealed: 'Auto-label works on the draft. Add photos to the draft first.',
  job_running: 'An auto-label job is already running for these photos.',
  source_unavailable: 'That source isn’t available for this project.',
};
export const autoError = (e: unknown) => (e instanceof ApiError && AUTO_ERRORS[e.problem.code ?? '']) || (e as Error).message;
export const isActive = (j?: AutoJob | null) => j?.status === 'queued' || j?.status === 'running';

export const useAutoSources = (vid: string) => useQuery({ queryKey: ['autolabel-sources', vid], queryFn: () => api<AutoSources>(`/dataset-versions/${vid}/autolabel-sources`), staleTime: 60_000 });

/** Newest job for the version (re-attaches after a reload); polls while it runs. */
export const useLatestJob = (vid: string) =>
  useQuery({
    queryKey: ['autolabel-jobs', vid],
    queryFn: async () => (await api<AutoJob[]>(`/dataset-versions/${vid}/autolabel-jobs`))[0] ?? null,
    refetchInterval: q => (isActive(q.state.data) ? 2000 : false),
  });

function useRefresh() {
  const qc = useQueryClient();
  return () => ['autolabel-jobs', 'suggestions', 'samples', 'versions'].forEach(k => qc.invalidateQueries({ queryKey: [k] }));
}
export const useStartAutolabel = (vid: string) => {
  const refresh = useRefresh();
  return useMutation({
    mutationFn: (body: { source: AutoSource; classes: string[]; prompts?: Record<string, string>; min_confidence: number; scope: 'unlabelled' | 'sample_ids'; sample_ids?: string[] }) =>
      api<AutoJob>(`/dataset-versions/${vid}/autolabel-jobs`, { method: 'POST', body }),
    onSuccess: refresh,
  });
};
export const useCancelAutolabel = () => {
  const refresh = useRefresh();
  return useMutation({ mutationFn: (id: string) => api<AutoJob>(`/autolabel-jobs/${id}:cancel`, { method: 'POST' }), onSuccess: refresh });
};
export const useSuggestions = (vid: string, state: ReviewState, poll: boolean) =>
  useQuery({ queryKey: ['suggestions', vid, state], queryFn: () => api<Suggestions>(`/dataset-versions/${vid}/suggestions`, { query: { state } }), refetchInterval: poll ? 3000 : false });
export const useReview = (vid: string) => {
  const refresh = useRefresh();
  return useMutation({ mutationFn: (body: ReviewBody) => api<ReviewOut>(`/dataset-versions/${vid}/suggestions:review`, { method: 'POST', body: body as Record<string, unknown> }), onSuccess: refresh });
};
