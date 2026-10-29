// Quick text labels (examples, no training) + ready-made sentiment. Contract from the backend.
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type { Project } from './types';

export interface TextSettings {
  mode: 'quick' | 'custom';
  threshold: number;
  default_threshold: number;
  ready_made: string[];
  labels: string[];
  embedding_model: { id: string; licence: string };
  sentiment_model: { id: string; licence: string } | null;
  has_trained_model: boolean;
}
export interface Example { id: string; text: string; label: string; created_at?: string; dataset_version_id?: string }
/** detail = the model's 5-way class, e.g. "very negative". */
export interface Sentiment { label: 'positive' | 'neutral' | 'negative'; score: number; detail?: string }
export interface TextPrediction {
  label: string | null;
  score: number;
  top3: { label: string; score: number }[];
  other: boolean;
  source: 'examples' | 'model' | 'sentiment_only';
  /** Similarity to everyday background text (why something went to Other). */
  background_score?: number;
  sentiment?: Sentiment;
  latency_ms: number;
  model_load_ms?: number;
}
export interface TextWarning { code: string; detail: string; other_label?: string }
type Hist = { count: number; mean: number | null; hist: { lo: number; hi: number; count: number }[] };
export interface TextReport {
  ready: boolean;
  examples: number;
  labels: number;
  loo_accuracy: number | null;
  current_threshold: number;
  suggested_threshold: number | null;
  genuine: Hist | null;
  impostor: Hist | null;
  per_label: { label: string; examples: number; loo_hits: number; warnings: TextWarning[] }[];
  warnings: TextWarning[];
}

const spec = (p: Project | undefined) => (p?.current_spec ?? {}) as { mode?: string; ready_made?: string[]; labels?: string[] };
export const isQuickText = (p: Project | undefined) => p?.task_type === 'text_classification' && spec(p).mode === 'quick';
export const hasSentiment = (p: Project | undefined) => p?.task_type === 'text_classification' && (spec(p).ready_made ?? []).includes('sentiment');
/** Ready-made sentiment with no labels of the user's own. */
export const isSentimentOnly = (p: Project | undefined) => isQuickText(p) && hasSentiment(p) && !(spec(p).labels ?? []).length;

function useRefresh(pid: string) {
  const qc = useQueryClient();
  return () => {
    for (const k of [['examples', pid], ['text-report', pid], ['text-settings', pid], ['project', pid], ['versions']]) qc.invalidateQueries({ queryKey: k });
  };
}

export const useTextSettings = (pid: string, enabled = true) => useQuery({ queryKey: ['text-settings', pid], queryFn: () => api<TextSettings>(`/projects/${pid}/text-settings`), enabled });
export const useSaveTextSettings = (pid: string) => {
  const refresh = useRefresh(pid);
  return useMutation({
    mutationFn: (body: { threshold?: number; ready_made?: string[] }) => api<TextSettings>(`/projects/${pid}/text-settings`, { method: 'PATCH', body }),
    onSuccess: refresh,
  });
};
export const useExamples = (pid: string, enabled = true) => useQuery({ queryKey: ['examples', pid], queryFn: () => api<Example[]>(`/projects/${pid}/examples`), enabled });
export const useAddExample = (pid: string) => {
  const refresh = useRefresh(pid);
  return useMutation({ mutationFn: (body: { text: string; label: string }) => api<Example>(`/projects/${pid}/examples`, { method: 'POST', body }), onSuccess: refresh });
};
export const useDeleteExample = (pid: string) => {
  const refresh = useRefresh(pid);
  return useMutation({ mutationFn: (id: string) => api<void>(`/projects/${pid}/examples/${id}`, { method: 'DELETE' }), onSuccess: refresh });
};
export const useTextReport = (pid: string, enabled = true) =>
  useQuery({ queryKey: ['text-report', pid], queryFn: () => api<TextReport>(`/projects/${pid}/text-report`), enabled });
export const predictText = (pid: string, text: string, signal?: AbortSignal) =>
  api<TextPrediction>(`/projects/${pid}/predict`, { method: 'POST', body: { text }, signal });

/** Examples per label we recommend (varied wording) and from which "Train for better accuracy" is offered. */
export const TRAIN_READY = 10;
/** Leave-one-out accuracy below this gets the "add more varied examples / train" nudge. */
export const LOO_OK = 0.7;
