// Training, evaluation, playground and export (PRD §6.6-6.9). Types come from the generated schema.
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useRef, useState } from 'react';
import { API_URL } from '@/lib/env';
import { api } from '@/lib/api';
import { useAuth } from '@/lib/auth';
import type { components } from './schema';

type S = components['schemas'];
export type Job = S['JobOut'];
export type JobCreated = S['JobCreated'];
export type Trial = S['TrialOut'];
export type ModelVersion = S['ModelVersionOut'];
export type ModelOut = S['ModelOut'];
export type SampleRef = S['SampleRef'];
export type ExportOut = S['ExportOut'];
export type BudgetMode = 'quick' | 'standard' | 'thorough';
export const TERMINAL = ['succeeded', 'failed', 'cancelled'] as const;

/** Live progress event published by the trainer (app/ml/*.py ctx.publish), merged into job.progress. */
export interface Progress {
  phase?: string;
  status?: Job['status'];
  message?: string;
  device?: string;
  budget?: string;
  trial?: number;
  trials_total?: number;
  epoch?: number;
  epochs_total?: number;
  loss?: number;
  metric_name?: string;
  current_metric?: number;
  best_metric?: number;
  eta_seconds?: number | null;
  error_code?: string | null;
  model_version_id?: string | null;
  ts?: number;
}

export interface Recommendation {
  kind: string;
  severity: 'high' | 'medium' | 'low';
  title: string;
  detail: string;
  action: string;
}
export interface PerClass {
  class: string;
  precision?: number | null;
  recall?: number | null;
  f1?: number | null;
  ap50?: number | null;
  ap50_95?: number | null;
  support: number;
}
export interface Mistake {
  sample_id: string;
  y_true: string;
  y_pred: string;
  confidence: number | null;
  error?: number | null;
  error_score?: number | null;
}
/** Shape documented in backend app/ml/evaluation.py. */
export interface EvalReport {
  task_type: string;
  primary_metric: string | null;
  n_test: number;
  metrics: Record<string, number | null>;
  baseline: { kind: string; metrics: Record<string, number | null> } | null;
  per_class: PerClass[];
  confusion: { labels: string[]; counts: number[][]; normalized: number[][]; cells: Record<string, string[]> } | null;
  regression: { scatter: { y_true: number; y_pred: number; sample_id: string }[]; residual_hist: { edges: number[]; counts: number[] } } | null;
  worst_mistakes: Mistake[];
  recommendations: Recommendation[];
  /** Near-duplicate share etc.; low=true means test data likely resembles training data. */
  data_diversity?: { low: boolean; near_duplicate_share?: number };
  /** Primary score ≥ 0.99 on low-diversity data: don't celebrate it. */
  suspicious_perfect?: boolean;
  /** EVL-4: reliability on calibrated test probs + how training calibrated (temperature scaling on val). */
  calibration?: {
    ece: number;
    bins: { lo: number; hi: number; confidence: number; accuracy: number; count: number }[];
    training?: { temperature: number; applied: boolean; val_ece_before: number; val_ece_after: number; reason?: 'val_too_small' | 'gain_too_small' | null; val_samples?: number } | null;
  } | null;
}
export type Evaluation = Omit<S['EvaluationOut'], 'report'> & { report: EvalReport | null };

export type Prediction =
  | { label: string; confidence: number; probs: Record<string, number> | number[]; latency_ms: number; model_load_ms?: number }
  | { value: number; latency_ms: number; model_load_ms?: number }
  | { boxes: { class_name: string; confidence: number; bbox: { x: number; y: number; w: number; h: number } }[]; latency_ms: number; model_load_ms?: number };

export const mk = {
  jobs: (pid: string) => ['jobs', pid] as const,
  models: (pid: string) => ['models', pid] as const,
};

export const useJobs = (projectId: string) =>
  useQuery({
    queryKey: mk.jobs(projectId),
    queryFn: () => api<Job[]>(`/projects/${projectId}/training-jobs`),
    refetchInterval: q => (q.state.data?.some(j => !TERMINAL.includes(j.status as never)) ? 5000 : false),
  });

export const useStartTraining = (projectId: string) => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { dataset_version_id: string; budget_mode: BudgetMode }) =>
      api<JobCreated>(`/projects/${projectId}/training-jobs`, { method: 'POST', body }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: mk.jobs(projectId) });
      qc.invalidateQueries({ queryKey: ['project', projectId] });
    },
  });
};

export const useCancelJob = (projectId: string) => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => api<Job>(`/training-jobs/${jobId}/cancel`, { method: 'POST' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: mk.jobs(projectId) }),
  });
};

export const useTrials = (jobId: string | undefined, live: boolean) =>
  useQuery({
    queryKey: ['trials', jobId],
    queryFn: () => api<Trial[]>(`/training-jobs/${jobId}/trials`),
    enabled: !!jobId,
    refetchInterval: live ? 5000 : false,
  });

/** TRN-7 live view: SSE with ?access_token (EventSource can't send headers). Falls back to the job snapshot. */
export function useJobEvents(job: Job | undefined, onDone?: () => void) {
  const [events, setEvents] = useState<Progress[]>([]);
  const [connected, setConnected] = useState(false);
  const token = useAuth(s => s.accessToken);
  const done = useRef(onDone);
  done.current = onDone;
  const live = !!job && !TERMINAL.includes(job.status as never);
  const id = job?.id;
  useEffect(() => {
    setEvents([]);
    if (!id || !live || !token) return;
    const url = new URL(`${API_URL}/training-jobs/${id}/events`, location.origin);
    url.searchParams.set('access_token', token);
    const es = new EventSource(url);
    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false); // EventSource retries by itself; a new token re-runs this effect
    const push = (e: MessageEvent) => setEvents(prev => [...prev.slice(-499), JSON.parse(e.data) as Progress]);
    es.addEventListener('progress', push);
    es.addEventListener('done', e => {
      push(e as MessageEvent);
      es.close();
      setConnected(false);
      done.current?.();
    });
    return () => es.close();
  }, [id, live, token]);
  return { events, connected };
}

export const useModels = (projectId: string) =>
  useQuery({ queryKey: mk.models(projectId), queryFn: () => api<ModelOut[]>(`/projects/${projectId}/models`) });

/** Newest version first across the project's models. */
export const allVersions = (models: ModelOut[] | undefined) =>
  (models ?? []).flatMap(m => m.versions).sort((a, b) => b.created_at.localeCompare(a.created_at));

export const useModelVersion = (id: string | undefined) =>
  useQuery({ queryKey: ['model-version', id], queryFn: () => api<ModelVersion>(`/model-versions/${id}`), enabled: !!id });

export const useEvaluation = (mvId: string | undefined) =>
  useQuery({
    queryKey: ['evaluation', mvId],
    queryFn: () => api<Evaluation>(`/model-versions/${mvId}/evaluation`),
    enabled: !!mvId,
    refetchInterval: q => (q.state.data?.status === 'pending' ? 3000 : false),
  });

export const useCell = (mvId: string, cell: { y_true: string; y_pred: string } | null) =>
  useQuery({
    queryKey: ['cell', mvId, cell],
    queryFn: () => api<SampleRef[]>(`/model-versions/${mvId}/evaluation/cell`, { query: cell! }),
    enabled: !!cell,
  });

export const useModelCard = (mvId: string | undefined, enabled: boolean) =>
  useQuery({ queryKey: ['model-card', mvId], queryFn: () => api<string>(`/model-versions/${mvId}/model-card`, { text: true }), enabled: !!mvId && enabled });

export const usePredict = (mvId: string | undefined) =>
  useMutation({
    mutationFn: ({ signal, ...input }: ({ file: Blob } | { text: string } | { row: Record<string, unknown> }) & { signal?: AbortSignal }) => {
      if ('file' in input) {
        const fd = new FormData();
        fd.append('file', input.file, 'input.jpg');
        return api<Prediction>(`/model-versions/${mvId}/predict`, { method: 'POST', body: fd, signal });
      }
      return api<Prediction>(`/model-versions/${mvId}/predict`, { method: 'POST', body: input, signal });
    },
  });

export const useSnippets = (mvId: string | undefined) =>
  useQuery({ queryKey: ['snippets', mvId], queryFn: () => api<Record<string, string>>(`/model-versions/${mvId}/snippets`), enabled: !!mvId });

export const useExports = (mvId: string | undefined) =>
  useQuery({
    queryKey: ['exports', mvId],
    queryFn: () => api<ExportOut[]>(`/model-versions/${mvId}/exports`),
    enabled: !!mvId,
    refetchInterval: q => (q.state.data?.some(e => e.status === 'queued' || e.status === 'running') ? 2000 : false),
  });

export const useCreateExport = (mvId: string | undefined) => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api<ExportOut>(`/model-versions/${mvId}/exports`, { method: 'POST', body: { format: 'onnx' } }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['exports', mvId] }),
  });
};

/** Metric display: names + whether higher is better. */
export const METRIC_NAMES: Record<string, string> = {
  accuracy: 'Accuracy', macro_f1: 'Macro F1', micro_f1: 'Micro F1', weighted_f1: 'Weighted F1', roc_auc: 'ROC-AUC',
  pr_auc: 'PR-AUC', log_loss: 'Log loss', ece: 'Calibration error (ECE)', rmse: 'RMSE', mae: 'MAE', r2: 'R²', mape: 'MAPE', map_50: 'mAP@0.5', map_50_95: 'mAP@0.5:0.95',
};
export const LOWER_IS_BETTER = new Set(['rmse', 'mae', 'mape', 'log_loss', 'ece']);
export const fmtMetric = (k: string, v: number | null | undefined) =>
  v == null ? '—' : k === 'mape' || k === 'ece' ? `${(v * 100).toFixed(1)}%` : LOWER_IS_BETTER.has(k) || k === 'r2' ? v.toPrecision(3) : `${(v * 100).toFixed(1)}%`;

/** One entry of an export bundle (server-reported; tokenizer files grouped). */
export interface BundleFile { name: string; size_bytes: number; description?: string; estimated?: boolean; files?: number }
export interface BundlePreview { files: BundleFile[]; total_bytes: number; estimated: boolean; includes_training_data: false }
/** Size + contents of the ONNX bundle before exporting (model.onnx estimated from the native weights). */
export const useExportPreview = (mvId: string | undefined, enabled: boolean) =>
  useQuery({ queryKey: ['export-preview', mvId], queryFn: () => api<BundlePreview>(`/model-versions/${mvId}/exports/preview`), enabled: !!mvId && enabled, staleTime: 60_000 });
