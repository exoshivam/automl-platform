// Ready-made object detector (COCO SSDlite, no training) vs "train your own". Contract from the backend.
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type { Project } from './types';

export interface PretrainedModel { id: string; name: string; licence: string; commercial_use: boolean }
export interface PretrainedClasses { model: PretrainedModel; classes: { id: number; name: string }[] }
export interface DetectorSettings { mode: 'pretrained' | 'custom'; score_threshold: number; default_score_threshold: number; classes: string[]; model: PretrainedModel }
export interface Detected {
  boxes: { class_name: string; confidence: number; bbox: { x: number; y: number; w: number; h: number }; source?: 'ready-made' | 'custom' }[];
  image: { width: number; height: number };
  latency_ms: number;
  model_load_ms?: number;
  custom_status?: CustomStatus;
}
/** outdated = the chosen classes changed since training: the model still runs, but only knows what it was trained on. */
export type CustomStatus = 'none' | 'not_trained' | 'training' | 'ready' | 'outdated' | 'needs_attention';

/** Projects created in ready-made mode carry spec.mode = "pretrained" (convert-to-custom flips it). */
export const isPretrained = (p: Pick<Project, 'task_type' | 'current_spec'> | undefined) =>
  p?.task_type === 'object_detection' && (p.current_spec as { mode?: string }).mode === 'pretrained';

export const usePretrainedClasses = () =>
  useQuery({ queryKey: ['pretrained-classes'], queryFn: () => api<PretrainedClasses>('/detection/pretrained-classes'), staleTime: Infinity });

export const useDetectorSettings = (pid: string) =>
  useQuery({ queryKey: ['detector-settings', pid], queryFn: () => api<DetectorSettings>(`/projects/${pid}/detector-settings`) });

export const useSaveDetectorSettings = (pid: string) => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { score_threshold?: number; classes?: string[] }) => api<DetectorSettings>(`/projects/${pid}/detector-settings`, { method: 'PATCH', body }),
    onSuccess: r => (qc.setQueryData(['detector-settings', pid], r), qc.invalidateQueries({ queryKey: ['project', pid] }), qc.invalidateQueries({ queryKey: ['projects'] })),
  });
};

export const useConvertToCustom = (pid: string) => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api<Project>(`/projects/${pid}:convert-to-custom`, { method: 'POST' }),
    onSuccess: p => (qc.setQueryData(['project', pid], p), qc.invalidateQueries({ queryKey: ['projects'] })),
  });
};

export const detect = (pid: string, blob: Blob, signal?: AbortSignal) => {
  const fd = new FormData();
  fd.append('file', blob, 'frame.jpg');
  return api<Detected>(`/projects/${pid}/predict`, { method: 'POST', body: fd, signal });
};

/** Hybrid: a ready-made project can also have the user's own trained objects. */
type HybridFields = { custom_labels?: string[] | null; custom_status?: CustomStatus | null; custom_model_version_id?: string | null };
export const customLabels = (p: Project | undefined) =>
  ((p as (Project & HybridFields) | undefined)?.custom_labels ?? (p?.current_spec as { custom_labels?: string[] } | undefined)?.custom_labels ?? []) as string[];
export const customStatus = (p: Project | undefined): CustomStatus => (p as (Project & HybridFields) | undefined)?.custom_status ?? 'none';
/** Classes the data/label/train screens work with: a ready-made project trains its custom objects only. */
export const trainableLabels = (p: Project | undefined) => (isPretrained(p) ? customLabels(p) : (p?.current_spec.labels ?? []));
/** Ready-made project with at least one custom object: show the training tabs. */
export const isHybrid = (p: Project | undefined) => isPretrained(p) && customLabels(p).length > 0;

function useHybridRefresh(pid: string) {
  const qc = useQueryClient();
  return (r: { project: Project }) => {
    qc.setQueryData(['project', pid], r.project);
    for (const k of [['projects'], ['detector-settings', pid], ['label-set'], ['versions']]) qc.invalidateQueries({ queryKey: k });
  };
}
export const useAddCustomObject = (pid: string) => {
  const done = useHybridRefresh(pid);
  return useMutation({
    mutationFn: (name: string) => api<{ custom_labels: string[]; label_set_id: string; project: Project }>(`/projects/${pid}/custom-objects`, { method: 'POST', body: { name } }),
    onSuccess: done,
  });
};
export const useRemoveCustomObject = (pid: string) => {
  const done = useHybridRefresh(pid);
  return useMutation({
    mutationFn: (name: string) => api<{ custom_labels: string[]; label_set_id: string; project: Project }>(`/projects/${pid}/custom-objects/${encodeURIComponent(name)}`, { method: 'DELETE' }),
    onSuccess: done,
  });
};
