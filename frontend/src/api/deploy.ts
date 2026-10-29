// Self-serve local deploy (the backend contract): one deployment per project, keys bound to it, POST /v1/predict.
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { ApiError, api } from '@/lib/api';
import type { TaskType } from './types';

export interface Deployment {
  id: string;
  project_id: string;
  task_type: TaskType;
  kind: 'model' | 'ready_made' | 'quick_text' | 'sentiment' | 'face' | 'speaker';
  model_version_id: string | null;
  model_version_no: number | null;
  status: 'active' | 'paused';
  rate_limit_per_min: number;
  key_count: number;
  created_by: string;
  created_at: string;
  updated_at: string;
}
export interface ApiKey { id: string; name: string; prefix: string; created_at: string; last_used_at: string | null; revoked_at: string | null }
export interface NewKey { id: string; name: string; prefix: string; created_at: string; key: string }
export interface Usage {
  window: '24h' | '7d';
  requests: number;
  errors: number;
  error_rate: number;
  avg_latency_ms: number | null;
  p95_latency_ms: number | null;
  series: { t: string; requests: number; errors: number; avg_latency_ms: number | null }[];
}
export interface DeployDocs {
  endpoint: string;
  task_type: TaskType;
  kind: Deployment['kind'];
  input: { content_type: 'multipart/form-data' | 'application/json'; field?: string; schema?: Record<string, unknown>; example?: Record<string, unknown> };
  output_example: Record<string, unknown>;
  labels: string[];
  curl: string;
}

export const DEPLOY_ERRORS: Record<string, string> = {
  deployment_exists: 'This project is already deployed.',
  no_model: 'Train a model first: there is no ready model version to deploy.',
  owner_only: 'Only the workspace owner can deploy face and speaker projects.',
};
export const deployError = (e: unknown) => (e instanceof ApiError && DEPLOY_ERRORS[e.problem.code ?? '']) || (e as Error).message;

const dk = {
  dep: (pid: string) => ['deployment', pid] as const,
  keys: (id: string) => ['deploy-keys', id] as const,
  usage: (id: string, w: string) => ['deploy-usage', id, w] as const,
  docs: (id: string) => ['deploy-docs', id] as const,
};

export const useDeployment = (pid: string) =>
  useQuery({
    queryKey: dk.dep(pid),
    queryFn: async () => {
      try {
        return await api<Deployment>(`/projects/${pid}/deployment`);
      } catch (e) {
        if (e instanceof ApiError && e.status === 404) return null; // not deployed yet
        throw e;
      }
    },
  });

function useRefresh(pid: string) {
  const qc = useQueryClient();
  return () => ['deployment', 'deploy-keys', 'deploy-docs', 'project'].forEach(k => qc.invalidateQueries({ queryKey: k === 'deployment' || k === 'project' ? [k, pid] : [k] }));
}
export const useDeploy = (pid: string) => {
  const refresh = useRefresh(pid);
  return useMutation({ mutationFn: (model_version_id?: string) => api<Deployment>(`/projects/${pid}/deployment`, { method: 'POST', body: model_version_id ? { model_version_id } : {} }), onSuccess: refresh });
};
export const useUpdateDeployment = (pid: string) => {
  const refresh = useRefresh(pid);
  return useMutation({
    mutationFn: ({ id, ...body }: { id: string; status?: Deployment['status']; model_version_id?: string }) => api<Deployment>(`/deployments/${id}`, { method: 'PATCH', body }),
    onSuccess: refresh,
  });
};
export const useDeleteDeployment = (pid: string) => {
  const refresh = useRefresh(pid);
  return useMutation({ mutationFn: (id: string) => api<void>(`/deployments/${id}`, { method: 'DELETE' }), onSuccess: refresh });
};

export const useKeys = (id: string) => useQuery({ queryKey: dk.keys(id), queryFn: () => api<ApiKey[]>(`/deployments/${id}/keys`) });
export const useCreateKey = (pid: string, id: string) => {
  const refresh = useRefresh(pid);
  return useMutation({ mutationFn: (name: string) => api<NewKey>(`/deployments/${id}/keys`, { method: 'POST', body: { name } }), onSuccess: refresh });
};
export const useRevokeKey = (pid: string, id: string) => {
  const refresh = useRefresh(pid);
  return useMutation({ mutationFn: (keyId: string) => api<void>(`/deployments/${id}/keys/${keyId}`, { method: 'DELETE' }), onSuccess: refresh });
};

export const useUsage = (id: string, window: Usage['window']) =>
  useQuery({ queryKey: dk.usage(id, window), queryFn: () => api<Usage>(`/deployments/${id}/usage`, { query: { window } }), refetchInterval: 30_000 });
export const useDocs = (id: string) => useQuery({ queryKey: dk.docs(id), queryFn: () => api<DeployDocs>(`/deployments/${id}/docs`) });
