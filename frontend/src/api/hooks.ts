import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { ApiError, api } from '@/lib/api';
import { useUi } from '@/stores/ui';
import { asPlan, type AuthResponse, type FromDescriptionResponse, type Me, type Membership, type Org, type OrgOut, type Page, type Project, type ProjectStatus, type RegisterResponse, type TaskSpec, type Template, type User } from './types';

export const qk = {
  me: ['me'] as const,
  authConfig: ['auth-config'] as const,
  templates: ['templates'] as const,
  projects: (orgId: string, filters: object) => ['projects', orgId, filters] as const,
  project: (id: string) => ['project', id] as const,
};

export const useAuthConfig = () =>
  useQuery({
    queryKey: qk.authConfig,
    queryFn: () => api<{ google_enabled: boolean }>('/auth/config', { noRefresh: true }),
    staleTime: Infinity,
  });

export const useLogin = () =>
  useMutation({ mutationFn: (body: { email: string; password: string }) => api<AuthResponse>('/auth/login', { method: 'POST', body, noRefresh: true }) });

/** ACC-1: registration does not log in; the user must verify their email first. */
export const useRegister = () =>
  useMutation({
    mutationFn: (body: { name: string; email: string; password: string }) => api<RegisterResponse>('/auth/register', { method: 'POST', body, noRefresh: true }),
  });

export const useVerifyEmail = () =>
  useMutation({ mutationFn: (token: string) => api<User>('/auth/verify-email', { method: 'POST', body: { token }, noRefresh: true }) });

export const useResendVerification = () =>
  useMutation({ mutationFn: (email: string) => api<void>('/auth/resend-verification', { method: 'POST', body: { email }, noRefresh: true }) });

export const useMe = () => useQuery({ queryKey: qk.me, queryFn: () => api<Me>('/me') });

export const toOrg = (m: Membership): Org => ({
  id: m.organization_id,
  name: m.organization_name,
  slug: m.slug,
  plan: asPlan(m.plan),
  role: m.role,
  personal: m.personal,
});

/** Current workspace: the stored choice if still a member, else the personal one, else the first. */
export function useCurrentOrg() {
  const me = useMe();
  const orgId = useUi(s => s.orgId);
  const orgs = (me.data?.memberships ?? []).map(toOrg);
  const org = orgs.find(o => o.id === orgId) ?? orgs.find(o => o.personal) ?? orgs[0];
  return { ...me, orgs, org };
}

export const useCreateOrg = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (name: string) => api<OrgOut>('/orgs', { method: 'POST', body: { name } }),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.me }),
  });
};

export const useTemplates = () => useQuery({ queryKey: qk.templates, queryFn: () => api<Template[]>('/templates'), staleTime: 5 * 60_000 });

export interface ProjectFilters {
  search?: string;
  status?: ProjectStatus | '';
  archived?: boolean;
}

export const useProjects = (orgId: string | undefined, filters: ProjectFilters) =>
  useQuery({
    queryKey: qk.projects(orgId ?? '-', filters),
    queryFn: () => api<Page<Project>>(`/orgs/${orgId}/projects`, { query: { ...filters, limit: 100 } }),
    enabled: !!orgId,
    placeholderData: keepPreviousData,
  });

export const useProject = (id: string) => useQuery({ queryKey: qk.project(id), queryFn: () => api<Project>(`/projects/${id}`) });

export const useCreateProject = (orgId: string | undefined) => {
  const qc = useQueryClient();
  return useMutation({
    /** Create from the reviewed spec (category path) or an LLM draft (with its llm_log_id). */
    mutationFn: (body: { name: string; spec: TaskSpec; llm_log_id?: string | null }) =>
      api<Project>(`/orgs/${orgId}/projects`, { method: 'POST', body: body as unknown as Record<string, unknown> }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['projects'] }),
  });
};

export const useFromDescription = (orgId: string | undefined) =>
  useMutation({
    mutationFn: (description: string) => api<FromDescriptionResponse>(`/orgs/${orgId}/projects/from-description`, { method: 'POST', body: { description } }),
  });

export const useArchiveProject = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ id, archived }: { id: string; archived: boolean }) => {
      // PRD §11: DELETE archives; PATCH archived:false restores.
      if (archived) await api<void>(`/projects/${id}`, { method: 'DELETE' });
      else await api<Project>(`/projects/${id}`, { method: 'PATCH', body: { archived: false } });
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ['projects'] }),
  });
};

/** Permanent purge (async). Drops the project from cached lists at once, then polls until GET /projects/{id} is 404. */
export const usePurgeProject = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, confirmName }: { id: string; confirmName: string }) =>
      api<{ project_id: string; status: 'Deleting'; task_id: string }>(`/projects/${id}:purge`, { method: 'POST', body: { confirm_name: confirmName } }),
    onSuccess: (_r, { id }) => {
      qc.setQueriesData<Page<Project>>({ queryKey: ['projects'] }, page => page && { ...page, items: page.items.filter(p => p.id !== id) });
      qc.removeQueries({ queryKey: qk.project(id) });
    },
  });
};

/** Resolves true once the purge finished (404), false after `timeoutMs`. */
export async function waitForPurge(id: string, timeoutMs = 15 * 60_000) {
  const until = Date.now() + timeoutMs;
  while (Date.now() < until) {
    try {
      await api<Project>(`/projects/${id}`);
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) return true;
    }
    await new Promise(r => setTimeout(r, 3000));
  }
  return false;
}
