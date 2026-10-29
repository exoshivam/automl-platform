import type { AuthResponse, Problem } from '@/api/types';
import { useAuth } from './auth';
import { API_URL } from './env';

export class ApiError extends Error {
  readonly problem: Problem;
  /** Seconds from the Retry-After header (429s). */
  readonly retryAfter: number | null;
  constructor(problem: Problem, retryAfter: number | null = null) {
    super(problem.detail || problem.title);
    this.problem = problem;
    this.retryAfter = retryAfter;
  }
  get status() {
    return this.problem.status;
  }
  get code() {
    return this.problem.code;
  }
  /** 422 field errors keyed by the last path segment, for react-hook-form. */
  fieldErrors(): Record<string, string> {
    return Object.fromEntries((this.problem.errors ?? []).map(e => [String(e.loc[e.loc.length - 1]), e.msg]));
  }
}

async function toProblem(res: Response): Promise<Problem> {
  try {
    const body = await res.json();
    if (body && typeof body === 'object' && 'title' in body) return body as Problem;
    return { title: res.statusText || 'Request failed', status: res.status, detail: JSON.stringify(body) };
  } catch {
    return { title: res.statusText || 'Request failed', status: res.status };
  }
}

/** Cookie-authenticated auth calls need credentials + the CSRF guard header. */
async function cookieAuth(path: '/auth/refresh' | '/auth/logout'): Promise<Response> {
  return fetch(`${API_URL}${path}`, { method: 'POST', credentials: 'include', headers: { 'X-CSRF': '1' } });
}

let refreshing: Promise<boolean> | null = null;

/** One refresh at a time; concurrent 401s wait for the same attempt. */
export function refreshSession(): Promise<boolean> {
  refreshing ??= (async () => {
    try {
      const res = await cookieAuth('/auth/refresh');
      if (!res.ok) {
        useAuth.getState().clear();
        return false;
      }
      const body = (await res.json()) as AuthResponse;
      useAuth.getState().setSession(body.access_token, body.user);
      return true;
    } catch {
      useAuth.getState().clear();
      return false;
    } finally {
      refreshing = null;
    }
  })();
  return refreshing;
}

export async function logout() {
  try {
    await cookieAuth('/auth/logout');
  } finally {
    useAuth.getState().clear();
  }
}

type Json = Record<string, unknown> | unknown[];
export interface RequestOptions extends Omit<RequestInit, 'body'> {
  body?: Json | FormData | Blob;
  query?: Record<string, string | number | boolean | null | undefined>;
  /** Skip the 401 → refresh → retry dance (used by the auth endpoints themselves). */
  noRefresh?: boolean;
  /** Return the body as text (e.g. text/markdown) instead of JSON. */
  text?: boolean;
}

export async function api<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const { body, query, noRefresh, text, headers, ...init } = opts;
  const url = new URL(`${API_URL}${path}`, location.origin);
  for (const [k, v] of Object.entries(query ?? {})) if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, String(v));

  const send = () => {
    const h = new Headers(headers);
    const token = useAuth.getState().accessToken;
    if (token) h.set('Authorization', `Bearer ${token}`);
    let payload: BodyInit | undefined;
    if (body instanceof FormData || body instanceof Blob) payload = body;
    else if (body !== undefined) {
      h.set('Content-Type', 'application/json');
      payload = JSON.stringify(body);
    }
    return fetch(url, { ...init, headers: h, body: payload, credentials: 'include' });
  };

  let res = await send();
  if (res.status === 401 && !noRefresh && (await refreshSession())) res = await send();
  if (!res.ok) {
    const ra = Number(res.headers.get('Retry-After'));
    throw new ApiError(await toProblem(res), Number.isFinite(ra) && ra > 0 ? ra : null);
  }
  if (res.status === 204) return undefined as T;
  if (text) return (await res.text()) as T;
  return (await res.json()) as T;
}
