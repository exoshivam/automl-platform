import { delay, HttpResponse } from 'msw';
import { API_URL } from '@/lib/env';
import { db } from './db';

export const u = (path: string) => `${API_URL}${path}`;

export function problem(status: number, title: string, extra: Record<string, unknown> = {}) {
  return HttpResponse.json({ type: 'about:blank', title, status, ...extra }, { status, headers: { 'Content-Type': 'application/problem+json' } });
}

/** Realistic latency so loading states are visible. */
export const lag = () => delay(250 + Math.random() * 350);

/** Bearer token format in mocks: "mock.<userId>". Returns the user or a 401 response. */
export function authed(request: Request) {
  const token = request.headers.get('Authorization')?.replace(/^Bearer /, '') ?? '';
  const user = db.users.find(x => `mock.${x.id}` === token);
  return user ?? null;
}

export function paginate<T>(items: T[], url: URL) {
  const limit = Number(url.searchParams.get('limit') ?? 50);
  const start = Number(url.searchParams.get('cursor') ?? 0);
  const page = items.slice(start, start + limit);
  return { items: page, next_cursor: start + limit < items.length ? String(start + limit) : null };
}
