/** Relative by default: dev goes through the Vite proxy, production serves SPA and API from one host. */
export const API_URL: string = (import.meta.env.VITE_API_URL as string | undefined) ?? '/api/v1';

function devOverride(): string | null {
  try {
    return localStorage.getItem('automl.mocks');
  } catch {
    return null;
  }
}

/**
 * Groups the backend backend does NOT serve yet. Only these are mocked by default; everything else
 * (auth, orgs, projects, templates, from-description) hits the real API. Remove a group when it lands.
 */
export const NOT_BUILT_YET: string[] = [];

/**
 * "all" | "none" | comma list of mock groups. Mocks are dev-only.
 * In dev, localStorage['automl.mocks'] overrides VITE_MOCKS without a server restart
 * (e.g. localStorage.setItem('automl.mocks','all') for a fully offline demo).
 */
export const MOCKS: string = import.meta.env.DEV ? (devOverride() ?? (import.meta.env.VITE_MOCKS as string | undefined) ?? NOT_BUILT_YET.join(',')) : 'none';

/** True when this group's screens show mock data (they then carry a "Demo data" badge). */
export function isMocked(group: string): boolean {
  if (MOCKS === 'none') return false;
  if (MOCKS === 'all') return true;
  return MOCKS.split(',').map(s => s.trim()).includes(group);
}
