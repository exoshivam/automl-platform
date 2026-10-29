import { setupWorker } from 'msw/browser';
import type { RequestHandler } from 'msw';
import { MOCKS } from '@/lib/env';
import { authHandlers } from './handlers/auth';
import { orgHandlers } from './handlers/orgs';
import { projectHandlers } from './handlers/projects';
import { dataHandlers, labelHandlers, validationHandlers } from './handlers/data';
import { starterHandlers } from './handlers/starters';
import { faceHandlers } from './handlers/face';

/** Mock groups. Turn any group off with VITE_MOCKS so those calls hit the real backend. */
const GROUPS: Record<string, RequestHandler[]> = {
  auth: authHandlers,
  orgs: orgHandlers,
  projects: projectHandlers, // includes templates and from-description
  data: dataHandlers, // uploads, fake S3, imports, versions, samples, rows, seal
  labels: labelHandlers, // labels:batch, box-suggestions (+ overlay on real samples)
  validation: validationHandlers, // report, acknowledge, fixes
  starters: starterHandlers, // DAT-13 starter dataset library (until the backend lands)
  face: faceHandlers, // §12 face recognition: overlay on the real API until the backend lands
  // training / evaluation handlers register here as they are built
};

async function removeStaleWorker() {
  // A worker left over from an earlier mocked session would still intercept; remove it.
  const regs = await navigator.serviceWorker?.getRegistrations?.();
  await Promise.all((regs ?? []).filter(r => r.active?.scriptURL.endsWith('/mockServiceWorker.js')).map(r => r.unregister()));
}

export async function startMocks() {
  const wanted = MOCKS === 'none' ? [] : MOCKS === 'all' ? Object.keys(GROUPS) : MOCKS.split(',').map(s => s.trim());
  const handlers = wanted.flatMap(g => GROUPS[g] ?? []);
  if (!handlers.length) return removeStaleWorker();
  // Unmocked requests pass through to the real API untouched.
  await setupWorker(...handlers).start({ onUnhandledFrame: 'bypass', quiet: true });
  console.info(`[mocks] active groups: ${wanted.filter(g => GROUPS[g]).join(', ')}`);
}

export { GROUPS };
