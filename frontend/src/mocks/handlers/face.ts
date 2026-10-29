import { bypass, http, HttpResponse } from 'msw';
import type { FaceReport, Identified, Person, PersonPhoto, RejectReason } from '@/api/face';
import type { Project, Template } from '@/api/types';
import { uuid } from '../db';
import { lag, problem, u } from '../util';

// Mirrors the backend face contract until the backend lands. Overlay mode: face projects live here (localStorage);
// every other project, the templates and the project list still come from the real API.
const KEY = 'automl.mockface.v1';
interface Store {
  projects: (Project & { threshold: number })[];
  people: (Person & { photos: PersonPhoto[] })[];
}
const load = (): Store => {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? '') as Store;
  } catch {
    return { projects: [], people: [] };
  }
};
let st = load();
const save = () => localStorage.setItem(KEY, JSON.stringify(st));
const proj = (id: string) => st.projects.find(p => p.id === id);
const person = (id: string) => st.people.find(p => p.id === id);
const peopleOf = (pid: string) => st.people.filter(p => p.project_id === pid).sort((a, b) => a.name.localeCompare(b.name));
const out = ({ photos: _p, ...rest }: Person & { photos: PersonPhoto[] }): Person => rest;

function refresh(pid: string) {
  const p = proj(pid)!;
  const people = peopleOf(pid);
  for (const x of people) {
    x.photo_count = x.photos.length;
    x.warnings = x.photos.length === 0 ? [{ code: 'no_photos', detail: 'No photos yet' }] : x.photos.length < 3 ? [{ code: 'few_photos', detail: 'Add at least 3 photos' }] : [];
  }
  p.status = (people.some(x => x.photos.length) ? 'Ready' : 'Data needed') as Project['status'];
  p.sample_count = people.reduce((n, x) => n + x.photos.length, 0);
  p.updated_at = new Date().toISOString();
  save();
}

/** Small JPEG thumbnail (data URL) so mock photos survive reloads. */
async function thumb(file: File): Promise<string | null> {
  try {
    const bmp = await createImageBitmap(file);
    const s = 160 / Math.max(bmp.width, bmp.height);
    const c = new OffscreenCanvas(Math.round(bmp.width * s), Math.round(bmp.height * s));
    c.getContext('2d')!.drawImage(bmp, 0, 0, c.width, c.height);
    const blob = await c.convertToBlob({ type: 'image/jpeg', quality: 0.7 });
    return await new Promise(r => { const fr = new FileReader(); fr.onload = () => r(fr.result as string); fr.readAsDataURL(blob); });
  } catch {
    return null;
  }
}

const settings = (p: Store['projects'][number]) => ({
  threshold: p.threshold,
  default_threshold: 0.4,
  embedding_model: { id: 'insightface-buffalo_l-arcface', name: 'ArcFace (InsightFace buffalo_l)', licence: 'non-commercial research only', commercial_use: false },
  detector: { id: 'retinaface', licence: 'non-commercial research only' },
  purpose: null,
  retention_until: null,
  people_count: peopleOf(p.id).length,
  photo_count: peopleOf(p.id).reduce((n, x) => n + x.photos.length, 0),
});

const hist = (mean: number, sd: number, n: number) => {
  const bins = Array.from({ length: 20 }, (_, i) => ({ lo: i / 20, hi: (i + 1) / 20, count: 0 }));
  for (let i = 0; i < n; i++) {
    const v = Math.min(0.999, Math.max(0, mean + sd * (Math.random() + Math.random() + Math.random() - 1.5)));
    bins[Math.floor(v * 20)]!.count++;
  }
  return { count: n, mean, hist: bins };
};

export const faceHandlers = [
  // Enable the Face card (the real template list still says "Coming soon" until the backend lands).
  http.get(u('/templates'), async ({ request }) => {
    const list = (await (await fetch(bypass(request))).json()) as Template[];
    const img = list.find(t => t.id === 'image_classification');
    return HttpResponse.json(
      list.map(t =>
        t.id === 'face_recognition'
          ? { ...t, enabled: true, description: 'Recognise enrolled people by face: attendance or check-in, with consent.', task_types: ['face_recognition'], default_spec: img?.default_spec && { ...img.default_spec, task_type: 'face_recognition', labels: [], primary_metric: 'tar_at_far', secondary_metrics: [] } }
          : t,
      ),
    );
  }),
  http.post(u('/orgs/:orgId/projects'), async ({ request, params }) => {
    const body = (await request.clone().json()) as { name: string; spec?: { task_type: string }; template?: string };
    if (body.spec?.task_type !== 'face_recognition' && body.template !== 'face_recognition') return; // real API
    await lag();
    const now = new Date().toISOString();
    const p = {
      id: uuid(), organization_id: params.orgId as string, name: body.name.trim(), task_type: 'face_recognition', category: 'image',
      status: 'Data needed', current_spec: { ...(body.spec ?? {}), task_type: 'face_recognition', labels: [] }, spec_version: 1, dataset_id: null,
      label_set_id: null, sample_count: 0, current_version_id: null, best_metric: null, created_at: now, updated_at: now, archived_at: null, threshold: 0.4,
    } as unknown as Store['projects'][number];
    st.projects.push(p);
    save();
    return HttpResponse.json(p, { status: 201 });
  }),
  http.get(u('/orgs/:orgId/projects'), async ({ request, params }) => {
    const real = (await (await fetch(bypass(request))).json()) as { items: Project[]; next_cursor: string | null };
    const url = new URL(request.url);
    const q = (url.searchParams.get('search') ?? '').toLowerCase();
    const mine = st.projects.filter(p => p.organization_id === params.orgId && !p.archived_at && p.name.toLowerCase().includes(q));
    return HttpResponse.json({ ...real, items: [...mine, ...real.items] });
  }),
  http.get(u('/projects/:id'), ({ params }) => {
    const p = proj(params.id as string);
    return p ? HttpResponse.json(p) : undefined;
  }),
  http.post(/\/projects\/[^/]+:purge$/, async ({ request }) => {
    const p = proj(new URL(request.url).pathname.split('/').at(-1)!.replace(/:purge$/, ''));
    if (!p) return;
    const { confirm_name } = (await request.json()) as { confirm_name: string };
    if (confirm_name !== p.name) return problem(422, 'Validation failed', { code: 'confirmation_mismatch' });
    st = { projects: st.projects.filter(x => x.id !== p.id), people: st.people.filter(x => x.project_id !== p.id) };
    save();
    return HttpResponse.json({ project_id: p.id, status: 'Deleting', task_id: uuid() }, { status: 202 });
  }),
  http.get(u('/projects/:id/storage'), ({ params }) =>
    proj(params.id as string) ? HttpResponse.json({ totals: { data_bytes: 0, manifests_bytes: 0, models_bytes: 0, exports_bytes: 0, total_bytes: 0 }, versions: [] }) : undefined),
  http.get(u('/projects/:id/models'), ({ params }) => (proj(params.id as string) ? HttpResponse.json([]) : undefined)),
  http.get(u('/projects/:id/training-jobs'), ({ params }) => (proj(params.id as string) ? HttpResponse.json([]) : undefined)),

  http.get(u('/projects/:id/face-settings'), async ({ params }) => {
    const p = proj(params.id as string);
    if (!p) return problem(404, 'Not found');
    await lag();
    return HttpResponse.json(settings(p));
  }),
  http.patch(u('/projects/:id/face-settings'), async ({ params, request }) => {
    const p = proj(params.id as string);
    if (!p) return problem(404, 'Not found');
    const b = (await request.json()) as { threshold?: number };
    if (b.threshold != null) p.threshold = Math.min(0.95, Math.max(0.05, b.threshold));
    save();
    return HttpResponse.json(settings(p));
  }),
  http.get(u('/projects/:id/people'), async ({ params }) => {
    await lag();
    return HttpResponse.json(peopleOf(params.id as string).map(out));
  }),
  http.post(u('/projects/:id/people'), async ({ params, request }) => {
    const pid = params.id as string;
    if (!proj(pid)) return problem(404, 'Not found');
    const b = (await request.json()) as { name: string; external_ref?: string; consent?: { attested?: boolean; purpose?: string } };
    if (b.consent?.attested !== true) return problem(422, 'Consent required', { code: 'consent_required', detail: 'Confirm that this person agreed before adding them' });
    await lag();
    const x = {
      id: uuid(), project_id: pid, name: b.name.trim(), external_ref: b.external_ref ?? null, status: 'active' as const, photo_count: 0,
      consent: { id: uuid(), method: 'checkbox_attested', attested_by: 'me', granted_at: new Date().toISOString(), purpose: b.consent.purpose ?? null },
      warnings: [], created_at: new Date().toISOString(), photos: [],
    };
    st.people.push(x);
    refresh(pid);
    return HttpResponse.json(out(x), { status: 201 });
  }),
  http.get(u('/people/:pid'), async ({ params }) => {
    const x = person(params.pid as string);
    if (!x) return problem(404, 'Not found');
    await lag();
    return HttpResponse.json(x);
  }),
  http.post(u('/people/:pid/photos'), async ({ params, request }) => {
    const x = person(params.pid as string);
    if (!x) return problem(404, 'Not found');
    const files = (await request.formData()).getAll('files') as File[];
    await lag();
    const accepted = [], rejected: { filename: string; reason: RejectReason; detail: string }[] = [];
    for (const f of files) {
      const reject = (reason: RejectReason, detail: string) => rejected.push({ filename: f.name, reason, detail });
      if (f.size > 10 * 1024 * 1024) { reject('file_too_large', 'Over 10 MB'); continue; }
      if (x.photos.length >= 10) { reject('too_many_photos', 'This person already has 10 photos'); continue; }
      const t = await thumb(f);
      if (!t) { reject('not_an_image', 'Could not read the image'); continue; }
      // ponytail: deterministic fake checks so both outcomes show; the real API runs the detector.
      if (/group|crowd/i.test(f.name)) { reject('multiple_faces', '2 faces found'); continue; }
      if (f.size < 4000) { reject('no_face', 'No face detected'); continue; }
      const ph = { id: uuid(), thumb_url: t, quality: +(0.6 + Math.random() * 0.4).toFixed(2), face_px: 120 + Math.round(Math.random() * 200), created_at: new Date().toISOString() };
      x.photos.push(ph);
      accepted.push({ photo_id: ph.id, filename: f.name, quality: ph.quality, face_px: ph.face_px, thumb_url: t });
    }
    refresh(x.project_id);
    return HttpResponse.json({ accepted, rejected, person: out(x) });
  }),
  http.delete(u('/people/:pid/photos/:photoId'), ({ params }) => {
    const x = person(params.pid as string);
    if (!x) return problem(404, 'Not found');
    x.photos = x.photos.filter(p => p.id !== params.photoId);
    refresh(x.project_id);
    return new HttpResponse(null, { status: 204 });
  }),
  http.delete(u('/people/:pid'), ({ params }) => {
    const x = person(params.pid as string);
    if (!x) return problem(404, 'Not found');
    st.people = st.people.filter(p => p.id !== x.id);
    refresh(x.project_id);
    return new HttpResponse(null, { status: 204 });
  }),
  http.get(u('/projects/:id/face-report'), async ({ params }) => {
    const p = proj(params.id as string);
    if (!p) return problem(404, 'Not found');
    await lag();
    const people = peopleOf(p.id);
    const photos = people.reduce((n, x) => n + x.photos.length, 0);
    const ready = people.filter(x => x.photos.length >= 2).length >= 2;
    const r: FaceReport = {
      people: people.length, photos, ready, current_threshold: p.threshold, suggested_threshold: ready ? 0.42 : null,
      genuine: ready ? hist(0.68, 0.12, photos * 3) : null, impostor: ready ? hist(0.18, 0.1, photos * 8) : null,
      tar_at_far: ready ? [{ far: 0.001, tar: 0.86, threshold: 0.48 }, { far: 0.01, tar: 0.93, threshold: 0.42 }, { far: 0.1, tar: 0.98, threshold: 0.33 }] : null,
      rank1_accuracy: ready ? 0.95 : null,
      per_person: people.map(x => ({ person_id: x.id, name: x.name, photos: x.photos.length, rank1_hits: Math.max(0, x.photos.length - (x.photos.length > 3 ? 0 : 1)), warnings: x.warnings })),
      warnings: ready ? [] : [{ code: 'not_ready', detail: 'Add at least 2 people with 2 or more photos each to measure accuracy.' }],
    };
    return HttpResponse.json(r);
  }),
  http.post(u('/projects/:id/identify'), async ({ params, request }) => {
    const p = proj(params.id as string);
    if (!p) return problem(404, 'Not found');
    const f = (await request.formData()).get('file') as File;
    const bmp = await createImageBitmap(f);
    await lag();
    const people = peopleOf(p.id).filter(x => x.photos.length);
    const s = Math.min(bmp.width, bmp.height) * 0.38;
    const scores = people.map(x => ({ person_id: x.id, name: x.name, score: +(0.2 + Math.random() * 0.6).toFixed(3) })).sort((a, b) => b.score - a.score);
    const best = scores[0];
    const r: Identified = {
      image: { width: bmp.width, height: bmp.height }, threshold: p.threshold, latency_ms: 40,
      faces: [{ box: { x: (bmp.width - s) / 2, y: (bmp.height - s) / 2.4, w: s, h: s * 1.2 }, person_id: best && best.score >= p.threshold ? best.person_id : null, name: best && best.score >= p.threshold ? best.name : null, score: best?.score ?? 0.1, top3: scores.slice(0, 3) }],
    };
    return HttpResponse.json(r);
  }),
];
