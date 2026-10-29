import { http, HttpResponse } from 'msw';
import type { FromDescriptionResponse, Project, TaskSpec, TaskType } from '@/api/types';
import { TEMPLATES } from '../templates';
import { db, save, uuid } from '../db';
import { authed, lag, paginate, problem, u } from '../util';

const canSee = (userId: string, orgId: string) => db.memberships.some(m => m.user_id === userId && m.organization_id === orgId);

/** Crude stand-in for the LLM (PRJ-2/PRJ-5), shaped like FromDescriptionOut. "[llm-down]" in the text gives the 503 fallback. */
function describe(text: string): FromDescriptionResponse | 'down' {
  const t = text.toLowerCase();
  if (t.includes('[llm-down]')) return 'down';
  const log = uuid();
  if (/generate (an? )?(image|picture|video)|text to image/.test(t))
    return { status: 'unsupported', message: "Generating images isn't supported. We can classify or detect objects in images you provide.", category: null, draft_spec: null, llm_log_id: log };
  if (/face|attendance/.test(t))
    return { status: 'unsupported', message: 'Face recognition is a Phase 2 template with consent requirements; it is not available yet.', category: null, draft_spec: null, llm_log_id: log };

  let tpl = 'image_classification';
  let taskType: TaskType = 'image_classification';
  if (/detect|locate|count|box|where/.test(t)) [tpl, taskType] = ['object_detection', 'object_detection'];
  if (/review|tweet|sentiment|comment|feedback|ticket|email|message|text|intent|spam/.test(t)) [tpl, taskType] = ['text_classification', 'text_classification'];
  if (/csv|excel|spreadsheet|churn|customer|row|column|price|how much|how many/.test(t))
    [tpl, taskType] = ['tabular', /price|how much|how many|eta|amount|revenue/.test(t) ? 'tabular_regression' : 'tabular_classification'];

  const spec: TaskSpec = { ...structuredClone(TEMPLATES.find(x => x.id === tpl)!.default_spec!), task_type: taskType };
  const words = ['helmet', 'vest', 'no_helmet', 'defect', 'ok', 'spam', 'not_spam', 'positive', 'negative', 'neutral'].filter(w => t.includes(w.replace('_', ' ')));
  if (taskType === 'tabular_regression') Object.assign(spec, { labels: [], primary_metric: 'rmse', secondary_metrics: ['mae', 'r2'] });
  else if (!taskType.startsWith('tabular')) spec.labels = words.length >= 2 ? words : ['class_a', 'class_b'];
  if (/android|phone|mobile/.test(t)) spec.constraints = { ...spec.constraints, target_platforms: ['android'], export_formats: ['tflite'], max_model_size_mb: 15, max_latency_ms: 50 };
  if (/iphone|ios/.test(t)) spec.constraints = { ...spec.constraints, target_platforms: ['ios'], export_formats: ['coreml'] };
  if (/browser|website|web app/.test(t)) spec.constraints = { ...spec.constraints, target_platforms: ['web'], export_formats: ['tfjs'] };
  if (/hindi|hinglish|indian/.test(t)) spec.language = ['en', 'hi', 'hi-Latn'];
  spec.notes = text.slice(0, 300);
  if (/stock price|guarantee|with certainty/.test(t))
    return { status: 'ambiguous', message: 'No model can predict prices with certainty; this drafts a regression that gives estimates with an error range.', category: 'tabular', draft_spec: spec, llm_log_id: log };
  return { status: 'ok', message: `Drafted a ${tpl.replace('_', ' ')} project from your description.`, category: TEMPLATES.find(x => x.id === tpl)!.category, draft_spec: spec, llm_log_id: log };
}

const STATUS_FIRST = 'Draft' as const;

export const projectHandlers = [
  http.get(u('/templates'), async () => {
    await lag();
    return HttpResponse.json(TEMPLATES);
  }),

  http.get(u('/orgs/:orgId/projects'), async ({ request, params }) => {
    const user = authed(request);
    if (!user) return problem(401, 'Not authenticated');
    if (!canSee(user.id, params.orgId as string)) return problem(404, 'Not found');
    await lag();
    const url = new URL(request.url);
    const q = url.searchParams.get('search')?.toLowerCase() ?? '';
    const status = url.searchParams.get('status');
    const archived = url.searchParams.get('archived') === 'true';
    const items = db.projects
      .filter(p => p.organization_id === params.orgId)
      .filter(p => (archived ? p.archived_at : !p.archived_at))
      .filter(p => !status || p.status === status)
      .filter(p => !q || p.name.toLowerCase().includes(q))
      .sort((a, b) => b.updated_at.localeCompare(a.updated_at));
    return HttpResponse.json(paginate(items, url));
  }),

  http.post(u('/orgs/:orgId/projects'), async ({ request, params }) => {
    const user = authed(request);
    if (!user) return problem(401, 'Not authenticated');
    if (!canSee(user.id, params.orgId as string)) return problem(404, 'Not found');
    await lag();
    const body = (await request.json()) as { name: string; spec?: TaskSpec; template?: string; labels?: string[]; task_type?: TaskType; llm_log_id?: string };
    if (!body.name?.trim()) return problem(422, 'Validation failed', { errors: [{ loc: ['body', 'name'], msg: 'Name is required' }] });
    let spec = body.spec;
    if (!spec && body.template) {
      const tpl = TEMPLATES.find(x => x.id === body.template);
      if (!tpl) return problem(422, 'Unknown template', { code: 'unknown_template' });
      if (!tpl.enabled) return problem(422, 'Coming soon', { code: 'template_coming_soon' });
      spec = { ...structuredClone(tpl.default_spec!), labels: body.labels ?? [], task_type: (body.task_type as TaskType) ?? tpl.default_spec!.task_type };
    }
    if (!spec) return problem(422, 'Validation failed', { code: 'invalid_spec', errors: [{ loc: '/', msg: 'spec or template is required' }] });
    const tid = spec.task_type.startsWith('tabular') ? 'tabular' : spec.task_type;
    const now = new Date().toISOString();
    const p: Project = {
      id: uuid(),
      organization_id: params.orgId as string,
      name: body.name.trim(),
      task_type: spec.task_type,
      category: TEMPLATES.find(x => x.id === tid)!.category,
      status: STATUS_FIRST,
      current_spec: spec,
      spec_version: 1,
      dataset_id: uuid(),
      label_set_id: uuid(),
      sample_count: 0,
      current_version_id: null,
      best_metric: null,
      created_at: now,
      updated_at: now,
      archived_at: null,
    };
    db.projects.push(p);
    save();
    return HttpResponse.json(p, { status: 201 });
  }),

  http.post(u('/orgs/:orgId/projects/from-description'), async ({ request }) => {
    if (!authed(request)) return problem(401, 'Not authenticated');
    await lag();
    await lag();
    const { description } = (await request.json()) as { description: string };
    if (!description || description.trim().length < 10)
      return problem(422, 'Validation failed', { errors: [{ loc: ['body', 'description'], msg: 'Describe the task in at least a sentence' }] });
    const out = describe(description);
    if (out === 'down') return problem(503, 'Task description service unavailable', { code: 'llm_unavailable', fallback: 'category', detail: 'Pick a category instead.' });
    return HttpResponse.json(out);
  }),

  http.get(u('/projects/:id'), async ({ request, params }) => {
    const user = authed(request);
    if (!user) return problem(401, 'Not authenticated');
    const p = db.projects.find(x => x.id === params.id);
    if (!p || !canSee(user.id, p.organization_id)) return problem(404, 'Not found');
    await lag();
    return HttpResponse.json(p);
  }),

  http.patch(u('/projects/:id'), async ({ request, params }) => {
    const user = authed(request);
    const p = db.projects.find(x => x.id === params.id);
    if (!user || !p || !canSee(user.id, p.organization_id)) return problem(404, 'Not found');
    const body = (await request.json()) as Partial<Pick<Project, 'name'>> & { archived?: boolean };
    if (body.name) p.name = body.name;
    if (body.archived !== undefined) p.archived_at = body.archived ? new Date().toISOString() : null;
    p.updated_at = new Date().toISOString();
    save();
    return HttpResponse.json(p);
  }),

  http.delete(u('/projects/:id'), async ({ request, params }) => {
    const user = authed(request);
    const p = db.projects.find(x => x.id === params.id);
    if (!user || !p || !canSee(user.id, p.organization_id)) return problem(404, 'Not found');
    p.archived_at = new Date().toISOString();
    save();
    return new HttpResponse(null, { status: 204 });
  }),

  http.put(u('/projects/:id/spec'), async ({ request, params }) => {
    const user = authed(request);
    const p = db.projects.find(x => x.id === params.id);
    if (!user || !p || !canSee(user.id, p.organization_id)) return problem(404, 'Not found');
    p.current_spec = ((await request.json()) as { spec: TaskSpec }).spec;
    p.spec_version += 1;
    p.updated_at = new Date().toISOString();
    save();
    return HttpResponse.json(p);
  }),
];
