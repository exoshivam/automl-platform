import type { OrgOut, Project, Role, User } from '@/api/types';

export type OrgRow = Omit<OrgOut, 'role'>;
export interface Mail {
  id: string;
  to: string;
  subject: string;
  text: string;
  sent_at: string;
}

/** In-memory mock database, persisted to localStorage so dev reloads keep data. */
interface Db {
  users: (User & { password: string })[];
  orgs: OrgRow[];
  memberships: { user_id: string; role: Role; organization_id: string }[];
  /** One-time email verification tokens -> user id. */
  verifyTokens: Record<string, string>;
  /** Dev mail outbox (mirrors the backend's GET /_dev/mail). */
  mail: Mail[];
  projects: Project[];
  /** Simulates the httpOnly refresh cookie: the logged-in user id. */
  session: string | null;
}

const KEY = 'automl.mockdb.v3';

export function uuid(): string {
  // UUIDv7-ish: time-ordered prefix, random tail.
  const t = Date.now().toString(16).padStart(12, '0');
  const r = crypto.randomUUID().replace(/-/g, '');
  return `${t.slice(0, 8)}-${t.slice(8, 12)}-7${r.slice(1, 4)}-${r.slice(4, 8)}-${r.slice(8, 20)}`;
}

function seed(): Db {
  const now = Date.now();
  const iso = (minsAgo: number) => new Date(now - minsAgo * 60_000).toISOString();
  const user = { id: uuid(), email: 'demo@automl.dev', name: 'Demo User', email_verified: true, password: 'demo1234' };
  const personal: OrgRow = { id: uuid(), name: "Demo's workspace", slug: 'demo', plan: 'pro', region: 'ap-south-1', personal: true, created_at: iso(60 * 24 * 30) };
  const team: OrgRow = { id: uuid(), name: 'Acme Factory', slug: 'acme', plan: 'team', region: 'ap-south-1', personal: false, created_at: iso(60 * 24 * 20) };
  const base = (o: Partial<Project> & Pick<Project, 'name' | 'task_type' | 'status'>): Project => ({
    id: uuid(),
    organization_id: personal.id,
    category: o.task_type.startsWith('tabular') ? 'tabular' : o.task_type === 'text_classification' ? 'text' : 'image',
    current_spec: {
      spec_version: '1.0',
      task_type: o.task_type,
      labels: o.task_type.startsWith('tabular') ? [] : ['helmet', 'vest', 'no_helmet'],
      input: { modality: o.task_type.startsWith('tabular') ? 'tabular' : o.task_type === 'text_classification' ? 'text' : 'image' },
      primary_metric: o.task_type === 'object_detection' ? 'map_50' : o.task_type === 'tabular_regression' ? 'rmse' : 'macro_f1',
      constraints: { target_platforms: ['server'], export_formats: ['onnx'] },
      budget_mode: 'quick',
      split: { train: 0.7, val: 0.15, test: 0.15, stratify: true, group_by: null, seed: 42 },
    },
    spec_version: 1,
    dataset_id: uuid(),
    label_set_id: uuid(),
    sample_count: 0,
    current_version_id: null,
    best_metric: null,
    created_at: iso(60 * 24 * 6),
    updated_at: iso(30),
    archived_at: null,
    ...o,
  });
  const projects: Project[] = [
    base({ name: 'Helmet & vest detector', task_type: 'object_detection', status: 'Trained', sample_count: 412, best_metric: 0.873, updated_at: iso(42) }),
    base({ name: 'Leaf disease classifier', task_type: 'image_classification', status: 'Deployed', sample_count: 1840, best_metric: 0.941, updated_at: iso(60 * 5) }),
    base({ name: 'Support ticket router', task_type: 'text_classification', status: 'Training', sample_count: 6210, updated_at: iso(3) }),
    base({ name: 'Customer churn', task_type: 'tabular_classification', status: 'Ready to train', sample_count: 12000, updated_at: iso(60 * 26) }),
    base({ name: 'Review sentiment (Hinglish)', task_type: 'text_classification', status: 'Validating', sample_count: 2300, updated_at: iso(14) }),
    base({ name: 'PCB defects', task_type: 'image_classification', status: 'Data needed', updated_at: iso(60 * 50) }),
    base({ name: 'Delivery ETA', task_type: 'tabular_regression', status: 'Draft', updated_at: iso(60 * 24 * 3) }),
    base({ name: 'Old shelf counter', task_type: 'object_detection', status: 'Trained', archived_at: iso(60 * 24 * 20), updated_at: iso(60 * 24 * 20) }),
    base({ name: 'Line 2 scratches', task_type: 'image_classification', status: 'Trained', organization_id: team.id, sample_count: 960, best_metric: 0.902 }),
  ];
  return {
    users: [user],
    orgs: [personal, team],
    memberships: [
      { user_id: user.id, role: 'owner', organization_id: personal.id },
      { user_id: user.id, role: 'member', organization_id: team.id },
    ],
    projects,
    verifyTokens: {},
    mail: [],
    session: null,
  };
}

function load(): Db {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) return JSON.parse(raw) as Db;
  } catch {
    /* fall through to seed */
  }
  return seed();
}

export const db: Db = load();
export function save() {
  try {
    localStorage.setItem(KEY, JSON.stringify(db));
  } catch {
    /* quota or private mode: in-memory only */
  }
}
export function resetDb() {
  localStorage.removeItem(KEY);
  location.reload();
}
