// Domain types for the PRD §11 API. Hand-written until the backend openapi.json lands;
// then `npm run gen:api` writes src/api/schema.d.ts and these become aliases of it.

export type UUID = string;
export type ISODate = string;

export interface Page<T> {
  items: T[];
  next_cursor: string | null;
}

/** RFC 9457 problem details. */
export interface Problem {
  type?: string;
  title: string;
  status: number;
  detail?: string;
  instance?: string;
  code?: string;
  errors?: { loc: (string | number)[]; msg: string }[];
}

// Auth, user and workspace shapes come straight from the backend OpenAPI (npm run gen:api).
import type { components } from './schema';
type S = components['schemas'];

export type User = S['UserOut'];
export type AuthResponse = S['TokenOut'];
export type RegisterResponse = S['RegisterOut'];
export type Me = S['MeOut'];
/** Flat membership row from GET /me. */
export type Membership = S['MembershipOut'];
export type OrgOut = S['OrgOut'];

export type Role = 'owner' | 'admin' | 'member' | 'viewer';
export type Plan = 'free' | 'pro' | 'team' | 'enterprise';
export const asPlan = (p: string | undefined): Plan => (p === 'pro' || p === 'team' || p === 'enterprise' ? p : 'free');

/** The workspace the UI is working in, normalised from a membership row. */
export interface Org {
  id: UUID;
  name: string;
  slug: string;
  plan: Plan;
  role: string;
  personal: boolean;
}

/** Phase 1 task types enabled by the backend (ENABLED_TASKS). Sentiment is a text_classification preset. */
export type TaskType = 'image_classification' | 'object_detection' | 'text_classification' | 'tabular_classification' | 'tabular_regression' | 'face_recognition' | 'speaker_recognition';

export type BudgetMode = 'quick' | 'standard' | 'thorough';
export type Platform = 'server' | 'android' | 'ios' | 'web' | 'edge_linux';
export type ExportFormat = 'onnx' | 'tflite' | 'coreml' | 'tfjs' | 'native';

/** Appendix A task spec (backend validates with additionalProperties:false). */
export interface TaskSpec {
  spec_version: '1.0';
  task_type: TaskType;
  multi_label?: boolean;
  labels?: string[];
  input: { modality: 'image' | 'text' | 'tabular' | 'audio'; min_resolution?: number };
  primary_metric: string;
  secondary_metrics?: string[];
  constraints: {
    target_platforms: Platform[];
    export_formats?: ExportFormat[];
    max_model_size_mb?: number | null;
    max_latency_ms?: number | null;
    latency_device?: string | null;
    quantization?: 'none' | 'fp16' | 'int8_dynamic' | 'int8_full';
  };
  budget_mode: BudgetMode;
  split: { train: number; val: number; test: number; stratify: boolean; group_by: string | null; seed: number };
  language?: string[] | null;
  target_column?: string | null;
  feature_columns?: string[] | null;
  id_columns?: string[] | null;
  group_column?: string | null;
  notes?: string | null;
}

/** GET /templates. `id` is the template id (image_classification | object_detection | text_classification | tabular | coming-soon ids). */
export type Template = Omit<S['TemplateOut'], 'default_spec'> & { default_spec: TaskSpec | null };

/** PRJ-7 statuses, exactly as the API sends them. */
export type ProjectStatus = S['ProjectOut']['status'];

/** best_metric is the value of current_spec.primary_metric (null until trained). */
export type Project = Omit<S['ProjectOut'], 'current_spec' | 'task_type'> & {
  task_type: TaskType;
  current_spec: TaskSpec;
};

export type FromDescriptionResponse = Omit<S['FromDescriptionOut'], 'draft_spec'> & { draft_spec: TaskSpec | null };
export type SpecVersion = Omit<S['SpecVersionOut'], 'spec'> & { spec: TaskSpec };
