import { z } from 'zod';
import type { TaskType } from '@/api/types';

/** Mirrors backend app/services/specs.py METRICS (primary_metric must be valid for the task_type). */
const CLASSIFICATION = ['macro_f1', 'accuracy', 'micro_f1', 'weighted_f1', 'roc_auc', 'pr_auc', 'per_class_recall', 'per_class_precision', 'log_loss'];
export const METRICS: Record<TaskType, string[]> = {
  image_classification: CLASSIFICATION,
  text_classification: CLASSIFICATION,
  tabular_classification: CLASSIFICATION,
  object_detection: ['map_50', 'map_50_95', 'per_class_ap'],
  tabular_regression: ['rmse', 'mae', 'r2', 'mape'],
  face_recognition: ['tar_at_far', 'rank1_accuracy'],
  speaker_recognition: ['tar_at_far', 'rank1_accuracy'],
};

export const METRIC_LABEL: Record<string, string> = {
  macro_f1: 'Macro F1',
  micro_f1: 'Micro F1',
  weighted_f1: 'Weighted F1',
  accuracy: 'Accuracy',
  roc_auc: 'ROC-AUC',
  pr_auc: 'PR-AUC',
  per_class_recall: 'Per-class recall',
  per_class_precision: 'Per-class precision',
  log_loss: 'Log loss',
  map_50: 'mAP@0.5',
  map_50_95: 'mAP@0.5:0.95',
  per_class_ap: 'Per-class AP',
  rmse: 'RMSE',
  mae: 'MAE',
  r2: 'R²',
  mape: 'MAPE',
};

export const NEEDS_LABELS = (t: TaskType) => !t.startsWith('tabular') && t !== 'face_recognition' && t !== 'speaker_recognition';
/** Galleries of enrolled people (face or voice): consent per person, no training, no export. */
export const isBiometric = (t: TaskType) => t === 'face_recognition' || t === 'speaker_recognition';

const label = z
  .string()
  .trim()
  .min(1, 'Empty class name')
  .max(64, 'Max 64 characters')
  .regex(/^[\p{L}\p{N} _.-]+$/u, 'Letters, numbers, spaces, _ . - only');

export const taskSpecSchema = z
  .object({
    spec_version: z.literal('1.0'),
    task_type: z.enum(['image_classification', 'object_detection', 'text_classification', 'tabular_classification', 'tabular_regression']),
    multi_label: z.boolean().optional(),
    labels: z.array(label).optional(),
    input: z.object({
      modality: z.enum(['image', 'text', 'tabular', 'audio']),
      min_resolution: z.number().int().min(16).max(4096).optional(),
    }),
    primary_metric: z.string(),
    secondary_metrics: z.array(z.string()).optional(),
    constraints: z.object({
      target_platforms: z.array(z.enum(['server', 'android', 'ios', 'web', 'edge_linux'])).min(1, 'Pick at least one platform'),
      export_formats: z.array(z.enum(['onnx', 'tflite', 'coreml', 'tfjs', 'native'])).optional(),
      max_model_size_mb: z.number().positive().max(10000).nullable().optional(),
      max_latency_ms: z.number().positive().max(60000).nullable().optional(),
      latency_device: z.string().nullable().optional(),
      quantization: z.enum(['none', 'fp16', 'int8_dynamic', 'int8_full']).optional(),
    }),
    mode: z.enum(['pretrained', 'custom']).optional(), // object detection: ready-made vs train your own
    score_threshold: z.number().min(0.05).max(0.95).optional(),
    budget_mode: z.enum(['quick', 'standard', 'thorough']),
    split: z.object({
      train: z.number().min(0.1).max(0.95),
      val: z.number().min(0.01).max(0.5),
      test: z.number().min(0.01).max(0.5),
      stratify: z.boolean(),
      group_by: z.string().nullable(),
      seed: z.number().int(),
    }),
    language: z.array(z.string()).nullable().optional(),
    target_column: z.string().nullable().optional(),
    feature_columns: z.array(z.string()).nullable().optional(),
    id_columns: z.array(z.string()).nullable().optional(),
    group_column: z.string().nullable().optional(),
    notes: z.string().max(2000).nullable().optional(),
  })
  .strict()
  .superRefine((s, ctx) => {
    if (s.task_type === 'tabular_regression' && s.labels?.length) ctx.addIssue({ code: 'custom', path: ['labels'], message: 'Regression has no class labels' });
    if (NEEDS_LABELS(s.task_type)) {
      const n = s.labels?.length ?? 0;
      const min = s.task_type === 'object_detection' ? 1 : 2; // a 1-class detector is valid
      if (n < min || n > 500) ctx.addIssue({ code: 'custom', path: ['labels'], message: `Add ${min} to 500 classes` });
      const seen = new Set<string>();
      for (const l of s.labels ?? []) {
        const k = l.trim().toLowerCase();
        if (seen.has(k)) ctx.addIssue({ code: 'custom', path: ['labels'], message: `Duplicate class "${l}"` });
        seen.add(k);
      }
    }
    if (!METRICS[s.task_type].includes(s.primary_metric))
      ctx.addIssue({ code: 'custom', path: ['primary_metric'], message: `Not a valid metric for ${s.task_type}` });
    const sum = s.split.train + s.split.val + s.split.test;
    if (Math.abs(sum - 1) > 0.001) ctx.addIssue({ code: 'custom', path: ['split'], message: `Splits must add up to 100% (now ${Math.round(sum * 100)}%)` });
  });

export type TaskSpecInput = z.infer<typeof taskSpecSchema>;

/** "constraints.max_model_size_mb: Too big" style lines for the JSON tab. */
export function specIssues(value: unknown): string[] {
  const r = taskSpecSchema.safeParse(value);
  return r.success ? [] : r.error.issues.map(i => `${i.path.join('.') || '(root)'}: ${i.message}`);
}
