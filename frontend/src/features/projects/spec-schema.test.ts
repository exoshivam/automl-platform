import { describe, expect, it } from 'vitest';
import { specIssues, taskSpecSchema } from './spec-schema';

const appendixA = {
  spec_version: '1.0',
  task_type: 'image_classification',
  multi_label: false,
  labels: ['healthy', 'early_blight', 'late_blight'],
  input: { modality: 'image', min_resolution: 224 },
  primary_metric: 'macro_f1',
  secondary_metrics: ['accuracy', 'per_class_recall'],
  constraints: {
    target_platforms: ['android'],
    export_formats: ['tflite'],
    max_model_size_mb: 15,
    max_latency_ms: 50,
    latency_device: 'mid_range_android_cpu',
    quantization: 'int8_full',
  },
  budget_mode: 'standard',
  split: { train: 0.7, val: 0.15, test: 0.15, stratify: true, group_by: null, seed: 42 },
  language: null,
  notes: 'Photos taken in the field with phone cameras; varied lighting.',
};

describe('task spec schema', () => {
  it('accepts the PRD Appendix A example', () => {
    expect(taskSpecSchema.safeParse(appendixA).success).toBe(true);
  });
  it('needs 2+ unique labels for classification', () => {
    expect(specIssues({ ...appendixA, labels: ['one'] })).toContain('labels: Add 2 to 500 classes');
    expect(specIssues({ ...appendixA, labels: ['a', 'A'] }).some(m => m.includes('Duplicate'))).toBe(true);
  });
  it('rejects a metric that does not fit the task', () => {
    expect(specIssues({ ...appendixA, primary_metric: 'rmse' }).some(m => m.startsWith('primary_metric'))).toBe(true);
  });
  it('regression needs no labels; splits must sum to 100%', () => {
    const reg = { ...appendixA, task_type: 'tabular_regression', labels: [], primary_metric: 'rmse', secondary_metrics: [], input: { modality: 'tabular' } };
    expect(specIssues(reg)).toEqual([]);
    expect(specIssues({ ...reg, split: { ...reg.split, train: 0.8 } })[0]).toMatch(/add up to 100%/);
    expect(specIssues({ ...reg, labels: ['a', 'b'] })).toContain('labels: Regression has no class labels');
  });
  it('rejects unknown keys like the backend (additionalProperties:false)', () => {
    expect(specIssues({ ...appendixA, colour: 'blue' }).length).toBeGreaterThan(0);
  });
});
