import type { TaskSpec, TaskType, Template } from '@/api/types';

/** Mirrors backend app/services/specs.py TEMPLATES / _base(). */
const MODALITY: Record<TaskType, TaskSpec['input']['modality']> = {
  image_classification: 'image',
  object_detection: 'image',
  text_classification: 'text',
  tabular_classification: 'tabular',
  tabular_regression: 'tabular',
  face_recognition: 'image',
  speaker_recognition: 'audio',
};

function base(task: TaskType, primary: string, secondary: string[], extra: Partial<TaskSpec> = {}): TaskSpec {
  return {
    spec_version: '1.0',
    task_type: task,
    multi_label: false,
    labels: [],
    input: { modality: MODALITY[task] },
    primary_metric: primary,
    secondary_metrics: secondary,
    constraints: { target_platforms: ['server'], export_formats: ['onnx'], max_model_size_mb: null, max_latency_ms: null, quantization: 'none' },
    budget_mode: 'quick',
    split: { train: 0.7, val: 0.15, test: 0.15, stratify: true, group_by: null, seed: 42 },
    language: null,
    notes: null,
    ...extra,
  };
}

const soon = (id: string, category: string, name: string): Template => ({ id, category, name, description: 'Coming soon', enabled: false, task_types: [id], default_spec: null });

export const TEMPLATES: Template[] = [
  {
    id: 'image_classification',
    category: 'image',
    name: 'Image classification',
    description: 'Sort photos into classes (defect / no defect, plant disease, product type).',
    enabled: true,
    task_types: ['image_classification'],
    default_spec: base('image_classification', 'macro_f1', ['accuracy'], { input: { modality: 'image', min_resolution: 224 } }),
  },
  {
    id: 'object_detection',
    category: 'image',
    name: 'Object detection',
    description: 'Draw boxes around objects (helmets, vehicles, shelf items).',
    enabled: true,
    task_types: ['object_detection'],
    default_spec: base('object_detection', 'map_50', ['map_50_95']),
  },
  {
    id: 'text_classification',
    category: 'text',
    name: 'Text / Sentiment',
    description: 'Route tickets, detect spam, or score sentiment (Indic and Hinglish supported).',
    enabled: true,
    task_types: ['text_classification'],
    default_spec: base('text_classification', 'macro_f1', ['accuracy'], { language: ['en'] }),
  },
  {
    id: 'tabular',
    category: 'tabular',
    name: 'Tabular (classification / regression)',
    description: 'Predict a column from the others: churn, fraud, price, ETA.',
    enabled: true,
    task_types: ['tabular_classification', 'tabular_regression'],
    default_spec: base('tabular_classification', 'macro_f1', ['accuracy', 'roc_auc'], { target_column: null, feature_columns: null, id_columns: null, group_column: null }),
  },
  soon('audio_classification', 'audio', 'Audio classification'),
  soon('keyword_spotting', 'audio', 'Keyword spotting'),
  soon('face_recognition', 'image', 'Face recognition'),
  soon('named_entity_extraction', 'text', 'Named entity extraction'),
  soon('aspect_sentiment', 'text', 'Aspect-based sentiment'),
  soon('forecasting', 'time_series', 'Forecasting'),
  soon('anomaly_detection', 'tabular', 'Anomaly detection'),
  soon('segmentation', 'image', 'Segmentation'),
];
