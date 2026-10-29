// Data, labeling and validation contract.
// Hand-written until these paths appear in openapi.json, then swap for generated types.
import type { ISODate, UUID } from './types';

export type UploadSource = 'files' | 'folder' | 'zip' | 'camera' | 'annotations';

export interface UploadSession {
  upload_id: UUID;
  key_prefix: string;
  expires_at: ISODate;
  max_file_bytes: number;
  part_size: number;
}

/** POST /uploads/{id}/sign. Covers PutObject, multipart create/part/complete, ListParts, Abort. */
export interface SignRequest {
  method: 'PUT' | 'POST' | 'GET' | 'DELETE';
  key: string;
  upload_id?: string;
  part_number?: number;
  size?: number;
}
export interface SignResponse {
  url: string;
  headers?: Record<string, string>;
}

export interface CompleteFile {
  key: string;
  name: string;
  relative_path: string | null;
  size: number;
  mime: string;
  sha256: string | null;
  label: string | null;
}

export type ImportFormat = 'csv' | 'xlsx' | 'parquet' | 'jsonl' | 'coco' | 'yolo' | 'voc';
export type ColumnRole = 'target' | 'feature' | 'text' | 'id' | 'timestamp' | 'group' | 'ignore';
export type ColumnType = 'categorical' | 'numeric' | 'text' | 'datetime' | 'boolean';

export interface ColumnPreview {
  name: string;
  inferred_type: ColumnType;
  sample_values: unknown[];
  unique_count: number;
  missing_pct: number;
}
export interface ImportPreview {
  columns: ColumnPreview[];
  row_count: number;
}
export interface ColumnMapping {
  columns: { name: string; role: ColumnRole; type: ColumnType }[];
}

export type IngestStatus = 'pending' | 'running' | 'ready' | 'failed';
export type ValidationStatus = 'pending' | 'running' | 'passed' | 'warnings' | 'failed';

export interface DatasetVersion {
  id: UUID;
  dataset_id: UUID;
  version_no: number;
  parent_version_id: UUID | null;
  origin: 'upload' | 'import' | 'labels' | 'fix';
  sample_count: number;
  label_stats: Record<string, { images: number; boxes?: number }>;
  split_counts: { train: number; val: number; test: number } | null;
  validation_status: ValidationStatus;
  ingest_status: IngestStatus;
  /** Skipped files, e.g. {file, code:'path_traversal'|'symlink'|...}. */
  ingest_errors: { file?: string; code?: string; message?: string }[];
  sealed_at: ISODate | null;
  /** The open draft: uploads/captures/imports append here until it is sealed. */
  is_draft: boolean;
  created_at: ISODate;
  /** Classes found in the data (e.g. new folder names) but missing from the label set. */
  unknown_classes?: string[];
}

export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}
export type LabelSource = 'user' | 'import' | 'ai_prelabel';
export interface SampleLabel {
  id?: UUID;
  class_name: string;
  /** Pixels of the original image, top-left origin (COCO style). Null for classification. */
  bbox: BBox | null;
  source?: LabelSource;
  confidence?: number | null;
}
export type Split = 'train' | 'val' | 'test';
export interface Sample {
  id: UUID;
  /** Decode failure reason (VAL-1) or null. */
  corrupt?: string | null;
  name?: string | null;
  sha256?: string | null;
  bytes?: number | null;
  /** Presigned MinIO URLs (cross-origin; fine for <img>). Null for text/tabular. */
  thumb_url: string | null;
  url: string | null;
  width: number | null;
  height: number | null;
  mime: string | null;
  split: Split | null;
  labels: SampleLabel[];
  no_objects: boolean;
  flags: string[];
  text?: string;
  /** Detection: the object the user said the photo is of (camera picker / upload selector). */
  capture_target?: string | null;
}

export interface RowsPage {
  columns: { name: string; type: ColumnType; role: ColumnRole }[];
  items: { row: number; values: Record<string, unknown>; split: Split | null }[];
  next_cursor: string | null;
}

export type Severity = 'error' | 'warning' | 'info';
export interface ValidationItem {
  code: string;
  severity: Severity;
  title: string;
  why: string;
  count: number;
  examples: UUID[];
  fix: { text: string; auto_fix_id?: string | null };
  /** Errors that can't be acknowledged must be fixed. */
  acknowledgeable?: boolean;
  acknowledged: boolean;
  /** Several items may share a code (e.g. VAL-16 per kind); details.kind tells them apart. */
  details?: Record<string, unknown>;
}
export interface ValidationReport {
  dataset_version_id?: UUID;
  status: ValidationStatus;
  summary: { errors: number; warnings: number; info: number };
  items: ValidationItem[];
  /** Images only (null for text/tabular). low=true adds a VAL-18 warning. */
  diversity?: Diversity | null;
}

export interface Diversity {
  samples: number;
  effective_samples: number;
  near_duplicate_share: number;
  sessions_per_class: Record<string, number>;
  min_sessions_per_class: number;
  camera_share?: number;
  low: boolean;
}

export interface LabelUpsert {
  sample_id: UUID;
  labels: SampleLabel[];
  no_objects?: boolean;
}

export interface BoxSuggestions {
  model: string;
  licence?: string;
  suggestions: Record<UUID, { class_name: string; bbox: BBox; confidence: number }[]>;
  /** Own model: best score per class for each sample, including below the floor ("best charger match 11%"). */
  max_scores?: Record<string, Record<string, number>>;
}

/** POST labels:batch result. On a sealed version the server forks a draft and sample ids change. */
export interface LabelsBatchOut {
  updated: number;
  dataset_version_id: UUID;
  forked: boolean;
  /** old sample id -> new sample id, when forked */
  sample_id_map: Record<UUID, UUID>;
}

export interface LabelSet {
  id: UUID;
  project_id: UUID;
  version: number;
  classes: string[];
}
export interface LabelSetChange {
  label_set: LabelSet;
  dataset_version_id: UUID | null;
  relabeled: number;
}
