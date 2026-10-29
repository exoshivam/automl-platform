import { AudioLines, Box, FileText, Image as ImageIcon, LineChart, Mic, ScanFace, Shapes, Sparkles, Table2, Tags, TrendingDown } from 'lucide-react';
import type { ProjectStatus, TaskType } from '@/api/types';
import { Badge } from '@/components/ui/badge';

/** Presentation for template cards, keyed by backend template id (GET /templates). */
export interface TemplateMeta {
  icon: typeof ImageIcon;
  tint: string;
  examples: string[];
}

const FALLBACK: TemplateMeta = { icon: Sparkles, tint: 'text-slate-600 bg-slate-500/12 dark:text-slate-300', examples: [] };

export const TEMPLATE_META: Record<string, TemplateMeta> = {
  image_classification: { icon: ImageIcon, tint: 'text-violet-600 bg-violet-500/12 dark:text-violet-300', examples: ['Defect / OK', 'Plant disease', 'Product type'] },
  object_detection: { icon: Box, tint: 'text-sky-600 bg-sky-500/12 dark:text-sky-300', examples: ['Helmets & vests', 'Shelf stock', 'Vehicles'] },
  text_classification: { icon: FileText, tint: 'text-emerald-600 bg-emerald-500/12 dark:text-emerald-300', examples: ['Ticket routing', 'Review sentiment', 'Hinglish'] },
  tabular: { icon: Table2, tint: 'text-amber-600 bg-amber-500/14 dark:text-amber-300', examples: ['Churn', 'Loan default', 'House price'] },
  audio_classification: { icon: AudioLines, tint: 'text-rose-600 bg-rose-500/12 dark:text-rose-300', examples: ['Machine faults', 'Baby cry'] },
  keyword_spotting: { icon: Mic, tint: 'text-pink-600 bg-pink-500/12 dark:text-pink-300', examples: ['Wake words', 'Kiosk commands'] },
  speaker_recognition: { icon: Mic, tint: 'text-fuchsia-600 bg-fuchsia-500/12 dark:text-fuchsia-300', examples: ['Who is speaking', 'Voice check-in'] },
  face_recognition: { icon: ScanFace, tint: 'text-slate-600 bg-slate-500/12 dark:text-slate-300', examples: ['Attendance', 'Check-in'] },
  named_entity_extraction: { icon: Tags, tint: 'text-teal-600 bg-teal-500/12 dark:text-teal-300', examples: ['Invoice fields', 'PII'] },
  aspect_sentiment: { icon: FileText, tint: 'text-lime-700 bg-lime-500/14 dark:text-lime-300', examples: ['Battery good, camera bad'] },
  forecasting: { icon: LineChart, tint: 'text-cyan-600 bg-cyan-500/12 dark:text-cyan-300', examples: ['Demand', 'Energy load'] },
  anomaly_detection: { icon: TrendingDown, tint: 'text-orange-600 bg-orange-500/12 dark:text-orange-300', examples: ['Transactions', 'Sensor drift'] },
  segmentation: { icon: Shapes, tint: 'text-fuchsia-600 bg-fuchsia-500/12 dark:text-fuchsia-300', examples: ['Road damage', 'Crop vs weed'] },
};

export const templateMeta = (id: string) => TEMPLATE_META[id] ?? FALLBACK;
/** Template id a task type belongs to (both tabular task types share the "tabular" template). */
export const templateIdOf = (t: TaskType) => (t.startsWith('tabular') ? 'tabular' : t);
export const metaOf = (t: TaskType) => templateMeta(templateIdOf(t));

export const TASK_LABEL: Record<TaskType, string> = {
  image_classification: 'Image classification',
  object_detection: 'Object detection',
  text_classification: 'Text classification',
  tabular_classification: 'Tabular classification',
  tabular_regression: 'Tabular regression',
  face_recognition: 'Face recognition',
  speaker_recognition: 'Speaker recognition',
};

/** Pipeline order (Deleting is transient and not a stage). */
export const STATUSES: ProjectStatus[] = ['Draft', 'Data needed', 'Validating', 'Ready to train', 'Training', 'Trained', 'Deployed'];

const STATUS_STYLE: Record<ProjectStatus, { variant: 'secondary' | 'warning' | 'info' | 'default' | 'success' | 'destructive'; pulse?: boolean }> = {
  Draft: { variant: 'secondary' },
  'Data needed': { variant: 'warning' },
  Validating: { variant: 'info', pulse: true },
  'Ready to train': { variant: 'default' },
  Training: { variant: 'info', pulse: true },
  Trained: { variant: 'success' },
  Deployed: { variant: 'success' },
  Ready: { variant: 'success' }, // face projects: gallery usable (no training)
  Deleting: { variant: 'destructive', pulse: true },
};

export function StatusBadge({ status }: { status: ProjectStatus }) {
  const s = STATUS_STYLE[status] ?? { variant: 'secondary' as const };
  return (
    <Badge variant={s.variant}>
      <span className={`size-1.5 rounded-full bg-current ${s.pulse ? 'animate-pulse' : ''}`} aria-hidden />
      {status}
    </Badge>
  );
}
