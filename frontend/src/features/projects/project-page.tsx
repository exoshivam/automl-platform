import { Link } from '@tanstack/react-router';
import { CheckCircle2, Circle, Database, FlaskConical, Gauge, Rocket, ShieldCheck, Tags } from 'lucide-react';
import { useProject } from '@/api/hooks';
import { StorageSummary } from '@/features/data/versions-panel';
import { PurgeProjectDialog } from './purge-dialog';
import { FaceOverview } from '@/features/face/face-overview';
import { isPretrained } from '@/api/detector';
import { ReadyOverview } from './ready-detector';
import { isQuickText } from '@/api/text';
import { TextOverview } from '@/features/text/text-pages';
import { SpeakerOverview } from '@/features/speaker/speaker-pages';
import { Button } from '@/components/ui/button';
import { Trash2 } from 'lucide-react';
import { useState } from 'react';
import type { ProjectStatus } from '@/api/types';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/misc';
import { cn } from '@/lib/utils';
import { STATUSES } from './meta';
import { METRIC_LABEL } from './spec-schema';

const ORDER: ProjectStatus[] = STATUSES;

/** The project's pipeline: each stage lights up as the status moves forward (PRD §5 flow). */
const STAGES = [
  { key: 'data', label: 'Data', desc: 'Upload or capture', icon: Database, doneAt: 'Validating', to: '/projects/$projectId/data' },
  { key: 'label', label: 'Label', desc: 'Classes and boxes', icon: Tags, doneAt: 'Validating', to: '/projects/$projectId/label' },
  { key: 'validate', label: 'Validate', desc: 'Fix data issues', icon: ShieldCheck, doneAt: 'Ready to train', to: '/projects/$projectId/dataset' },
  { key: 'train', label: 'Train', desc: 'Budget and live progress', icon: FlaskConical, doneAt: 'Trained', to: '/projects/$projectId/train' },
  { key: 'evaluate', label: 'Evaluate', desc: 'Metrics and playground', icon: Gauge, doneAt: 'Trained', to: '/projects/$projectId/evaluate' },
  { key: 'deploy', label: 'Export & deploy', desc: 'ONNX files', icon: Rocket, doneAt: 'Deployed', to: '/projects/$projectId/export' },
] as const satisfies readonly { key: string; label: string; desc: string; icon: unknown; doneAt: ProjectStatus; to?: string }[];

export function ProjectPage({ projectId }: { projectId: string }) {
  const { data: p } = useProject(projectId);
  const [purge, setPurge] = useState(false);
  if (!p) return <Skeleton className="h-96 rounded-xl" />;
  const at = ORDER.indexOf(p.status);
  const spec = p.current_spec;
  const danger = (
    <>
      <Card className="border-destructive/30">
        <CardHeader className="flex-row flex-wrap items-center justify-between gap-3">
          <div>
            <CardTitle className="text-base">Danger zone</CardTitle>
            <CardDescription>Permanently delete this project and everything in it{p.task_type === 'face_recognition' ? ', including all face photos, face data and consent records' : p.task_type === 'speaker_recognition' ? ', including all voice clips, voice data and consent records' : ': all photos, labels, versions, models and exports'}.</CardDescription>
          </div>
          <Button variant="destructive" size="sm" onClick={() => setPurge(true)}>
            <Trash2 /> Delete project permanently
          </Button>
        </CardHeader>
      </Card>
      {purge && <PurgeProjectDialog project={p} onClose={() => setPurge(false)} />}
    </>
  );
  if (p.task_type === 'speaker_recognition')
    return (
      <div className="grid gap-6">
        <SpeakerOverview projectId={projectId} />
        {danger}
      </div>
    );
  if (isQuickText(p))
    return (
      <div className="grid gap-6">
        <TextOverview projectId={projectId} />
        {danger}
      </div>
    );
  if (isPretrained(p))
    return (
      <div className="grid gap-6">
        <ReadyOverview projectId={projectId} />
        {danger}
      </div>
    );
  if (p.task_type === 'face_recognition')
    return (
      <div className="grid gap-6">
        <FaceOverview projectId={projectId} />
        {danger}
      </div>
    );

  return (
    <div className="grid gap-6">
      <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6" aria-label="Pipeline">
        {STAGES.map(s => {
          const done = at >= ORDER.indexOf(s.doneAt);
          return (
            <li key={s.key}>
              <Card className={cn('h-full p-4', done && 'border-success/40')}>
                <div className="flex items-center justify-between">
                  <s.icon className="size-5 text-muted-foreground" />
                  {done ? <CheckCircle2 className="size-4 text-success" aria-label="done" /> : <Circle className="size-4 text-muted-foreground/50" aria-label="to do" />}
                </div>
                <div className="mt-3 font-medium">{s.label}</div>
                <p className="text-xs text-muted-foreground">{s.desc}</p>
                {'to' in s ? (
                  <Link to={s.to} params={{ projectId }} className="mt-3 inline-block text-xs font-medium text-primary hover:underline">
                    Open →
                  </Link>
                ) : (
                  <Badge variant="secondary" className="mt-3">
                    In progress
                  </Badge>
                )}
              </Card>
            </li>
          );
        })}
      </ol>

      <Card>
        <CardHeader>
          <CardTitle>Task spec v{p.spec_version}</CardTitle>
          <CardDescription>What the training search optimises and where the model must run.</CardDescription>
        </CardHeader>
        <CardContent>
          <dl className="grid gap-x-6 gap-y-4 text-sm sm:grid-cols-2 lg:grid-cols-4">
            <div>
              <dt className="text-muted-foreground">Classes</dt>
              <dd className="font-medium">{spec.labels?.length ? spec.labels.join(', ') : 'From target column'}</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Platforms</dt>
              <dd className="font-medium capitalize">{spec.constraints.target_platforms.join(', ')}</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Primary metric</dt>
              <dd className="font-medium">{METRIC_LABEL[spec.primary_metric] ?? spec.primary_metric}</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Budget</dt>
              <dd className="font-medium capitalize">{spec.budget_mode}</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Max size / latency</dt>
              <dd className="font-medium">
                {spec.constraints.max_model_size_mb ? `${spec.constraints.max_model_size_mb} MB` : 'Any'} / {spec.constraints.max_latency_ms ? `${spec.constraints.max_latency_ms} ms` : 'Any'}
              </dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Split</dt>
              <dd className="font-medium tabular-nums">
                {Math.round(spec.split.train * 100)}/{Math.round(spec.split.val * 100)}/{Math.round(spec.split.test * 100)}
              </dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Samples</dt>
              <dd className="font-medium tabular-nums">{p.sample_count.toLocaleString()}</dd>
            </div>
          </dl>
        </CardContent>
      </Card>
      <StorageSummary projectId={projectId} />
      {danger}
    </div>
  );
}
