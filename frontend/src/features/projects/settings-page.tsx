import { PhraseCard } from '@/features/speaker/speaker-pages';
import { hasSentiment, isQuickText } from '@/api/text';
import { TextSettingsPage } from '@/features/text/text-pages';
import { isPretrained } from '@/api/detector';
import { useProject } from '@/api/hooks';
import { Card } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/misc';
import { FaceSettingsPage } from '@/features/face/face-settings-page';
import { ReadySettingsPage } from './ready-detector';

/** /settings: face threshold or ready-made detector threshold, by project kind. */
export function SettingsPage({ projectId }: { projectId: string }) {
  const { data: p } = useProject(projectId);
  if (!p) return <Skeleton className="h-80 rounded-xl" />;
  if (p.task_type === 'face_recognition') return <FaceSettingsPage projectId={projectId} />;
  if (p.task_type === 'speaker_recognition')
    return (
      <div className="grid gap-5">
        <PhraseCard projectId={projectId} />
        <FaceSettingsPage projectId={projectId} />
      </div>
    );
  if (isPretrained(p)) return <ReadySettingsPage projectId={projectId} />;
  if (isQuickText(p) || hasSentiment(p)) return <TextSettingsPage projectId={projectId} />;
  return <Card className="p-6 text-sm text-muted-foreground">This project has no settings page.</Card>;
}
