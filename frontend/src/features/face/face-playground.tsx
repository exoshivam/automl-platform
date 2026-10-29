import { UserRound } from 'lucide-react';
import { usePeople, type Identified } from '@/api/face';
import { BoxPlayground } from '@/components/box-playground';
import { DemoBadge } from '@/components/ui/misc';
import { api } from '@/lib/api';

/** Identify every face: name + score, or "Other" below the threshold; top-3 candidates on hover. */
export function FacePlayground({ projectId }: { projectId: string }) {
  const people = usePeople(projectId);
  return (
    <BoxPlayground
      title="Who is this?"
      badge={<DemoBadge group="face" />}
      description="Upload a photo or use the camera. Every face gets a box; unknown faces show as “Other”."
      names={(people.data ?? []).map(p => p.name)}
      emptyText="No face found in this image."
      icon={UserRound}
      detect={async (blob, signal) => {
        const fd = new FormData();
        fd.append('file', blob, 'frame.jpg');
        const r = await api<Identified>(`/projects/${projectId}/identify`, { method: 'POST', body: fd, signal });
        return { ...r, items: r.faces.map(f => ({ box: f.box, label: f.name, score: f.score, top3: f.top3 })) };
      }}
    />
  );
}
