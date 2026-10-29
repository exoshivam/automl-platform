import { Link } from '@tanstack/react-router';
import { CheckCircle2, Circle, ShieldCheck, SlidersHorizontal, Users, Wand2 } from 'lucide-react';
import { useFaceSettings, usePeople } from '@/api/face';
import { useCurrentOrg, useProject } from '@/api/hooks';
import { canExportFace } from './face-export';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/utils';

/** Face projects: enrol people → tune threshold → try it. No dataset versions or training (PRD §12); export lives on its own tab. */
export function FaceOverview({ projectId }: { projectId: string }) {
  const s = useFaceSettings(projectId).data;
  const project = useProject(projectId).data;
  const { orgs } = useCurrentOrg();
  const role = orgs.find(o => o.id === project?.organization_id)?.role;
  const people = usePeople(projectId).data ?? [];
  const enrolled = people.filter(p => p.photo_count >= 3).length;
  const steps = [
    { label: 'Add people', desc: s ? `${s.people_count} people · ${s.photo_count} photos` : 'With consent, 3–10 photos each', icon: Users, done: enrolled > 0, to: '/projects/$projectId/people' as const },
    { label: 'Tune threshold', desc: s ? `Currently ${s.threshold.toFixed(2)}` : 'Strict vs lenient matching', icon: SlidersHorizontal, done: !!s && s.threshold !== s.default_threshold, to: '/projects/$projectId/settings' as const },
    { label: 'Try it', desc: 'Photo or live camera', icon: Wand2, done: false, to: '/projects/$projectId/playground' as const },
  ];
  return (
    <div className="grid gap-6">
      <ol className="grid gap-3 sm:grid-cols-3" aria-label="Steps">
        {steps.map(x => (
          <li key={x.label}>
            <Card className={cn('h-full p-4', x.done && 'border-success/40')}>
              <div className="flex items-center justify-between">
                <x.icon className="size-5 text-muted-foreground" />
                {x.done ? <CheckCircle2 className="size-4 text-success" aria-label="done" /> : <Circle className="size-4 text-muted-foreground/50" aria-label="to do" />}
              </div>
              <div className="mt-3 font-medium">{x.label}</div>
              <p className="text-xs text-muted-foreground">{x.desc}</p>
              <Link to={x.to} params={{ projectId }} className="mt-3 inline-block text-xs font-medium text-primary hover:underline">Open →</Link>
            </Card>
          </li>
        ))}
      </ol>
      {canExportFace(role) && (
        <p className="text-sm text-muted-foreground">
          To recognise people offline on your own computer, see <Link to="/projects/$projectId/export" params={{ projectId }} className="font-medium text-primary hover:underline">Export →</Link>
        </p>
      )}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base"><ShieldCheck className="size-5" /> Using face recognition responsibly</CardTitle>
          <CardDescription>This is a gallery of people who agreed to be recognised, not a trained classifier.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-1.5 text-sm text-muted-foreground">
          <p>• Only add people who consented. Each enrolment records who confirmed it and when.</p>
          <p>• Removing a person deletes all their photos and face data immediately.</p>
          <p>• Not for covert surveillance, or as the only basis for decisions about someone without human review.</p>
          {s && <p>• Face model licence: <span className="font-medium text-foreground">{s.embedding_model.licence}</span>.</p>}
        </CardContent>
      </Card>
    </div>
  );
}
