import { Link, Outlet } from '@tanstack/react-router';
import { ArrowLeft, Database, Rocket, FlaskConical, Gauge, LayoutDashboard, ListChecks, Package, Rows3, SlidersHorizontal, Tags, Users, Wand2 } from 'lucide-react';
import { useProject } from '@/api/hooks';
import { Card } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/misc';
import { cn, timeAgo } from '@/lib/utils';
import { metaOf, StatusBadge, TASK_LABEL } from './meta';
import { useFaceSettings } from '@/api/face';
import { useCurrentOrg } from '@/api/hooks';
import { canDeploy } from '@/features/deploy/deploy-page';
import { canExportFace } from '@/features/face/face-export';
import { customLabels, isHybrid, isPretrained } from '@/api/detector';
import { TRAIN_READY, isQuickText, isSentimentOnly, useExamples, useTextSettings } from '@/api/text';

const TABS = [
  { to: '/projects/$projectId', label: 'Overview', icon: LayoutDashboard, exact: true },
  { to: '/projects/$projectId/data', label: 'Add data', icon: Database },
  { to: '/projects/$projectId/dataset', label: 'Dataset', icon: Rows3 },
  { to: '/projects/$projectId/label', label: 'Label', icon: Tags },
  { to: '/projects/$projectId/train', label: 'Train', icon: FlaskConical },
  { to: '/projects/$projectId/evaluate', label: 'Evaluate', icon: Gauge },
  { to: '/projects/$projectId/playground', label: 'Playground', icon: Wand2 },
  { to: '/projects/$projectId/export', label: 'Export', icon: Package },
] as const;
/** Ready-made (COCO) detector: nothing to train; Settings = threshold + chosen classes. */
const READY_TABS = [
  { to: '/projects/$projectId', label: 'Overview', icon: LayoutDashboard, exact: true },
  { to: '/projects/$projectId/settings', label: 'Settings', icon: SlidersHorizontal },
  { to: '/projects/$projectId/playground', label: 'Playground', icon: Wand2 },
] as const;
/** Local self-serve API (key per deployment); face/speaker only for the workspace owner. */
const DEPLOY_TAB = { to: '/projects/$projectId/deploy', label: 'Deploy', icon: Rocket } as const;
/** Ready-made + the user's own objects: training tabs for the custom objects only (no Export: Phase 2). */
const HYBRID_TABS = [
  { to: '/projects/$projectId', label: 'Overview', icon: LayoutDashboard, exact: true },
  { to: '/projects/$projectId/settings', label: 'Settings', icon: SlidersHorizontal },
  { to: '/projects/$projectId/data', label: 'Add data', icon: Database },
  { to: '/projects/$projectId/dataset', label: 'Dataset', icon: Rows3 },
  { to: '/projects/$projectId/label', label: 'Label', icon: Tags },
  { to: '/projects/$projectId/train', label: 'Train', icon: FlaskConical },
  { to: '/projects/$projectId/evaluate', label: 'Evaluate', icon: Gauge },
  { to: '/projects/$projectId/playground', label: 'Playground', icon: Wand2 },
] as const;
/** Quick text labels: examples instead of a dataset; training tabs appear once there's enough data (or a model). */
const QUICK_TEXT_TABS = [
  { to: '/projects/$projectId', label: 'Overview', icon: LayoutDashboard, exact: true },
  { to: '/projects/$projectId/examples', label: 'Examples', icon: ListChecks },
  { to: '/projects/$projectId/settings', label: 'Settings', icon: SlidersHorizontal },
  { to: '/projects/$projectId/playground', label: 'Playground', icon: Wand2 },
] as const;
const QUICK_TEXT_TRAIN_TABS = [
  { to: '/projects/$projectId/dataset', label: 'Dataset', icon: Rows3 },
  { to: '/projects/$projectId/train', label: 'Train', icon: FlaskConical },
  { to: '/projects/$projectId/evaluate', label: 'Evaluate', icon: Gauge },
] as const;
const SENTIMENT_TABS = [
  { to: '/projects/$projectId', label: 'Overview', icon: LayoutDashboard, exact: true },
  { to: '/projects/$projectId/settings', label: 'Settings', icon: SlidersHorizontal },
  { to: '/projects/$projectId/playground', label: 'Playground', icon: Wand2 },
] as const;
/** Speaker recognition: the face layout with a mic. */
const SPEAKER_TABS = [
  { to: '/projects/$projectId', label: 'Overview', icon: LayoutDashboard, exact: true },
  { to: '/projects/$projectId/speakers', label: 'Speakers', icon: Users },
  { to: '/projects/$projectId/settings', label: 'Settings', icon: SlidersHorizontal },
  { to: '/projects/$projectId/playground', label: 'Playground', icon: Wand2 },
] as const;
/** Face recognition is a gallery, not a trained classifier (PRD §12): no data/label/train/evaluate/export steps. */
/** Same place and icon as Export on trained-model projects; owner/admin only (biometric data). */
const FACE_EXPORT_TAB = { to: '/projects/$projectId/export', label: 'Export', icon: Package } as const;
const FACE_TABS = [
  { to: '/projects/$projectId', label: 'Overview', icon: LayoutDashboard, exact: true },
  { to: '/projects/$projectId/people', label: 'People', icon: Users },
  { to: '/projects/$projectId/settings', label: 'Settings', icon: SlidersHorizontal },
  { to: '/projects/$projectId/playground', label: 'Playground', icon: Wand2 },
] as const;

/** Shared header + tabs for everything under /projects/$projectId. */
export function ProjectLayout({ projectId }: { projectId: string }) {
  const { data: p, isLoading, error } = useProject(projectId);
  const { org } = useCurrentOrg();
  // Quick text: show the training tabs once every label has ~20 examples, or a model already exists.
  const quickOn = isQuickText(p);
  const ts = useTextSettings(projectId, quickOn).data;
  const exs = useExamples(projectId, quickOn).data;
  const quickTrain = !!ts && (ts.has_trained_model || (ts.labels.length >= 2 && ts.labels.every(l => (exs ?? []).filter(e => e.label === l).length >= TRAIN_READY)));
  if (isLoading) return <Skeleton className="h-24 rounded-xl" />;
  if (error || !p) return <Card className="p-8 text-sm text-destructive">Project not found or you don't have access.</Card>;
  if (p.status === 'Deleting')
    return (
      <Card className="p-8 text-sm">
        <p className="font-medium">“{p.name}” is being deleted.</p>
        <Link to="/projects" className="mt-2 inline-block text-primary hover:underline">Back to projects</Link>
      </Card>
    );
  const m = metaOf(p.task_type);
  const labelTab = p.task_type.startsWith('tabular') ? null : 'label';
  return (
    <div className="grid gap-5">
      <div>
        <Link to="/projects" className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
          <ArrowLeft className="size-4" /> Projects
        </Link>
        <div className="mt-2 flex flex-wrap items-center gap-3">
          <span className={cn('grid size-10 shrink-0 place-items-center rounded-lg', m.tint)}>
            <m.icon className="size-5" />
          </span>
          <div className="min-w-0">
            <h1 className="truncate text-2xl font-semibold tracking-tight">{p.name}</h1>
            <p className="text-sm text-muted-foreground">
              {TASK_LABEL[p.task_type]} · {p.task_type === 'face_recognition' || p.task_type === 'speaker_recognition' ? <FaceCount projectId={p.id} fallback={p.sample_count} /> : isPretrained(p) ? `Ready-made · ${p.current_spec.labels?.length ?? 0} objects${customLabels(p).length ? ` + ${customLabels(p).length} custom` : ''}` : `${p.sample_count.toLocaleString()} samples`} · updated {timeAgo(p.updated_at)}
            </p>
          </div>
          <StatusBadge status={p.status} />
        </div>
      </div>
      <nav aria-label="Project" className="-mx-1 flex gap-1 overflow-x-auto border-b px-1">
        {[...(p.task_type === 'face_recognition' ? [...FACE_TABS, ...(canExportFace(org?.role) ? [FACE_EXPORT_TAB] : [])] : p.task_type === 'speaker_recognition' ? SPEAKER_TABS : isHybrid(p) ? HYBRID_TABS : isPretrained(p) ? READY_TABS : isSentimentOnly(p) ? SENTIMENT_TABS : isQuickText(p) ? [...QUICK_TEXT_TABS, ...(quickTrain ? QUICK_TEXT_TRAIN_TABS : [])] : TABS.filter(t => labelTab || t.label !== 'Label')), ...(canDeploy(p, org?.role) ? [DEPLOY_TAB] : [])].map(t => (
          <Link
            key={t.to}
            to={t.to}
            params={{ projectId }}
            activeOptions={{ exact: 'exact' in t }}
            className="-mb-px flex shrink-0 items-center gap-2 border-b-2 border-transparent px-3 py-2.5 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground [&.active]:border-primary [&.active]:text-primary"
          >
            <t.icon className="size-4" />
            {t.label}
          </Link>
        ))}
      </nav>
      <Outlet />
    </div>
  );
}

/** Face projects: "N people · M photos" (face settings are refreshed on every enrolment change). */
function FaceCount({ projectId, fallback }: { projectId: string; fallback: number }) {
  const s = useFaceSettings(projectId).data;
  return <>{s ? `${s.people_count} ${s.people_count === 1 ? 'person' : 'people'} · ${s.photo_count} ${s.kind === 'speaker' ? (s.photo_count === 1 ? 'clip' : 'clips') : s.photo_count === 1 ? 'photo' : 'photos'}` : `${fallback.toLocaleString()} photos`}</>;
}
