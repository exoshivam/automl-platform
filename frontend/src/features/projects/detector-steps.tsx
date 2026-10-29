import { useNavigate } from '@tanstack/react-router';
import { ArrowLeft, ArrowRight, Check, GraduationCap, Loader2, Search, Zap } from 'lucide-react';
import { useMemo, useState } from 'react';
import { toast } from 'sonner';
import { usePretrainedClasses } from '@/api/detector';
import { useCreateProject, useCurrentOrg } from '@/api/hooks';
import type { TaskSpec } from '@/api/types';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Skeleton } from '@/components/ui/misc';
import { ApiError } from '@/lib/api';
import { cn } from '@/lib/utils';
import type { Draft } from './new-project-page';

/** The user's explicit choice: ready-made COCO detector (instant) or train your own. */
export function DetectionModeStep({ onBack, onPick }: { onBack: () => void; onPick: (m: 'pretrained' | 'custom') => void }) {
  const opt = (m: 'pretrained' | 'custom', Icon: typeof Zap, title: string, body: string, tags: string[]) => (
    <button
      type="button"
      onClick={() => onPick(m)}
      className="flex cursor-pointer flex-col gap-3 rounded-xl border bg-card p-5 text-left shadow-sm transition-all hover:-translate-y-0.5 hover:border-primary/50 hover:shadow-md focus-visible:ring-2 focus-visible:ring-ring"
    >
      <span className="grid size-11 place-items-center rounded-lg bg-primary/12 text-primary"><Icon className="size-5" /></span>
      <div>
        <div className="text-base font-semibold">{title}</div>
        <p className="text-sm text-muted-foreground">{body}</p>
      </div>
      <div className="mt-auto flex flex-wrap gap-1">{tags.map(t => <span key={t} className="rounded bg-muted px-1.5 py-0.5 text-[11px] text-muted-foreground">{t}</span>)}</div>
    </button>
  );
  return (
    <div className="grid gap-5">
      <div>
        <h2 className="text-lg font-semibold">Which detector do you want?</h2>
        <p className="text-sm text-muted-foreground">You can start ready-made and switch to training your own later; your classes are kept.</p>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        {opt('pretrained', Zap, 'Ready-made', 'Detect common objects instantly (80 everyday things like person, phone, bottle, cup, laptop, car, dog). No photos or training needed.', ['Instant', 'No data needed', '80 classes'])}
        {opt('custom', GraduationCap, 'Train your own', 'For anything else (your products, parts, defects). Needs labelled photos and a short training.', ['Any objects', 'Your photos', 'A few minutes of training'])}
      </div>
      <div><Button variant="ghost" onClick={onBack}><ArrowLeft /> Back</Button></div>
    </div>
  );
}

/** Name + searchable multi-select over the 80 COCO classes. Unknown term → offer "train your own" with it pre-filled. */
export function PretrainedStep({ draft, onBack, onTrainOwn }: { draft: Draft; onBack: () => void; onTrainOwn: (name: string, labels: string[]) => void }) {
  const classes = usePretrainedClasses();
  const { org } = useCurrentOrg();
  const create = useCreateProject(org?.id);
  const navigate = useNavigate();
  const [name, setName] = useState(draft.name);
  const [q, setQ] = useState('');
  const [picked, setPicked] = useState<string[]>([]);
  const all = useMemo(() => (classes.data?.classes ?? []).map(c => c.name), [classes.data]);
  const term = q.trim().toLowerCase();
  const shown = term ? all.filter(c => c.toLowerCase().includes(term)) : all;
  const toggle = (c: string) => setPicked(p => (p.includes(c) ? p.filter(x => x !== c) : [...p, c]));

  const submit = () =>
    create.mutate(
      { name: name.trim(), spec: { ...draft.spec, task_type: 'object_detection', mode: 'pretrained', labels: picked, score_threshold: 0.5 } as TaskSpec },
      {
        onSuccess: p => (toast.success(`"${p.name}" is ready: try it in the Playground`), navigate({ to: '/projects/$projectId/playground', params: { projectId: p.id } })),
        onError: e => toast.error(e instanceof ApiError && e.problem.code === 'unknown_pretrained_class' ? `Not in the ready-made list: ${((e.problem as { unknown?: string[] }).unknown ?? []).join(', ')}` : e.message),
      },
    );

  return (
    <Card className="grid gap-5 p-6">
      <div>
        <h2 className="text-lg font-semibold">Ready-made detector</h2>
        <p className="text-sm text-muted-foreground">Pick what it should look for. It works straight away; there's nothing to train.</p>
      </div>
      <label className="grid gap-1.5 text-sm">
        <span className="font-medium">Project name</span>
        <Input value={name} onChange={e => setName(e.target.value)} placeholder="e.g. Desk objects" maxLength={200} />
      </label>
      <div className="grid gap-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="text-sm font-medium">Objects to detect <span className="text-muted-foreground">({picked.length} of {all.length})</span></span>
          <div className="flex gap-1">
            <Button size="sm" variant="ghost" onClick={() => setPicked(all)} disabled={!all.length}>Select all</Button>
            <Button size="sm" variant="ghost" onClick={() => setPicked([])} disabled={!picked.length}>Clear</Button>
          </div>
        </div>
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-2.5 size-4 text-muted-foreground" />
          <Input value={q} onChange={e => setQ(e.target.value)} placeholder="Search: person, phone, cup…" className="pl-8" aria-label="Search objects" />
        </div>
        {classes.isLoading ? <Skeleton className="h-40" /> : term && !shown.length ? (
          <div className="flex flex-wrap items-center gap-3 rounded-lg border border-warning/40 bg-warning/8 p-3 text-sm" role="status">
            <span>“{q.trim()}” isn't in the ready-made list. You'll need to train your own detector for it.</span>
            <Button size="sm" variant="outline" onClick={() => onTrainOwn(name.trim(), [...picked, q.trim()])}>
              Train my own for “{q.trim()}” <ArrowRight />
            </Button>
          </div>
        ) : (
          <div className="flex max-h-64 flex-wrap gap-1.5 overflow-y-auto rounded-lg border p-2" role="group" aria-label="Ready-made objects">
            {shown.map(c => {
              const on = picked.includes(c);
              return (
                <button key={c} type="button" aria-pressed={on} onClick={() => toggle(c)}
                  className={cn('inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs transition-colors', on ? 'border-primary bg-primary/12 text-primary' : 'hover:bg-accent')}>
                  {on && <Check className="size-3" />}{c}
                </button>
              );
            })}
          </div>
        )}
        {picked.length > 0 && <div className="flex flex-wrap gap-1">{picked.slice(0, 12).map(c => <Badge key={c} variant="secondary">{c}</Badge>)}{picked.length > 12 && <Badge variant="outline">+{picked.length - 12}</Badge>}</div>}
        {classes.data && <p className="text-[11px] text-muted-foreground">Model: {classes.data.model.name} · {classes.data.model.licence}</p>}
      </div>
      <div className="flex justify-between">
        <Button variant="ghost" onClick={onBack}><ArrowLeft /> Back</Button>
        <Button disabled={!name.trim() || !picked.length || create.isPending} onClick={submit}>
          {create.isPending ? <Loader2 className="animate-spin" /> : <Zap />} Create ready-made detector
        </Button>
      </div>
    </Card>
  );
}
