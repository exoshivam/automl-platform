import { useNavigate } from '@tanstack/react-router';
import { ArrowLeft, GraduationCap, Loader2, MessageSquareText, Smile, Zap } from 'lucide-react';
import { useState } from 'react';
import { toast } from 'sonner';
import { useCreateProject, useCurrentOrg } from '@/api/hooks';
import type { TaskSpec } from '@/api/types';
import { TagInput } from '@/components/tag-input';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Checkbox } from '@/components/ui/misc';
import type { Draft } from './new-project-page';

export type TextMode = 'sentiment' | 'quick' | 'custom';

/** Text: ready-made sentiment, quick labels from a few examples, or train your own (existing flow). */
export function TextModeStep({ onBack, onPick }: { onBack: () => void; onPick: (m: TextMode, withSentiment: boolean) => void }) {
  const [sentiment, setSentiment] = useState(true);
  const opt = (m: TextMode, Icon: typeof Zap, title: string, body: string, tags: string[]) => (
    <button
      type="button"
      onClick={() => onPick(m, m === 'sentiment' || sentiment)}
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
        <h2 className="text-lg font-semibold">How do you want to sort your texts?</h2>
        <p className="text-sm text-muted-foreground">Start quick and train later on the same examples; nothing is lost.</p>
      </div>
      <div className="grid gap-3 md:grid-cols-3">
        {opt('sentiment', Smile, 'Ready-made sentiment', 'Positive / negative / neutral instantly in many languages including Hindi (Hinglish is weaker). No data needed.', ['Instant', 'No data', 'Non-commercial licence'])}
        {opt('quick', MessageSquareText, 'Your own labels from a few examples', 'Name your labels and add 5–10 example texts each. Works instantly, no training (e.g. refund / delivery late / praise).', ['Instant', '5–10 examples per label'])}
        {opt('custom', GraduationCap, 'Train your own', 'For lots of labelled data. Upload a CSV and train a model (best accuracy).', ['Hundreds of examples', 'Training'])}
      </div>
      <label className="flex items-center gap-2 text-sm">
        <Checkbox checked={sentiment} onCheckedChange={v => setSentiment(v === true)} />
        Also show sentiment (positive / negative / neutral) next to your labels
      </label>
      <div><Button variant="ghost" onClick={onBack}><ArrowLeft /> Back</Button></div>
    </div>
  );
}

/** Ready-made sentiment (labels []) or quick labels (≥2), created straight away: "Ready" with no training. */
export function QuickTextStep({ draft, mode, withSentiment, onBack }: { draft: Draft; mode: 'sentiment' | 'quick'; withSentiment: boolean; onBack: () => void }) {
  const { org } = useCurrentOrg();
  const create = useCreateProject(org?.id);
  const navigate = useNavigate();
  const [name, setName] = useState(draft.name);
  const [labels, setLabels] = useState<string[]>(draft.labels.length ? draft.labels : []);
  const quick = mode === 'quick';
  const ok = name.trim() && (!quick || labels.length >= 2);
  const submit = () =>
    create.mutate(
      { name: name.trim(), spec: { ...draft.spec, task_type: 'text_classification', mode: 'quick', labels: quick ? labels : [], ready_made: withSentiment ? ['sentiment'] : [] } as unknown as TaskSpec },
      {
        onSuccess: p => (toast.success(`"${p.name}" is ready`), navigate({ to: quick ? '/projects/$projectId/examples' : '/projects/$projectId/playground', params: { projectId: p.id } })),
        onError: e => toast.error(e.message),
      },
    );
  return (
    <Card className="grid gap-5 p-6">
      <div>
        <h2 className="text-lg font-semibold">{quick ? 'Your own labels' : 'Ready-made sentiment'}</h2>
        <p className="text-sm text-muted-foreground">
          {quick ? 'Name the labels now; you’ll add 5–10 example texts for each next. It works right away and gets better with every example.' : 'Positive, negative or neutral for any text, in many languages including Hindi. Hinglish is weaker; for that, your own labels with examples work better. The sentiment model is licensed for non-commercial use.'}
        </p>
      </div>
      <label className="grid gap-1.5 text-sm">
        <span className="font-medium">Project name</span>
        <Input value={name} onChange={e => setName(e.target.value)} maxLength={200} placeholder={quick ? 'e.g. Support tickets' : 'e.g. Review sentiment'} />
      </label>
      {quick && (
        <div className="grid gap-1.5 text-sm">
          <span className="font-medium">Labels</span>
          <TagInput id="quick-labels" value={labels} onChange={setLabels} placeholder="refund, delivery late, praise" />
          <span className="text-xs text-muted-foreground">At least 2. Texts that match none of them show as “Other”.{withSentiment && ' Sentiment is shown too.'}</span>
        </div>
      )}
      <div className="flex justify-between">
        <Button variant="ghost" onClick={onBack}><ArrowLeft /> Back</Button>
        <Button disabled={!ok || create.isPending} onClick={submit}>{create.isPending ? <Loader2 className="animate-spin" /> : <Zap />} Create</Button>
      </div>
    </Card>
  );
}
