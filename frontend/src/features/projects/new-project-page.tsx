import { useState } from 'react';
import { Link, useNavigate } from '@tanstack/react-router';
import { AlertTriangle, ArrowLeft, ArrowRight, Info, Loader2, Sparkles, Wand2 } from 'lucide-react';
import { toast } from 'sonner';
import { useCreateProject, useCurrentOrg, useFromDescription, useTemplates } from '@/api/hooks';
import { useImportStarter, useStarters, type Starter } from '@/api/starters';
import { StarterCard, importError } from '@/features/data/starter-library';
import { DetectionModeStep, PretrainedStep } from './detector-steps';
import { QuickTextStep, TextModeStep, type TextMode } from './text-steps';
import type { FromDescriptionResponse, TaskSpec, TaskType, Template } from '@/api/types';
import { TagInput } from '@/components/tag-input';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input, Textarea } from '@/components/ui/input';
import { Field, Label } from '@/components/ui/label';
import { Checkbox, Skeleton } from '@/components/ui/misc';
import { ApiError, api } from '@/lib/api';
import { cn } from '@/lib/utils';
import { TASK_LABEL, templateIdOf, templateMeta } from './meta';
import { SpecEditor, useSpecForm } from './spec-editor';
import { NEEDS_LABELS, isBiometric } from './spec-schema';

type Step = 'category' | 'details' | 'spec' | 'describe' | 'detmode' | 'pretrained' | 'textmode' | 'quicktext';

export interface Draft {
  template: Template;
  taskType: TaskType;
  name: string;
  labels: string[];
  spec: TaskSpec;
  /** From the describe path: the draft must be created with its llm_log_id. */
  llmLogId: string | null;
  note: { status: FromDescriptionResponse['status']; message: string } | null;
  /** DAT-13: imported into the new project's draft right after creation. */
  starter?: Starter | null;
}

/** Sentiment is a labels preset of text classification (the backend has no separate task type). */
const SENTIMENT = ['positive', 'negative', 'neutral'];

/** Default spec for a task type: the template default, adjusted for tabular regression. */
function defaultSpec(t: Template, taskType: TaskType): TaskSpec {
  const spec = structuredClone(t.default_spec!);
  if (taskType === 'tabular_regression')
    return { ...spec, task_type: taskType, labels: [], primary_metric: 'rmse', secondary_metrics: ['mae', 'r2'], split: { ...spec.split, stratify: false } };
  return { ...spec, task_type: taskType };
}

const LABEL_EXAMPLES: Partial<Record<TaskType, string>> = {
  image_classification: 'e.g. healthy, early_blight, late_blight',
  object_detection: 'e.g. helmet, vest, no_helmet',
  text_classification: 'e.g. billing, delivery, refund',
};

export function NewProjectPage() {
  const templates = useTemplates();
  const [step, setStep] = useState<Step>('category');
  const [draft, setDraft] = useState<Draft | null>(null);
  const [textMode, setTextMode] = useState<{ mode: TextMode; withSentiment: boolean }>({ mode: 'quick', withSentiment: true });

  const templateById = (id: string) => templates.data?.find(x => x.id === id);

  const pickTemplate = (t: Template) => {
    const taskType = t.task_types[0] as TaskType;
    setDraft({ template: t, taskType, name: '', labels: [], spec: defaultSpec(t, taskType), llmLogId: null, note: null });
    setStep(t.id === 'object_detection' ? 'detmode' : t.id === 'text_classification' ? 'textmode' : 'details');
  };

  const pickStarter = (s: Starter) => {
    const t = templateById(templateIdOf(s.task_type));
    if (!t?.default_spec) return toast.error('That task type is not available in this build.');
    const base = defaultSpec(t, s.task_type);
    const spec = { ...base, labels: s.classes, ...(s.target ? { target_column: s.target } : {}) } as TaskSpec;
    setDraft({ template: t, taskType: s.task_type, name: s.name, labels: s.classes, spec, llmLogId: null, note: null, starter: s });
    setStep('details');
  };

  const steps: [Step, string][] = [
    ['category', 'Category'],
    ['details', 'Name & classes'],
    ['spec', 'Task spec'],
  ];
  const index = step === 'describe' ? 0 : ['detmode', 'pretrained', 'textmode', 'quicktext'].includes(step) ? 1 : steps.findIndex(([s]) => s === step);

  return (
    <div className="mx-auto grid max-w-5xl gap-6">
      <div>
        <Link to="/projects" className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
          <ArrowLeft className="size-4" /> Projects
        </Link>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight">New project</h1>
      </div>

      <ol className="flex items-center gap-2 text-sm" aria-label="Progress">
        {steps.map(([s, label], i) => (
          <li key={s} className="flex items-center gap-2" aria-current={i === index ? 'step' : undefined}>
            <span
              className={cn(
                'grid size-6 place-items-center rounded-full border text-xs font-semibold',
                i < index && 'border-primary bg-primary text-primary-foreground',
                i === index && 'border-primary text-primary',
                i > index && 'text-muted-foreground',
              )}
            >
              {i + 1}
            </span>
            <span className={cn('hidden sm:inline', i === index ? 'font-medium' : 'text-muted-foreground')}>{label}</span>
            {i < steps.length - 1 && <span className="mx-1 h-px w-6 bg-border sm:w-10" aria-hidden />}
          </li>
        ))}
      </ol>

      {step === 'category' && <CategoryStep loading={templates.isLoading} templates={templates.data} onPick={pickTemplate} onDescribe={() => setStep('describe')} onStarter={pickStarter} />}
      {step === 'describe' && (
        <DescribeStep
          onBack={() => setStep('category')}
          onDraft={(r, spec, name) => {
            const template = templateById(templateIdOf(spec.task_type));
            if (!template) return;
            setDraft({ template, taskType: spec.task_type, name, labels: spec.labels ?? [], spec, llmLogId: r.llm_log_id, note: { status: r.status, message: r.message } });
            setStep('details');
          }}
        />
      )}
      {step === 'details' && draft && (
        <DetailsStep
          draft={draft}
          onBack={() => setStep('category')}
          onNext={d => {
            setDraft(d);
            setStep('spec');
          }}
        />
      )}
      {step === 'spec' && draft && <SpecStep draft={draft} onBack={() => setStep('details')} />}
      {step === 'textmode' && draft && (
        <TextModeStep
          onBack={() => setStep('category')}
          onPick={(m, withSentiment) => {
            setTextMode({ mode: m, withSentiment });
            if (m === 'custom') {
              setDraft({ ...draft, spec: { ...draft.spec, ready_made: withSentiment ? ['sentiment'] : [] } as TaskSpec });
              setStep('details');
            } else setStep('quicktext');
          }}
        />
      )}
      {step === 'quicktext' && draft && textMode.mode !== 'custom' && (
        <QuickTextStep draft={draft} mode={textMode.mode} withSentiment={textMode.withSentiment} onBack={() => setStep('textmode')} />
      )}
      {step === 'detmode' && draft && <DetectionModeStep onBack={() => setStep('category')} onPick={m => setStep(m === 'pretrained' ? 'pretrained' : 'details')} />}
      {step === 'pretrained' && draft && (
        <PretrainedStep
          draft={draft}
          onBack={() => setStep('detmode')}
          onTrainOwn={(name, labels) => (setDraft({ ...draft, name, labels }), setStep('details'))}
        />
      )}
    </div>
  );
}

function CategoryStep({ loading, templates, onPick, onDescribe, onStarter }: { loading: boolean; templates?: Template[]; onPick: (t: Template) => void; onDescribe: () => void; onStarter: (s: Starter) => void }) {
  // Enabled templates first, then "Coming soon" (enabled:false) ones.
  const cards = [...(templates ?? [])].sort((x, y) => Number(y.enabled) - Number(x.enabled));
  return (
    <div className="grid gap-6">
      <div>
        <h2 className="text-lg font-semibold">What do you want your model to do?</h2>
        <p className="text-sm text-muted-foreground">Pick a category. You can fine-tune every setting afterwards.</p>
      </div>
      {loading || !templates ? (
        <SkeletonCards />
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {cards.map(t => {
            const m = templateMeta(t.id);
            const Icon = m.icon;
            const disabled = !t.enabled || !t.default_spec;
            return (
              <button
                key={t.id}
                type="button"
                disabled={disabled}
                onClick={() => onPick(t)}
                className={cn(
                  'group relative flex cursor-pointer flex-col gap-3 rounded-xl border bg-card p-5 text-left shadow-sm transition-all focus-visible:ring-2 focus-visible:ring-ring',
                  disabled ? 'cursor-not-allowed opacity-60' : 'hover:-translate-y-0.5 hover:border-primary/50 hover:shadow-md',
                )}
                aria-describedby={`cat-${t.id}-desc`}
              >
                <div className="flex items-start justify-between">
                  <span className={cn('grid size-11 place-items-center rounded-lg', m.tint)}>
                    <Icon className="size-5" />
                  </span>
                  {!t.enabled && <Badge variant="secondary">Coming soon</Badge>}
                </div>
                <div>
                  <div className="font-semibold">{t.name}</div>
                  <p id={`cat-${t.id}-desc`} className="text-sm text-muted-foreground">
                    {t.enabled ? t.description : 'Planned for a later phase.'}
                  </p>
                </div>
                {m.examples.length > 0 && (
                  <div className="mt-auto flex flex-wrap gap-1">
                    {m.examples.map(e => (
                      <span key={e} className="rounded bg-muted px-1.5 py-0.5 text-[11px] text-muted-foreground">
                        {e}
                      </span>
                    ))}
                  </div>
                )}
              </button>
            );
          })}
        </div>
      )}
      <Card className="flex flex-col items-start gap-3 border-dashed bg-gradient-to-r from-primary/6 to-transparent p-5 sm:flex-row sm:items-center">
        <span className="grid size-10 shrink-0 place-items-center rounded-lg bg-primary/12 text-primary">
          <Sparkles className="size-5" />
        </span>
        <div className="flex-1">
          <div className="font-medium">Not sure? Describe your task.</div>
          <p className="text-sm text-muted-foreground">Write what you need in plain English. We'll pick the category and draft the settings for you to review.</p>
        </div>
        <Button variant="outline" onClick={onDescribe}>
          <Wand2 /> Describe it
        </Button>
      </Card>
      <StarterPicker enabled={new Set((templates ?? []).filter(t => t.enabled).map(t => t.id))} onPick={onStarter} />
    </div>
  );
}

/** "Start from a starter dataset": pre-fills name + classes; the data is imported after the project is created. */
function StarterPicker({ enabled, onPick }: { enabled: Set<string>; onPick: (s: Starter) => void }) {
  const starters = useStarters(undefined);
  const list = (starters.data ?? []).filter(s => enabled.has(templateIdOf(s.task_type)));
  if (!starters.isLoading && !list.length) return null;
  return (
    <section className="grid gap-3" aria-labelledby="starters-h">
      <div>
        <h3 id="starters-h" className="font-semibold">Or start from a starter dataset</h3>
        <p className="text-sm text-muted-foreground">Ready-made public data to try the whole flow in minutes. You can add your own data later.</p>
      </div>
      {starters.isLoading ? <SkeletonCards /> : (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {list.map(s => <StarterCard key={s.id} s={s} onUse={() => onPick(s)} cta="Start with this" />)}
        </div>
      )}
    </section>
  );
}

const EXAMPLES = [
  'Detect whether workers are wearing helmets and vests in CCTV photos. It must run on an Android phone offline.',
  'Sort customer reviews in Hinglish into positive, negative and neutral.',
  'Predict which customers will churn next month from our CSV export.',
];

function DescribeStep({ onBack, onDraft }: { onBack: () => void; onDraft: (r: FromDescriptionResponse, spec: TaskSpec, name: string) => void }) {
  const { org } = useCurrentOrg();
  const describe = useFromDescription(org?.id);
  const [text, setText] = useState('');
  const err = describe.error instanceof ApiError ? describe.error : null;
  // 503 llm_unavailable and 422 llm_invalid_output both carry fallback:"category".
  const fallback = !!err && ((err.problem as { fallback?: string }).fallback === 'category' || err.code === 'llm_unavailable' || err.code === 'llm_invalid_output');
  const fieldErr = err && !fallback ? err.fieldErrors().description : undefined;
  const result = describe.data;
  const suggestedName = text.split(/[.!?]/)[0]!.split(/\s+/).slice(0, 6).join(' ');

  return (
    <Card className="grid gap-5 p-6">
      <div>
        <h2 className="text-lg font-semibold">Describe your task</h2>
        <p className="text-sm text-muted-foreground">Mention what goes in, what should come out, and where it must run. Only this text is sent to the AI, never your data.</p>
      </div>
      <form
        className="grid gap-4"
        onSubmit={e => {
          e.preventDefault();
          describe.mutate(text, {
            onSuccess: r => {
              // ok (and ambiguous with a draft) goes straight to review; unsupported stays here with the explanation.
              if (r.draft_spec && r.status !== 'unsupported') onDraft(r, r.draft_spec, suggestedName.charAt(0).toUpperCase() + suggestedName.slice(1));
            },
          });
        }}
      >
        <Field id="description" label="What should the model do?" error={fieldErr} hint="10 to 2,000 characters">
          <Textarea id="description" rows={4} value={text} maxLength={2000} onChange={e => setText(e.target.value)} placeholder={EXAMPLES[0]} aria-invalid={!!fieldErr} />
        </Field>
        <div className="flex flex-wrap gap-2">
          <span className="text-xs text-muted-foreground">Try:</span>
          {EXAMPLES.map(ex => (
            <button key={ex} type="button" onClick={() => setText(ex)} className="max-w-full cursor-pointer truncate rounded-full border bg-card px-2.5 py-0.5 text-xs text-muted-foreground hover:bg-accent sm:max-w-80">
              {ex}
            </button>
          ))}
        </div>
        {result && (result.status === 'unsupported' || !result.draft_spec) && (
          <div role="alert" className="flex gap-3 rounded-md border border-warning/40 bg-warning/10 p-4 text-sm">
            <AlertTriangle className="mt-0.5 size-4 shrink-0 text-warning" />
            <div className="grid gap-2">
              <p>{result.message}</p>
              <Button type="button" variant="outline" size="sm" className="w-fit" onClick={onBack}>
                Choose a category
              </Button>
            </div>
          </div>
        )}
        {fallback && (
          <div role="alert" className="flex flex-col gap-3 rounded-md border border-warning/40 bg-warning/10 p-4 text-sm sm:flex-row sm:items-center">
            <AlertTriangle className="size-5 shrink-0 text-warning" />
            <span className="flex-1">The AI helper couldn't draft this right now. Pick a category instead; it takes the same few seconds.</span>
            <Button type="button" variant="outline" size="sm" onClick={onBack}>
              Choose a category
            </Button>
          </div>
        )}
        {err && !fallback && !fieldErr && (
          <p role="alert" className="text-sm text-destructive">
            {err.message}
          </p>
        )}
        <div className="flex justify-between gap-2">
          <Button type="button" variant="ghost" onClick={onBack}>
            <ArrowLeft /> Back to categories
          </Button>
          <Button type="submit" disabled={text.trim().length < 10 || describe.isPending}>
            {describe.isPending ? (
              <>
                <Loader2 className="animate-spin" /> Drafting spec…
              </>
            ) : (
              <>
                <Sparkles /> Draft my project
              </>
            )}
          </Button>
        </div>
      </form>
    </Card>
  );
}

function DetailsStep({ draft, onBack, onNext }: { draft: Draft; onBack: () => void; onNext: (d: Draft) => void }) {
  const t = draft.template;
  const m = templateMeta(t.id);
  const [name, setName] = useState(draft.name);
  const [taskType, setTaskType] = useState(draft.taskType);
  const [labels, setLabels] = useState(draft.labels);
  const [touched, setTouched] = useState(false);
  const needsLabels = NEEDS_LABELS(taskType);
  const isSentiment = taskType === 'text_classification' && SENTIMENT.every(l => labels.includes(l)) && labels.length === 3;
  const nameErr = touched && !name.trim() ? 'Give the project a name' : undefined;
  const minLabels = taskType === 'object_detection' ? 1 : 2;
  const labelsErr = touched && needsLabels && labels.length < minLabels ? `Add at least ${minLabels} ${minLabels === 1 ? 'class' : 'classes'}` : undefined;

  return (
    <Card className="grid gap-6 p-6">
      {draft.starter && (
        <div className="flex gap-2 rounded-md border border-info/40 bg-info/8 p-3 text-sm">
          <Info className="mt-0.5 size-4 shrink-0 text-info" />
          <span>
            Starter dataset <span className="font-medium">{draft.starter.name}</span> ({draft.starter.sample_count.toLocaleString()} samples, {draft.starter.licence}) will be imported into the
            project's draft right after it's created.
            {taskType !== draft.starter.task_type && <span className="text-warning"> It's for {TASK_LABEL[draft.starter.task_type]}, so it won't be imported with this task type.</span>}
          </span>
        </div>
      )}
      {draft.note && (
        <div className={cn('flex gap-2 rounded-md border p-3 text-sm', draft.note.status === 'ok' ? 'border-info/40 bg-info/8' : 'border-warning/40 bg-warning/10')}>
          {draft.note.status === 'ok' ? <Info className="mt-0.5 size-4 shrink-0 text-info" /> : <AlertTriangle className="mt-0.5 size-4 shrink-0 text-warning" />}
          <span>
            {draft.note.message} <span className="text-muted-foreground">Review everything below before creating the project.</span>
          </span>
        </div>
      )}
      <div className="flex items-center gap-3">
        <span className={cn('grid size-10 place-items-center rounded-lg', m.tint)}>
          <m.icon className="size-5" />
        </span>
        <div>
          <h2 className="font-semibold">{t.name}</h2>
          <p className="text-sm text-muted-foreground">{t.description}</p>
        </div>
      </div>

      {t.id === 'tabular' && (
        <fieldset className="grid gap-2">
          <legend className="mb-2 text-sm font-medium">What do you predict?</legend>
          <div className="grid gap-2 sm:grid-cols-2">
            {(t.task_types as TaskType[]).map(tt => (
              <label key={tt} className={cn('flex cursor-pointer gap-3 rounded-lg border p-3', taskType === tt ? 'border-primary bg-primary/6' : 'hover:bg-accent/60')}>
                <input type="radio" name="task-type" className="mt-1 accent-[var(--primary)]" checked={taskType === tt} onChange={() => setTaskType(tt)} />
                <span>
                  <span className="block text-sm font-medium">{TASK_LABEL[tt]}</span>
                  <span className="block text-xs text-muted-foreground">{tt === 'tabular_regression' ? 'A number: price, ETA, demand.' : 'A category: churn yes/no, fraud, segment.'}</span>
                </span>
              </label>
            ))}
          </div>
        </fieldset>
      )}

      {taskType === 'text_classification' && (
        <fieldset className="grid gap-2">
          <legend className="mb-2 text-sm font-medium">Kind of text task</legend>
          <div className="grid gap-2 sm:grid-cols-2">
            {[
              { id: 'custom', title: 'Your own classes', desc: 'Ticket routing, spam, intent, topics.', on: !isSentiment, pick: () => isSentiment && setLabels([]) },
              { id: 'sentiment', title: 'Sentiment', desc: 'Starts with positive / negative / neutral; edit freely.', on: isSentiment, pick: () => setLabels(SENTIMENT) },
            ].map(o => (
              <label key={o.id} className={cn('flex cursor-pointer gap-3 rounded-lg border p-3', o.on ? 'border-primary bg-primary/6' : 'hover:bg-accent/60')}>
                <input type="radio" name="text-kind" className="mt-1 accent-[var(--primary)]" checked={o.on} onChange={o.pick} />
                <span>
                  <span className="block text-sm font-medium">{o.title}</span>
                  <span className="block text-xs text-muted-foreground">{o.desc}</span>
                </span>
              </label>
            ))}
          </div>
        </fieldset>
      )}

      <Field id="project-name" label="Project name" error={nameErr}>
        <Input id="project-name" value={name} onChange={e => setName(e.target.value)} placeholder={`e.g. ${m.examples[0] ?? 'My'} model`} autoFocus aria-invalid={!!nameErr} maxLength={80} />
      </Field>

      {needsLabels ? (
        <div className="grid gap-1.5">
          <Label htmlFor="classes">Class names</Label>
          <TagInput id="classes" value={labels} onChange={setLabels} placeholder={LABEL_EXAMPLES[taskType]} invalid={!!labelsErr} describedBy="classes-hint" />
          <p id="classes-hint" className={cn('text-xs', labelsErr ? 'text-destructive' : 'text-muted-foreground')} role={labelsErr ? 'alert' : undefined}>
            {labelsErr ?? `Press Enter or comma after each class, or paste a list. ${minLabels} to 500 classes; the order sets the number-key shortcuts when labeling.`}
          </p>
        </div>
      ) : (
        <p className="flex gap-2 rounded-md bg-muted p-3 text-sm text-muted-foreground">
          <Info className="mt-0.5 size-4 shrink-0" />
          {taskType === 'tabular_regression'
            ? 'Regression predicts a number, so there are no classes. You pick the target column when you upload your CSV.'
            : 'Classes come from your target column; you pick it when you upload your CSV or Excel file.'}
        </p>
      )}

      <div className="flex justify-between gap-2">
        <Button variant="ghost" onClick={onBack}>
          <ArrowLeft /> Back
        </Button>
        <Button
          onClick={() => {
            setTouched(true);
            if (!name.trim() || (needsLabels && labels.length < minLabels)) return;
            const base = taskType === draft.spec.task_type ? draft.spec : defaultSpec(t, taskType);
            onNext({ ...draft, taskType, name: name.trim(), labels, spec: { ...base, task_type: taskType, labels: needsLabels ? labels : [] } });
          }}
        >
          Next: task spec <ArrowRight />
        </Button>
      </div>
    </Card>
  );
}

function SpecStep({ draft, onBack }: { draft: Draft; onBack: () => void }) {
  const { org } = useCurrentOrg();
  const form = useSpecForm(draft.spec);
  const create = useCreateProject(org?.id);
  const importStarter = useImportStarter();
  const navigate = useNavigate();
  const starter = draft.starter?.task_type === draft.taskType ? draft.starter : null;
  const face = isBiometric(draft.taskType);
  const voice = draft.taskType === 'speaker_recognition';
  const [phrase, setPhrase] = useState('');
  const createFace = () =>
    create.mutate(
      { name: draft.name, spec: { ...draft.spec, task_type: draft.taskType, labels: [] } },
      {
        onSuccess: async p => {
          if (voice && phrase.trim()) await api(`/projects/${p.id}/face-settings`, { method: 'PATCH', body: { phrase: phrase.trim() } }).catch(() => toast.error('Couldn’t save the phrase; set it in Settings'));
          toast.success(`Project "${p.name}" created`);
          navigate({ to: voice ? '/projects/$projectId/speakers' : '/projects/$projectId/people', params: { projectId: p.id } });
        },
        onError: e => toast.error(e.message),
      },
    );
  const [agree, setAgree] = useState(false);

  const submit = form.handleSubmit(spec =>
    create.mutate(
      { name: draft.name, spec: spec as TaskSpec, llm_log_id: draft.llmLogId },
      {
        onSuccess: p => {
          toast.success(`Project "${p.name}" created`);
          if (!starter) return navigate({ to: '/projects/$projectId', params: { projectId: p.id } });
          // The draft bar on the Dataset page shows the ingest; an import failure leaves a usable empty project.
          importStarter.mutate(
            { projectId: p.id, starterId: starter.id },
            {
              onSuccess: () => toast.info(`Importing "${starter.name}" into the draft…`),
              onError: e => toast.error(`Couldn't import the starter dataset: ${importError(e)} Use Add data → Starter datasets to retry.`),
            },
          );
          navigate({ to: '/projects/$projectId/dataset', params: { projectId: p.id }, search: {} });
        },
        onError: e => toast.error(e.message),
      },
    ),
  );

  if (face)
    return (
      <Card className="grid gap-5 p-6">
        <div>
          <h2 className="text-lg font-semibold">{draft.name}</h2>
          <p className="text-sm text-muted-foreground">{voice ? 'Speaker recognition matches voices' : 'Face recognition matches faces'} against people you add. There's no training step: add people, tune the threshold, try it.</p>
        </div>
        <ul className="grid gap-1.5 rounded-lg border bg-muted/40 p-4 text-sm">
          <li>• {voice ? 'A voice is biometric data: every' : 'Every'} person needs their consent before you add their {voice ? 'recordings' : 'photos'}; you confirm it per person.</li>
          <li>• Removing a person deletes their {voice ? 'clips and voice' : 'photos and face'} data immediately. Deleting the project removes everything.</li>
          <li>• Not for covert surveillance of the public, or as the only basis for decisions about people without human review.</li>
          {!voice && <li>• The face model is licensed for non-commercial research use only in this build.</li>}
        </ul>
        {voice && (
          <label className="grid gap-1.5 text-sm">
            <span className="font-medium">Phrase everyone says (optional)</span>
            <Input value={phrase} onChange={e => setPhrase(e.target.value)} maxLength={300} placeholder="Ask everyone to say: “Good morning, this is my voice”" />
            <span className="text-xs text-muted-foreground">Shown while recording. The same words for everyone make matching more reliable. You can change it later in Settings.</span>
          </label>
        )}
        <label className="flex items-start gap-3 text-sm">
          <Checkbox checked={agree} onCheckedChange={v => setAgree(v === true)} className="mt-0.5" />
          <span>I'll only use this for a consented purpose such as attendance or check-in.</span>
        </label>
        <div className="flex justify-between">
          <Button variant="ghost" onClick={onBack}><ArrowLeft /> Back</Button>
          <Button disabled={!agree || create.isPending} onClick={createFace}>
            {create.isPending ? <Loader2 className="animate-spin" /> : null} Create project <ArrowRight />
          </Button>
        </div>
      </Card>
    );

  return (
    <Card className="grid gap-6 p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">{draft.name}</h2>
          <p className="text-sm text-muted-foreground">
            {TASK_LABEL[draft.taskType]}
            {draft.labels.length > 0 && ` · ${draft.labels.length} classes: ${draft.labels.slice(0, 6).join(', ')}${draft.labels.length > 6 ? '…' : ''}`}
          </p>
        </div>
      </div>
      <SpecEditor form={form} />
      <div className="flex justify-between gap-2 border-t pt-5">
        <Button variant="ghost" onClick={onBack}>
          <ArrowLeft /> Back
        </Button>
        <Button onClick={submit} disabled={create.isPending}>
          {create.isPending ? (
            <>
              <Loader2 className="animate-spin" /> Creating…
            </>
          ) : (
            <>
              Create project <ArrowRight />
            </>
          )}
        </Button>
      </div>
    </Card>
  );
}

export function SkeletonCards() {
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      {Array.from({ length: 8 }, (_, i) => (
        <Skeleton key={i} className="h-44 rounded-xl" />
      ))}
    </div>
  );
}
