import { useEffect, useState } from 'react';
import { Controller, useForm, type UseFormReturn } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { Braces, CheckCircle2, Clock, FormInput, AlertTriangle } from 'lucide-react';
import type { BudgetMode, ExportFormat, Platform, TaskSpec, TaskType } from '@/api/types';
import { Badge } from '@/components/ui/badge';
import { Input, Textarea } from '@/components/ui/input';
import { Field, Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { cn } from '@/lib/utils';
import { METRIC_LABEL, METRICS, specIssues, taskSpecSchema, type TaskSpecInput } from './spec-schema';

const PLATFORMS: { id: Platform; label: string; format: ExportFormat }[] = [
  { id: 'server', label: 'Server / API', format: 'onnx' },
  { id: 'android', label: 'Android', format: 'tflite' },
  { id: 'ios', label: 'iOS', format: 'coreml' },
  { id: 'web', label: 'Web browser', format: 'tfjs' },
  { id: 'edge_linux', label: 'Edge Linux device', format: 'onnx' },
];

/** EXP-1 MVP export matrix per task. */
const FORMATS_FOR: Record<TaskType, ExportFormat[]> = {
  image_classification: ['onnx', 'tflite', 'coreml', 'tfjs'],
  object_detection: ['onnx', 'tflite', 'coreml', 'tfjs'],
  face_recognition: [], // no export in v1 (FACE-7: no raw embeddings leave the workspace)
  speaker_recognition: [],
  text_classification: ['onnx', 'tflite', 'coreml'],
  tabular_classification: ['onnx', 'native'],
  tabular_regression: ['onnx', 'native'],
};
const FORMAT_LABEL: Record<ExportFormat, string> = { onnx: 'ONNX', tflite: 'TFLite', coreml: 'Core ML', tfjs: 'TF.js', native: 'Native (LightGBM/XGBoost)' };

/** Local build: only Quick runs; Standard/Thorough are shown as coming soon (no plans or prices). */
export const BUDGETS: { id: BudgetMode; label: string; time: string; available: boolean }[] = [
  { id: 'quick', label: 'Quick', time: '~10 min', available: true },
  { id: 'standard', label: 'Standard', time: '30–60 min', available: false },
  { id: 'thorough', label: 'Thorough', time: 'several hours', available: false },
];

const LANGUAGES = [
  ['en', 'English'],
  ['hi', 'Hindi'],
  ['hi-Latn', 'Hinglish (romanised)'],
  ['bn', 'Bengali'],
  ['ta', 'Tamil'],
  ['te', 'Telugu'],
  ['mr', 'Marathi'],
  ['gu', 'Gujarati'],
  ['kn', 'Kannada'],
  ['ml', 'Malayalam'],
  ['pa', 'Punjabi'],
  ['or', 'Odia'],
] as const;

function Chips<T extends string>({ options, value, onChange, label }: { options: readonly { id: T; label: string; disabled?: boolean }[]; value: T[]; onChange: (v: T[]) => void; label: string }) {
  return (
    <div className="flex flex-wrap gap-2" role="group" aria-label={label}>
      {options.map(o => {
        const on = value.includes(o.id);
        return (
          <button
            key={o.id}
            type="button"
            aria-pressed={on}
            disabled={o.disabled}
            onClick={() => onChange(on ? value.filter(v => v !== o.id) : [...value, o.id])}
            className={cn(
              'cursor-pointer rounded-full border px-3 py-1 text-sm transition-colors disabled:cursor-not-allowed disabled:opacity-50',
              on ? 'border-primary bg-primary/10 font-medium text-primary' : 'bg-card hover:bg-accent',
            )}
          >
            {on && <CheckCircle2 className="-ml-0.5 mr-1 inline size-3.5" />}
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

/** Appendix A key order, so the JSON reads like the PRD example. */
const KEY_ORDER = ['spec_version', 'task_type', 'multi_label', 'labels', 'input', 'primary_metric', 'secondary_metrics', 'constraints', 'budget_mode', 'split', 'language', 'target_column', 'feature_columns', 'id_columns', 'group_column', 'notes'];
function ordered(spec: object): object {
  const entries = Object.entries(spec).filter(([, v]) => v !== undefined);
  return Object.fromEntries(entries.sort(([a], [b]) => (KEY_ORDER.indexOf(a) + 1 || 99) - (KEY_ORDER.indexOf(b) + 1 || 99)));
}

const nullableNumber = { setValueAs: (v: unknown) => (v === '' || v === null || v === undefined ? null : Number(v)) };

export function useSpecForm(initial: TaskSpec) {
  return useForm<TaskSpecInput>({ resolver: zodResolver(taskSpecSchema), defaultValues: initial as TaskSpecInput, mode: 'onChange' });
}

/** PRJ-3: form view and raw JSON view of the same spec, with live schema validation. */
export function SpecEditor({ form }: { form: UseFormReturn<TaskSpecInput> }) {
  const [tab, setTab] = useState('form');
  const [json, setJson] = useState('');
  const [jsonIssues, setJsonIssues] = useState<string[]>([]);
  const taskType = form.watch('task_type');
  const { errors } = form.formState;

  // Entering the JSON tab shows the current form values.
  useEffect(() => {
    if (tab === 'json') {
      setJson(JSON.stringify(ordered(form.getValues()), null, 2));
      setJsonIssues([]);
    }
  }, [tab, form]);

  const onJson = (text: string) => {
    setJson(text);
    try {
      const parsed = JSON.parse(text);
      const issues = specIssues(parsed);
      setJsonIssues(issues);
      // Valid JSON that matches the schema flows straight back into the form.
      if (!issues.length) form.reset(parsed, { keepDefaultValues: true });
    } catch (e) {
      setJsonIssues([`JSON syntax: ${(e as Error).message}`]);
    }
  };

  const isText = taskType === 'text_classification';

  return (
    <Tabs value={tab} onValueChange={setTab}>
      <TabsList>
        <TabsTrigger value="form">
          <FormInput /> Form
        </TabsTrigger>
        <TabsTrigger value="json">
          <Braces /> Raw JSON
          {jsonIssues.length > 0 && tab === 'json' && <span className="size-1.5 rounded-full bg-destructive" aria-label="has errors" />}
        </TabsTrigger>
      </TabsList>

      <TabsContent value="form" className="grid gap-6">
        <section className="grid gap-3">
          <Label id="platforms-label">Where will the model run?</Label>
          <Controller
            control={form.control}
            name="constraints.target_platforms"
            render={({ field }) => (
              <Chips
                label="Target platforms"
                options={PLATFORMS}
                value={field.value ?? []}
                onChange={v => {
                  field.onChange(v);
                  // PRJ-6: platforms drive the default export formats.
                  const allowed = FORMATS_FOR[taskType];
                  const formats = [...new Set(v.map(p => PLATFORMS.find(x => x.id === p)!.format))].filter(f => allowed.includes(f));
                  form.setValue('constraints.export_formats', formats.length ? formats : ['onnx'], { shouldValidate: true });
                }}
              />
            )}
          />
          {errors.constraints?.target_platforms && <p className="text-xs text-destructive">{errors.constraints.target_platforms.message}</p>}
        </section>

        <section className="grid gap-3">
          <Label>Export formats</Label>
          <Controller
            control={form.control}
            name="constraints.export_formats"
            render={({ field }) => (
              <Chips label="Export formats" options={FORMATS_FOR[taskType].map(f => ({ id: f, label: FORMAT_LABEL[f] }))} value={field.value ?? []} onChange={field.onChange} />
            )}
          />
          {taskType === 'object_detection' && <p className="text-xs text-muted-foreground">TF.js for detection is best effort; NMS is included in the graph or documented in the snippet.</p>}
        </section>

        <div className="grid gap-4 sm:grid-cols-3">
          <Field id="max-size" label="Max model size (MB)" hint="Leave empty for no limit" error={errors.constraints?.max_model_size_mb?.message}>
            <Input id="max-size" type="number" min={1} inputMode="decimal" placeholder="No limit" {...form.register('constraints.max_model_size_mb', nullableNumber)} />
          </Field>
          <Field id="max-latency" label="Max latency (ms)" hint="Per prediction on the target device" error={errors.constraints?.max_latency_ms?.message}>
            <Input id="max-latency" type="number" min={1} inputMode="numeric" placeholder="No limit" {...form.register('constraints.max_latency_ms', nullableNumber)} />
          </Field>
          <Field id="quant" label="Quantisation" hint="Accuracy is re-measured after quantising">
            <select id="quant" className="h-9 rounded-md border border-input bg-card px-3 text-sm" {...form.register('constraints.quantization')}>
              <option value="none">None (FP32)</option>
              <option value="fp16">FP16</option>
              <option value="int8_dynamic">INT8 dynamic</option>
              <option value="int8_full">INT8 full (calibrated)</option>
            </select>
          </Field>
        </div>

        <section className="grid gap-3">
          <Label id="budget-label">Training budget</Label>
          <Controller
            control={form.control}
            name="budget_mode"
            render={({ field }) => (
              <div role="radiogroup" aria-labelledby="budget-label" className="grid gap-3 sm:grid-cols-3">
                {BUDGETS.map(b => {
                  const locked = !b.available;
                  const on = field.value === b.id;
                  return (
                    <button
                      key={b.id}
                      type="button"
                      role="radio"
                      aria-checked={on}
                      disabled={locked}
                      onClick={() => field.onChange(b.id)}
                      className={cn(
                        'cursor-pointer rounded-lg border bg-card p-4 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-60',
                        on ? 'border-primary ring-2 ring-primary/30' : 'hover:bg-accent/60',
                      )}
                    >
                      <div className="flex items-center justify-between font-medium">
                        {b.label}
                        {locked && <Badge variant="secondary">Coming soon</Badge>}
                      </div>
                      <div className="mt-2 grid gap-1 text-xs text-muted-foreground">
                        <span className="flex items-center gap-1.5">
                          <Clock className="size-3.5" /> {b.time}
                        </span>
                      </div>
                    </button>
                  );
                })}
              </div>
            )}
          />
        </section>

        <div className="grid gap-4 sm:grid-cols-2">
          <Field id="metric" label="Primary metric" hint="The search optimises this" error={errors.primary_metric?.message}>
            <select id="metric" className="h-9 rounded-md border border-input bg-card px-3 text-sm" {...form.register('primary_metric')}>
              {METRICS[taskType].map(m => (
                <option key={m} value={m}>
                  {METRIC_LABEL[m] ?? m}
                </option>
              ))}
            </select>
          </Field>
          <Controller
            control={form.control}
            name="split"
            render={({ field }) => (
              <fieldset className="grid gap-1.5">
                <legend className="mb-1.5 text-sm font-medium">Train / val / test split (%)</legend>
                <div className="grid grid-cols-3 gap-2">
                  {(['train', 'val', 'test'] as const).map(k => (
                    <Input
                      key={k}
                      type="number"
                      min={1}
                      max={95}
                      aria-label={`${k} percent`}
                      value={Math.round(field.value[k] * 100)}
                      onChange={e => field.onChange({ ...field.value, [k]: Number(e.target.value) / 100 })}
                    />
                  ))}
                </div>
                {errors.split && <p className="text-xs text-destructive">{errors.split.message ?? errors.split.root?.message}</p>}
              </fieldset>
            )}
          />
        </div>

        {isText && (
          <section className="grid gap-3">
            <Label>Languages in your text</Label>
            <Controller
              control={form.control}
              name="language"
              render={({ field }) => (
                <Chips label="Languages" options={LANGUAGES.map(([id, label]) => ({ id, label }))} value={(field.value ?? []) as string[]} onChange={v => field.onChange(v)} />
              )}
            />
            <p className="text-xs text-muted-foreground">Indic or code-mixed text above 10% switches to multilingual encoders (MuRIL, IndicBERT, XLM-R).</p>
          </section>
        )}

        <Field id="notes" label="Notes (optional)" hint="Anything about how the data was collected">
          <Textarea id="notes" rows={2} placeholder="e.g. Phone photos in the field, varied lighting" {...form.register('notes', { setValueAs: v => (v === '' ? null : v) })} />
        </Field>
      </TabsContent>

      <TabsContent value="json" className="grid gap-3">
        <Textarea
          value={json}
          onChange={e => onJson(e.target.value)}
          spellCheck={false}
          rows={22}
          className="font-mono text-xs leading-relaxed"
          aria-label="Task spec JSON"
          aria-invalid={jsonIssues.length > 0}
          aria-describedby="json-status"
        />
        <div id="json-status" aria-live="polite">
          {jsonIssues.length ? (
            <ul className="grid gap-1 rounded-md border border-destructive/30 bg-destructive/6 p-3 text-xs text-destructive">
              {jsonIssues.map(i => (
                <li key={i} className="flex gap-1.5">
                  <AlertTriangle className="mt-px size-3.5 shrink-0" />
                  {i}
                </li>
              ))}
            </ul>
          ) : (
            <p className="flex items-center gap-1.5 text-xs text-success">
              <CheckCircle2 className="size-3.5" /> Valid task spec (schema v1.0). Changes are applied to the form.
            </p>
          )}
        </div>
      </TabsContent>
    </Tabs>
  );
}
