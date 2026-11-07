import { Link } from '@tanstack/react-router';
import { ArrowRight, Check, FolderGit2, Image, MessageSquareText, Mic, ScanFace, ScanSearch, Table2, type LucideIcon } from 'lucide-react';
import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react';
import { Button } from '@/components/ui/button';
import { Logo } from '@/features/shell/logo';
import { ThemeToggle } from '@/features/shell/theme-toggle';
import { useAuth } from '@/lib/auth';
import { cn } from '@/lib/utils';

const GITHUB = 'https://github.com/exoshivam/automl-platform';

const HERO_STEPS: [string, string][] = [
  ['Upload data', 'Photos, CSV, camera or starter set'],
  ['Label with AI help', 'Auto-suggested boxes and classes'],
  ['Validate', 'Duplicates, leaks, imbalance'],
  ['Train', 'Optuna picks the best settings'],
  ['Evaluate', 'Test set, baseline, calibration'],
  ['Deploy', 'API key or ONNX export'],
];

const STATS: [string, string][] = [
  ['6', 'Kinds of AI you can build'],
  ['15+', 'Automatic data-quality checks'],
  ['1 click', 'From labelled data to trained model'],
  ['0', 'Lines of code for the user'],
];

/** Each bullet is [bold lead-in, rest of the sentence]. */
const FLOW: { t: string; d: string; li: [string, string][]; g: string[] }[] = [
  {
    t: 'Add data',
    d: 'Bring photos, a CSV, a zip, a camera feed or a ready-made starter dataset.',
    li: [
      ['Background ingest', ' opens every file, rejects corrupt ones and fingerprints each with sha256.'],
      ['Files go to MinIO', ', metadata to Postgres, so the API never blocks.'],
      ['Import', ' existing labels in COCO, YOLO or VOC format. Bad records are reported, never silently dropped.'],
    ],
    g: ['MinIO', 'Celery', 'sha256'],
  },
  {
    t: 'Label with AI help',
    d: 'Click a class, or draw a box. Or let the platform propose labels for you.',
    li: [
      ['Auto-label', ' asks a detector for boxes: your own model first, then ready-made COCO, then a text prompt through OWLv2.'],
      ['Suggestions only.', ' Nothing becomes a label until you accept it.'],
      ['Mark "no objects"', ' so the model also learns what empty looks like.'],
    ],
    g: ['OWLv2', 'COCO', 'Human in the loop'],
  },
  {
    t: 'Seal and validate',
    d: 'Freeze the dataset and let the platform check its health before any training.',
    li: [
      ['Stratified train / val / test split', ' keeps class ratios equal in every part.'],
      ['Near-duplicates stay together', ' so the test set is never a copy of training.'],
      ['Checks:', ' corrupt files, tiny classes, imbalance, bad boxes, conflicting labels, target leakage.'],
      ['Errors block training.', ' Warnings must be acknowledged.'],
    ],
    g: ['Perceptual hash', 'Content hash', 'Leakage check'],
  },
  {
    t: 'Train',
    d: 'One click. The platform picks the model, searches settings and streams progress live.',
    li: [
      ['Pretrained backbone', ', new head, fine-tuned on your data.'],
      ['Optuna', ' searches learning rate and architecture, stopping weak trials early.'],
      ['Safe on a laptop:', ' one heavy job at a time, memory check, batch shrinks on low RAM.'],
      ['Live progress', ' reaches the browser through Redis and server-sent events.'],
    ],
    g: ['Optuna', 'PyTorch', 'SSE'],
  },
  {
    t: 'Evaluate',
    d: 'The model sits an exam on photos it has never seen.',
    li: [
      ['Metrics:', ' accuracy, macro-F1, mAP for detection, RMSE for regression.'],
      ['Confusion matrix', ' where each cell opens the exact photos behind it.'],
      ['Baseline comparison and calibration', ' show whether the model really learned.'],
      ['Suspiciously perfect?', ' You get a warning instead of applause.'],
    ],
    g: ['Test set', 'ECE', 'Baseline'],
  },
  {
    t: 'Use it',
    d: 'Try it in the playground, then export or deploy.',
    li: [
      ['Playground:', ' drop in a new photo or text and see the prediction.'],
      ['ONNX export', ' with a model card, labels and README. Output is verified against the original model.'],
      ['Deploy:', ' an API key and a single predict endpoint with usage tracking.'],
    ],
    g: ['ONNX', 'API key', 'Rate limit'],
  },
];

const BUILD: { icon: LucideIcon; color: string; t: string; d: string; m: string }[] = [
  { icon: Image, color: 'text-violet-500 bg-violet-500/14', t: 'Image classification', d: 'Sort photos into classes: defect or fine, plant disease, product type.', m: 'MobileNetV3 / EfficientNet, fine-tuned' },
  { icon: ScanSearch, color: 'text-sky-500 bg-sky-500/14', t: 'Object detection', d: 'Draw boxes around helmets, vehicles, shelf items. AI suggests boxes so you label faster.', m: 'SSDlite + OWLv2 auto-label' },
  { icon: MessageSquareText, color: 'text-emerald-500 bg-emerald-500/14', t: 'Text & sentiment', d: 'Route tickets, detect spam, score sentiment. Works with Hindi and Hinglish.', m: 'Multilingual MiniLM' },
  { icon: Table2, color: 'text-amber-500 bg-amber-500/14', t: 'Tabular prediction', d: 'Predict churn, fraud or price from a CSV. Classification and regression.', m: 'LightGBM + Optuna search' },
  { icon: ScanFace, color: 'text-pink-500 bg-pink-500/14', t: 'Face recognition', d: 'Recognise enrolled, consenting people. Add or remove a person instantly, no retraining.', m: 'Embedding gallery in pgvector' },
  { icon: Mic, color: 'text-rose-500 bg-rose-500/14', t: 'Speaker recognition', d: 'Know who is speaking. Strangers come back as Unknown. No training needed.', m: 'WavLM voice embeddings' },
];

const TRUST: [string, string][] = [
  ['Leak-proof splits', 'Duplicate and near-identical photos always stay in the same split, so the test set is a fair exam.'],
  ['Data checked before training', 'Corrupt files, tiny classes, imbalance, bad boxes and target leakage block or warn you first.'],
  ['Beat the baseline', 'Results are compared with a trivial "always guess the common class" model, so you see real learning.'],
  ['Calibrated confidence', '"90% sure" should mean right about 90% of the time. The platform measures and corrects this.'],
  ['Secure by default', 'Argon2 password hashing, API keys stored only as hashes, role-based access and rate limits.'],
  ['Take it anywhere', 'Export to ONNX with an automatic output check, or deploy locally behind an API key.'],
];

const STACK: [string, string][] = [
  ['React', ' · TypeScript · Vite'],
  ['FastAPI', ' · Python 3.12'],
  ['PostgreSQL', ' + pgvector'],
  ['Redis', ' · Celery'],
  ['MinIO', ' object storage'],
  ['MLflow', ''],
  ['PyTorch', ' · timm · torchvision'],
  ['Hugging Face', ' Transformers'],
  ['LightGBM', ' · Optuna'],
  ['ONNX', ' Runtime'],
  ['Docker', ' Compose'],
];

const NAV: [string, string][] = [
  ['#how', 'How it works'],
  ['#build', 'What you can build'],
  ['#under', 'Under the hood'],
  ['#trust', 'Reliability'],
  ['#stack', 'Tech stack'],
];

const GRAD_TEXT = 'bg-gradient-to-r from-primary via-violet-500 to-teal-500 bg-clip-text text-transparent dark:to-teal-400';
const PRIMARY_BTN = 'border-0 bg-gradient-to-br from-primary to-violet-500 text-white shadow-lg shadow-primary/25 hover:from-primary hover:to-violet-600';

const reducedMotion = () => typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

/** Fades children up once they scroll into view; instant for reduced motion. */
function Reveal({ children, className, style }: { children: ReactNode; className?: string; style?: CSSProperties }) {
  const ref = useRef<HTMLDivElement>(null);
  const [shown, setShown] = useState(reducedMotion);
  useEffect(() => {
    const el = ref.current;
    if (shown || !el) return;
    const io = new IntersectionObserver(es => es.some(e => e.isIntersecting) && (setShown(true), io.disconnect()), { threshold: 0.12 });
    io.observe(el);
    return () => io.disconnect();
  }, [shown]);
  return (
    <div ref={ref} style={style} className={cn('transition-[opacity,transform] duration-700 motion-reduce:transition-none', !shown && 'translate-y-5 opacity-0', className)}>
      {children}
    </div>
  );
}

function SectionHead({ eyebrow, title, sub }: { eyebrow: string; title: string; sub?: string }) {
  return (
    <Reveal className="mb-9">
      <div className="text-[13px] font-bold uppercase tracking-[0.14em] text-primary">{eyebrow}</div>
      <h2 className="mt-3 mb-3.5 text-[clamp(28px,3.8vw,44px)] font-extrabold leading-[1.1] tracking-tight">{title}</h2>
      {sub && <p className="max-w-[640px] text-[17px] text-muted-foreground">{sub}</p>}
    </Reveal>
  );
}

/** The hero card: the six steps light up in turn, like a pipeline running. */
function HeroPipeline() {
  const [cur, setCur] = useState(() => (reducedMotion() ? 3 : 0));
  useEffect(() => {
    if (reducedMotion()) return;
    const t = setInterval(() => setCur(c => (c + 1) % (HERO_STEPS.length + 1)), 2100);
    return () => clearInterval(t);
  }, []);
  return (
    <div className="min-w-0 overflow-hidden rounded-2xl border bg-card shadow-2xl shadow-primary/10" aria-label="The six pipeline steps">
      <div className="flex items-center gap-1.5 border-b bg-muted/60 px-4 py-3">
        <i className="size-2.5 rounded-full bg-[#ff5f57]" />
        <i className="size-2.5 rounded-full bg-[#febc2e]" />
        <i className="size-2.5 rounded-full bg-[#28c840]" />
        <span className="ml-3 truncate rounded-md bg-background px-3 py-0.5 font-mono text-xs text-muted-foreground">127.0.0.1:5180 / projects</span>
      </div>
      <ol className="grid gap-2.5 p-5">
        {HERO_STEPS.map(([b, s], i) => {
          const on = i === cur;
          const done = i < cur;
          return (
            <li key={b} className={cn('flex items-center gap-3.5 rounded-xl border bg-background/60 px-3.5 py-3 transition-all duration-300 motion-reduce:transition-none', on ? 'border-primary opacity-100 ring-3 ring-primary/18' : done ? 'opacity-90' : 'opacity-55')}>
              <span className={cn('grid size-[30px] shrink-0 place-items-center rounded-lg text-[13px] font-bold transition-colors', on ? 'bg-gradient-to-br from-primary to-violet-500 text-white' : done ? 'bg-teal-500/20 text-teal-600 dark:text-teal-400' : 'bg-muted text-muted-foreground')}>
                {i + 1}
              </span>
              <div className="min-w-0 flex-1">
                <b className="block text-sm leading-snug">{b}</b>
                <small className="text-xs text-muted-foreground">{s}</small>
                <div className="mt-1.5 h-1 overflow-hidden rounded bg-border">
                  <div className={cn('h-full rounded bg-gradient-to-r from-primary to-teal-500', on ? 'w-full transition-[width] duration-[2000ms] ease-linear motion-reduce:transition-none' : done ? 'w-full' : 'w-0')} />
                </div>
              </div>
              <span className={cn('ml-auto text-xs', on || done ? 'text-teal-600 dark:text-teal-400' : 'text-muted-foreground')}>{done ? 'Done' : on ? 'Running' : ''}</span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function WorkflowExplorer() {
  const [i, setI] = useState(0);
  const f = FLOW[i]!;
  return (
    <Reveal className="grid gap-4.5 md:grid-cols-[300px_1fr]">
      <div className="flex gap-2 overflow-x-auto pb-1 md:flex-col md:overflow-visible md:pb-0" role="tablist" aria-label="Workflow steps">
        {FLOW.map((x, j) => (
          <button
            key={x.t}
            type="button"
            role="tab"
            aria-selected={i === j}
            onClick={() => setI(j)}
            className={cn(
              'flex shrink-0 cursor-pointer items-center gap-3 whitespace-nowrap rounded-xl border bg-card px-4 py-3.5 text-left text-muted-foreground transition-colors hover:border-primary hover:text-foreground md:whitespace-normal',
              i === j && 'border-primary bg-gradient-to-br from-primary/15 to-card text-foreground',
            )}
          >
            <span className={cn('grid size-7 shrink-0 place-items-center rounded-lg text-[13px] font-bold', i === j ? 'bg-gradient-to-br from-primary to-violet-500 text-white' : 'bg-muted')}>{j + 1}</span>
            {x.t}
          </button>
        ))}
      </div>
      <div className="min-h-[420px] min-w-0 rounded-2xl border bg-card p-5 shadow-xl shadow-primary/5 sm:p-7.5" role="tabpanel" aria-label={f.t}>
        <div key={i} className="motion-safe:animate-[landing-fadeup_.35s_ease]">
          <h3 className="mb-1.5 text-2xl font-bold tracking-tight">{f.t}</h3>
          <p className="mb-4.5 text-muted-foreground">{f.d}</p>
          <ul className="grid gap-2.5">
            {f.li.map(([b, rest]) => (
              <li key={b} className="flex gap-3 rounded-xl border bg-background/60 px-3.5 py-3 text-[14.5px] text-muted-foreground">
                <ArrowRight className="mt-0.5 size-4 shrink-0 text-primary" aria-hidden />
                <span>
                  <b className="text-foreground">{b}</b>
                  {rest}
                </span>
              </li>
            ))}
          </ul>
          <div className="mt-4.5 flex flex-wrap gap-2">
            {f.g.map(t => (
              <span key={t} className="rounded-full border bg-background/60 px-3 py-1 text-xs text-muted-foreground">{t}</span>
            ))}
          </div>
        </div>
      </div>
    </Reveal>
  );
}

function Node({ b, s, tone }: { b: string; s: ReactNode; tone?: 'brand' | 'accent' }) {
  return (
    <div className={cn('min-w-0 rounded-xl border bg-background/60 px-3 py-4 text-center text-[13px]', tone === 'brand' && 'border-primary bg-primary/10', tone === 'accent' && 'border-teal-500 bg-teal-500/10')}>
      <b className="mb-0.5 block text-sm">{b}</b>
      <small className="text-[11.5px] text-muted-foreground">{s}</small>
    </div>
  );
}

export function LandingPage() {
  const authed = useAuth(s => s.status === 'authenticated');
  return (
    <div className="relative min-h-dvh overflow-x-clip bg-background text-foreground">
      {/* backdrop: a faint grid fading out from the top, and two soft colour orbs */}
      <div
        aria-hidden
        className="pointer-events-none fixed inset-0 -z-0 [mask-image:radial-gradient(ellipse_at_50%_0%,#000_30%,transparent_75%)]"
        style={{ backgroundImage: 'linear-gradient(color-mix(in oklch, var(--primary) 8%, transparent) 1px, transparent 1px), linear-gradient(90deg, color-mix(in oklch, var(--primary) 8%, transparent) 1px, transparent 1px)', backgroundSize: '48px 48px' }}
      />
      <div aria-hidden className="pointer-events-none fixed -top-44 -left-28 size-[520px] rounded-full bg-primary opacity-20 blur-[90px]" />
      <div aria-hidden className="pointer-events-none fixed top-28 -right-40 size-[460px] rounded-full bg-violet-500 opacity-15 blur-[90px]" />

      <header className="sticky top-0 z-50 border-b bg-background/75 backdrop-blur-md">
        <div className="mx-auto flex h-16 max-w-[1160px] items-center justify-between gap-3 px-5">
          <a href="#top" aria-label="AutoML, back to top"><Logo /></a>
          <nav aria-label="Sections" className="hidden gap-6.5 text-sm text-muted-foreground lg:flex">
            {NAV.map(([h, l]) => <a key={h} href={h} className="hover:text-foreground">{l}</a>)}
          </nav>
          <div className="flex items-center gap-2">
            <ThemeToggle />
            <Button asChild size="sm" className={PRIMARY_BTN}>
              <Link to="/projects">{authed ? 'Open app' : 'Sign in'}</Link>
            </Button>
          </div>
        </div>
      </header>

      <main id="top" className="relative mx-auto max-w-[1160px] px-5">
        <section className="grid items-center gap-12 pt-13 pb-14 lg:grid-cols-[1.05fr_.95fr] lg:pt-21">
          <div className="min-w-0">
            <span className="inline-flex items-center gap-2 rounded-full border border-primary/35 bg-primary/14 px-3.5 py-1.5 text-[13px] text-primary">
              <span className="size-[7px] rounded-full bg-teal-400 shadow-[0_0_10px] shadow-teal-400 motion-safe:animate-pulse" />
              Runs 100% on your own computer
            </span>
            <h1 className="mt-5 mb-5 text-[clamp(38px,5.6vw,66px)] font-extrabold leading-[1.04] tracking-[-0.035em]">
              Build a machine-learning model <span className={GRAD_TEXT}>without writing code.</span>
            </h1>
            <p className="max-w-[540px] text-[clamp(16px,1.6vw,19px)] text-muted-foreground">
              Add your data, label it with AI help, train with one click, test the result and ship it behind an API key. Every step is checked so a model that looks good is also a model you can trust.
            </p>
            <div className="mt-7.5 mb-5.5 flex flex-wrap gap-3">
              <Button asChild size="lg" className={PRIMARY_BTN}>
                <Link to="/projects">Open live demo <ArrowRight /></Link>
              </Button>
              <Button asChild size="lg" variant="outline">
                <a href="#how">See how it works</a>
              </Button>
            </div>
            <div className="flex flex-wrap gap-5 text-[13px] text-muted-foreground">
              {['No cloud needed', 'Your data never leaves the laptop', 'Export to ONNX'].map(t => (
                <span key={t} className="inline-flex items-center gap-1"><Check className="size-3.5 text-teal-500" aria-hidden />{t}</span>
              ))}
            </div>
          </div>
          <HeroPipeline />
        </section>

        <div className="grid grid-cols-2 gap-3.5 md:grid-cols-4">
          {STATS.map(([v, l]) => (
            <div key={l} className="rounded-2xl border bg-card px-5 py-5.5 text-center">
              <b className={cn('block text-[34px] font-extrabold tracking-tight', GRAD_TEXT)}>{v}</b>
              <span className="text-[13px] text-muted-foreground">{l}</span>
            </div>
          ))}
        </div>

        <section id="how" className="scroll-mt-20 pt-23">
          <SectionHead eyebrow="The workflow" title="From raw photos to a live API in six steps" sub="Click a step to see what the platform does behind the scenes." />
          <WorkflowExplorer />
        </section>

        <section id="build" className="scroll-mt-20 pt-23">
          <SectionHead eyebrow="What you can build" title="One platform, many kinds of models" sub="Pick a task, bring your data. The platform chooses the right model family for it." />
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {BUILD.map(b => (
              <Reveal key={b.t} className="rounded-2xl border bg-card p-6.5 transition-[opacity,transform,border-color,box-shadow] hover:-translate-y-1 hover:border-primary hover:shadow-xl hover:shadow-primary/15 motion-reduce:hover:translate-y-0">
                <div className={cn('mb-4 grid size-[46px] place-items-center rounded-xl', b.color)}><b.icon className="size-[22px]" aria-hidden /></div>
                <h4 className="mb-1.5 text-lg font-semibold tracking-tight">{b.t}</h4>
                <p className="text-[14.5px] text-muted-foreground">{b.d}</p>
                <small className="mt-3.5 block text-[12.5px] text-muted-foreground/80">{b.m}</small>
              </Reveal>
            ))}
          </div>
        </section>

        <section id="under" className="scroll-mt-20 pt-23">
          <SectionHead eyebrow="Under the hood" title="We don't train from zero. We adapt." sub="Large pretrained models already know how to see. The platform keeps that knowledge and teaches it only what is new: your classes." />
          <div className="grid items-stretch gap-4.5 md:grid-cols-2">
            <Reveal className="min-w-0 rounded-2xl border bg-card p-6 sm:p-7">
              <h3 className="mb-1 text-xl font-bold">Transfer learning</h3>
              <p className="mb-4.5 text-[14.5px] text-muted-foreground">The old decision layer is swapped for a new one that matches your classes.</p>
              <div className="flex flex-wrap items-center gap-2 sm:flex-nowrap">
                <div className="min-w-0 flex-1 basis-full rounded-xl border border-primary/45 bg-primary/14 px-2 py-4 text-center text-[12.5px] font-semibold sm:basis-0">Backbone<small className="mt-0.5 block text-[11px] font-normal text-muted-foreground">edges · shapes · textures</small></div>
                <ArrowRight className="hidden size-4 shrink-0 text-muted-foreground sm:block" aria-hidden />
                <div className="min-w-0 flex-1 basis-full rounded-xl border bg-background/60 px-2 py-4 text-center text-[12.5px] font-semibold line-through opacity-55 sm:basis-0">Old head<small className="mt-0.5 block text-[11px] font-normal text-muted-foreground">1000 classes</small></div>
                <ArrowRight className="hidden size-4 shrink-0 text-muted-foreground sm:block" aria-hidden />
                <div className="min-w-0 flex-1 basis-full rounded-xl border border-teal-500 bg-teal-500/18 px-2 py-4 text-center text-[12.5px] font-semibold shadow-teal-500/50 motion-safe:animate-[landing-glow_2.4s_infinite] sm:basis-0">New head<small className="mt-0.5 block text-[11px] font-normal text-muted-foreground">your classes</small></div>
              </div>
              <div className="mt-4 flex flex-wrap gap-4 text-[12.5px] text-muted-foreground">
                <span className="inline-flex items-center gap-1.5"><i className="size-2.5 rounded-[3px] bg-primary" />Kept from pretraining</span>
                <span className="inline-flex items-center gap-1.5"><i className="size-2.5 rounded-[3px] bg-teal-500" />Trained on your data</span>
              </div>
            </Reveal>
            <Reveal className="min-w-0 rounded-2xl border bg-card p-6 sm:p-7">
              <h3 className="mb-1 text-xl font-bold">Auto search with Optuna</h3>
              <p className="mb-4.5 text-[14.5px] text-muted-foreground">The platform tries several settings and keeps the best one. You never tune a knob by hand.</p>
              <ul className="grid gap-2.5 text-[14.5px] text-muted-foreground">
                {['Learning rate, weight decay, architecture', 'Weak trials are stopped early', 'Memory check first, batch size shrinks on low RAM', 'Every trial recorded in MLflow'].map(t => (
                  <li key={t} className="flex gap-2.5"><span className="mt-2 size-1.5 shrink-0 rotate-45 bg-primary" aria-hidden />{t}</li>
                ))}
              </ul>
            </Reveal>
          </div>
          <Reveal className="mt-4.5 rounded-3xl border bg-card p-5 shadow-xl shadow-primary/5 sm:p-7.5">
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-5">
              <Node tone="brand" b="React app" s={<>JWT in memory<br />refresh via cookie</>} />
              <Node tone="brand" b="FastAPI" s="auth · roles · validation" />
              <Node b="Postgres" s="projects · labels · pgvector" />
              <Node b="Redis" s="job queue · live events" />
              <Node tone="accent" b="Celery workers" s="ingest · train · evaluate" />
            </div>
            <div className="flex items-center gap-2 py-3 text-center text-xs text-muted-foreground">
              <span className="h-px flex-1 bg-gradient-to-r from-transparent to-border" />
              <span className="max-w-[80%]">Train takes the heavy queue, one job at a time, so a 16 GB laptop stays responsive</span>
              <span className="h-px flex-1 bg-gradient-to-r from-border to-transparent" />
            </div>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
              <Node b="MinIO" s="photos, models, exports" />
              <Node b="MLflow" s="experiment tracking" />
              <Node b="SSE stream" s="live training progress in the browser" />
            </div>
          </Reveal>
        </section>

        <section id="trust" className="scroll-mt-20 pt-23">
          <SectionHead eyebrow="Built to be trusted" title="A high score is not enough. We check it is real." sub="The most common mistake in machine learning is a model that looks perfect and fails in real life. The platform guards against it." />
          <div className="grid gap-4 md:grid-cols-2">
            {TRUST.map(([b, s], i) => (
              <Reveal key={b} className="flex gap-3.5 rounded-2xl border bg-card p-5">
                <span className="grid size-[34px] shrink-0 place-items-center rounded-lg bg-teal-500/16 font-extrabold text-teal-600 dark:text-teal-400">{i + 1}</span>
                <div>
                  <b className="mb-0.5 block">{b}</b>
                  <span className="text-sm text-muted-foreground">{s}</span>
                </div>
              </Reveal>
            ))}
          </div>
        </section>

        <section id="deploy" className="scroll-mt-20 pt-23">
          <SectionHead eyebrow="Deploy" title="One key. One endpoint." sub="Create a key in the Deploy tab. The key alone decides which model answers." />
          <Reveal>
            <pre className="overflow-x-auto rounded-2xl border bg-[#050812] px-5.5 py-5 font-mono text-[13.5px] leading-relaxed text-[#cdd7f5]">
              <span className="text-[#6b7898]"># every deployed model is served here, rate limited to 60 requests per minute</span>
              {'\n'}
              <span className="text-[#a78bfa]">POST</span> http://127.0.0.1:8010/v1/predict
              {'\n'}
              <span className="text-[#a78bfa]">X-API-Key:</span> <span className="text-[#2dd4bf]">ak_&lt;prefix&gt;_&lt;secret&gt;</span> <span className="text-[#6b7898]"># shown once, stored only as a hash</span>
            </pre>
          </Reveal>
        </section>

        <section id="stack" className="scroll-mt-20 pt-23">
          <SectionHead eyebrow="Tech stack" title="Modern, open and fully local" />
          <Reveal className="flex flex-wrap gap-2.5">
            {STACK.map(([b, rest]) => (
              <span key={b} className="rounded-full border bg-card px-4 py-2 text-sm text-muted-foreground">
                <b className="font-semibold text-foreground">{b}</b>
                {rest}
              </span>
            ))}
          </Reveal>
        </section>

        <Reveal className="mt-24 rounded-3xl border border-primary/40 bg-gradient-to-br from-primary/25 via-card to-violet-500/20 px-4.5 py-10 text-center sm:px-7 sm:py-14.5">
          <h2 className="mb-3.5 text-[clamp(28px,3.8vw,44px)] font-extrabold leading-[1.1] tracking-tight">See it work, live.</h2>
          <p className="mx-auto mb-6.5 max-w-[640px] text-[17px] text-muted-foreground">Sign in with the demo account and create your first model in a few minutes.</p>
          <div className="flex flex-wrap justify-center gap-3">
            <Button asChild size="lg" className={PRIMARY_BTN}>
              <Link to="/projects">Open the app <ArrowRight /></Link>
            </Button>
            <Button asChild size="lg" variant="outline">
              <a href={GITHUB} target="_blank" rel="noopener noreferrer"><FolderGit2 /> View source on GitHub</a>
            </Button>
          </div>
        </Reveal>
      </main>

      <footer className="relative px-5 pt-11 pb-14 text-center text-[13.5px] text-muted-foreground">AutoML Platform · Final year project · Shivam Kumar Yadav</footer>
    </div>
  );
}
