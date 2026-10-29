import { CheckCircle2 } from 'lucide-react';
import { API_URL } from '@/lib/env';
import { Logo } from '@/features/shell/logo';
import { ThemeToggle } from '@/features/shell/theme-toggle';

const POINTS = [
  'Pick a category, add your data, get an evaluated model in under 30 minutes',
  'Capture training photos straight from your phone camera',
  'Export to ONNX, TFLite, Core ML or TF.js, or deploy an API in one click',
];

export function AuthLayout({ title, subtitle, children }: { title: string; subtitle: string; children: React.ReactNode }) {
  return (
    <div className="grid min-h-dvh lg:grid-cols-2">
      <section className="relative hidden overflow-hidden bg-primary p-10 text-primary-foreground lg:flex lg:flex-col">
        <div className="absolute -right-24 -top-24 size-96 rounded-full bg-white/10 blur-3xl" aria-hidden />
        <div className="absolute -bottom-32 -left-16 size-96 rounded-full bg-black/10 blur-3xl" aria-hidden />
        <Logo className="relative text-lg [&_rect]:fill-white/20" />
        <div className="relative mt-auto max-w-md">
          <h2 className="text-3xl font-semibold leading-tight">Your data. Your model. Deployed anywhere.</h2>
          <ul className="mt-6 grid gap-3 text-sm/6 opacity-95">
            {POINTS.map(p => (
              <li key={p} className="flex gap-2">
                <CheckCircle2 className="mt-0.5 size-4 shrink-0" />
                {p}
              </li>
            ))}
          </ul>
        </div>
      </section>
      <section className="flex flex-col p-6 sm:p-10">
        <div className="flex items-center justify-between">
          <Logo className="lg:invisible" />
          <ThemeToggle />
        </div>
        <div className="mx-auto my-auto w-full max-w-sm py-10">
          <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
          <p className="mt-1 text-sm text-muted-foreground">{subtitle}</p>
          <div className="mt-8">{children}</div>
        </div>
      </section>
    </div>
  );
}

/** Google sign-in is Phase 2 for the local build; flip GOOGLE_LOGIN to bring it back (it also needs google_enabled). */
const GOOGLE_LOGIN = false;

export function GoogleButton({ enabled }: { enabled: boolean | undefined }) {
  if (!GOOGLE_LOGIN || !enabled) return null;
  return (
    <>
      <a
        href={`${API_URL}/auth/google/start`}
        className="inline-flex h-10 w-full items-center justify-center gap-2 rounded-md border border-input bg-card text-sm font-medium shadow-xs transition-colors hover:bg-accent"
      >
        <svg viewBox="0 0 24 24" className="size-4" aria-hidden>
          <path fill="#4285F4" d="M23.5 12.3c0-.8-.1-1.6-.2-2.3H12v4.5h6.5a5.5 5.5 0 0 1-2.4 3.6v3h3.9c2.3-2.1 3.5-5.2 3.5-8.8z" />
          <path fill="#34A853" d="M12 24c3.2 0 6-1.1 8-2.9l-3.9-3c-1.1.7-2.5 1.2-4.1 1.2-3.1 0-5.8-2.1-6.7-5H1.3v3.1A12 12 0 0 0 12 24z" />
          <path fill="#FBBC05" d="M5.3 14.3a7.2 7.2 0 0 1 0-4.6V6.6h-4a12 12 0 0 0 0 10.8l4-3.1z" />
          <path fill="#EA4335" d="M12 4.8c1.8 0 3.3.6 4.6 1.8l3.4-3.4A11.5 11.5 0 0 0 12 0 12 12 0 0 0 1.3 6.6l4 3.1c.9-2.9 3.6-4.9 6.7-4.9z" />
        </svg>
        Continue with Google
      </a>
      <div className="my-6 flex items-center gap-3 text-xs text-muted-foreground">
        <span className="h-px flex-1 bg-border" /> or <span className="h-px flex-1 bg-border" />
      </div>
    </>
  );
}
