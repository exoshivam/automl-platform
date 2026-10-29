import { zodResolver } from '@hookform/resolvers/zod';
import { Link, useNavigate, useSearch } from '@tanstack/react-router';
import { useEffect, useRef } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { AlertCircle, CheckCircle2, Loader2, MailCheck } from 'lucide-react';
import { useState } from 'react';
import { useForm, type UseFormSetError, type FieldValues, type Path } from 'react-hook-form';
import { z } from 'zod';
import { useAuthConfig, useLogin, useRegister, useResendVerification, useVerifyEmail } from '@/api/hooks';
import type { AuthResponse } from '@/api/types';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Field } from '@/components/ui/label';
import { ApiError } from '@/lib/api';
import { useAuth } from '@/lib/auth';
import { API_URL, isMocked } from '@/lib/env';
import { AuthLayout, GoogleButton } from './auth-layout';

const loginSchema = z.object({
  email: z.email('Enter a valid email'),
  password: z.string().min(1, 'Enter your password'),
});
const registerSchema = z.object({
  name: z.string().trim().min(2, 'Enter your name'),
  email: z.email('Enter a valid email'),
  password: z.string().min(8, 'Use at least 8 characters'),
});

/** Put 422 field errors from the server onto the form; return a form-level message otherwise. */
function applyServerErrors<T extends FieldValues>(err: unknown, setError: UseFormSetError<T>, fields: string[]): string | undefined {
  if (!(err instanceof ApiError)) return 'Network error. Check your connection and try again.';
  const fe = err.fieldErrors();
  let mapped = false;
  for (const f of fields)
    if (fe[f]) {
      setError(f as Path<T>, { message: fe[f] });
      mapped = true;
    }
  return mapped ? undefined : err.message;
}

function useFinishAuth() {
  const setSession = useAuth(s => s.setSession);
  const navigate = useNavigate();
  const qc = useQueryClient();
  const search = useSearch({ strict: false }) as { redirect?: string };
  return (res: AuthResponse) => {
    setSession(res.access_token, res.user);
    qc.clear();
    navigate({ to: search.redirect?.startsWith('/') ? search.redirect : '/projects' });
  };
}

function FormError({ message }: { message?: string }) {
  if (!message) return null;
  return (
    <div role="alert" className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/8 p-3 text-sm text-destructive">
      <AlertCircle className="mt-0.5 size-4 shrink-0" />
      {message}
    </div>
  );
}

export function LoginPage() {
  const config = useAuthConfig();
  const login = useLogin();
  const finish = useFinishAuth();
  const form = useForm({ resolver: zodResolver(loginSchema), defaultValues: { email: '', password: '' } });
  const { errors } = form.formState;
  const [formError, setFormError] = useState<string>();
  const [unverified, setUnverified] = useState<string | null>(null);

  return (
    <AuthLayout title="Welcome back" subtitle="Log in to continue to your workspace.">
      <GoogleButton enabled={config.data?.google_enabled} />
      <form className="grid gap-4" noValidate onSubmit={form.handleSubmit(v => {
          setFormError(undefined);
          setUnverified(null);
          login.mutate(v, {
            onSuccess: finish,
            onError: e => {
              if (e instanceof ApiError && e.code === 'email_not_verified') return setUnverified(v.email);
              if (e instanceof ApiError && e.status === 429)
                return setFormError(`Too many attempts. Try again in ${e.retryAfter ? `${e.retryAfter} seconds` : 'a minute'}.`);
              setFormError(applyServerErrors(e, form.setError, ['email', 'password']));
            },
          });
        })}>
        <FormError message={formError} />
        {unverified && <UnverifiedNotice email={unverified} />}
        <Field id="email" label="Email" error={errors.email?.message}>
          <Input id="email" type="email" autoComplete="email" aria-invalid={!!errors.email} {...form.register('email')} />
        </Field>
        <Field id="password" label="Password" error={errors.password?.message}>
          <Input id="password" type="password" autoComplete="current-password" aria-invalid={!!errors.password} {...form.register('password')} />
        </Field>
        <Button type="submit" size="lg" disabled={login.isPending}>
          {login.isPending ? 'Logging in…' : 'Log in'}
        </Button>
      </form>
      <p className="mt-6 text-center text-sm text-muted-foreground">
        New here?{' '}
        <Link to="/register" className="font-medium text-primary hover:underline">
          Create an account
        </Link>
      </p>
      {isMocked('auth') && <p className="mt-8 rounded-md bg-muted p-3 text-xs text-muted-foreground">Mock login: demo@automl.dev / demo1234</p>}
    </AuthLayout>
  );
}

export function RegisterPage() {
  const config = useAuthConfig();
  const register = useRegister();
  const [sentTo, setSentTo] = useState<string | null>(null);
  const form = useForm({ resolver: zodResolver(registerSchema), defaultValues: { name: '', email: '', password: '' } });
  const { errors } = form.formState;
  const [formError, setFormError] = useState<string>();

  if (sentTo) return <CheckEmail email={sentTo} />;
  return (
    <AuthLayout title="Create your account" subtitle="Train your own models from your own data.">
      <GoogleButton enabled={config.data?.google_enabled} />
      <form className="grid gap-4" noValidate onSubmit={form.handleSubmit(v => {
          setFormError(undefined);
          register.mutate(v, { onSuccess: res => setSentTo(res.user.email), onError: e => setFormError(applyServerErrors(e, form.setError, ['name', 'email', 'password'])) });
        })}>
        <FormError message={formError} />
        <Field id="name" label="Name" error={errors.name?.message}>
          <Input id="name" autoComplete="name" aria-invalid={!!errors.name} {...form.register('name')} />
        </Field>
        <Field id="email" label="Work email" error={errors.email?.message}>
          <Input id="email" type="email" autoComplete="email" aria-invalid={!!errors.email} {...form.register('email')} />
        </Field>
        <Field id="password" label="Password" hint="At least 8 characters." error={errors.password?.message}>
          <Input id="password" type="password" autoComplete="new-password" aria-invalid={!!errors.password} {...form.register('password')} />
        </Field>
        <Button type="submit" size="lg" disabled={register.isPending}>
          {register.isPending ? 'Creating account…' : 'Create account'}
        </Button>
        <p className="text-xs text-muted-foreground">We send a link to verify your email. Your data is only ever used to train your own models.</p>
      </form>
      <p className="mt-6 text-center text-sm text-muted-foreground">
        Already have an account?{' '}
        <Link to="/login" className="font-medium text-primary hover:underline">
          Log in
        </Link>
      </p>
    </AuthLayout>
  );
}

function ResendButton({ email }: { email: string }) {
  const resend = useResendVerification();
  return (
    <Button type="button" variant="outline" size="sm" disabled={resend.isPending || resend.isSuccess} onClick={() => resend.mutate(email)}>
      {resend.isSuccess ? 'Sent. Check your inbox' : resend.isPending ? 'Sending…' : 'Resend the email'}
    </Button>
  );
}

/** Dev only: the backend (and the mocks) expose sent mail at /_dev/mail; show the newest link for this address. */
function DevMailLink({ email }: { email: string }) {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!import.meta.env.DEV) return;
    let stop = false;
    const look = async () => {
      try {
        const res = await fetch(`${API_URL}/_dev/mail`);
        const mails = (await res.json()) as unknown[];
        const mine = mails.find(m => JSON.stringify(m).toLowerCase().includes(email.toLowerCase()));
        const link = JSON.stringify(mine ?? '').match(/https?:\/\/[^\s"\\]*verify-email\?token=[\w.%-]+/);
        if (!stop) setUrl(link ? link[0] : null);
      } catch {
        /* no dev mail endpoint */
      }
    };
    void look();
    const t = setInterval(look, 3000);
    return () => {
      stop = true;
      clearInterval(t);
    };
  }, [email]);
  if (!url) return null;
  const path = url.replace(/^https?:\/\/[^/]+/, '');
  return (
    <a href={path} className="block rounded-md bg-muted p-3 text-xs text-muted-foreground hover:text-foreground">
      Dev: open the verification email link
    </a>
  );
}

function UnverifiedNotice({ email }: { email: string }) {
  return (
    <div role="alert" className="grid gap-3 rounded-md border border-warning/40 bg-warning/10 p-3 text-sm">
      <p>
        Please verify <b>{email}</b> before logging in. We sent you a link when you signed up.
      </p>
      <div className="flex flex-wrap gap-2">
        <ResendButton email={email} />
      </div>
      <DevMailLink email={email} />
    </div>
  );
}

function CheckEmail({ email }: { email: string }) {
  return (
    <AuthLayout title="Check your email" subtitle="One more step before you can log in.">
      <div className="grid gap-5">
        <span className="grid size-12 place-items-center rounded-full bg-primary/12 text-primary">
          <MailCheck className="size-6" />
        </span>
        <p className="text-sm">
          We sent a verification link to <b>{email}</b>. Open it to activate your account and your free workspace.
        </p>
        <div className="flex flex-wrap items-center gap-3">
          <ResendButton email={email} />
          <Link to="/login" className="text-sm font-medium text-primary hover:underline">
            Back to log in
          </Link>
        </div>
        <DevMailLink email={email} />
      </div>
    </AuthLayout>
  );
}

/** /verify-email?token=… from the email link. */
export function VerifyEmailPage() {
  const { token } = useSearch({ strict: false }) as { token?: string };
  const verify = useVerifyEmail();
  const [email, setEmail] = useState('');
  // Tokens are single-use: submit each one exactly once (StrictMode re-runs effects in dev).
  const sent = useRef<string | null>(null);
  useEffect(() => {
    if (!token || sent.current === token) return;
    sent.current = token;
    verify.mutate(token);
  }, [token, verify]);

  return (
    <AuthLayout title="Verify your email" subtitle="Confirming your account.">
      {!token ? (
        <p className="text-sm text-destructive">This link is missing its token. Open the link from the email again.</p>
      ) : verify.isPending || verify.isIdle ? (
        <p className="flex items-center gap-2 text-sm text-muted-foreground" role="status">
          <Loader2 className="size-4 animate-spin" /> Verifying…
        </p>
      ) : verify.isSuccess ? (
        <div className="grid gap-4">
          <p className="flex items-center gap-2 text-sm text-success" role="status">
            <CheckCircle2 className="size-5" /> {verify.data.email} is verified.
          </p>
          <Button asChild size="lg">
            <Link to="/login">Log in</Link>
          </Button>
        </div>
      ) : (
        <div className="grid gap-4">
          <FormError message="This link is invalid or has already been used. Request a new one below." />
          <Field id="resend-email" label="Your email">
            <Input id="resend-email" type="email" value={email} onChange={e => setEmail(e.target.value)} />
          </Field>
          {email.includes('@') && <ResendButton email={email} />}
        </div>
      )}
    </AuthLayout>
  );
}
