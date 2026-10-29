import { http, HttpResponse } from 'msw';
import type { Membership } from '@/api/types';
import { db, save, uuid } from '../db';
import { authed, lag, problem, u } from '../util';

const pub = (userId: string) => {
  const { password: _pw, ...user } = db.users.find(x => x.id === userId)!;
  void _pw;
  return user;
};
const token = (userId: string) => ({ access_token: `mock.${userId}`, token_type: 'bearer', expires_in: 900, user: pub(userId) });

function sendVerification(userId: string) {
  const t = crypto.randomUUID();
  db.verifyTokens[t] = userId;
  const user = db.users.find(x => x.id === userId)!;
  db.mail.unshift({
    id: uuid(),
    to: user.email,
    subject: 'Verify your email',
    text: `Hi ${user.name}, confirm your email: ${location.origin}/verify-email?token=${t}`,
    sent_at: new Date().toISOString(),
  });
}

const attempts = new Map<string, number>();

export const authHandlers = [
  http.get(u('/auth/config'), () => HttpResponse.json({ google_enabled: true })),

  http.post(u('/auth/login'), async ({ request }) => {
    await lag();
    const { email, password } = (await request.json()) as { email: string; password: string };
    const key = email.trim().toLowerCase();
    const n = (attempts.get(key) ?? 0) + 1;
    attempts.set(key, n);
    if (n > 5) return HttpResponse.json({ title: 'Too many attempts', status: 429, code: 'too_many_attempts' }, { status: 429, headers: { 'Retry-After': '60' } });
    const user = db.users.find(x => x.email.toLowerCase() === key);
    if (!user || user.password !== password) return problem(401, 'Invalid email or password', { code: 'invalid_credentials' });
    if (!user.email_verified) return problem(403, 'Verify your email first', { code: 'email_not_verified', detail: 'Check your inbox for the verification link.' });
    attempts.delete(key);
    db.session = user.id;
    save();
    return HttpResponse.json(token(user.id));
  }),

  // ACC-1: no tokens and no cookie until the email is verified; the personal workspace is created now.
  http.post(u('/auth/register'), async ({ request }) => {
    await lag();
    const { email, password, name } = (await request.json()) as { email: string; password: string; name: string };
    if (db.users.some(x => x.email.toLowerCase() === email.trim().toLowerCase()))
      return problem(422, 'Validation failed', { code: 'validation_error', errors: [{ loc: ['body', 'email'], msg: 'An account with this email already exists' }] });
    if (password.length < 8) return problem(422, 'Validation failed', { errors: [{ loc: ['body', 'password'], msg: 'Use at least 8 characters' }] });
    const user = { id: uuid(), email: email.trim(), name: name.trim(), email_verified: false, password };
    const org = { id: uuid(), name: `${user.name.split(' ')[0]}'s workspace`, slug: user.name.toLowerCase().replace(/\W+/g, '-'), plan: 'free', region: 'ap-south-1', personal: true, created_at: new Date().toISOString() };
    db.users.push(user);
    db.orgs.push(org);
    db.memberships.push({ user_id: user.id, role: 'owner', organization_id: org.id });
    sendVerification(user.id);
    save();
    return HttpResponse.json({ user: pub(user.id), verification_required: true }, { status: 201 });
  }),

  http.post(u('/auth/verify-email'), async ({ request }) => {
    await lag();
    const { token: t } = (await request.json()) as { token: string };
    const userId = db.verifyTokens[t];
    if (!userId) return problem(400, 'This link is invalid or has expired', { code: 'invalid_token' });
    delete db.verifyTokens[t];
    db.users.find(x => x.id === userId)!.email_verified = true;
    save();
    return HttpResponse.json(pub(userId));
  }),

  http.post(u('/auth/resend-verification'), async ({ request }) => {
    await lag();
    const { email } = (await request.json()) as { email: string };
    const user = db.users.find(x => x.email.toLowerCase() === email.trim().toLowerCase());
    // Always 202 so the endpoint does not reveal which emails exist.
    if (user && !user.email_verified) {
      sendVerification(user.id);
      save();
    }
    return new HttpResponse(null, { status: 202 });
  }),

  http.get(u('/_dev/mail'), () => HttpResponse.json(db.mail)),

  http.post(u('/auth/refresh'), async ({ request }) => {
    if (request.headers.get('X-CSRF') !== '1') return problem(403, 'CSRF header missing');
    await lag();
    if (!db.session) return problem(401, 'No session', { code: 'refresh_invalid' });
    return HttpResponse.json(token(db.session));
  }),

  http.post(u('/auth/logout'), () => {
    db.session = null;
    save();
    return new HttpResponse(null, { status: 204 });
  }),

  // Real backend redirects to Google; the mock pretends consent succeeded.
  http.get(u('/auth/google/start'), () => {
    db.session = db.users[0]!.id;
    save();
    return HttpResponse.redirect(`${location.origin}/projects`, 302);
  }),

  http.get(u('/me'), async ({ request }) => {
    const user = authed(request);
    if (!user) return problem(401, 'Not authenticated');
    await lag();
    const memberships: Membership[] = db.memberships
      .filter(m => m.user_id === user.id)
      .map(m => {
        const o = db.orgs.find(x => x.id === m.organization_id)!;
        return { organization_id: o.id, organization_name: o.name, slug: o.slug, personal: o.personal, plan: o.plan, role: m.role };
      });
    return HttpResponse.json({ user: pub(user.id), memberships });
  }),
];
