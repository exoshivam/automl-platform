import { http, HttpResponse } from 'msw';
import { db, save, uuid } from '../db';
import { authed, lag, problem, u } from '../util';

export const orgHandlers = [
  http.get(u('/orgs'), ({ request }) => {
    const user = authed(request);
    if (!user) return problem(401, 'Not authenticated');
    const ids = db.memberships.filter(m => m.user_id === user.id).map(m => m.organization_id);
    const role = (id: string) => db.memberships.find(m => m.user_id === user.id && m.organization_id === id)?.role;
    return HttpResponse.json({ items: db.orgs.filter(o => ids.includes(o.id)).map(o => ({ ...o, role: role(o.id) })), next_cursor: null });
  }),
  http.post(u('/orgs'), async ({ request }) => {
    const user = authed(request);
    if (!user) return problem(401, 'Not authenticated');
    await lag();
    const { name } = (await request.json()) as { name: string };
    if (!name?.trim()) return problem(422, 'Validation failed', { errors: [{ loc: ['body', 'name'], msg: 'Name is required' }] });
    const org = { id: uuid(), name: name.trim(), slug: name.trim().toLowerCase().replace(/\W+/g, '-'), plan: 'free', region: 'ap-south-1', personal: false, created_at: new Date().toISOString() };
    db.orgs.push(org);
    db.memberships.push({ user_id: user.id, role: 'owner', organization_id: org.id });
    save();
    return HttpResponse.json({ ...org, role: 'owner' }, { status: 201 });
  }),
];
