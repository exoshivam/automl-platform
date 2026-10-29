#!/usr/bin/env python3
"""Acceptance run for ACC-1/2/3/4/6 and NFR-SEC3 (org paths) from CHECKLIST.md.

Standard library only. Creates its own users/orgs with a unique run id (never touches demo data).
    python3 tests/acceptance/acc_run.py            # AUTOML_API overrides http://127.0.0.1:8010/api/v1
Exit code 0 = all checks pass. Full request/response log: tests/acceptance/.out/acc_<run>.json
"""
import base64
import http.client
import json
import os
import re
import sys
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

API = os.environ.get("AUTOML_API", "http://127.0.0.1:8010/api/v1")
U = urllib.parse.urlsplit(API)
RUN = datetime.now().strftime("%m%d%H%M%S")
PW = "Str0ng!pass-" + RUN
OUT = Path(__file__).resolve().parent / ".out"
OUT.mkdir(exist_ok=True)
LOG, RESULTS = [], []
STATE = {}
PROBLEM_KEYS = {"type", "title", "status", "detail", "instance", "code"}


class R:
    def __init__(self, status, headers, body, req):
        self.status, self.headers, self.raw, self.req = status, headers, body, req
        try:
            self.json = json.loads(body) if body else None
        except ValueError:
            self.json = None

    def header(self, name):
        return [v for k, v in self.headers if k.lower() == name.lower()]

    @property
    def code(self):
        return (self.json or {}).get("code") if isinstance(self.json, dict) else None

    def brief(self):
        return {"request": self.req, "status": self.status, "body": self.raw[:800]}


def call(method, path, body=None, tok=None, key=None, xkey=None, cookie=None, csrf=False, headers=None):
    h = {"Accept": "application/json", **(headers or {})}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    if key:
        h["Authorization"] = f"Bearer {key}"
    if xkey:
        h["X-API-Key"] = xkey
    if cookie:
        h["Cookie"] = f"refresh_token={cookie}"
    if csrf:
        h["X-CSRF"] = "1"
    conn = http.client.HTTPConnection(U.hostname, U.port, timeout=15)
    full = U.path + path
    conn.request(method, full, body=data, headers=h)
    resp = conn.getresponse()
    raw = resp.read().decode("utf-8", "replace")
    conn.close()
    shown = {k: ("<redacted>" if k in ("Authorization", "X-API-Key", "Cookie") else v) for k, v in h.items()}
    r = R(resp.status, resp.getheaders(), raw, {"method": method, "path": full, "headers": shown, "body": _redact(body)})
    LOG.append({"req": r.req, "status": r.status, "headers": [(k, _scrub(v)) for k, v in r.headers], "body": _scrub(raw[:4000])})
    return r


_SECRETS = [(re.compile(r'"(key|access_token|token)"\s*:\s*"[^"]*"'), r'"\1":"<redacted>"'),
            (re.compile(r"ak_[0-9a-f]{8}_[A-Za-z0-9_\-]+"), "ak_<redacted>"), (re.compile(r"refresh_token=[^;\s]+"), "refresh_token=<redacted>")]


def _scrub(text):
    """Keep credentials out of the .out log: API keys, access/verify tokens, refresh cookies."""
    for pattern, repl in _SECRETS:
        text = pattern.sub(repl, text)
    return text


def _redact(body):
    if isinstance(body, dict):
        return {k: ("<redacted>" if k in ("password", "token") else v) for k, v in body.items()}
    return body


def check(cid, ok, why="", r=None):
    RESULTS.append({"id": cid, "pass": bool(ok), "why": why, **({"evidence": r.brief()} if (r is not None and not ok) else {})})
    print(f"{'PASS' if ok else 'FAIL'} {cid}{'' if ok else ' -- ' + why}")
    return ok


def problem_ok(r):
    ct = ";".join(r.header("content-type"))
    return "application/problem+json" in ct and isinstance(r.json, dict) and PROBLEM_KEYS <= set(r.json)


def expect(cid, r, status, code=None):
    ok = r.status == status and (code is None or r.code == code) and (status < 400 or problem_ok(r))
    why = f"want {status}{' ' + code if code else ''} problem+json, got {r.status} {r.code}"
    return check(cid, ok, why, r)


def refresh_cookie(r):
    for v in r.header("set-cookie"):
        m = re.match(r"refresh_token=([^;]*)", v)
        if m:
            return m.group(1), v
    return None, None


def jwt_claims(tok):
    p = tok.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(p + "=" * (-len(p) % 4)))


def mail_token(email):
    r = call("GET", "/_dev/mail")
    for m in r.json or []:
        if m.get("to", "").lower() == email.lower():
            t = re.search(r"Token:\s*(\S+)", m.get("body", ""))
            if t:
                return t.group(1)
    return None


def user(tag, verify=True):
    email = f"acc-{RUN}-{tag}@test.local"
    r = call("POST", "/auth/register", {"email": email, "password": PW, "name": f"Acc {tag}"})
    assert r.status == 201, r.brief()
    if verify:
        t = mail_token(email)
        assert t, f"no verification mail for {email}"
        v = call("POST", "/auth/verify-email", {"token": t})
        assert v.status in (200, 204), v.brief()
    return email, r


def login(email):
    r = call("POST", "/auth/login", {"email": email, "password": PW})
    assert r.status == 200, r.brief()
    c, _ = refresh_cookie(r)
    return r.json["access_token"], c, r


def personal_org(tok):
    me = call("GET", "/me", tok=tok).json
    return next(m["organization_id"] for m in me["memberships"] if m["personal"])


def main():
    h = call("GET", "/health")
    if h.status != 200:
        sys.exit(f"API not healthy at {API}: {h.status} {h.raw[:200]}")
    print(f"run {RUN} against {API}")

    # ---------- ACC-1 ----------
    a_email, reg = user("a", verify=False)
    STATE["a_email"] = a_email
    check("ACC-1 register 201 no tokens", reg.json.get("verification_required") is True and "access_token" not in reg.raw and not refresh_cookie(reg)[0]
          and reg.json["user"]["email_verified"] is False, "register response shape", reg)
    r = call("POST", "/auth/login", {"email": a_email, "password": PW})
    expect("ACC-1c login before verify -> 403 email_not_verified", r, 403, "email_not_verified")
    check("ACC-1c no cookie before verify", not refresh_cookie(r)[0], "cookie set", r)
    t = mail_token(a_email)
    check("ACC-1 verification mail has Token line", bool(t), "no Token: line in /_dev/mail")
    v = call("POST", "/auth/verify-email", {"token": t})
    check("ACC-1 verify-email -> 2xx", v.status in (200, 204), f"got {v.status} {v.code}", v)
    v = call("POST", "/auth/verify-email", {"token": "nope"})
    check("ACC-1 verify-email bogus token -> 4xx problem", 400 <= v.status < 500 and problem_ok(v), f"got {v.status}", v)
    for variant in (a_email.upper(), a_email.split("@")[0] + "@TEST.local"):
        expect(f"ACC-1a duplicate {variant}", call("POST", "/auth/register", {"email": variant, "password": PW}), 409)
    mixed = f"Mixed{RUN}@TEST.local"
    r = call("POST", "/auth/register", {"email": mixed, "password": PW})
    check("ACC-1a domain lowercased on register", r.status == 201 and r.json["user"]["email"] == f"Mixed{RUN}@test.local", f"stored {(r.json or {}).get('user', {}).get('email')}", r)
    for name, body in [("short password", {"email": f"acc-{RUN}-x@test.local", "password": "123456789"}),
                       ("long password", {"email": f"acc-{RUN}-y@test.local", "password": "x" * 257}),
                       ("missing email", {"password": PW}), ("malformed email", {"email": "not-an-email", "password": PW})]:
        r = call("POST", "/auth/register", body)
        ok = r.status == 422 and problem_ok(r) and isinstance(r.json.get("errors"), list) and r.json["errors"]
        check(f"ACC-1b {name} -> 422 with errors[]", ok, f"got {r.status}", r)

    # rate limit on a dedicated user so A is never locked out
    rl_email, _ = user("rl")
    bodies = []
    for i in range(1, 7):
        r = call("POST", "/auth/login", {"email": rl_email, "password": "wrong-password-" + str(i)})
        bodies.append(r)
    strip = lambda r: {k: v for k, v in (r.json or {}).items() if k != "instance"}
    check("ACC-1d attempts 1-5 -> 401 invalid_credentials", all(b.status == 401 and b.code == "invalid_credentials" and problem_ok(b) for b in bodies[:5]),
          f"statuses {[b.status for b in bodies[:5]]}", bodies[0])
    check("ACC-1d attempts 1-5 identical bodies", len({json.dumps(strip(b), sort_keys=True) for b in bodies[:5]}) == 1, "bodies differ", bodies[1])
    unk = call("POST", "/auth/login", {"email": f"nobody-{RUN}@test.local", "password": "whatever-123"})
    check("ACC-1d unknown email body == wrong password body", unk.status == 401 and strip(unk) == strip(bodies[0]), f"{unk.status} {strip(unk)} vs {strip(bodies[0])}", unk)
    check("ACC-1d attempt 6 -> 429 too_many_attempts + Retry-After", bodies[5].status == 429 and bodies[5].code == "too_many_attempts" and bodies[5].header("retry-after"),
          f"got {bodies[5].status} {bodies[5].code} retry-after={bodies[5].header('retry-after')}", bodies[5])
    r = call("POST", "/auth/login", {"email": rl_email, "password": PW})
    check("ACC-1d locked: correct password also refused while limited", r.status == 429, f"got {r.status} (record: lockout applies to correct password?)", r)

    r = call("GET", "/auth/google/start")
    if r.status == 404 and r.code == "google_disabled":  # Google sign-in deferred to Phase 2 (user scope decision 2026-10-22)
        RESULTS.append({"id": "ACC-1e/1f Google OAuth", "pass": True, "skipped": True, "why": "deferred: Phase 2 (google_disabled in this env)"})
        print("SKIP ACC-1e/1f Google OAuth -- deferred: Phase 2")
        return run_rest()
    loc = (r.header("location") or [""])[0]
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(loc).query)
    ok = r.status in (302, 303, 307) and "accounts.google.com" in loc and q.get("response_type") == ["code"] and "openid" in " ".join(q.get("scope", [])) \
        and "email" in " ".join(q.get("scope", [])) and q.get("state") and (q.get("nonce") or q.get("code_challenge"))
    check("ACC-1e google start redirect params", ok, f"{r.status} {loc[:300]}", r)
    r = call("GET", "/auth/google/callback?code=x&state=forged")
    check("ACC-1f google callback forged state -> 4xx, no cookie", 400 <= r.status < 500 and not refresh_cookie(r)[0], f"got {r.status}", r)

    return run_rest()


def run_rest():
    a_email = STATE["a_email"]
    # ---------- ACC-2 ----------
    tok_a, c1, lr = login(a_email)
    cl = jwt_claims(tok_a)
    check("ACC-2a access token TTL 900s", cl.get("exp", 0) - cl.get("iat", 0) == 900 and lr.json.get("expires_in") == 900, f"exp-iat={cl.get('exp', 0) - cl.get('iat', 0)} expires_in={lr.json.get('expires_in')}")
    check("ACC-2a no secrets in JWT", "sub" in cl and not any(k in json.dumps(cl).lower() for k in ("password", "hash", "secret")), f"claims {sorted(cl)}")
    _, sc = refresh_cookie(lr)
    attrs = [a.strip().lower() for a in sc.split(";")]
    ok = "httponly" in attrs and "samesite=lax" in attrs and "path=/api/v1/auth" in attrs and "max-age=2592000" in attrs
    check("ACC-2b cookie HttpOnly; SameSite=lax; Path=/api/v1/auth; Max-Age=2592000", ok, sc)
    check("ACC-2b Secure absent in dev (expected; prod must set it)", "secure" not in attrs, sc)
    r = call("POST", "/auth/refresh", cookie=c1)
    expect("ACC-2c refresh without X-CSRF -> 403 csrf_header_missing", r, 403, "csrf_header_missing")
    r = call("POST", "/auth/refresh", cookie=c1, csrf=True)
    c2, _ = refresh_cookie(r)
    check("ACC-2c refresh rotates", r.status == 200 and r.json.get("access_token") and c2 and c2 != c1, f"got {r.status}", r)
    r = call("POST", "/auth/refresh", cookie=c1, csrf=True)
    expect("ACC-2d reuse old refresh -> 401 refresh_reuse", r, 401, "refresh_reuse")
    r = call("POST", "/auth/refresh", cookie=c2, csrf=True)
    expect("ACC-2d sibling revoked -> 401 session_expired", r, 401, "session_expired")
    tok_a, c3, _ = login(a_email)
    r = call("POST", "/auth/logout", cookie=c3, csrf=True)
    check("ACC-2e logout 204", r.status == 204, f"got {r.status}", r)
    expect("ACC-2e refresh after logout -> 401", call("POST", "/auth/refresh", cookie=c3, csrf=True), 401)
    tok_a, _, _ = login(a_email)
    head, payload, sig = tok_a.split(".")
    bad = f"{head}.{payload}.{sig[:-2] + ('AA' if sig[-2:] != 'AA' else 'BB')}"
    expect("ACC-2f tampered signature -> 401", call("GET", "/me", tok=bad), 401)
    forged = base64.urlsafe_b64encode(json.dumps({**cl, "exp": int(time.time()) - 10}).encode()).decode().rstrip("=")
    expect("ACC-2f forged expired payload (old sig) -> 401", call("GET", "/me", tok=f"{head}.{forged}.{sig}"), 401)
    check("ACC-2f real expiry via TTL", True, "SKIPPED here: needs a server with ACCESS_TOKEN_TTL_SECONDS=5; backend pytest covers it")
    RESULTS[-1]["skipped"] = True

    # ---------- ACC-3 ----------
    me = call("GET", "/me", tok=tok_a)
    ms = (me.json or {}).get("memberships", [])
    check("ACC-3a personal org auto-created, owner", me.status == 200 and len(ms) == 1 and ms[0]["personal"] and ms[0]["role"] == "owner", f"memberships {ms}", me)
    org_a = ms[0]["organization_id"]
    expect("ACC-3c delete personal org -> 409 personal_workspace", call("DELETE", f"/orgs/{org_a}", tok=tok_a), 409, "personal_workspace")
    t1 = call("POST", "/orgs", {"name": f"Acc Team {RUN}"}, tok=tok_a)
    t2 = call("POST", "/orgs", {"name": f"Acc Team {RUN}"}, tok=tok_a)
    check("ACC-3b create org, owner", t1.status == 201 and t1.json.get("role") == "owner" and not t1.json.get("personal"), f"got {t1.status}", t1)
    check("ACC-3b same name -> distinct slug or 409", t2.status == 409 or (t2.status == 201 and t2.json["slug"] != t1.json["slug"]), f"{t2.status} {(t2.json or {}).get('slug')} vs {t1.json.get('slug')}", t2)
    org_t = t1.json["id"]

    # ---------- ACC-4 role matrix (team org) ----------
    roles = {}
    for role in ("admin", "member", "viewer"):
        email, _ = user(role)
        r = call("POST", f"/orgs/{org_t}/members", {"email": email, "role": role}, tok=tok_a)
        expect(f"ACC-4 owner adds {role} -> 201", r, 201)
        roles[role] = (email, login(email)[0], r.json.get("user_id") if r.json else None)
    roles["owner"] = (a_email, tok_a, jwt_claims(tok_a)["sub"])
    extra_email, _ = user("extra")
    unverified_email, _ = user("unverified", verify=False)
    expect("ACC-4 add existing member -> 409", call("POST", f"/orgs/{org_t}/members", {"email": roles["member"][0], "role": "member"}, tok=tok_a), 409)
    expect("ACC-4 add unknown email -> 404 user_not_found", call("POST", f"/orgs/{org_t}/members", {"email": f"ghost-{RUN}@test.local", "role": "member"}, tok=tok_a), 404, "user_not_found")
    expect("ACC-4 add unverified email -> 404 user_not_found", call("POST", f"/orgs/{org_t}/members", {"email": unverified_email, "role": "member"}, tok=tok_a), 404, "user_not_found")

    def matrix(cid, method, path, allowed, ok_status=200, body=None):
        for role in ("owner", "admin", "member", "viewer"):
            r = call(method, path, body, tok=roles[role][1])
            want = ok_status if role in allowed else 403
            if role in allowed:
                check(f"{cid} [{role}] -> {want}", r.status == want or (want == 200 and r.status in (200, 201, 204)), f"got {r.status} {r.code}", r)
            else:
                expect(f"{cid} [{role}] -> 403", r, 403)

    every = ("owner", "admin", "member", "viewer")
    oa = ("owner", "admin")
    matrix("ACC-4 GET org", "GET", f"/orgs/{org_t}", every)
    matrix("ACC-4 GET members", "GET", f"/orgs/{org_t}/members", every)
    matrix("ACC-4 GET usage", "GET", f"/orgs/{org_t}/usage", every)
    matrix("ACC-4 PATCH org", "PATCH", f"/orgs/{org_t}", oa, body={"name": f"Acc Team {RUN} renamed"})
    matrix("ACC-4 GET api-keys", "GET", f"/orgs/{org_t}/api-keys", oa)
    matrix("ACC-4 POST api-keys", "POST", f"/orgs/{org_t}/api-keys", oa, 201, body={"name": "matrix", "scopes": ["projects:read"]})
    matrix("ACC-4 GET audit-logs", "GET", f"/orgs/{org_t}/audit-logs", oa)
    for role in ("member", "viewer"):
        expect(f"ACC-4 POST members [{role}] -> 403", call("POST", f"/orgs/{org_t}/members", {"email": extra_email, "role": "viewer"}, tok=roles[role][1]), 403)
    expect("ACC-4 POST members [admin] -> 201", call("POST", f"/orgs/{org_t}/members", {"email": extra_email, "role": "viewer"}, tok=roles["admin"][1]), 201)
    owner_id = roles["owner"][2]
    adm = roles["admin"][1]
    expect("ACC-4 admin demotes owner -> 403", call("PATCH", f"/orgs/{org_t}/members/{owner_id}", {"role": "member"}, tok=adm), 403)
    expect("ACC-4 admin removes owner -> 403", call("DELETE", f"/orgs/{org_t}/members/{owner_id}", tok=adm), 403)
    expect("ACC-4 admin grants owner -> 403", call("PATCH", f"/orgs/{org_t}/members/{roles['member'][2]}", {"role": "owner"}, tok=adm), 403)
    expect("ACC-4 owner demotes last owner (self) -> 409 last_owner", call("PATCH", f"/orgs/{org_t}/members/{owner_id}", {"role": "admin"}, tok=tok_a), 409, "last_owner")
    expect("ACC-4 owner removes last owner (self) -> 409 last_owner", call("DELETE", f"/orgs/{org_t}/members/{owner_id}", tok=tok_a), 409, "last_owner")
    for role in ("admin", "member", "viewer"):
        expect(f"ACC-4 DELETE org [{role}] -> 403", call("DELETE", f"/orgs/{org_t}", tok=roles[role][1]), 403)

    # ---------- ACC-6 API keys (personal org A) ----------
    exp = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    r = call("POST", f"/orgs/{org_a}/api-keys", {"name": "ci", "scopes": ["projects:read"], "expires_at": exp}, tok=tok_a)
    key_ro = (r.json or {}).get("key", "")
    check("ACC-6a create -> 201, key ak_<8hex>_<secret> shown once", r.status == 201 and re.fullmatch(r"ak_[0-9a-f]{8}_.+", key_ro) and r.json.get("prefix") and r.json.get("scopes") == ["projects:read"],
          f"got {r.status} key={key_ro[:12]}…", r)
    lst = call("GET", f"/orgs/{org_a}/api-keys", tok=tok_a)
    secret = key_ro.split("_", 2)[-1] if key_ro else "<none>"
    check("ACC-6b list shows no full key / hash", lst.status == 200 and key_ro not in lst.raw and secret not in lst.raw and "key_hash" not in lst.raw and '"key"' not in lst.raw, "key material in list", lst)
    r1 = call("GET", f"/orgs/{org_a}", key=key_ro)
    r2 = call("GET", f"/orgs/{org_a}", xkey=key_ro)
    check("ACC-6d read key works (Bearer)", r1.status == 200, f"got {r1.status} {r1.code}", r1)
    check("ACC-6d read key works (X-API-Key)", r2.status == 200, f"got {r2.status} {r2.code}", r2)
    lst = call("GET", f"/orgs/{org_a}/api-keys", tok=tok_a)
    ci = next((k for k in lst.json or [] if k["name"] == "ci"), {}) if isinstance(lst.json, list) else next((k for k in (lst.json or {}).get("items", []) if k["name"] == "ci"), {})
    check("ACC-6d last_used_at set", bool(ci.get("last_used_at")), f"key row {ci}", lst)
    expect("ACC-6e read key PATCH org -> 403", call("PATCH", f"/orgs/{org_a}", {"name": "nope"}, key=key_ro), 403)
    r = call("POST", f"/orgs/{org_a}/api-keys", {"name": "short", "scopes": ["projects:read"], "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=3)).isoformat()}, tok=tok_a)
    short = (r.json or {}).get("key", "")
    time.sleep(4)
    expect("ACC-6f expired key -> 401", call("GET", f"/orgs/{org_a}", key=short), 401)
    expect("ACC-6f unknown key -> 401", call("GET", f"/orgs/{org_a}", key="ak_00000000_nope"), 401)
    r = call("POST", f"/orgs/{org_a}/api-keys", {"name": "rev", "scopes": ["projects:read"]}, tok=tok_a)
    rev_key, rev_id = (r.json or {}).get("key", ""), (r.json or {}).get("id")
    d = call("DELETE", f"/orgs/{org_a}/api-keys/{rev_id}", tok=tok_a)
    check("ACC-6g revoke -> 204", d.status == 204, f"got {d.status}", d)
    expect("ACC-6g revoked key -> 401", call("GET", f"/orgs/{org_a}", key=rev_key), 401)
    expect("ACC-6h unknown scope -> 422", call("POST", f"/orgs/{org_a}/api-keys", {"name": "bad", "scopes": ["admin:everything"]}, tok=tok_a), 422)
    r = call("POST", f"/orgs/{org_a}/api-keys", {"name": "all", "scopes": ["inference", "projects:read", "projects:write", "training"]}, tok=tok_a)
    key_all = (r.json or {}).get("key", "")
    check("ACC-6 create all-scope key", r.status == 201 and key_all, f"got {r.status}", r)
    expect("ACC-6i all-scope key POST members -> 403", call("POST", f"/orgs/{org_a}/members", {"email": extra_email, "role": "viewer"}, key=key_all), 403)
    expect("ACC-6i all-scope key POST api-keys -> 403", call("POST", f"/orgs/{org_a}/api-keys", {"name": "x", "scopes": ["projects:read"]}, key=key_all), 403)
    expect("ACC-6i all-scope key PATCH org -> 403", call("PATCH", f"/orgs/{org_a}", {"name": "x"}, key=key_all), 403)
    expect("ACC-6i all-scope key GET /me -> 403", call("GET", "/me", key=key_all), 403)
    expect("ACC-6i all-scope key on other org -> 404", call("GET", f"/orgs/{org_t}", key=key_all), 404)

    # ---------- NFR-SEC3 org paths: user B vs ORG_A / ORG_T ----------
    b_email, _ = user("b")
    tok_b, _, _ = login(b_email)
    org_b = personal_org(tok_b)
    rb = call("POST", f"/orgs/{org_b}/api-keys", {"name": "b-all", "scopes": ["inference", "projects:read", "projects:write", "training"]}, tok=tok_b)
    key_b = (rb.json or {}).get("key", "")
    ci_id = ci.get("id", "00000000-0000-0000-0000-000000000000")
    targets = [
        ("GET", f"/orgs/{{o}}", None), ("PATCH", f"/orgs/{{o}}", {"name": "pwned"}), ("DELETE", f"/orgs/{{o}}", None),
        ("GET", f"/orgs/{{o}}/members", None), ("POST", f"/orgs/{{o}}/members", {"email": b_email, "role": "owner"}),
        ("PATCH", f"/orgs/{{o}}/members/{owner_id}", {"role": "viewer"}), ("DELETE", f"/orgs/{{o}}/members/{owner_id}", None),
        ("GET", f"/orgs/{{o}}/api-keys", None), ("POST", f"/orgs/{{o}}/api-keys", {"name": "pwn", "scopes": ["projects:read"]}),
        ("DELETE", f"/orgs/{{o}}/api-keys/{ci_id}", None), ("GET", f"/orgs/{{o}}/usage", None), ("GET", f"/orgs/{{o}}/audit-logs", None),
    ]
    for o_name, o in (("ORG_A", org_a), ("ORG_T", org_t)):
        for method, p, body in targets:
            path = p.format(o=o)
            expect(f"SEC3 B {method} {p.replace('{o}', o_name)} -> 404", call(method, path, body, tok=tok_b), 404)
            expect(f"SEC3 B-key {method} {p.replace('{o}', o_name)} -> 404", call(method, path, body, key=key_b), 404)
    me_b = call("GET", "/me", tok=tok_b)
    check("SEC3 B /me has no A orgs", org_a not in me_b.raw and org_t not in me_b.raw, "A org id in B /me", me_b)
    ob = call("GET", "/orgs", tok=tok_b)
    check("SEC3 B GET /orgs lists only B", ob.status == 200 and org_a not in ob.raw and org_t not in ob.raw, "A org id in B /orgs", ob)
    # A's data is untouched after B's attempts
    ga = call("GET", f"/orgs/{org_t}", tok=tok_a)
    check("SEC3 ORG_T unchanged after attacks", ga.status == 200 and "pwned" not in ga.raw, f"got {ga.status}", ga)
    km = call("GET", f"/orgs/{org_a}/api-keys", tok=tok_a)
    check("SEC3 A's key 'ci' not revoked by B", "ci" in km.raw and "pwn" not in km.raw, "keys changed", km)
    mm = call("GET", f"/orgs/{org_t}/members", tok=tok_a)
    check("SEC3 B not added to ORG_T", b_email not in mm.raw, "B is a member", mm)
    # random org id
    expect("SEC3 random org id -> 404", call("GET", "/orgs/00000000-0000-7000-8000-000000000000", tok=tok_b), 404)

    # cleanup: owner deletes the team org (also the owner-only DELETE check)
    d = call("DELETE", f"/orgs/{org_t}", tok=tok_a)
    check("ACC-4 DELETE org [owner] -> 2xx", d.status in (200, 202, 204), f"got {d.status}", d)

    fails = [r for r in RESULTS if not r["pass"]]
    path = OUT / f"acc_{RUN}.json"
    path.write_text(json.dumps({"run": RUN, "api": API, "results": RESULTS, "log": LOG}, indent=1))
    print(f"\n{len(RESULTS) - len(fails)}/{len(RESULTS)} pass, {len(fails)} fail, {sum(1 for r in RESULTS if r.get('skipped'))} skipped. Log: {path}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
