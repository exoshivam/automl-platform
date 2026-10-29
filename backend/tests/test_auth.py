import time

import jwt

from tests.conftest import last_mail_token, make_user

API = "/api/v1"
PW = "Str0ng!pass-123"
CSRF = {"X-CSRF": "1"}


def problem(r, status, code=None):
    assert r.status_code == status, r.text
    assert r.headers["content-type"].startswith("application/problem+json")
    body = r.json()
    assert {"type", "title", "status", "detail"} <= body.keys()
    if code:
        assert body["code"] == code, body
    return body


def test_health_and_problem_details(client):
    r = client.get(f"{API}/health")
    assert r.json()["checks"] == {"database": "ok", "redis": "ok"}
    problem(client.get(f"{API}/nope"), 404, "not_found")
    problem(client.post(f"{API}/auth/register", json={}), 422, "validation_error")


def test_register_verify_login_flow(client):
    r = client.post(f"{API}/auth/register", json={"email": "Ann@Test.local", "password": PW})
    assert r.status_code == 201 and r.json()["verification_required"] is True
    problem(client.post(f"{API}/auth/register", json={"email": "ann@test.LOCAL", "password": PW}), 409, "email_taken")
    problem(client.post(f"{API}/auth/register", json={"email": "x@test.local", "password": "123"}), 422)
    problem(client.post(f"{API}/auth/register", json={"email": "not-an-email", "password": PW}), 422)
    problem(client.post(f"{API}/auth/login", json={"email": "ann@test.local", "password": PW}), 403, "email_not_verified")

    token = last_mail_token(client, "Ann@Test.local")
    assert client.post(f"{API}/auth/verify-email", json={"token": token}).json()["email_verified"] is True
    problem(client.post(f"{API}/auth/verify-email", json={"token": token}), 400, "invalid_token")  # single use

    r = client.post(f"{API}/auth/login", json={"email": "ANN@test.local", "password": PW})
    assert r.status_code == 200
    body = r.json()
    claims = jwt.decode(body["access_token"], options={"verify_signature": False})
    assert claims["exp"] - claims["iat"] == 900 and body["expires_in"] == 900
    assert "password" not in str(claims)
    cookie = r.headers["set-cookie"]
    assert "refresh_token=" in cookie and "HttpOnly" in cookie and "SameSite=lax" in cookie
    assert "Path=/api/v1/auth" in cookie and "Max-Age=" in cookie

    me = client.get(f"{API}/me", headers={"Authorization": f"Bearer {body['access_token']}"}).json()
    assert len(me["memberships"]) == 1 and me["memberships"][0]["role"] == "owner"
    assert me["memberships"][0]["personal"] is True


def test_login_failures_identical_then_rate_limited(client, alice):
    bodies = []
    for _ in range(5):
        r = client.post(f"{API}/auth/login", json={"email": alice.email, "password": "wrong-password"})
        bodies.append(problem(r, 401, "invalid_credentials"))
    unknown = problem(client.post(f"{API}/auth/login", json={"email": "ghost@test.local", "password": "x" * 12}), 401)
    strip = lambda b: {k: v for k, v in b.items() if k != "instance"}
    assert all(strip(b) == strip(unknown) for b in bodies)  # no user enumeration
    r = client.post(f"{API}/auth/login", json={"email": alice.email, "password": "wrong-password"})
    problem(r, 429, "too_many_attempts")
    assert int(r.headers["retry-after"]) > 0
    # even the right password is blocked while limited
    problem(client.post(f"{API}/auth/login", json={"email": alice.email, "password": PW}), 429)


def test_refresh_rotation_and_reuse_detection(client, alice):
    r1 = client.cookies.get("refresh_token")
    problem(client.post(f"{API}/auth/refresh"), 403, "csrf_header_missing")
    r = client.post(f"{API}/auth/refresh", headers=CSRF)
    assert r.status_code == 200 and r.json()["access_token"]
    r2 = client.cookies.get("refresh_token")
    assert r2 and r2 != r1
    # replay R1: reuse -> whole family revoked, R2 dies too
    problem(client.post(f"{API}/auth/refresh", headers=CSRF, cookies={"refresh_token": r1}), 401, "refresh_reuse")
    problem(client.post(f"{API}/auth/refresh", headers=CSRF, cookies={"refresh_token": r2}), 401, "session_expired")


def test_logout_revokes(client, alice):
    last = client.cookies.get("refresh_token")
    assert client.post(f"{API}/auth/logout", headers=CSRF).status_code == 204
    problem(client.post(f"{API}/auth/refresh", headers=CSRF, cookies={"refresh_token": last}), 401)


def test_bad_and_expired_access_tokens(client, alice, settings, monkeypatch):
    head, payload, sig = alice.token.split(".")
    tampered = f"{head}.{payload}.{sig[:-2]}{'AA' if sig[-2:] != 'AA' else 'BB'}"
    problem(client.get(f"{API}/me", headers={"Authorization": f"Bearer {tampered}"}), 401)
    problem(client.get(f"{API}/me"), 401)
    monkeypatch.setattr(settings, "access_token_ttl_seconds", 1)
    short = client.post(f"{API}/auth/login", json={"email": alice.email, "password": PW}).json()["access_token"]
    assert client.get(f"{API}/me", headers={"Authorization": f"Bearer {short}"}).status_code == 200
    time.sleep(2.1)
    problem(client.get(f"{API}/me", headers={"Authorization": f"Bearer {short}"}), 401)


def test_google_disabled_by_default(client):
    assert client.get(f"{API}/auth/config").json() == {"google_enabled": False}
    problem(client.get(f"{API}/auth/google/start", follow_redirects=False), 404, "google_disabled")


def test_google_start_redirect_when_configured(client, settings, monkeypatch):
    monkeypatch.setattr(settings, "google_client_id", "cid.apps.googleusercontent.com")
    monkeypatch.setattr(settings, "google_client_secret", "secret")
    monkeypatch.setattr(settings, "google_redirect_uri", "http://localhost:8010/api/v1/auth/google/callback")
    r = client.get(f"{API}/auth/google/start", follow_redirects=False)
    assert r.status_code == 302
    loc = r.headers["location"]
    assert loc.startswith("https://accounts.google.com/") and "response_type=code" in loc
    assert "scope=openid+email" in loc and "state=" in loc and "nonce=" in loc and "code_challenge=" in loc
    problem(client.get(f"{API}/auth/google/callback?code=x&state=forged", follow_redirects=False), 400)
    assert "refresh_token" not in client.cookies


def test_second_user_independent(client, alice):
    bob = make_user(client, "bob2@test.local")
    assert bob.org_id != alice.org_id
