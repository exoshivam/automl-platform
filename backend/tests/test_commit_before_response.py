"""Regression: a mutation must be committed before its response is sent (revoked key used right after 204)."""

import contextlib
import pathlib
import socket
import threading
import time
import typing

import httpx
import uvicorn

import app as app_pkg
from app.db import get_db
from app.deps import DbDep
from app.main import app
from tests.conftest import make_user
from tests.test_auth import CSRF, PW

API = "/api/v1"


def test_db_dependency_commits_before_response():
    """DbDep must end (commit) before the response is sent, and nothing may declare get_db another way."""
    dep = typing.get_args(DbDep)[1]
    assert dep.dependency is get_db and dep.scope == "function"
    offenders = [str(p) for p in pathlib.Path(app_pkg.__file__).parent.rglob("*.py")
                 if "Depends(get_db" in p.read_text() and p.name != "deps.py"]
    assert offenders == []


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def live_server():
    """Live uvicorn (TestClient waits for teardown, so it can't show the race)."""
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    while not server.started:
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}{API}"
    finally:
        server.should_exit = True
        t.join(5)


def test_revoked_key_rejected_immediately(alice):
    with live_server() as base, httpx.Client(base_url=base, headers=alice.h) as c:
        org = alice.org_id
        for _ in range(20):
            k = c.post(f"/orgs/{org}/api-keys", json={"name": "rev", "scopes": ["projects:read"]}).json()
            assert c.delete(f"/orgs/{org}/api-keys/{k['id']}").status_code == 204
            r = c.get(f"/orgs/{org}", headers={"Authorization": f"Bearer {k['key']}"})
            assert r.status_code == 401, "revoked key still worked right after 204"


def test_removed_member_and_revoked_session_rejected_immediately(client, alice):
    bob = make_user(client, "racer@test.local")
    with live_server() as base, httpx.Client(base_url=base) as c:
        for _ in range(10):
            r = c.post(f"/orgs/{alice.org_id}/members", json={"email": bob.email, "role": "member"}, headers=alice.h)
            assert r.status_code == 201
            assert c.delete(f"/orgs/{alice.org_id}/members/{_uid(c, bob)}", headers=alice.h).status_code == 204
            assert c.get(f"/orgs/{alice.org_id}", headers=bob.h).status_code == 404, "removed member still in"
        for _ in range(10):
            login = c.post("/auth/login", json={"email": bob.email, "password": PW})
            cookie = login.cookies["refresh_token"]
            assert c.post("/auth/logout", headers=CSRF, cookies={"refresh_token": cookie}).status_code == 204
            r = c.post("/auth/refresh", headers=CSRF, cookies={"refresh_token": cookie})
            assert r.status_code == 401, "logged-out refresh token still worked"
            rot = c.post("/auth/login", json={"email": bob.email, "password": PW}).cookies["refresh_token"]
            assert c.post("/auth/refresh", headers=CSRF, cookies={"refresh_token": rot}).status_code == 200
            assert c.post("/auth/refresh", headers=CSRF, cookies={"refresh_token": rot}).status_code == 401


def _uid(c, u):
    return c.get("/me", headers=u.h).json()["user"]["id"]
