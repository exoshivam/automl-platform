from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from app import models as m
from app.db import session_factory
from tests.conftest import make_user
from tests.test_auth import problem

API = "/api/v1"


def add(owner, user, role):
    r = owner.client.post(f"{API}/orgs/{owner.org_id}/members", json={"email": user.email, "role": role}, headers=owner.h)
    assert r.status_code == 201, r.text


def test_create_org_unique_slug(client, alice):
    a = client.post(f"{API}/orgs", json={"name": "Acc A Team"}, headers=alice.h)
    b = client.post(f"{API}/orgs", json={"name": "Acc A Team"}, headers=alice.h)
    assert a.status_code == b.status_code == 201
    assert a.json()["role"] == "owner" and a.json()["slug"] != b.json()["slug"]
    assert len(client.get(f"{API}/orgs", headers=alice.h).json()) == 3


def test_role_matrix_members_keys_org(client, alice):
    admin, member, viewer = (make_user(client, f"{r}@test.local") for r in ("admin", "member", "viewer"))
    for u, role in ((admin, "admin"), (member, "member"), (viewer, "viewer")):
        add(alice, u, role)
    team = client.post(f"{API}/orgs", json={"name": "Team"}, headers=alice.h).json()["id"]
    for u, role in ((admin, "admin"), (member, "member"), (viewer, "viewer")):
        client.post(f"{API}/orgs/{team}/members", json={"email": u.email, "role": role}, headers=alice.h)
    org = alice.org_id
    members = client.get(f"{API}/orgs/{org}/members", headers=viewer.h).json()
    assert sorted(x["role"] for x in members) == ["admin", "member", "owner", "viewer"]

    key_body = {"name": "k", "scopes": ["projects:read"]}
    for u, want in ((alice, 201), (admin, 201), (member, 403), (viewer, 403)):
        assert client.post(f"{API}/orgs/{org}/api-keys", json=key_body, headers=u.h).status_code == want, u.email
    for u, want in ((alice, 200), (admin, 200), (member, 403), (viewer, 403)):
        assert client.patch(f"{API}/orgs/{org}", json={"name": "Renamed"}, headers=u.h).status_code == want
    for u in (admin, member, viewer):
        problem(client.delete(f"{API}/orgs/{team}", headers=u.h), 403, "forbidden")
    assert client.delete(f"{API}/orgs/{team}", headers=alice.h).status_code == 204
    problem(client.delete(f"{API}/orgs/{org}", headers=alice.h), 409, "personal_workspace")

    # admin can't touch the owner role; owner can't leave the org owner-less
    problem(client.patch(f"{API}/orgs/{org}/members/{_uid(alice)}", json={"role": "admin"}, headers=admin.h), 403)
    problem(client.post(f"{API}/orgs/{org}/members", json={"email": "x@y.z", "role": "owner"}, headers=admin.h), 403)
    problem(client.patch(f"{API}/orgs/{org}/members/{_uid(alice)}", json={"role": "admin"}, headers=alice.h), 409,
            "last_owner")
    assert client.patch(f"{API}/orgs/{org}/members/{_uid(viewer)}", json={"role": "member"},
                        headers=admin.h).json()["role"] == "member"
    # member can leave on its own
    assert client.delete(f"{API}/orgs/{org}/members/{_uid(member)}", headers=member.h).status_code == 204


def _uid(u):
    return u.client.get(f"{API}/me", headers=u.h).json()["user"]["id"]


def test_add_member_rules(client, alice):
    problem(client.post(f"{API}/orgs/{alice.org_id}/members", json={"email": "ghost@test.local"}, headers=alice.h), 404,
            "user_not_found")
    bob = make_user(client, "b@test.local")
    add(alice, bob, "member")
    problem(client.post(f"{API}/orgs/{alice.org_id}/members", json={"email": bob.email}, headers=alice.h), 409)


def test_api_keys(client, alice):
    org = alice.org_id
    r = client.post(f"{API}/orgs/{org}/api-keys", json={"name": "ci", "scopes": ["projects:read"],
                                                         "expires_at": (datetime.now(UTC) + timedelta(days=1)).isoformat()},
                    headers=alice.h)
    assert r.status_code == 201
    created = r.json()
    key = created["key"]
    assert key.startswith(f"ak_{created['prefix']}_")
    listed = client.get(f"{API}/orgs/{org}/api-keys", headers=alice.h).json()
    assert key not in str(listed) and "key_hash" not in str(listed) and listed[0]["last_used_at"] is None

    with session_factory()() as db:
        row = db.scalar(select(m.ApiKey))
        assert row.key_hash != key and key not in row.key_hash and len(row.key_hash) == 64

    kh = {"Authorization": f"Bearer {key}"}
    assert client.get(f"{API}/orgs/{org}", headers=kh).status_code == 200
    assert client.get(f"{API}/orgs/{org}", headers={"X-API-Key": key}).status_code == 200
    assert client.get(f"{API}/orgs/{org}/api-keys", headers=alice.h).json()[0]["last_used_at"] is not None
    # keys never manage members, keys or the org
    full = client.post(f"{API}/orgs/{org}/api-keys", headers=alice.h,
                       json={"name": "all", "scopes": ["inference", "projects:read", "projects:write", "training"]})
    fh = {"Authorization": f"Bearer {full.json()['key']}"}
    problem(client.get(f"{API}/orgs/{org}/api-keys", headers=fh), 403)
    problem(client.post(f"{API}/orgs/{org}/members", json={"email": "a@b.c"}, headers=fh), 403)
    problem(client.patch(f"{API}/orgs/{org}", json={"name": "x"}, headers=fh), 403)
    problem(client.get(f"{API}/me", headers=fh), 403)

    problem(client.post(f"{API}/orgs/{org}/api-keys", json={"name": "x", "scopes": ["admin:everything"]},
                        headers=alice.h), 422)
    past = client.post(f"{API}/orgs/{org}/api-keys", headers=alice.h, json={
        "name": "old", "scopes": ["projects:read"], "expires_at": (datetime.now(UTC) - timedelta(minutes=1)).isoformat()})
    problem(client.get(f"{API}/orgs/{org}", headers={"Authorization": f"Bearer {past.json()['key']}"}), 401)
    assert client.delete(f"{API}/orgs/{org}/api-keys/{created['id']}", headers=alice.h).status_code == 204
    problem(client.get(f"{API}/orgs/{org}", headers=kh), 401)
    problem(client.get(f"{API}/orgs/{org}", headers={"Authorization": "Bearer ak_deadbeef_nope"}), 401)


def test_cross_tenant_orgs(client, alice, bob):
    a = alice.org_id
    for path in (f"/orgs/{a}", f"/orgs/{a}/members", f"/orgs/{a}/api-keys", f"/orgs/{a}/usage", f"/orgs/{a}/audit-logs"):
        body = problem(client.get(f"{API}{path}", headers=bob.h), 404)
        assert a not in str({k: v for k, v in body.items() if k != "instance"})
    problem(client.patch(f"{API}/orgs/{a}", json={"name": "pwned"}, headers=bob.h), 404)
    problem(client.post(f"{API}/orgs/{a}/members", json={"email": bob.email, "role": "owner"}, headers=bob.h), 404)
    bkey = client.post(f"{API}/orgs/{bob.org_id}/api-keys", headers=bob.h,
                       json={"name": "b", "scopes": ["projects:read", "projects:write"]}).json()["key"]
    problem(client.get(f"{API}/orgs/{a}", headers={"Authorization": f"Bearer {bkey}"}), 404)
    assert [o["id"] for o in client.get(f"{API}/orgs", headers=bob.h).json()] == [bob.org_id]
    assert client.get(f"{API}/orgs/{a}", headers=alice.h).json()["name"] != "pwned"


def test_audit_log_records_key_creation(client, alice):
    client.post(f"{API}/orgs/{alice.org_id}/api-keys", json={"name": "k", "scopes": ["inference"]}, headers=alice.h)
    items = client.get(f"{API}/orgs/{alice.org_id}/audit-logs", headers=alice.h).json()["items"]
    assert "api_key.create" in [i["action"] for i in items]
