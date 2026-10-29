"""NFR-SEC3 defense in depth: every ORM UPDATE/DELETE on a tenant table carries organization_id."""

import re

from sqlalchemy import event

import app.models  # noqa: F401  (registers every table on Base.metadata before TENANT_TABLES is computed)
from app.db import Base, engine
from tests.conftest import make_user

API = "/api/v1"
TENANT_TABLES = {t.name for t in Base.metadata.sorted_tables
                 if "organization_id" in t.c and not t.c.organization_id.nullable}


def test_tenant_tables_are_mapped_with_org_identity():
    from app import models as m

    for cls in Base.registry.mappers:
        if cls.local_table.name in TENANT_TABLES:
            assert [c.name for c in cls.primary_key] == ["id", "organization_id"], cls.class_
    assert {"api_keys", "projects", "memberships", "samples", "labels"} <= TENANT_TABLES
    assert m.AuditLog.__table__.name not in TENANT_TABLES  # nullable org, append-only


def test_update_and_delete_statements_are_org_scoped(client, alice):
    seen = []

    def capture(conn, cursor, statement, params, context, executemany):
        if re.match(r"\s*(UPDATE|DELETE)", statement, re.IGNORECASE):
            seen.append(statement)

    event.listen(engine(), "before_cursor_execute", capture)
    try:
        org = alice.org_id
        k = client.post(f"{API}/orgs/{org}/api-keys", json={"name": "k", "scopes": ["projects:read"]},
                        headers=alice.h).json()
        client.get(f"{API}/orgs/{org}", headers={"Authorization": f"Bearer {k['key']}"})  # last_used_at
        client.delete(f"{API}/orgs/{org}/api-keys/{k['id']}", headers=alice.h)  # revoked_at
        bob = make_user(client, "scoped@test.local")
        client.post(f"{API}/orgs/{org}/members", json={"email": bob.email}, headers=alice.h)
        uid = client.get(f"{API}/me", headers=bob.h).json()["user"]["id"]
        client.patch(f"{API}/orgs/{org}/members/{uid}", json={"role": "viewer"}, headers=alice.h)
        client.delete(f"{API}/orgs/{org}/members/{uid}", headers=alice.h)
    finally:
        event.remove(engine(), "before_cursor_execute", capture)
    tenant_stmts = [s for s in seen if re.search(r"(UPDATE|DELETE FROM)\s+(\w+)", s, re.IGNORECASE).group(2) in TENANT_TABLES]
    assert any(s.startswith("UPDATE api_keys SET last_used_at") for s in tenant_stmts)
    assert any("revoked_at" in s for s in tenant_stmts)
    assert any(s.startswith("DELETE FROM memberships") for s in tenant_stmts)
    for s in tenant_stmts:
        assert "organization_id" in s.split("WHERE", 1)[1], s
