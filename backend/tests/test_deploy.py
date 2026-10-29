"""Self-serve local deploy: one deployment per project, keys bound to it, POST /v1/predict picks the model by key."""

import uuid

from sqlalchemy import select

from app import models as m
from app.db import session_factory
from tests.conftest import make_user
from tests.test_auth import problem
from tests.test_orgs import add
from tests.test_quicktext import add as add_examples
from tests.test_quicktext import quick
from tests.test_speakers import sp_project
from tests.test_training import ready_image_version
from tests.test_uploads import png

API = "/api/v1"


def train(c, u, p, vid):
    r = c.post(f"{API}/projects/{p['id']}/training-jobs", headers=u.h, json={"dataset_version_id": vid})
    assert r.status_code == 202, r.text
    job = c.get(f"{API}/training-jobs/{r.json()['job']['id']}", headers=u.h).json()
    assert job["status"] == "succeeded", job
    return job["model_version_id"]


def deploy(c, u, p, **body):
    r = c.post(f"{API}/projects/{p['id']}/deployment", headers=u.h, json=body)
    assert r.status_code == 201, r.text
    return r.json()


def new_key(c, u, d, name="app"):
    r = c.post(f"{API}/deployments/{d['id']}/keys", headers=u.h, json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()


def call(c, key, **kw):
    return c.post("/v1/predict", headers={"Authorization": f"Bearer {key}"} if key else {}, **kw)


def test_image_deploy_auth_version_switch_pause_rate_limit(client, alice, bob):
    p, vid = ready_image_version(client, alice)
    problem(client.post(f"{API}/projects/{p['id']}/deployment", headers=alice.h, json={}), 409, "no_model")
    mv1 = train(client, alice, p, vid)
    mv2 = train(client, alice, p, vid)
    d = deploy(client, alice, p)
    assert d["kind"] == "model" and d["model_version_id"] == mv2 and d["status"] == "active"  # latest by default
    problem(client.post(f"{API}/projects/{p['id']}/deployment", headers=alice.h, json={}), 409, "deployment_exists")
    k = new_key(client, alice, d)
    assert k["key"].startswith("ak_") and k["prefix"] in k["key"]
    listed = client.get(f"{API}/deployments/{d['id']}/keys", headers=alice.h).json()
    assert [x["id"] for x in listed] == [k["id"]] and "key" not in listed[0]  # plaintext only once

    img = {"files": {"file": ("x.png", png((5, 5, 5), (64, 64)))}}
    r = call(client, k["key"], **img)
    assert r.status_code == 200, r.text
    assert r.json()["label"] in ("cat", "dog") and "probs" in r.json() and "latency_ms" in r.json()
    assert r.headers["x-model-version"] == mv2
    # auth: none / garbage / session token / workspace key / revoked
    problem(call(client, None, **img), 401)
    problem(call(client, "ak_deadbeef_nope", **img), 401)
    problem(client.post("/v1/predict", headers=alice.h, **img), 401)  # a session token is never accepted here
    ws = client.post(f"{API}/orgs/{alice.org_id}/api-keys", headers=alice.h,
                     json={"name": "ws", "scopes": ["inference"]}).json()["key"]
    problem(call(client, ws, **img), 401)  # not bound to a deployment
    k2 = new_key(client, alice, d, "second")
    assert client.delete(f"{API}/deployments/{d['id']}/keys/{k2['id']}", headers=alice.h).status_code == 204
    problem(call(client, k2["key"], **img), 401)
    # wrong input for this model
    problem(call(client, k["key"], json={"text": "hi"}), 422)
    # switch / roll back the version: same key
    s = client.patch(f"{API}/deployments/{d['id']}", headers=alice.h, json={"model_version_id": mv1}).json()
    assert s["model_version_id"] == mv1
    assert call(client, k["key"], **img).headers["x-model-version"] == mv1
    # pause / resume
    client.patch(f"{API}/deployments/{d['id']}", headers=alice.h, json={"status": "paused"})
    problem(call(client, k["key"], **img), 403, "deployment_paused")
    client.patch(f"{API}/deployments/{d['id']}", headers=alice.h, json={"status": "active"})
    # rate limit (fixed window per key)
    with session_factory()() as db:
        db.get(m.Deployment, (uuid.UUID(d["id"]), uuid.UUID(alice.org_id))).rate_limit_per_min = 1
        db.commit()
    k3 = new_key(client, alice, d, "burst")
    assert call(client, k3["key"], **img).status_code == 200
    r = call(client, k3["key"], **img)
    assert r.status_code == 429 and r.json()["code"] == "rate_limited" and 0 < int(r.headers["retry-after"]) <= 60
    # cross tenant: bob can't see or manage alice's deployment; bob's own key only reaches bob's project
    problem(client.get(f"{API}/deployments/{d['id']}/keys", headers=bob.h), 404)
    problem(client.post(f"{API}/deployments/{d['id']}/keys", headers=bob.h, json={"name": "x"}), 404)
    problem(client.patch(f"{API}/deployments/{d['id']}", headers=bob.h, json={"status": "paused"}), 404)
    # usage + docs (401s are rejected before a deployment is known, so they are not counted per deployment)
    u = client.get(f"{API}/deployments/{d['id']}/usage?window=24h", headers=alice.h).json()
    assert u["requests"] == 6 and u["errors"] == 3 and u["p95_latency_ms"] is not None and len(u["series"]) >= 24
    doc = client.get(f"{API}/deployments/{d['id']}/docs", headers=alice.h).json()
    assert doc["endpoint"].endswith("/v1/predict") and doc["input"]["field"] == "file" and set(doc["labels"]) == {
        "cat", "dog"}
    keys = client.get(f"{API}/orgs/{alice.org_id}/api-keys", headers=alice.h).json()
    mine = next(x for x in keys if x["id"] == k["id"])
    assert mine["deployment_id"] == d["id"] and mine["project_id"] == p["id"] and mine["project_name"] == p["name"]
    # delete revokes its keys
    assert client.delete(f"{API}/deployments/{d['id']}", headers=alice.h).status_code == 204
    problem(call(client, k["key"], **img), 401)


def test_text_deploy_and_owner_only_biometrics(client, alice):
    p = quick(client, alice)
    add_examples(client, alice, p)
    d = deploy(client, alice, p)
    assert d["kind"] == "quick_text" and d["model_version_id"] is None
    k = new_key(client, alice, d)
    r = call(client, k["key"], json={"text": "refund please, paisa wapas chahiye"})
    assert r.status_code == 200 and r.json()["label"] == "refund" and r.json()["source"] == "examples", r.text
    problem(call(client, k["key"], json={}), 422)
    # face / speaker deployments: workspace owner only
    member = make_user(client, "dep-member@test.local")
    add(alice, member, "member")
    sp = sp_project(client, alice)
    problem(client.post(f"{API}/projects/{sp['id']}/deployment", headers=member.h, json={}), 403)
    sd = deploy(client, alice, sp)
    assert sd["kind"] == "speaker"
    with session_factory()() as db:
        assert db.scalar(select(m.AuditLog).where(m.AuditLog.action == "deployment.create",
                                                  m.AuditLog.target_id == sd["id"]))
