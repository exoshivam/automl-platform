"""Deleting / pruning versions: shared blobs survive, unique blobs go, guards hold, tenants are isolated."""

import uuid

from sqlalchemy import select

from app import models as m
from app.db import session_factory
from app.services import storage
from tests.test_auth import problem
from tests.test_uploads import folder_files, png, project, upload, version

API = "/api/v1"


def keys_of(vid):
    with session_factory()() as db:
        return {s.object_key for s in db.scalars(select(m.Sample).where(m.Sample.dataset_version_id == uuid.UUID(vid)))}


def sealed(c, u, p, files):
    _, r = upload(c, u, p, files)
    vid = r.json()["dataset_version_id"]
    c.post(f"{API}/dataset-versions/{vid}/seal", headers=u.h)
    return vid


def test_delete_old_version_gc_and_guards(client, alice):
    p = project(client, alice)
    corrupt = ("cat/broken.png", png()[:60])  # held ONLY by v1 once the fix excludes it
    v1 = sealed(client, alice, p, folder_files(12) + [corrupt])
    problem(client.delete(f"{API}/dataset-versions/{v1}", headers=alice.h), 409, "version_in_use")  # current
    v2 = client.post(f"{API}/dataset-versions/{v1}/validation/fixes", headers=alice.h,
                     json={"fix_ids": ["exclude_corrupt"]}).json()["dataset_version_id"]
    assert version(client, alice, v2)["parent_version_id"] == v1
    shared, only_v1 = keys_of(v2), keys_of(v1) - keys_of(v2)
    assert len(only_v1) == 1
    rows = {r["id"]: r for r in client.get(f"{API}/projects/{p['id']}/storage", headers=alice.h).json()["versions"]}
    assert rows[v1]["reclaimable_bytes"] == 60 and rows[v2]["is_current"]
    assert client.delete(f"{API}/dataset-versions/{v1}", headers=alice.h).status_code == 204
    assert all(storage.head(k) for k in shared)  # shared blobs survive
    assert not any(storage.head(k) for k in only_v1)  # the unique blob is gone
    assert version(client, alice, v2)["parent_version_id"] is None  # lineage re-linked
    problem(client.get(f"{API}/dataset-versions/{v1}", headers=alice.h), 404)
    with session_factory()() as db:
        assert db.scalar(select(m.AuditLog).where(m.AuditLog.action == "dataset_version.delete"))


def test_unique_blobs_removed_and_prune(client, alice):
    p = project(client, alice)
    v1 = sealed(client, alice, p, folder_files(12))
    # a label change on the sealed current version forks a draft; deleting a sample-unique blob path:
    unique_png = png((250, 1, 250))
    v2 = sealed(client, alice, p, [("dog/unique.png", unique_png)])
    v3_draft_id = upload(client, alice, p, [("dog/later.png", png((3, 4, 5)))])[1].json()["dataset_version_id"]
    prev = client.post(f"{API}/datasets/{p['dataset_id']}/versions:prune", headers=alice.h, json={}).json()
    assert prev["dry_run"] and {r["id"] for r in prev["deleted"]} == {v1, v2}
    reasons = {r["id"]: r["reason"] for r in prev["kept"]}
    assert reasons[v3_draft_id] in ("current", "draft")
    assert version(client, alice, v1)["id"] == v1  # dry run deleted nothing
    keep = client.post(f"{API}/datasets/{p['dataset_id']}/versions:prune", headers=alice.h,
                       json={"keep_latest": 1, "dry_run": False}).json()
    assert {r["id"] for r in keep["deleted"]} == {v1} and {r["id"] for r in keep["kept"]} >= {v2, v3_draft_id}
    listed = {x["id"] for x in client.get(f"{API}/datasets/{p['dataset_id']}/versions", headers=alice.h).json()["items"]}
    assert listed == {v2, v3_draft_id}


def test_gc_deletes_only_unreferenced(client, alice):
    from app.tasks import gc_objects

    p = project(client, alice)
    vid = sealed(client, alice, p, folder_files(12))
    keep = min(keys_of(vid))
    orphan = storage.org_prefix(alice.org_id) + "datasets/x/objects/aa/orphan.png"
    storage.put_bytes(orphan, b"x")
    foreign = f"org/{uuid.uuid4()}/datasets/x/objects/bb/other.png"
    storage.put_bytes(foreign, b"y")
    assert gc_objects(alice.org_id, [keep, orphan, foreign]) == 1
    assert storage.head(keep) and not storage.head(orphan) and storage.head(foreign)  # other tenants untouched
    storage.delete(foreign)


def test_version_with_models_guard_and_model_delete(client, alice):
    from tests.test_training import ready_image_version

    p, vid = ready_image_version(client, alice)
    jid = client.post(f"{API}/projects/{p['id']}/training-jobs", headers=alice.h,
                      json={"dataset_version_id": vid}).json()["job"]["id"]
    mv = client.get(f"{API}/training-jobs/{jid}", headers=alice.h).json()["model_version_id"]
    upload(client, alice, p, [("cat/new.png", png((9, 9, 9)))])  # vid is no longer current
    body = problem(client.delete(f"{API}/dataset-versions/{vid}", headers=alice.h), 409, "version_has_models")
    assert body["models"][0]["model_version_id"] == mv
    with session_factory()() as db:
        prefix = db.scalar(select(m.ModelVersion.artifact_prefix).where(m.ModelVersion.id == uuid.UUID(mv)))
    assert storage.existing_sizes(prefix)
    assert client.delete(f"{API}/model-versions/{mv}", headers=alice.h).status_code == 204
    assert storage.existing_sizes(prefix) == {}
    problem(client.get(f"{API}/model-versions/{mv}", headers=alice.h), 404)
    assert client.delete(f"{API}/dataset-versions/{vid}", headers=alice.h).status_code == 204


def test_storage_endpoints_cross_tenant_and_roles(client, alice, bob):
    from tests.conftest import make_user

    p = project(client, alice)
    v1 = sealed(client, alice, p, folder_files(12))
    sealed(client, alice, p, [("cat/n.png", png((5, 5, 5)))])
    for method, path, body in (("delete", f"/dataset-versions/{v1}", None),
                               ("post", f"/datasets/{p['dataset_id']}/versions:prune", {"dry_run": False}),
                               ("get", f"/projects/{p['id']}/storage", None)):
        problem(client.request(method.upper(), f"{API}{path}", json=body, headers=bob.h), 404)
    viewer = make_user(client, "v@test.local")
    client.post(f"{API}/orgs/{alice.org_id}/members", json={"email": viewer.email, "role": "viewer"}, headers=alice.h)
    problem(client.delete(f"{API}/dataset-versions/{v1}", headers=viewer.h), 403)
    assert client.get(f"{API}/projects/{p['id']}/storage", headers=viewer.h).status_code == 200
    assert version(client, alice, v1)["id"] == v1


def samples_of(c, u, vid):
    return c.get(f"{API}/dataset-versions/{vid}/samples?limit=200", headers=u.h).json()["items"]


def test_remove_samples_from_draft(client, alice, bob):
    from tests.conftest import make_user
    from tests.test_orgs import add

    p = project(client, alice)
    v1 = sealed(client, alice, p, folder_files(12))  # sealed v1: its blobs must survive removals in later drafts
    _, r = upload(client, alice, p, [("cat/burst1.png", png((1, 1, 1))), ("cat/burst2.png", png((2, 2, 2)))])
    draft = r.json()["dataset_version_id"]
    assert draft != v1 and version(client, alice, draft)["sample_count"] == 26
    rows = samples_of(client, alice, draft)
    shared = next(s for s in rows if s["name"].startswith("cat/0"))  # carried over from v1: same blob
    burst = [s for s in rows if "burst" in (s["name"] or "")]
    url = f"{API}/dataset-versions/{draft}/samples:remove"
    viewer = make_user(client, "rm-viewer@test.local")
    add(alice, viewer, "viewer")
    problem(client.post(url, headers=viewer.h, json={"sample_ids": [shared["id"]]}), 403)
    problem(client.post(url, headers=bob.h, json={"sample_ids": [shared["id"]]}), 404)
    problem(client.post(f"{API}/dataset-versions/{v1}/samples:remove", headers=alice.h,
                        json={"sample_ids": [samples_of(client, alice, v1)[0]["id"]]}), 409, "version_sealed")
    problem(client.post(url, headers=alice.h, json={"sample_ids": [samples_of(client, alice, v1)[0]["id"]]}), 404)
    with session_factory()() as db:
        shared_key = db.scalar(select(m.Sample.object_key).where(m.Sample.id == uuid.UUID(shared["id"])))
        burst_keys = list(db.scalars(select(m.Sample.object_key).where(
            m.Sample.id.in_([uuid.UUID(s["id"]) for s in burst]))))
    r = client.post(url, headers=alice.h, json={"sample_ids": [shared["id"], burst[0]["id"]]})
    assert r.status_code == 200, r.text
    assert r.json() == {"removed": 2, "sample_count": 24, "dataset_version_id": draft}
    assert version(client, alice, draft)["sample_count"] == 24
    assert storage.head(shared_key)  # still used by sealed v1
    assert not storage.head(burst_keys[0]) and storage.head(burst_keys[1])  # unique blob GC'd, the other untouched
    assert len(samples_of(client, alice, v1)) == 24  # the sealed version is unchanged
    with session_factory()() as db:
        assert db.scalar(select(m.AuditLog).where(m.AuditLog.action == "samples.remove",
                                                  m.AuditLog.target_id == draft)).details["removed"] == 2


def test_undo_upload_burst(client, alice, bob):
    p = project(client, alice)
    _, r1 = upload(client, alice, p, folder_files(3))
    up2, r2 = upload(client, alice, p, [("dog/x1.png", png((9, 1, 1))), ("dog/x2.png", png((9, 2, 2)))],
                     source="camera")
    draft = r2.json()["dataset_version_id"]
    assert r1.json()["dataset_version_id"] == draft and version(client, alice, draft)["sample_count"] == 8
    problem(client.post(f"{API}/uploads/{up2['upload_id']}:undo", headers=bob.h), 404)
    r = client.post(f"{API}/uploads/{up2['upload_id']}:undo", headers=alice.h)
    assert r.status_code == 200 and r.json()["removed"] == 2 and r.json()["sample_count"] == 6, r.text
    assert client.post(f"{API}/uploads/{up2['upload_id']}:undo", headers=alice.h).json()["removed"] == 0
    assert all("x" not in (s["name"] or "") for s in samples_of(client, alice, draft))
    client.post(f"{API}/dataset-versions/{draft}/seal", headers=alice.h)
    problem(client.post(f"{API}/uploads/{r1.json()['upload_id']}:undo", headers=alice.h), 409, "version_sealed")
