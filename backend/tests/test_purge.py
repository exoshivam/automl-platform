"""Permanent project purge: every row and every project-owned object goes; other projects/tenants are untouched."""

import uuid

from sqlalchemy import func, select

from app import models as m
from app.db import session_factory
from app.services import storage
from tests.conftest import make_user
from tests.test_auth import problem
from tests.test_orgs import add
from tests.test_training import ready_image_version
from tests.test_uploads import folder_files, png, project, upload

API = "/api/v1"
OWNED = (m.Dataset, m.DatasetVersion, m.Sample, m.Label, m.LabelSet, m.TaskSpecVersion, m.Upload, m.TrainingJob,
         m.Trial, m.Model, m.ModelVersion, m.Export)


def counts(org_id) -> dict:
    with session_factory()() as db:
        return {t.__name__: db.scalar(select(func.count()).select_from(t).where(t.organization_id == uuid.UUID(org_id)))
                for t in OWNED}


def objects(prefix) -> set[str]:
    return set(storage.existing_sizes(prefix))


def test_purge_removes_every_row_and_object(client, alice, bob):
    keep = project(client, alice, labels=("cat", "dog"))  # a sibling project in the same org must survive
    upload(client, alice, keep, folder_files(3))
    keep_objects, keep_rows = objects(storage.org_prefix(alice.org_id)), counts(alice.org_id)

    p, vid = ready_image_version(client, alice)
    job = client.post(f"{API}/projects/{p['id']}/training-jobs", headers=alice.h,
                      json={"dataset_version_id": vid}).json()["job"]
    assert client.get(f"{API}/training-jobs/{job['id']}", headers=alice.h).json()["status"] == "succeeded"
    got = client.get(f"{API}/projects/{p['id']}", headers=alice.h).json()
    assert got["best_metric_name"] and got["best_metric"] is not None and got["best_metric_split"] in ("test", "val")
    with session_factory()() as db:  # an export bundle + a job still "running" (its worker never answers)
        mv = db.scalar(select(m.ModelVersion).where(m.ModelVersion.organization_id == uuid.UUID(alice.org_id),
                                                    m.ModelVersion.status == "ready"))
        key = mv.artifact_prefix + "exports/onnx-none.zip"
        storage.put_bytes(key, b"zip")
        db.add(m.Export(organization_id=mv.organization_id, model_version_id=mv.id, format="onnx", status="ready",
                        artifact_key=key))
        db.add(m.TrainingJob(organization_id=mv.organization_id, project_id=uuid.UUID(p["id"]),
                             dataset_version_id=uuid.UUID(vid), spec_version=1, budget_mode="fast_test",
                             status="running"))
        db.commit()
    assert counts(alice.org_id)["Export"] == 1

    url = f"{API}/projects/{p['id']}:purge"
    problem(client.post(url, headers=bob.h, json={"confirm_name": p["name"]}), 404)  # cross-tenant
    viewer, member = make_user(client, "viewer@test.local"), make_user(client, "member@test.local")
    add(alice, viewer, "viewer")
    add(alice, member, "member")
    problem(client.post(url, headers=viewer.h, json={"confirm_name": p["name"]}), 403)
    problem(client.post(url, headers=member.h, json={"confirm_name": p["name"]}), 403)
    problem(client.post(url, headers=alice.h, json={"confirm_name": p["name"] + "x"}), 422, "confirmation_mismatch")

    r = client.post(url, headers=alice.h, json={"confirm_name": p["name"]})
    assert r.status_code == 202 and r.json()["status"] == "Deleting" and r.json()["task_id"], r.text

    problem(client.get(f"{API}/projects/{p['id']}", headers=alice.h), 404)
    assert counts(alice.org_id) == keep_rows  # zero rows of the purged project remain
    assert objects(storage.org_prefix(alice.org_id)) == keep_objects  # zero of its objects remain, sibling's intact
    listed = {x["id"] for x in client.get(f"{API}/orgs/{alice.org_id}/projects", headers=alice.h).json()["items"]}
    assert p["id"] not in listed and keep["id"] in listed
    with session_factory()() as db:
        acts = set(db.scalars(select(m.AuditLog.action).where(m.AuditLog.target_id == p["id"])))
    assert {"project.purge_requested", "project.purged"} <= acts

    from app.tasks import purge_project  # idempotent: a re-run is a no-op

    assert purge_project.apply(args=[p["id"]]).get()["status"] == "gone"


def test_deleting_project_is_hidden_and_read_only(client, alice):
    p = project(client, alice, labels=("cat", "dog"))
    with session_factory()() as db:
        db.get(m.Project, (uuid.UUID(p["id"]), uuid.UUID(alice.org_id))).status = "Deleting"
        db.commit()
    assert client.get(f"{API}/projects/{p['id']}", headers=alice.h).json()["status"] == "Deleting"
    listed = {x["id"] for x in client.get(f"{API}/orgs/{alice.org_id}/projects", headers=alice.h).json()["items"]}
    assert p["id"] not in listed
    problem(client.patch(f"{API}/projects/{p['id']}", headers=alice.h, json={"name": "x"}), 409, "project_deleting")


def test_purge_removes_everything_for_every_project_kind(client, alice):
    """Face + speaker + trained image (with export, deployment, key, usage): the storage preview counts gallery
    media, and after purging all three, NOTHING of them remains: 0 project rows in every tenant table, 0 objects."""
    from app.db import Base
    from tests.test_deploy import call, deploy, new_key, train
    from tests.test_faces import _seed_template
    from tests.test_speakers import A, clips, person, sp_project
    from tests.test_training import ready_image_version

    org = uuid.UUID(alice.org_id)
    root = storage.org_prefix(alice.org_id)
    # face gallery: a person with an embedding + photo/thumbnail/export objects under the gallery prefix
    face = project(client, alice, template="face_recognition", labels=None)
    pid = _seed_template(alice.org_id, face["id"])
    with session_factory()() as db:
        g = db.scalar(select(m.FaceGallery).where(m.FaceGallery.project_id == uuid.UUID(face["id"])))
    for suffix in (f"{pid}/a.jpg", f"{pid}/a.thumb.webp", "exports/old.zip"):
        storage.put_bytes(f"{root}faces/{g.id}/{suffix}", b"x" * 1000)
    # speaker gallery through the real enrolment path (clips stored as WAV)
    sp = sp_project(client, alice)
    a = person(client, alice, sp, "Reader A").json()
    assert len(clips(client, alice, a["id"], A[:2]).json()["accepted"]) == 2
    for prj in (face, sp):  # the delete dialog's size used to be 0 B here
        t = client.get(f"/api/v1/projects/{prj['id']}/storage", headers=alice.h).json()["totals"]
        parts = ("data_bytes", "manifests_bytes", "models_bytes", "exports_bytes", "gallery_bytes")
        assert t["gallery_bytes"] > 0 and t["total_bytes"] == sum(t[k] for k in parts)
        assert 0 < t["gallery_photos_bytes"] <= t["gallery_bytes"]
    t = client.get(f"/api/v1/projects/{face['id']}/storage", headers=alice.h).json()["totals"]
    assert t["gallery_bytes"] - t["gallery_photos_bytes"] == 1000  # the old export bundle isn't "your photos"
    # trained image model + export artefact + deployment + key + usage
    img, vid = ready_image_version(client, alice)
    mv = train(client, alice, img, vid)
    with session_factory()() as db:
        row = db.get(m.ModelVersion, (uuid.UUID(mv), org))
        db.add(m.Export(organization_id=org, model_version_id=row.id, format="onnx", status="ready",
                        artifact_key=row.artifact_prefix + "exports/onnx-none.zip"))
        db.commit()
        storage.put_bytes(row.artifact_prefix + "exports/onnx-none.zip", b"zip")
    d = deploy(client, alice, img)
    k = new_key(client, alice, d)
    assert call(client, k["key"], files={"file": ("x.png", png((5, 5, 5), (64, 64)))}).status_code == 200
    assert storage.existing_sizes(root)

    for prj in (face, sp, img):
        r = client.post(f"/api/v1/projects/{prj['id']}:purge", headers=alice.h, json={"confirm_name": prj["name"]})
        assert r.status_code == 202, r.text

    keep = {"audit_logs", "memberships", "quotas", "llm_logs", "usage_events", "api_keys"}  # account-level rows
    with session_factory()() as db:
        left = {t.name: db.scalar(select(func.count()).select_from(t).where(t.c.organization_id == org))
                for t in Base.metadata.sorted_tables if "organization_id" in t.c and t.name not in keep}
        assert not {k: v for k, v in left.items() if v}, left
        assert db.scalar(select(func.count()).select_from(m.UsageEvent).where(  # metering keeps no project link
            m.UsageEvent.organization_id == org, m.UsageEvent.project_id.is_not(None))) == 0
        key = db.get(m.ApiKey, (uuid.UUID(k["id"]), org))
        assert key.revoked_at is not None  # deployment keys are revoked, not just unbound
    assert storage.existing_sizes(root) == {}  # 0 objects: datasets, uploads, models, exports, gallery media
    problem(call(client, k["key"], files={"file": ("x.png", png())}), 401)
