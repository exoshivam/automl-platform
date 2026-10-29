"""DAT-13 starter datasets: catalog, import through the normal ingest pipeline, guards, tenant isolation."""

import json
import uuid

import pytest
from sqlalchemy import select

from app import models as m
from app.db import session_factory
from app.services import starters, storage
from tests.test_auth import problem
from tests.test_uploads import COCO, _det_images, folder_files, project, upload, version

API = "/api/v1"
LIC = {"source": "test", "licence": "CC0", "source_url": None, "description": "tiny"}


@pytest.fixture
def catalog(monkeypatch):
    monkeypatch.setattr(starters, "PREFIX", f"starters-test-{uuid.uuid4()}/")
    text_csv = "text,label\n" + "".join(f"this is {'great' if i % 2 else 'awful'} number {i},"
                                        f"{'pos' if i % 2 else 'neg'}\n" for i in range(40))
    tab_csv = "a,b,y\n" + "".join(f"{i},{i % 3},{'yes' if i % 2 else 'no'}\n" for i in range(60))
    entries = [
        starters.publish_files({"id": "tiny_img", "name": "Tiny images", "task_type": "image_classification",
                                "classes": ["cat", "dog"], **LIC},
                               [(rel, data, rel.split("/")[0]) for rel, data in folder_files(12)]),
        starters.publish_files({"id": "tiny_det", "name": "Tiny boxes", "task_type": "object_detection",
                                "classes": ["helmet", "vest"], **LIC},
                               [(rel, data, None) for rel, data in _det_images()], json.dumps(COCO).encode()),
        starters.publish_table({"id": "tiny_text", "name": "Tiny text", "task_type": "text_classification",
                                "classes": ["neg", "pos"], **LIC}, text_csv.encode(),
                               [{"name": "text", "role": "text", "type": "text"},
                                {"name": "label", "role": "target", "type": "categorical"}], 40),
        starters.publish_table({"id": "tiny_tab", "name": "Tiny table", "task_type": "tabular_classification",
                                "classes": ["no", "yes"], "target": "y", **LIC}, tab_csv.encode(),
                               [{"name": "y", "role": "target", "type": "categorical"}], 60),
    ]
    starters.write_catalog(entries)
    yield {e["id"]: e for e in entries}
    for k in storage.existing_sizes(starters.PREFIX):
        storage.delete(k)


def start(c, u, p, sid):
    return c.post(f"{API}/projects/{p['id']}/starter-imports", headers=u.h, json={"starter_id": sid})


def test_catalog_listing(client, alice, catalog):
    rows = client.get(f"{API}/starter-datasets", headers=alice.h).json()
    assert {r["id"] for r in rows} == set(catalog)
    img = client.get(f"{API}/starter-datasets?task_type=image_classification", headers=alice.h).json()
    assert [r["id"] for r in img] == ["tiny_img"] and img[0]["sample_count"] == 24 and len(img[0]["preview_urls"]) == 6
    assert img[0]["licence"] == "CC0" and img[0]["size_bytes"] > 0
    assert client.get(f"{API}/starter-datasets?task_type=nope", headers=alice.h).json() == []
    assert client.get(f"{API}/starter-datasets").status_code == 401


def test_image_import_adds_classes_and_appends_uploads(client, alice, catalog):
    p = project(client, alice, labels=("cat", "bird"))
    r = start(client, alice, p, "tiny_img")
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["status"] == "ingesting" and body["sample_count"] == 24
    v = version(client, alice, body["dataset_version_id"])
    assert v["ingest_status"] == "ready" and v["sample_count"] == 24 and v["is_draft"]
    assert {k: x["images"] for k, x in v["label_stats"].items()} == {"cat": 12, "dog": 12}
    got = client.get(f"{API}/projects/{p['id']}", headers=alice.h).json()
    classes = client.get(f"{API}/label-sets/{got['label_set_id']}", headers=alice.h).json()["classes"]
    assert classes == ["cat", "bird", "dog"]  # dog was added
    with session_factory()() as db:
        keys = list(db.scalars(select(m.Sample.object_key).where(
            m.Sample.dataset_version_id == uuid.UUID(body["dataset_version_id"]))))
        up = db.get(m.Upload, (uuid.UUID(body["upload_id"]), uuid.UUID(alice.org_id)))
        assert up.status == "completed" and up.options["starter_id"] == "tiny_img"
    assert all(k.startswith(storage.org_prefix(alice.org_id)) for k in keys)  # copied into the tenant's prefix
    assert not storage.existing_sizes(f"{storage.org_prefix(alice.org_id)}uploads/{body['upload_id']}/")
    assert len(storage.existing_sizes(starters.PREFIX + "tiny_img/files/")) == 24  # the starter stays intact
    _, r2 = upload(client, alice, p, [("cat/mine.png", folder_files(13)[12][1])])  # user upload lands in the draft
    assert r2.json()["dataset_version_id"] == body["dataset_version_id"]
    assert version(client, alice, body["dataset_version_id"])["sample_count"] == 25
    assert client.post(f"{API}/dataset-versions/{body['dataset_version_id']}/seal", headers=alice.h).status_code == 202


@pytest.mark.parametrize("sid,kw,count", [
    ("tiny_det", {"template": "object_detection", "labels": ("helmet", "person")}, 4),
    ("tiny_text", {"template": "text_classification", "labels": ("neg", "neutral")}, 40),
    ("tiny_tab", {"template": "tabular", "task_type": "tabular_classification", "labels": None}, 60)])
def test_other_kinds_import(client, alice, catalog, sid, kw, count):
    p = project(client, alice, **kw)
    r = start(client, alice, p, sid)
    assert r.status_code == 202, r.text
    v = version(client, alice, r.json()["dataset_version_id"])
    assert v["ingest_status"] == "ready" and v["sample_count"] == count, v
    if sid == "tiny_det":
        assert set(v["label_stats"]) == {"helmet", "vest"}


def test_guards(client, alice, bob, catalog):
    p = project(client, alice)
    problem(start(client, alice, p, "nope"), 404)
    r = problem(start(client, alice, p, "tiny_text"), 422, "task_mismatch")
    assert r["starter_task_type"] == "text_classification"
    problem(start(client, bob, p, "tiny_img"), 404)  # another tenant's project
    with session_factory()() as db:  # an upload still ingesting blocks a second import
        up = m.Upload(organization_id=uuid.UUID(alice.org_id), project_id=uuid.UUID(p["id"]),
                      dataset_id=uuid.UUID(p["dataset_id"]), files=[], status="ingesting")
        db.add(up)
        db.commit()
        up_id = up.id
    problem(start(client, alice, p, "tiny_img"), 409, "import_running")
    with session_factory()() as db:
        db.get(m.Upload, (up_id, uuid.UUID(alice.org_id))).status = "failed"
        db.commit()
    assert start(client, alice, p, "tiny_img").status_code == 202
    problem(start(client, alice, p, "tiny_img"), 409, "already_imported")  # twice would duplicate every sample


def test_purge_keeps_starters(client, alice, catalog):
    p = project(client, alice)
    assert start(client, alice, p, "tiny_img").status_code == 202
    before = storage.existing_sizes(starters.PREFIX)
    assert client.post(f"{API}/projects/{p['id']}:purge", headers=alice.h,
                       json={"confirm_name": p["name"]}).status_code == 202
    assert storage.existing_sizes(starters.PREFIX) == before
    problem(start(client, alice, p, "tiny_img"), 404)
