"""Face recognition (PRD §12, FACE-1..8): alignment, consent gate, enrol -> identify -> Unknown, delete, tenancy."""

import io
import json
import pathlib
import subprocess
import sys
import uuid

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import func, select

from app import models as m
from app.db import session_factory
from app.ml import face
from app.services import storage
from tests.test_auth import problem
from tests.test_uploads import png, project

API = "/api/v1"
FIX = pathlib.Path(__file__).parent / "fixtures" / "faces"
needs_models = pytest.mark.skipif(not list(pathlib.Path(face.get_settings().face_model_dir).glob("det_*.onnx")),
                                  reason="InsightFace buffalo pack not downloaded")
needs_faces = pytest.mark.skipif(not (FIX / "a").exists(), reason="face fixtures missing")


def fp(*parts) -> list[pathlib.Path]:
    return sorted(p for p in (FIX.joinpath(*parts)).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def face_project(c, u):
    return project(c, u, template="face_recognition", labels=None)


def person(c, u, p, name, consent=True):
    return c.post(f"{API}/projects/{p['id']}/people", headers=u.h,
                  json={"name": name, **({"consent": {"attested": True}} if consent else {})})


def send(c, u, pid, paths_or_bytes):
    files = [("files", (p.name, p.read_bytes()) if isinstance(p, pathlib.Path) else ("x.png", p))
             for p in paths_or_bytes]
    return c.post(f"{API}/people/{pid}/photos", headers=u.h, files=files)


def identify(c, u, p, img: pathlib.Path | bytes):
    data = img.read_bytes() if isinstance(img, pathlib.Path) else img
    return c.post(f"{API}/projects/{p['id']}/identify", headers=u.h, files={"file": ("x.jpg", data)}).json()


def test_alignment_maps_landmarks_onto_the_template():
    rng = np.random.default_rng(0)
    for _ in range(5):  # random scale / rotation / shift of the template = a "detected" face
        a, s, t = rng.uniform(-0.6, 0.6), rng.uniform(1.2, 4.0), rng.uniform(20, 200, 2)
        r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        lm = (face.ARCFACE_112 @ r.T) * s + t + rng.normal(0, 0.3, (5, 2))
        mtx = face._similarity(lm, face.ARCFACE_112)
        assert np.abs(lm @ mtx[:, :2].T + mtx[:, 2] - face.ARCFACE_112).max() < 1.5
    # the PIL warp puts a marker at landmark 0 onto template point 0 (within ~1 px)
    img = Image.new("RGB", (400, 400))
    lm = face.ARCFACE_112 * 3 + [40, 30]
    x, y = lm[0].round().astype(int)
    img.paste((255, 255, 255), (x - 3, y - 3, x + 4, y + 4))
    crop = np.asarray(face.align(img, lm).convert("L"), dtype=float)
    cy, cx = np.argwhere(crop > 128).mean(0)
    assert abs(cx - face.ARCFACE_112[0, 0]) < 1.5 and abs(cy - face.ARCFACE_112[0, 1]) < 1.5


def test_template_consent_and_tenancy(client, alice, bob):
    t = {x["id"]: x for x in client.get(f"{API}/templates").json()}
    assert t["face_recognition"]["enabled"]
    p = face_project(client, alice)
    assert p["task_type"] == "face_recognition" and p["status"] == "Draft"
    problem(person(client, alice, p, "Ann", consent=False), 422, "consent_required")
    problem(client.post(f"{API}/projects/{p['id']}/people", headers=alice.h,
                        json={"name": "Ann", "consent": {"attested": False}}), 422, "consent_required")
    ann = person(client, alice, p, "Ann")
    assert ann.status_code == 201, ann.text
    a = ann.json()
    assert a["consent"]["method"] == "checkbox_attested" and a["warnings"][0]["code"] == "no_photos"
    s = client.get(f"{API}/projects/{p['id']}/face-settings", headers=alice.h).json()
    assert s["threshold"] == s["default_threshold"] == 0.4 and not s["embedding_model"]["commercial_use"]
    assert "non-commercial" in s["embedding_model"]["licence"]
    assert client.patch(f"{API}/projects/{p['id']}/face-settings", headers=alice.h,
                        json={"threshold": 0.5}).json()["threshold"] == 0.5
    problem(client.patch(f"{API}/projects/{p['id']}/face-settings", headers=alice.h, json={"threshold": 2}), 422)
    # another tenant sees nothing
    problem(client.get(f"{API}/people/{a['id']}", headers=bob.h), 404)
    problem(client.delete(f"{API}/people/{a['id']}", headers=bob.h), 404)
    problem(person(client, bob, p, "Eve"), 404)
    problem(client.get(f"{API}/projects/{p['id']}/face-report", headers=bob.h), 404)
    problem(client.post(f"{API}/projects/{p['id']}/identify", headers=bob.h, files={"file": ("x.png", png())}), 404)
    for path in ("storage", "training-jobs", "models"):  # shared screens call these for every project
        assert client.get(f"{API}/projects/{p['id']}/{path}", headers=alice.h).status_code == 200, path
    spec = t["face_recognition"]["default_spec"]
    via_spec = client.post(f"{API}/orgs/{alice.org_id}/projects", headers=alice.h, json={"name": "S", "spec": spec})
    assert via_spec.status_code == 201 and client.get(f"{API}/projects/{via_spec.json()['id']}/face-settings",
                                                      headers=alice.h).status_code == 200
    other = project(client, alice)  # an image classification project has no gallery
    problem(person(client, alice, other, "Ann"), 422, "not_face_project")
    with session_factory()() as db:
        assert db.scalar(select(m.AuditLog).where(m.AuditLog.action == "face.identity_created",
                                                  m.AuditLog.target_id == a["id"]))


@needs_models
def test_photo_rejections_without_faces(client, alice):
    p = face_project(client, alice)
    pid = person(client, alice, p, "Ann").json()["id"]
    r = send(client, alice, pid, [png(), b"not an image"]).json()
    assert r["accepted"] == [] and [x["reason"] for x in r["rejected"]] == ["no_face", "not_an_image"]
    out = identify(client, alice, p, png(size=(320, 240)))
    assert out["faces"] == [] and out["image"] == {"width": 320, "height": 240} and "latency_ms" in out


@needs_models
@needs_faces
def test_enrol_identify_unknown_report_delete(client, alice):
    p = face_project(client, alice)
    a = person(client, alice, p, "Person A").json()
    b = person(client, alice, p, "Person B").json()
    ra, rb = send(client, alice, a["id"], fp("a")[:-1]).json(), send(client, alice, b["id"], fp("b")[:-1]).json()
    assert ra["rejected"] == [] and rb["rejected"] == [], (ra["rejected"], rb["rejected"])
    assert ra["person"]["photo_count"] == len(fp("a")) - 1
    assert client.get(f"{API}/projects/{p['id']}", headers=alice.h).json()["status"] == "Ready"

    held_a, held_b = identify(client, alice, p, fp("a")[-1]), identify(client, alice, p, fp("b")[-1])  # unseen photos
    assert [f["name"] for f in held_a["faces"]] == ["Person A"], held_a
    assert [f["name"] for f in held_b["faces"]] == ["Person B"], held_b
    assert held_a["faces"][0]["score"] >= held_a["threshold"] and len(held_a["faces"][0]["top3"]) == 2
    stranger = identify(client, alice, p, fp("c")[0])
    assert len(stranger["faces"]) == 1 and stranger["faces"][0]["name"] is None, stranger  # Unknown
    if (FIX / "group").exists():
        assert len(identify(client, alice, p, fp("group")[0])["faces"]) >= 2
        multi = send(client, alice, a["id"], fp("group")[:1]).json()
        assert multi["rejected"][0]["reason"] == "multiple_faces"

    rep = client.get(f"{API}/projects/{p['id']}/face-report", headers=alice.h).json()
    assert rep["ready"] and rep["genuine"]["mean"] > rep["impostor"]["mean"] and rep["rank1_accuracy"] == 1.0
    assert 0.25 <= rep["suggested_threshold"] <= 0.9 and [x["far"] for x in rep["tar_at_far"]] == [0.001, 0.01, 0.1]
    assert {x["name"] for x in rep["per_person"]} == {"Person A", "Person B"}

    detail = client.get(f"{API}/people/{a['id']}", headers=alice.h).json()
    prefix = f"{storage.org_prefix(alice.org_id)}faces/"
    a_keys = [k for k in storage.existing_sizes(prefix) if a["id"] in k]
    assert len(a_keys) == 2 * len(detail["photos"]) and detail["photos"][0]["thumb_url"]
    assert client.delete(f"{API}/people/{a['id']}", headers=alice.h).status_code == 204
    problem(client.get(f"{API}/people/{a['id']}", headers=alice.h), 404)
    assert not [k for k in storage.existing_sizes(prefix) if a["id"] in k]  # photos + thumbs gone from MinIO
    with session_factory()() as db:
        aid = uuid.UUID(a["id"])
        assert db.scalar(select(func.count()).select_from(m.FaceEmbedding).where(m.FaceEmbedding.identity_id == aid)) == 0
        assert db.scalar(select(func.count()).select_from(m.ConsentRecord).where(
            m.ConsentRecord.id == uuid.UUID(a["consent"]["id"]))) == 0
        cert = db.scalar(select(m.AuditLog).where(m.AuditLog.action == "face.identity_deleted",
                                                  m.AuditLog.target_id == a["id"]))
        assert cert.details["certificate"]["embeddings_deleted"] == len(detail["photos"]) + 1  # + the template
    # A is now unknown; instant, nothing retrained
    assert identify(client, alice, p, fp("a")[-1])["faces"][0]["name"] is None

    # purge removes the rest of the gallery (rows + biometric objects)
    assert client.post(f"{API}/projects/{p['id']}:purge", headers=alice.h,
                       json={"confirm_name": p["name"]}).status_code == 202
    assert not storage.existing_sizes(prefix)
    with session_factory()() as db:
        org = uuid.UUID(alice.org_id)
        for t in (m.FaceGallery, m.FaceIdentity, m.FaceEmbedding, m.ConsentRecord):
            assert db.scalar(select(func.count()).select_from(t).where(t.organization_id == org)) == 0, t


def _seed_template(org_id, project_id, name="Seeded"):
    """A person with a template row only (no real face needed): enough for export plumbing tests."""
    with session_factory()() as db:
        org = uuid.UUID(org_id)
        g = db.scalar(select(m.FaceGallery).where(m.FaceGallery.project_id == uuid.UUID(project_id)))
        c = m.ConsentRecord(organization_id=org, gallery_id=g.id, consent_method="checkbox_attested",
                            granted_at=g.created_at)
        db.add(c)
        db.flush()
        i = m.FaceIdentity(organization_id=org, gallery_id=g.id, display_name=name, consent_record_id=c.id)
        db.add(i)
        db.flush()
        v = np.random.default_rng(1).normal(size=512)
        for tpl in (False, True):
            db.add(m.FaceEmbedding(organization_id=org, identity_id=i.id, gallery_id=g.id, is_template=tpl,
                                   embedding=(v / np.linalg.norm(v)).tolist()))
        db.commit()
        return str(i.id)


def _unzip(client, out, dest: pathlib.Path) -> pathlib.Path:
    import zipfile

    import httpx

    data = httpx.get(out["download_url"], timeout=60).content
    assert len(data) == out["size_bytes"]
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(dest)
    return dest


@needs_models
def test_face_export_guards_and_bundle(client, alice, bob, tmp_path):
    from tests.conftest import make_user
    from tests.test_orgs import add

    p = face_project(client, alice)
    url = f"{API}/projects/{p['id']}/face-export"
    problem(client.post(url, headers=alice.h, json={"acknowledge_biometric": True}), 409, "gallery_empty")
    pid = _seed_template(alice.org_id, p["id"])
    assert client.get(f"{API}/projects/{p['id']}", headers=alice.h).json()["sample_count"] == 1  # photos, not 0
    problem(client.post(url, headers=alice.h, json={}), 422, "acknowledgement_required")
    problem(client.post(url, headers=bob.h, json={"acknowledge_biometric": True}), 404)
    viewer, member = make_user(client, "fv@test.local"), make_user(client, "fm@test.local")
    add(alice, viewer, "viewer")
    add(alice, member, "member")
    for u in (viewer, member):
        problem(client.post(url, headers=u.h, json={"acknowledge_biometric": True}), 403)
    pre = client.get(f"{url}/preview", headers=alice.h)
    assert pre.status_code == 200, pre.text
    pre = pre.json()
    assert pre["people"] == 1 and pre["photos_excluded"] == 1  # the seeded photo row is NOT in the bundle
    assert [f["name"] for f in pre["files"]] == ["det_10g.onnx", "w600k_r50.onnx", "gallery.json", "identify.py",
                                                 "README.md", "LICENSE-NOTICE.txt"]
    for u in (viewer, member):
        problem(client.get(f"{url}/preview", headers=u.h), 403)
    problem(client.get(f"{url}/preview", headers=bob.h), 404)
    r = client.post(url, headers=alice.h, json={"acknowledge_biometric": True})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["size_bytes"] == pre["total_bytes"]  # the preview is the exact zip size, not an estimate
    assert out["people"] == 1 and out["filename"].endswith(".zip")
    d = _unzip(client, out, tmp_path / "bundle")
    assert {x.name for x in d.iterdir()} == {"det_10g.onnx", "w600k_r50.onnx", "gallery.json", "identify.py",
                                             "README.md", "LICENSE-NOTICE.txt"}
    gal = json.loads((d / "gallery.json").read_text())
    assert [x["name"] for x in gal["people"]] == ["Seeded"] and set(gal["people"][0]) == {"name", "template"}
    assert len(gal["people"][0]["template"]) == 512 and gal["threshold"] == 0.4
    assert "non-commercial" in (d / "LICENSE-NOTICE.txt").read_text()
    (d / "none.png").write_bytes(png())
    res = subprocess.run([sys.executable, "-I", str(d / "identify.py"), str(d / "none.png")], capture_output=True, check=False,
                         text=True, cwd=d, timeout=120)
    assert res.returncode == 0 and "no face" in res.stdout, res.stderr
    with session_factory()() as db:
        assert db.scalar(select(m.AuditLog).where(m.AuditLog.action == "face_export",
                                                  m.AuditLog.target_id.is_not(None))).details["people"] == 1
    exports = f"{storage.org_prefix(alice.org_id)}faces/"
    assert [k for k in storage.existing_sizes(exports) if "/exports/" in k]
    assert client.delete(f"{API}/people/{pid}", headers=alice.h).status_code == 204
    assert not [k for k in storage.existing_sizes(exports) if "/exports/" in k]  # bundles held the template


@needs_models
@needs_faces
def test_exported_script_matches_identify(client, alice, tmp_path):
    p = face_project(client, alice)
    for who in ("a", "b"):
        pid = person(client, alice, p, f"Person {who.upper()}").json()["id"]
        assert not send(client, alice, pid, fp(who)[:-1]).json()["rejected"]
    out = client.post(f"{API}/projects/{p['id']}/face-export", headers=alice.h,
                      json={"acknowledge_biometric": True}).json()
    d = _unzip(client, out, tmp_path / "bundle")
    for img in (fp("a")[-1], fp("b")[-1], fp("c")[0]):
        api = identify(client, alice, p, img)["faces"][0]["name"] or "Other"
        res = subprocess.run([sys.executable, "-I", str(d / "identify.py"), str(img)], capture_output=True, text=True, check=False,
                             timeout=120)
        assert res.returncode == 0, res.stderr
        assert res.stdout.split(": ", 1)[1].split("\t")[0] == api, (img, res.stdout, api)


@needs_models
def test_idle_unload_frees_models(client, alice, monkeypatch):
    import time

    from app.ml import idle, loader

    p = face_project(client, alice)
    monkeypatch.setattr(face.get_settings(), "face_idle_unload_s", 3600)  # the background reaper must not race the load checks
    idle.sweep(time.monotonic() + 3600)  # start cold
    first = identify(client, alice, p, png(size=(200, 200)))
    assert face.loaded() and first["model_load_ms"] > 0  # this call loaded the models (UI: "loading model...")
    assert identify(client, alice, p, png(size=(200, 200)))["model_load_ms"] == 0  # warm
    monkeypatch.setattr(face.get_settings(), "face_idle_unload_s", 0.2)
    time.sleep(0.3)
    idle.sweep()  # (the background reaper may have freed them already)
    assert not face.loaded()  # freed after the idle limit
    assert identify(client, alice, p, png(size=(200, 200)))["model_load_ms"] > 0  # reloads on the next request
    # the predict cache follows the same rule
    monkeypatch.setattr(face.get_settings(), "predict_idle_unload_s", 0.2)
    loader._CACHE["fake"], loader._USED["fake"] = object(), time.monotonic()
    time.sleep(0.3)
    idle.register("predict", loader._unload_idle)
    idle.sweep()
    assert "fake" not in loader._CACHE


def _seed_people(org_id, project_id, n_people=2, n_photos=3):
    """People with photo objects + accurate (buffalo_l-like) clustered embeddings; no real faces involved."""
    rng = np.random.default_rng(7)
    with session_factory()() as db:
        org = uuid.UUID(org_id)
        g = db.scalar(select(m.FaceGallery).where(m.FaceGallery.project_id == uuid.UUID(project_id)))
        for p_i in range(n_people):
            c = m.ConsentRecord(organization_id=org, gallery_id=g.id, consent_method="checkbox_attested",
                                granted_at=g.created_at)
            db.add(c)
            db.flush()
            ident = m.FaceIdentity(organization_id=org, gallery_id=g.id, display_name=f"P{p_i}", consent_record_id=c.id)
            db.add(ident)
            db.flush()
            centre = rng.normal(size=512)
            vecs = []
            for k in range(n_photos):
                v = centre + 0.3 * rng.normal(size=512)
                v /= np.linalg.norm(v)
                vecs.append(v)
                key = f"{storage.org_prefix(org_id)}faces/{g.id}/{ident.id}/{p_i}-{k}.jpg"
                storage.put_bytes(key, f"photo {p_i} {k}".encode())
                db.add(m.FaceEmbedding(organization_id=org, identity_id=ident.id, gallery_id=g.id,
                                       embedding=v.tolist(), photo_object_key=key))
            t = np.mean(vecs, axis=0)
            db.add(m.FaceEmbedding(organization_id=org, identity_id=ident.id, gallery_id=g.id, is_template=True,
                                   embedding=(t / np.linalg.norm(t)).tolist()))
        db.commit()


@needs_models
def test_small_pack_reembeds_calibrates_and_ships_no_photos(client, alice, monkeypatch, tmp_path):
    import zipfile

    from app.ml import face as face_mod

    p = face_project(client, alice)
    _seed_people(alice.org_id, p["id"])
    url = f"{API}/projects/{p['id']}/face-export"
    acc = client.get(f"{url}/preview?pack=accurate", headers=alice.h).json()
    assert acc["pack"] == "accurate" and acc["threshold_source"] == "gallery" and acc["quality"]["photos"] == 6
    small_dir = tmp_path / "buffalo_s"
    small_dir.mkdir()
    (small_dir / "det_500m.onnx").write_bytes(b"d" * 2500)
    (small_dir / "w600k_mbf.onnx").write_bytes(b"r" * 13000)
    real_dir = face_mod.pack_dir
    monkeypatch.setattr(face_mod, "pack_dir", lambda pack="accurate": small_dir if pack == "small" else real_dir(pack))
    calls = []

    def fake_small(pack, images):  # MobileFaceNet stand-in: a different embedding space, per photo
        calls.append(len(images))
        out = []
        for data in images:
            who = int(data.split()[1])
            v = np.random.default_rng(1000 * who + int(data.split()[2])).normal(size=512) * 0.4
            v[who * 10:(who + 1) * 10] += 3.0
            out.append([face_core_face(v / np.linalg.norm(v))])
        return out

    monkeypatch.setattr(face_mod, "analyse_with", fake_small)
    sm = client.get(f"{url}/preview?pack=small", headers=alice.h).json()
    assert sm["available"] and sm["pack_label"].startswith("Small") and sm["threshold_source"] == "calibrated"
    assert 0.2 <= sm["threshold"] <= 0.9 and sm["quality"]["recognised"] == 6 and sm["not_detected"] == 0
    assert [f["name"] for f in sm["files"]][:2] == ["det_500m.onnx", "w600k_mbf.onnx"]
    assert sm["total_bytes"] < acc["total_bytes"]
    out = client.post(url, headers=alice.h, json={"acknowledge_biometric": True, "pack": "small"}).json()
    assert out["pack"] == "small" and out["size_bytes"] == sm["total_bytes"]  # exact, per pack
    assert calls == [6]  # re-embedded once: the export reused the preview's cached vectors
    d = _unzip(client, out, tmp_path / "bundle")
    names = {x.name for x in d.iterdir()}
    assert names == {"det_500m.onnx", "w600k_mbf.onnx", "gallery.json", "identify.py", "README.md",
                     "LICENSE-NOTICE.txt"}  # no photos
    gal = json.loads((d / "gallery.json").read_text())
    assert gal["pack"] == "small" and gal["threshold"] == sm["threshold"] and len(gal["people"]) == 2
    assert all(len(x["template"]) == 512 and set(x) == {"name", "template"} for x in gal["people"])
    assert "buffalo_s" in (d / "LICENSE-NOTICE.txt").read_text() and "w600k_mbf.onnx" in (d / "README.md").read_text()
    with zipfile.ZipFile(io.BytesIO(storage.get_bytes(next(k for k in storage.existing_sizes(
            f"{storage.org_prefix(alice.org_id)}faces/") if k.endswith(".zip"))))) as z:
        assert not [n for n in z.namelist() if n.endswith((".jpg", ".png", ".webp"))]


def face_core_face(vec):
    from app.ml.face_core import Face

    return Face((0, 0, 100, 100), 0.99, np.zeros((5, 2)), embedding=vec)


@needs_models
@needs_faces
@pytest.mark.skipif(not (face.pack_dir("small") / "w600k_mbf.onnx").exists(), reason="buffalo_s not installed")
def test_small_bundle_script_matches_small_pipeline(client, alice, tmp_path):
    """identify.py in a Small bundle (det_500m + w600k_mbf) gives the same name as our pipeline run with buffalo_s
    against the same re-embedded gallery."""
    from app.ml import face_core

    p = face_project(client, alice)
    for who in ("a", "b"):
        pid = person(client, alice, p, f"Person {who.upper()}").json()["id"]
        assert not send(client, alice, pid, fp(who)[:-1]).json()["rejected"]
    out = client.post(f"{API}/projects/{p['id']}/face-export", headers=alice.h,
                      json={"acknowledge_biometric": True, "pack": "small"}).json()
    d = _unzip(client, out, tmp_path / "bundle")
    gal = json.loads((d / "gallery.json").read_text())
    for img in (fp("a")[-1], fp("b")[-1], fp("c")[0]):
        ours = face.analyse_with("small", [img.read_bytes()])[0]
        expected = face_core.match(gal, ours[0].embedding)[0] or "Other"
        res = subprocess.run([sys.executable, "-I", str(d / "identify.py"), str(img)], capture_output=True, text=True,
                             check=False, timeout=120)
        assert res.returncode == 0 and res.stdout.split(": ", 1)[1].split("\t")[0] == expected, res.stdout
