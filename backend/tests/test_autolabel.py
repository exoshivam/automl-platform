"""Bulk auto-label: suggestions are NOT labels until accepted; review semantics; no duplicates; cancel; forks."""

import json
import uuid

from sqlalchemy import func, select

from app import models as m
from app.db import session_factory
from app.ml import autolabel
from tests.test_auth import problem
from tests.test_uploads import COCO, _det_images, _import, png, project, upload, version

API = "/api/v1"


class FakeDetector:
    """Two boxes per photo: a confident one at the same place as COCO's first helmet box, and a weak one."""

    model_id = "fake-owl"

    def suggest(self, width, height, classes, image_bytes):
        return [{"class_name": classes[0], "confidence": 0.9, "bbox": {"x": 10, "y": 5, "w": 20, "h": 30}},
                {"class_name": classes[-1], "confidence": 0.4, "bbox": {"x": 60, "y": 40, "w": 15, "h": 15}}]

    def unload(self):
        self.unloaded = True


def setup(client, alice, monkeypatch):
    fake = FakeDetector()
    monkeypatch.setattr(autolabel, "detectors_for", lambda db, p, by, prompts, floor: {"text_prompt": fake})
    monkeypatch.setattr("app.services.detector.text_prompt_available", lambda model_id=None: True)
    p = project(client, alice, template="object_detection", labels=("helmet", "vest"))
    vid = _import(client, alice, p, "coco.json", json.dumps(COCO).encode(), "coco",
                  extra_files=_det_images()).json()["dataset_version_id"]  # 4 photos WITH user boxes
    _, r = upload(client, alice, p, [(f"new{i}.png", png((i * 30, 9, 9), size=(100, 80))) for i in range(3)],
                  label_from_folder=False)
    assert r.json()["dataset_version_id"] == vid  # the same draft: 3 unlabelled photos
    return p, vid, fake


def counts(vid):
    with session_factory()() as db:
        labels = db.scalar(select(func.count()).select_from(m.Label).join(
            m.Sample, m.Sample.id == m.Label.sample_id).where(m.Sample.dataset_version_id == uuid.UUID(vid)))
        sugg = db.scalar(select(func.count()).select_from(m.BoxSuggestion).where(
            m.BoxSuggestion.dataset_version_id == uuid.UUID(vid)))
    return labels, sugg


def test_job_suggests_only_and_is_ignored_until_accepted(client, alice, monkeypatch):
    _, vid, fake = setup(client, alice, monkeypatch)
    before_stats = version(client, alice, vid)["label_stats"]
    labels0, _ = counts(vid)
    r = client.post(f"{API}/dataset-versions/{vid}/autolabel-jobs", headers=alice.h,
                    json={"source": "text_prompt", "classes": ["helmet", "vest"], "prompts": {"helmet": "hard hat"}})
    assert r.status_code == 202, r.text
    job = client.get(f"{API}/autolabel-jobs/{r.json()['id']}", headers=alice.h).json()
    assert job["status"] == "done" and job["progress"]["done"] == job["progress"]["total"] == 3  # unlabelled only
    assert job["counts"] == {"suggested": 6, "samples_with_boxes": 3, "none_found": 0} and fake.unloaded
    # SAFETY: suggestions are not labels -> stats, validation, seal and training data don't see them
    labels1, sugg = counts(vid)
    assert labels1 == labels0 and sugg == 6
    assert version(client, alice, vid)["label_stats"] == before_stats
    from app.ml.train_tasks import _detection_data

    with session_factory()() as db:
        v = db.get(m.DatasetVersion, (uuid.UUID(vid), uuid.UUID(alice.org_id)))
        items = _detection_data(db, v, ["helmet", "vest"])["items"]
        new = {s.id for s in db.scalars(select(m.Sample).where(m.Sample.dataset_version_id == v.id,
                                                               m.Sample.original_name.like("new%")))}
    assert all(not boxes for sid, boxes, _, _ in items if sid in new)  # training sees no box on them
    lst = client.get(f"{API}/dataset-versions/{vid}/suggestions?state=needs_review", headers=alice.h).json()
    assert lst["counts"]["needs_review"] == 3 and lst["counts"]["pending_boxes"] == 6
    row = lst["samples"][0]
    assert {"width", "height", "url", "thumb_url", "labels"} <= set(row) and row["boxes"][0]["confidence"] == 0.9
    problem(client.post(f"{API}/dataset-versions/{vid}/autolabel-jobs", headers=alice.h,
                        json={"classes": ["dragon"]}), 422, "unknown_class")

    # bulk review: accept >= 0.6, reject the rest
    rv = client.post(f"{API}/dataset-versions/{vid}/suggestions:review", headers=alice.h,
                     json={"accept_above": 0.6, "reject_remaining": True})
    assert rv.status_code == 200, rv.text
    assert rv.json()["accepted"] == 3 and rv.json()["rejected"] == 3 and not rv.json()["forked"]
    labels2, _ = counts(vid)
    assert labels2 == labels0 + 3
    with session_factory()() as db:
        auto = db.scalars(select(m.Label).where(m.Label.source == "auto")).all()
        assert len(auto) == 3 and all(a.confidence == 0.9 and a.class_name == "helmet" for a in auto)
        assert db.scalar(select(m.AuditLog).where(m.AuditLog.action == "autolabel.review")).details["models"] == [
            "fake-owl"]
    st = version(client, alice, vid)["label_stats"]
    assert st["helmet"]["boxes"] == before_stats["helmet"]["boxes"] + 3  # only ACCEPTED boxes count
    assert client.get(f"{API}/dataset-versions/{vid}/suggestions", headers=alice.h).json()["counts"]["accepted"] == 3


def test_no_duplicates_over_user_boxes_edit_and_per_photo(client, alice, monkeypatch):
    _, vid, _ = setup(client, alice, monkeypatch)
    rows = client.get(f"{API}/dataset-versions/{vid}/samples?limit=50", headers=alice.h).json()["items"]
    boxed = next(s for s in rows if s["labels"])  # a photo where the user already drew boxes
    plain = [s for s in rows if not s["labels"]]
    r = client.post(f"{API}/dataset-versions/{vid}/autolabel-jobs", headers=alice.h,
                    json={"source": "text_prompt", "scope": "sample_ids", "sample_ids": [boxed["id"], plain[0]["id"]]})
    assert r.status_code == 202, r.text
    lst = client.get(f"{API}/dataset-versions/{vid}/suggestions", headers=alice.h).json()["samples"]
    on_boxed = next(x for x in lst if x["sample_id"] == boxed["id"])
    user_box = boxed["labels"][0]["bbox"]
    assert all(abs(b["bbox"]["x"] - user_box["x"]) > 5 for b in on_boxed["boxes"])  # the overlapping one was dropped
    sid = plain[0]["id"]
    mine = next(x for x in lst if x["sample_id"] == sid)["boxes"]
    rv = client.post(f"{API}/dataset-versions/{vid}/suggestions:review", headers=alice.h, json={
        "edit": [{"id": mine[0]["id"], "bbox": {"w": 25}, "class_name": "vest"}], "reject_samples": [sid]}).json()
    assert rv["accepted"] == 1 and rv["edited"] == 1 and rv["rejected"] == 1  # edit wins over the photo's reject
    after = client.get(f"{API}/dataset-versions/{vid}/samples?limit=50", headers=alice.h).json()["items"]
    got = next(s for s in after if s["id"] == sid)["labels"]
    assert len(got) == 1 and got[0]["class_name"] == "vest" and got[0]["bbox"]["w"] == 25
    problem(client.post(f"{API}/dataset-versions/{vid}/suggestions:review", headers=alice.h,
                        json={"accept": [mine[0]["id"]]}), 404)  # no longer pending


def test_cancel_sealed_stale_and_fork_migration(client, alice, bob, monkeypatch):
    _, vid, _ = setup(client, alice, monkeypatch)
    with session_factory()() as db:  # a queued job cancelled before the worker starts it
        j = m.AutolabelJob(organization_id=uuid.UUID(alice.org_id), dataset_version_id=uuid.UUID(vid),
                           source="text_prompt", classes=["helmet"], min_confidence=0.3, progress={}, results={},
                           status="queued")
        db.add(j)
        db.commit()
        jid = str(j.id)
    assert client.post(f"{API}/autolabel-jobs/{jid}:cancel", headers=alice.h).json()["status"] == "cancelled"
    autolabel.autolabel(jid)
    assert client.get(f"{API}/autolabel-jobs/{jid}", headers=alice.h).json()["progress"].get("done", 0) == 0
    problem(client.get(f"{API}/autolabel-jobs/{jid}", headers=bob.h), 404)
    # a cancel requested mid-run stops after the current photo
    with session_factory()() as db:
        j2 = m.AutolabelJob(organization_id=uuid.UUID(alice.org_id), dataset_version_id=uuid.UUID(vid),
                            source="text_prompt", classes=["helmet"], min_confidence=0.3, progress={}, results={},
                            status="queued", cancel_requested=True)
        db.add(j2)
        db.commit()
        jid2 = str(j2.id)
    autolabel.autolabel(jid2)
    assert client.get(f"{API}/autolabel-jobs/{jid2}", headers=alice.h).json()["status"] == "cancelled"

    assert client.post(f"{API}/dataset-versions/{vid}/autolabel-jobs", headers=alice.h,
                       json={"source": "text_prompt"}).status_code == 202
    client.post(f"{API}/dataset-versions/{vid}/seal", headers=alice.h)  # sealed WITH pending suggestions
    problem(client.post(f"{API}/dataset-versions/{vid}/autolabel-jobs", headers=alice.h, json={}), 409,
            "version_sealed")
    rv = client.post(f"{API}/dataset-versions/{vid}/suggestions:review", headers=alice.h, json={"accept_above": 0.6})
    assert rv.status_code == 200 and rv.json()["forked"], rv.text
    child = rv.json()["dataset_version_id"]
    assert counts(vid)[1] == 0  # the fork moved every pending suggestion with its photo; the sealed version keeps none
    lst = client.get(f"{API}/dataset-versions/{child}/suggestions", headers=alice.h).json()
    assert lst["counts"]["pending_boxes"] == 3 and lst["counts"]["needs_review"] == 3  # weak ones still to review
    assert counts(child)[0] == counts(vid)[0] + 3  # the accepted boxes are labels in the NEW draft only
    problem(client.post(f"{API}/dataset-versions/{vid}/suggestions:review", headers=alice.h,
                        json={"reject_remaining": True}), 409, "stale_version")


def test_text_prompt_detector_uses_transformers5_postprocess():
    """transformers 5 removed processor.post_process_object_detection (a live job failed on it). The detector must
    call post_process_grounded_object_detection and map 0-based labels / xyxy boxes through the shared tidy()."""
    import io as _io

    import torch
    from PIL import Image

    from app.services.detector import TextPromptDetector

    class Out(dict):
        pass

    class Inputs(dict):
        def to(self, device):
            return self

    class FakeProc:
        def __call__(self, text, images, return_tensors):
            self.queries = text
            return Inputs()

        def post_process_grounded_object_detection(self, outputs, threshold, target_sizes):
            self.target = target_sizes.tolist()
            return [{"boxes": torch.tensor([[10.0, 20, 50, 80], [12, 22, 52, 82]]),
                     "scores": torch.tensor([0.8, 0.6]), "labels": torch.tensor([1, 1])}]

    class FakeOwlViT:
        def __call__(self, **kw):
            return Out(logits=torch.zeros(1))

    det = TextPromptDetector(model_id="google/owlvit-base-patch32", prompts={"square": "white square"}, floor=0.3)
    proc = FakeProc()
    det._m = (proc, FakeOwlViT())
    buf = _io.BytesIO()
    Image.new("RGB", (120, 90)).save(buf, "PNG")
    boxes = det.suggest(120, 90, ["circle", "square"], buf.getvalue())
    assert proc.queries == [["a photo of a circle", "a photo of a white square"]]  # per-label prompt override
    assert proc.target == [[90, 120]]  # OWL-ViT: (h, w); OWLv2 uses the padded square side instead
    assert [b["class_name"] for b in boxes] == ["square"]  # class-wise NMS merged the overlapping pair
    assert boxes[0]["bbox"] == {"x": 10.0, "y": 20.0, "w": 40.0, "h": 60.0}
