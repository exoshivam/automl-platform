"""Ready-made detector mode (COCO SSDlite, no training): classes, create rules, predict filter + threshold, convert."""

import time

import pytest
import torch

from app.ml import idle, pretrained
from tests.test_auth import problem
from tests.test_uploads import png

API = "/api/v1"


def create(c, u, labels, **kw):
    return c.post(f"{API}/orgs/{u.org_id}/projects", headers=u.h,
                  json={"name": "Ready", "template": "object_detection", "mode": "pretrained", "labels": labels, **kw})


def predict(c, u, p, img=None):
    return c.post(f"{API}/projects/{p['id']}/predict", headers=u.h, files={"file": ("x.png", img or png((9, 9, 9),
                                                                                                      (320, 240)))})


class FakeSSD:
    """Stands in for the COCO model: a person, a dog (overlapping the person) and a weak car."""

    def __call__(self, imgs):
        return [{"boxes": torch.tensor([[10.0, 10, 110, 210], [12, 12, 112, 212], [200, 50, 300, 120]]),
                 "scores": torch.tensor([0.6, 0.9, 0.3]), "labels": torch.tensor([1, 18, 3])}]


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(pretrained, "_model", lambda: FakeSSD())


def test_classes_and_create_rules(client, alice):
    r = client.get(f"{API}/detection/pretrained-classes", headers=alice.h).json()
    assert len(r["classes"]) == 80 and {"id": 18, "name": "dog"} in r["classes"] and r["model"]["commercial_use"]
    assert client.get(f"{API}/detection/pretrained-classes").status_code == 401
    bad = problem(create(client, alice, ["person", "unicorn", "Helmet"]), 422, "unknown_pretrained_class")
    assert bad["unknown"] == ["unicorn", "Helmet"]
    problem(create(client, alice, []), 422, "invalid_spec")
    p = create(client, alice, ["Person", "dog", "person"])
    assert p.status_code == 201, p.text
    p = p.json()
    assert p["status"] == "Ready" and p["current_spec"]["labels"] == ["person", "dog"]  # canonical, de-duplicated
    assert p["current_spec"]["mode"] == "pretrained" and p["current_spec"]["score_threshold"] == 0.5
    problem(client.post(f"{API}/orgs/{alice.org_id}/projects", headers=alice.h,
                        json={"name": "x", "template": "image_classification", "mode": "pretrained",
                              "labels": ["cat", "dog"]}), 422, "invalid_spec")
    one = client.post(f"{API}/orgs/{alice.org_id}/projects", headers=alice.h,
                      json={"name": "One", "template": "object_detection", "labels": ["helmet"]})
    assert one.status_code == 201 and one.json()["status"] == "Draft"  # custom 1-class detector is valid


def test_predict_filters_classes_and_threshold(client, alice, fake_model):
    p = create(client, alice, ["dog", "car"]).json()
    out = predict(client, alice, p)
    assert out.status_code == 200, out.text
    body = out.json()
    assert [b["class_name"] for b in body["boxes"]] == ["dog"]  # person not chosen; car below 0.5
    assert body["image"] == {"width": 320, "height": 240} and set(body["boxes"][0]["bbox"]) == {"x", "y", "w", "h"}
    s = client.patch(f"{API}/projects/{p['id']}/detector-settings", headers=alice.h, json={"score_threshold": 0.25})
    assert s.json()["score_threshold"] == 0.25 and s.json()["mode"] == "pretrained"
    assert sorted(b["class_name"] for b in predict(client, alice, p).json()["boxes"]) == ["car", "dog"]
    problem(client.patch(f"{API}/projects/{p['id']}/detector-settings", headers=alice.h,
                         json={"score_threshold": 0.99}), 422)
    q = create(client, alice, ["person"]).json()  # an unchosen overlapping dog never suppresses the person
    assert [b["class_name"] for b in predict(client, alice, q).json()["boxes"]] == ["person"]
    both = create(client, alice, ["person", "dog"]).json()  # class-wise NMS: a person holding a dog keeps both
    assert sorted(b["class_name"] for b in predict(client, alice, both).json()["boxes"]) == ["dog", "person"]


def test_convert_training_gate_and_tenancy(client, alice, bob, fake_model):
    p = create(client, alice, ["person", "dog"]).json()
    for call in (lambda u: predict(client, u, p),
                 lambda u: client.get(f"{API}/projects/{p['id']}/detector-settings", headers=u.h),
                 lambda u: client.post(f"{API}/projects/{p['id']}:convert-to-custom", headers=u.h)):
        problem(call(bob), 404)
    problem(client.put(f"{API}/projects/{p['id']}/spec", headers=alice.h,
                       json={"spec": {**p["current_spec"], "mode": "custom"}}), 422, "invalid_spec")
    c = client.post(f"{API}/projects/{p['id']}:convert-to-custom", headers=alice.h)
    assert c.status_code == 200, c.text
    c = c.json()
    assert c["current_spec"]["mode"] == "custom" and c["current_spec"]["labels"] == ["person", "dog"]
    assert c["status"] != "Ready"
    problem(client.post(f"{API}/projects/{p['id']}:convert-to-custom", headers=alice.h), 409, "already_custom")
    problem(predict(client, alice, p), 409, "not_pretrained")
    custom = client.post(f"{API}/orgs/{alice.org_id}/projects", headers=alice.h,
                         json={"name": "c", "template": "object_detection", "labels": ["helmet"]}).json()
    from app import models as m
    from app.db import session_factory
    from app.services.detector import PretrainedDetector, StubDetector, detector

    with session_factory()() as db:
        assert isinstance(detector(db, db.get(m.Project, (p["id"], alice.org_id))), PretrainedDetector)
        assert isinstance(detector(db, db.get(m.Project, (custom["id"], alice.org_id))), StubDetector)


def test_real_model_loads_lazily_and_unloads(client, alice, monkeypatch):
    from app.settings import get_settings

    p = create(client, alice, ["person"]).json()
    monkeypatch.setattr(get_settings(), "predict_idle_unload_s", 3600)  # the background reaper must not race the load checks
    idle.sweep(time.monotonic() + 3600)
    first = predict(client, alice, p).json()
    assert pretrained.loaded() and first["model_load_ms"] > 0 and isinstance(first["boxes"], list)
    assert predict(client, alice, p).json()["model_load_ms"] == 0
    monkeypatch.setattr(get_settings(), "predict_idle_unload_s", 0.2)
    time.sleep(0.3)
    idle.sweep()
    assert not pretrained.loaded()


def test_tidy_is_class_wise():
    from app.ml.detection import tidy

    det = {"boxes": torch.tensor([[10.0, 10, 110, 210], [12, 12, 112, 212], [14, 14, 114, 214]]),
           "scores": torch.tensor([0.9, 0.8, 0.7]), "labels": torch.tensor([1, 2, 1])}
    out = tidy(det, ["person", "dog"], 0.5)
    assert sorted((b["class_name"], round(b["confidence"], 1)) for b in out) == [("dog", 0.8), ("person", 0.9)]


def test_custom_objects_rules(client, alice, bob):
    p = create(client, alice, ["person", "cup"]).json()
    url = f"{API}/projects/{p['id']}/custom-objects"
    bad = problem(client.post(url, headers=alice.h, json={"name": "Cup"}), 422, "ready_made_class")
    assert bad["name"] == "cup" and bad["class_id"] == 47
    problem(client.post(url, headers=bob.h, json={"name": "helmet"}), 404)
    r = client.post(url, headers=alice.h, json={"name": "helmet"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["custom_labels"] == ["helmet"] and body["project"]["custom_status"] == "not_trained"
    assert body["project"]["status"] == "Ready" and body["project"]["current_spec"]["labels"] == ["person", "cup"]
    ls = client.get(f"{API}/label-sets/{body['label_set_id']}", headers=alice.h).json()
    assert ls["classes"] == ["helmet"]  # labeling / validation / training see only the custom objects
    problem(client.post(url, headers=alice.h, json={"name": "HELMET"}), 409, "duplicate_class")
    s = client.patch(f"{API}/projects/{p['id']}/detector-settings", headers=alice.h, json={"classes": ["Dog", "cup"]})
    assert s.status_code == 200 and s.json()["classes"] == ["dog", "cup"]
    problem(client.patch(f"{API}/projects/{p['id']}/detector-settings", headers=alice.h,
                         json={"classes": ["dragon"]}), 422, "unknown_pretrained_class")
    assert client.delete(f"{url}/Helmet", headers=alice.h).json()["custom_labels"] == []
    problem(client.delete(f"{url}/helmet", headers=alice.h), 404)
    custom = client.post(f"{API}/orgs/{alice.org_id}/projects", headers=alice.h,
                         json={"name": "c", "template": "object_detection", "labels": ["vest"]}).json()
    problem(client.post(f"{API}/projects/{custom['id']}/custom-objects", headers=alice.h, json={"name": "x"}),
            409, "not_pretrained")


def test_hybrid_single_model(client, alice, monkeypatch, fake_model):
    import json as _json
    import uuid as _uuid

    from app.ml import loader
    from app.services import storage
    from tests.test_uploads import _import as imp

    monkeypatch.setenv("TRAIN_PRETRAINED", "false")  # read by the training child process
    key = f"replay-test-{_uuid.uuid4()}/"
    rows = []
    for i in range(12):  # a tiny COCO-style replay set (synthetic): 'person' boxes, one unchosen class
        storage.put_bytes(f"{key}images/r{i}.png", png((i * 20, 9, 99), size=(96, 96)))
        rows.append({"key": f"{key}images/r{i}.png", "width": 96, "height": 96,
                     "boxes": [{"class_name": "person", "bbox": {"x": 10, "y": 10, "w": 40, "h": 60}},
                               {"class_name": "zebra", "bbox": {"x": 60, "y": 5, "w": 20, "h": 20}}]})
    storage.put_bytes(f"{key}index.json", _json.dumps(rows).encode())
    monkeypatch.setenv("REPLAY_INDEX_KEY", f"{key}index.json")

    p = create(client, alice, ["person"]).json()
    client.post(f"{API}/projects/{p['id']}/custom-objects", headers=alice.h, json={"name": "helmet"})
    before = predict(client, alice, p).json()  # before training: the stock COCO model + not_trained
    assert before["custom_status"] == "not_trained" and {b["source"] for b in before["boxes"]} == {"ready-made"}
    images, anns = [], []
    for i in range(20):  # the user boxes ONLY the custom object
        images.append((f"images/h{i:02d}.png", png((i * 9 % 255, 70, 180 - i), size=(96, 96))))
        anns.append({"id": i + 1, "image_id": i + 1, "category_id": 1, "bbox": [5 + i % 10, 20, 30, 40]})
    coco = {"images": [{"id": i + 1, "file_name": f"h{i:02d}.png", "width": 96, "height": 96} for i in range(20)],
            "categories": [{"id": 1, "name": "helmet"}], "annotations": anns}
    vid = imp(client, alice, p, "coco.json", _json.dumps(coco).encode(), "coco",
              extra_files=images).json()["dataset_version_id"]
    client.post(f"{API}/dataset-versions/{vid}/seal", headers=alice.h)
    rep = client.get(f"{API}/dataset-versions/{vid}/validation", headers=alice.h).json()
    assert rep["summary"]["errors"] == 0, rep["items"]  # no 'too few boxes for person': ready-made isn't validated
    acks = sorted({i["code"] for i in rep["items"] if i["acknowledgeable"]})
    if acks:
        client.post(f"{API}/dataset-versions/{vid}/validation/acknowledge", headers=alice.h, json={"codes": acks})
    from app.settings import get_settings

    train = f"{API}/projects/{p['id']}/training-jobs"
    bad = problem(client.post(train, headers=alice.h, json={"dataset_version_id": vid}), 409, "replay_unavailable")
    assert bad["replay_images"] == 12 and bad["required"] == 50  # replay is mandatory: refuse rather than forget
    monkeypatch.setattr(get_settings(), "hybrid_min_replay", 10)
    r = client.post(train, headers=alice.h, json={"dataset_version_id": vid})
    assert r.status_code == 202, r.text
    job = client.get(f"{API}/training-jobs/{r.json()['job']['id']}", headers=alice.h).json()
    assert job["status"] == "succeeded", job
    mv = client.get(f"{API}/model-versions/{job['model_version_id']}", headers=alice.h).json()
    assert mv["model_json"]["labels"] == ["person", "helmet"]  # ONE model: ready-made + custom classes
    hy = mv["model_json"]["output"]["hybrid"]
    assert hy["ready_made"] == ["person"] and hy["custom"] == ["helmet"] and hy["replay_train"] > 0
    assert {"coco_map50_before", "coco_map50_after", "coco_drop", "custom_map50", "pseudo_boxes"} <= set(hy)
    from app import models as m
    from app.db import session_factory

    with session_factory()() as db:
        repro = db.get(m.TrainingJob, (_uuid.UUID(job["id"]), _uuid.UUID(alice.org_id))).repro_manifest["hybrid"]
    assert repro["pseudo_score"] == 0.5 and len(repro["stock_model"]["sha256"]) == 64 and repro["replay_index"]
    got = client.get(f"{API}/projects/{p['id']}", headers=alice.h).json()
    assert got["custom_status"] == "ready" and got["custom_model_version_id"] == mv["id"] and got["status"] == "Ready"
    assert isinstance(loader.load_model(mv["id"]).predict([images[0][1]])[0]["boxes"], list)  # real model loads

    assert hy["ready_made_from"] in ("stock", "fine-tuned")

    class OneModel:  # stands in for the fine-tuned model's output
        def __init__(self, source="fine-tuned"):
            self.meta = {"output": {"hybrid": {"ready_made_from": source}}}

        def predict(self, batch, threshold=None):
            box = {"x": 1, "y": 2, "w": 3, "h": 4}
            return [{"boxes": [{"class_name": c, "confidence": 0.8, "bbox": box} for c in ("person", "helmet")]}]

    fake = OneModel()
    monkeypatch.setattr(loader, "load_model_timed", lambda mv_id: (fake, 7))
    monkeypatch.setattr(__import__("app.ml.pretrained", fromlist=["x"]), "_model",
                        lambda: (_ for _ in ()).throw(AssertionError("stock model must not run after training")))
    after = predict(client, alice, p).json()
    assert after["custom_status"] == "ready" and after["model_load_ms"] == 7
    assert sorted((b["class_name"], b["source"]) for b in after["boxes"]) == [("helmet", "custom"),
                                                                              ("person", "ready-made")]
    # an untrusted fine-tune (no replay / got worse): ready-made boxes come from the stock model, custom from ours
    fake.meta = OneModel("stock").meta
    monkeypatch.setattr(__import__("app.ml.pretrained", fromlist=["x"]), "_model", lambda: FakeSSD())
    two = predict(client, alice, p).json()["boxes"]
    assert sorted((b["class_name"], b["source"], round(b["confidence"], 1)) for b in two) == [
        ("helmet", "custom", 0.8), ("person", "ready-made", 0.6)]  # person 0.6 = the stock model's box
    with session_factory()() as db:  # a fine-tune that forgot the ready-made classes is flagged, not shipped quietly
        row = db.get(m.ModelVersion, (_uuid.UUID(mv["id"]), _uuid.UUID(alice.org_id)))
        row.metrics = {**row.metrics, "val": {**row.metrics["val"], "hybrid/warning": "Ready-made objects got worse"}}
        db.commit()
    assert client.get(f"{API}/projects/{p['id']}", headers=alice.h).json()["custom_status"] == "needs_attention"
    with session_factory()() as db:
        row = db.get(m.ModelVersion, (_uuid.UUID(mv["id"]), _uuid.UUID(alice.org_id)))
        row.metrics = {**row.metrics, "val": {k: v for k, v in row.metrics["val"].items() if k != "hybrid/warning"}}
        db.commit()
    client.patch(f"{API}/projects/{p['id']}/detector-settings", headers=alice.h, json={"classes": ["person", "dog"]})
    assert client.get(f"{API}/projects/{p['id']}", headers=alice.h).json()["custom_status"] == "outdated"
    conv = client.post(f"{API}/projects/{p['id']}:convert-to-custom", headers=alice.h).json()
    assert conv["current_spec"]["labels"] == ["person", "dog", "helmet"] and "custom_labels" not in conv["current_spec"]
    for k in storage.existing_sizes(key):
        storage.delete(k)


def test_no_single_image_batches_regression():
    """Job 01a169ec (user's Test1) died: 34 photos + replay left a final batch of ONE, and BatchNorm in train mode
    can't normalise a 1-sample 1x1 extras map. Batches must never end with a lone image; hybrid BN stays frozen."""
    from app.ml.detection import batch_end, build, build_hybrid, freeze_bn

    for total in range(1, 60):
        for size in (1, 2, 4, 8):
            k, sizes = 0, []
            while k < total:
                end = batch_end(k, size, total)
                sizes.append(end - k)
                k = end
            assert sum(sizes) == total and (size == 1 or total == 1 or min(sizes) > 1), (total, size, sizes)
    x, t = [torch.rand(3, 120, 160)], [{"boxes": torch.tensor([[10.0, 10, 80, 80]]), "labels": torch.tensor([1])}]
    m = build(2, False)
    m.train()
    with pytest.raises(ValueError, match="more than 1 value per channel"):
        m(x, t)  # the original failure mode, reproduced
    freeze_bn(m)
    assert sum(m(x, t).values()).isfinite()  # frozen BN: a single image trains fine
    h = build_hybrid(["person", "helmet"], {"person": 1}, False)
    h.train()
    freeze_bn(h)
    assert sum(h(x, t).values()).isfinite()


def test_pseudo_boxes_never_sit_on_user_boxes():
    """User boxed a 'charger'; the stock model calls it 'cell phone'. That auto box must be dropped (any class)."""
    from app.ml.detection import keep_pseudo

    charger = {"x": 100, "y": 100, "w": 60, "h": 120}
    found = [{"class_name": "cell phone", "confidence": 0.8, "bbox": {"x": 104, "y": 98, "w": 58, "h": 118}},  # IoU high
             {"class_name": "cell phone", "confidence": 0.7, "bbox": {"x": 110, "y": 120, "w": 30, "h": 40}},  # inside
             {"class_name": "person", "confidence": 0.9, "bbox": {"x": 300, "y": 20, "w": 80, "h": 200}}]  # elsewhere
    assert [f["class_name"] for f in keep_pseudo(found, [charger])] == ["person"]
    assert len(keep_pseudo(found, [])) == 3


def test_ready_made_preview_uses_the_training_rule(client, alice, monkeypatch, fake_model):
    import json as _json

    from tests.test_uploads import _import as imp

    p = create(client, alice, ["person", "dog"]).json()
    client.post(f"{API}/projects/{p['id']}/custom-objects", headers=alice.h, json={"name": "helmet"})
    coco = {"images": [{"id": 1, "file_name": "a.png", "width": 320, "height": 240}],
            "categories": [{"id": 1, "name": "helmet"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [12, 12, 100, 200]}]}  # on the "dog"
    vid = imp(client, alice, p, "coco.json", _json.dumps(coco).encode(), "coco",
              extra_files=[("images/a.png", png((9, 9, 9), (320, 240)))]).json()["dataset_version_id"]
    sid = client.get(f"{API}/dataset-versions/{vid}/samples", headers=alice.h).json()["items"][0]["id"]
    r = client.post(f"{API}/dataset-versions/{vid}/ready-made-preview", headers=alice.h, json={"sample_ids": [sid]})
    assert r.status_code == 200, r.text
    # FakeSSD: person 0.6 + dog 0.9 overlap the user's helmet box -> both dropped; the car is below 0.5
    assert r.json()["suggestions"][sid] == []
