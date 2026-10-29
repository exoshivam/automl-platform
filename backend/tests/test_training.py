"""Training slice: gate, image smoke run end to end, artifacts contract, load_model, SSE, cancel, memory guard."""

import io
import json
import sys
import uuid

import pytest
from sqlalchemy import select

from app import models as m
from app.db import session_factory
from app.ml import artifacts
from app.ml.loader import load_model
from app.services import storage
from tests.test_auth import problem
from tests.test_uploads import _import, folder_files, png, project, upload

API = "/api/v1"


def ready_image_version(c, u, per_class=32):
    p = project(c, u, labels=("cat", "dog"))
    _, r = upload(c, u, p, folder_files(per_class))
    vid = r.json()["dataset_version_id"]
    c.post(f"{API}/dataset-versions/{vid}/seal", headers=u.h)
    return p, vid


def test_api_does_not_import_torch():
    import subprocess

    code = "import app.main, sys; print('torch' in sys.modules, 'lightgbm' in sys.modules, 'transformers' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False False False"


def test_gate(client, alice, bob):
    p = project(client, alice, labels=("cat", "dog"))
    _, r = upload(client, alice, p, folder_files(5))
    vid = r.json()["dataset_version_id"]
    url = f"{API}/projects/{p['id']}/training-jobs"
    problem(client.post(url, headers=alice.h, json={"dataset_version_id": vid}), 409, "version_not_sealed")
    client.post(f"{API}/dataset-versions/{vid}/seal", headers=alice.h)  # 5/class -> VAL-2 error
    problem(client.post(url, headers=alice.h, json={"dataset_version_id": vid}), 409, "validation_not_passed")
    pb, vb = ready_image_version(client, bob, per_class=12)
    problem(client.post(url, headers=alice.h, json={"dataset_version_id": vb}), 404)  # B's version in A's project
    problem(client.post(f"{API}/projects/{pb['id']}/training-jobs", headers=alice.h, json={"dataset_version_id": vb}), 404)
    t = project(client, alice, template="text_classification", labels=("a", "b"))
    problem(client.post(f"{API}/projects/{t['id']}/training-jobs", headers=alice.h,
                        json={"dataset_version_id": vid}), 404)  # version of another project


def test_image_smoke_end_to_end(client, alice):
    p, vid = ready_image_version(client, alice)
    r = client.post(f"{API}/projects/{p['id']}/training-jobs", headers=alice.h, json={"dataset_version_id": vid})
    assert r.status_code == 202, r.text
    job = client.get(f"{API}/training-jobs/{r.json()['job']['id']}", headers=alice.h).json()
    assert job["status"] == "succeeded", job
    assert job["model_version_id"] and job["progress"]["phase"] == "succeeded"
    trials = client.get(f"{API}/training-jobs/{job['id']}/trials", headers=alice.h).json()
    assert len(trials) == 1 and trials[0]["state"] == "complete" and "macro_f1" in trials[0]["metrics"]
    mv = client.get(f"{API}/model-versions/{job['model_version_id']}", headers=alice.h).json()
    assert mv["status"] == "ready" and mv["framework"] == "timm" and mv["labels"] == ["cat", "dog"]
    meta = mv["model_json"]
    assert meta["schema_version"] == "1.0" and meta["task_type"] == "image_classification"
    assert meta["input"]["size"] == [96, 96] and meta["training"]["device"] == "cpu"
    # metrics.test: null from training; the on_model_ready evaluation fills it when present
    assert meta["metrics"]["test"] is None or "accuracy" in meta["metrics"]["test"]
    assert meta["base_model"]["licence"] == "Apache-2.0"
    assert meta["output"]["temperature"] > 0 and "val_ece_after" in meta["output"]["calibration"]
    with session_factory()() as db:
        row = db.scalar(select(m.ModelVersion).where(m.ModelVersion.id == uuid.UUID(mv["id"])))
        prefix = row.artifact_prefix
    assert prefix == artifacts.prefix(alice.org_id, mv["model_id"], 1)
    assert storage.head(prefix + "native/model.pt") and storage.head(prefix + "training.json")
    preds = artifacts.read_predictions(prefix + "test_predictions.parquet")
    test_count = sum(1 for s in client.get(f"{API}/dataset-versions/{vid}/samples?split=test&limit=200",
                                           headers=alice.h).json()["items"])
    assert len(preds) == test_count == 10
    assert set(preds.columns) == {"sample_id", "y_true", "y_pred", "probs", "confidence"}
    assert all(len(p) == 2 and abs(sum(p) - 1) < 1e-4 for p in preds["probs"])
    model = load_model(mv["id"])
    out = model.predict([png((1, 2, 3)), png((200, 30, 9))])
    assert {o["label"] for o in out} <= {"cat", "dog"} and abs(sum(out[0]["probs"].values()) - 1) < 1e-4
    proj = client.get(f"{API}/projects/{p['id']}", headers=alice.h).json()
    assert proj["status"] == "Trained"
    models = client.get(f"{API}/projects/{p['id']}/models", headers=alice.h).json()
    assert models[0]["versions"][0]["id"] == mv["id"]
    # SSE for a finished job: snapshot + done; token via query string (EventSource)
    with client.stream("GET", f"{API}/training-jobs/{job['id']}/events?access_token={alice.token}") as s:
        body = "".join(s.iter_text())
    assert "event: progress" in body and "event: done" in body and '"succeeded"' in body
    problem(client.get(f"{API}/training-jobs/{job['id']}?access_token={alice.token}"), 401)  # query token: SSE only
    # a second job while one runs -> 409
    with session_factory()() as db:
        db.add(m.TrainingJob(organization_id=uuid.UUID(alice.org_id), project_id=uuid.UUID(p["id"]),
                             dataset_version_id=uuid.UUID(vid), spec_version=1, budget_mode="quick", status="running"))
        db.flush()
        from app.services.project_status import refresh_status

        refresh_status(db, db.scalar(select(m.Project).where(m.Project.id == uuid.UUID(p["id"]))))
        db.commit()
    problem(client.post(f"{API}/projects/{p['id']}/training-jobs", headers=alice.h, json={"dataset_version_id": vid}),
            409, "job_running")
    assert client.get(f"{API}/projects/{p['id']}", headers=alice.h).json()["status"] == "Training"


def test_cancel_and_memory_guard(client, alice, monkeypatch):
    from app.ml import train_tasks

    p, vid = ready_image_version(client, alice)
    with session_factory()() as db:
        job = m.TrainingJob(organization_id=uuid.UUID(alice.org_id), project_id=uuid.UUID(p["id"]),
                            dataset_version_id=uuid.UUID(vid), spec_version=1, budget_mode="quick", status="queued")
        db.add(job)
        db.commit()
        jid = job.id
    r = client.post(f"{API}/training-jobs/{jid}/cancel", headers=alice.h).json()
    assert r["status"] == "cancelled" and r["cancel_requested"]
    train_tasks.run(str(jid))  # a worker picking it up later must not run it
    assert client.get(f"{API}/training-jobs/{jid}", headers=alice.h).json()["status"] == "cancelled"
    assert client.get(f"{API}/projects/{p['id']}", headers=alice.h).json()["status"] == "Ready to train"

    monkeypatch.setenv("TRAIN_MAX_MEMORY_GB", "0.001")  # read by the training child process
    r = client.post(f"{API}/projects/{p['id']}/training-jobs", headers=alice.h, json={"dataset_version_id": vid})
    job = client.get(f"{API}/training-jobs/{r.json()['job']['id']}", headers=alice.h).json()
    assert job["status"] == "failed" and job["error_code"] == "not_enough_memory"
    assert "Not enough memory" in job["error_message"]


def test_cross_tenant_training(client, alice, bob):
    p, vid = ready_image_version(client, alice)
    jid = client.post(f"{API}/projects/{p['id']}/training-jobs", headers=alice.h,
                      json={"dataset_version_id": vid}).json()["job"]["id"]
    mv = client.get(f"{API}/training-jobs/{jid}", headers=alice.h).json()["model_version_id"]
    for method, path in (("get", f"/training-jobs/{jid}"), ("post", f"/training-jobs/{jid}/cancel"),
                         ("get", f"/training-jobs/{jid}/trials"), ("get", f"/projects/{p['id']}/training-jobs"),
                         ("get", f"/projects/{p['id']}/models"), ("get", f"/model-versions/{mv}"),
                         ("get", f"/training-jobs/{jid}/events")):
        problem(client.request(method.upper(), f"{API}{path}", headers=bob.h), 404)


def _lightgbm_ok():
    try:
        import lightgbm  # noqa: F401

        return True
    except OSError:
        return False


@pytest.mark.skipif(not _lightgbm_ok(), reason="lightgbm needs libomp (brew install libomp)")
def test_tabular_smoke(client, alice):
    from tests.test_uploads import CSV

    p = project(client, alice, template="tabular", labels=None)
    mapping = {"columns": [{"name": "churn", "role": "target"}, {"name": "customer_id", "role": "id"},
                           {"name": "patient_id", "role": "ignore"}]}
    r = _import(client, alice, p, "clean.csv", CSV.encode(), "csv", mapping)
    vid = r.json()["dataset_version_id"]
    client.post(f"{API}/dataset-versions/{vid}/seal", headers=alice.h)
    rep = client.get(f"{API}/dataset-versions/{vid}/validation", headers=alice.h).json()
    acks = sorted({i["code"] for i in rep["items"] if i["acknowledgeable"]})
    if acks:
        client.post(f"{API}/dataset-versions/{vid}/validation/acknowledge", headers=alice.h, json={"codes": acks})
    r = client.post(f"{API}/projects/{p['id']}/training-jobs", headers=alice.h, json={"dataset_version_id": vid})
    assert r.status_code == 202, r.text
    job = client.get(f"{API}/training-jobs/{r.json()['job']['id']}", headers=alice.h).json()
    assert job["status"] == "succeeded", job
    mv = client.get(f"{API}/model-versions/{job['model_version_id']}", headers=alice.h).json()
    assert mv["framework"] == "lightgbm" and mv["labels"] == ["0", "1"]
    model = load_model(mv["id"])
    out = model.predict([{"tenure": 3, "monthly_charges": 50.5, "contract": "monthly", "region": "north"}])
    assert out[0]["label"] in ("0", "1")
    _ = (io, json)


@pytest.mark.parametrize("pretrained", ["false", "true"])
def test_detection_smoke_and_own_model_suggestions(client, alice, monkeypatch, pretrained):
    """'true' = the production path with the real (cached) COCO weights: regression for the reduced-tail
    state_dict mismatch that only appeared when training and finalize/loader built different architectures."""
    from tests.test_uploads import _import as imp

    monkeypatch.setenv("TRAIN_PRETRAINED", pretrained)  # read by the training child process

    images, anns = [], []
    for i in range(20):
        images.append((f"images/d{i:02d}.png", png((i * 9 % 255, 40, 200 - i), size=(96, 96))))
        for k, (cat, x) in enumerate(((1, 5 + i % 10), (2, 50))):
            anns.append({"id": len(anns) + 1, "image_id": i + 1, "category_id": cat, "bbox": [x, 20, 30, 40]})
    coco = {"images": [{"id": i + 1, "file_name": f"d{i:02d}.png", "width": 96, "height": 96} for i in range(20)],
            "categories": [{"id": 1, "name": "helmet"}, {"id": 2, "name": "vest"}], "annotations": anns}
    p = project(client, alice, template="object_detection", labels=("helmet", "vest"))
    r = imp(client, alice, p, "coco.json", json.dumps(coco).encode(), "coco", extra_files=images)
    vid = r.json()["dataset_version_id"]
    client.post(f"{API}/dataset-versions/{vid}/seal", headers=alice.h)
    rep = client.get(f"{API}/dataset-versions/{vid}/validation", headers=alice.h).json()
    assert rep["summary"]["errors"] == 0, rep["items"]
    acks = sorted({i["code"] for i in rep["items"] if i["acknowledgeable"]})
    if acks:
        client.post(f"{API}/dataset-versions/{vid}/validation/acknowledge", headers=alice.h, json={"codes": acks})
    r = client.post(f"{API}/projects/{p['id']}/training-jobs", headers=alice.h, json={"dataset_version_id": vid})
    assert r.status_code == 202, r.text
    job = client.get(f"{API}/training-jobs/{r.json()['job']['id']}", headers=alice.h).json()
    assert job["status"] == "succeeded", job
    mv = client.get(f"{API}/model-versions/{job['model_version_id']}", headers=alice.h).json()
    assert mv["framework"] == "torchvision" and mv["base_model"]["licence"].startswith("BSD-3-Clause")
    assert mv["model_json"]["output"]["type"] == "boxes" and "map_50" in mv["metrics"]["val"]
    with session_factory()() as db:
        prefix = db.scalar(select(m.ModelVersion.artifact_prefix).where(m.ModelVersion.id == uuid.UUID(mv["id"])))
    preds = artifacts.read_predictions(prefix + "test_predictions.parquet")
    assert len(preds) >= 1
    truth = json.loads(preds["y_true"].iloc[0])
    assert {b["class_name"] for b in truth} == {"helmet", "vest"} and set(truth[0]["bbox"]) == {"x", "y", "w", "h"}
    assert isinstance(json.loads(preds["y_pred"].iloc[0]), list)
    out = load_model(mv["id"]).predict([images[0][1]])
    assert isinstance(out[0]["boxes"], list)
    sid = client.get(f"{API}/dataset-versions/{vid}/samples?limit=1", headers=alice.h).json()["items"][0]["id"]
    sug = client.post(f"{API}/dataset-versions/{vid}/box-suggestions", headers=alice.h, json={"sample_ids": [sid]}).json()
    assert sug["model"] == f"own:{mv['id']}"


def test_explicit_image_size_env_wins_over_preset(monkeypatch):
    from app.ml.common import budget
    from app.settings import get_settings

    s = get_settings()
    monkeypatch.setattr(s, "train_fast_test", False)
    monkeypatch.setattr(s, "train_low_memory", True)
    monkeypatch.setattr(s, "train_image_size", None)
    assert budget("quick")["image_size"] == 112  # low-memory preset
    monkeypatch.setattr(s, "train_image_size", 96)
    assert budget("quick")["image_size"] == 96 and budget("standard")["image_size"] == 96


DEMO = __import__("pathlib").Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "demo" / "detection"


@pytest.mark.skipif(not (DEMO / "coco.json").exists() or not __import__("os").environ.get("RUN_SLOW"),
                    reason="slow: RUN_SLOW=1 and the demo detection fixtures")
def test_detection_demo_boxes_are_few_and_right(client, alice, monkeypatch):
    """Real path (COCO weights, low-memory budget): predict returns a handful of boxes, not hundreds."""
    from tests.test_uploads import _import as imp

    for k, v in {"TRAIN_FAST_TEST": "false", "TRAIN_PRETRAINED": "true", "TRAIN_DEVICE": "mps"}.items():
        monkeypatch.setenv(k, v)
    images = [(f"images/{q.name}", q.read_bytes()) for q in sorted((DEMO / "images").iterdir())]
    coco = json.loads((DEMO / "coco.json").read_text())
    labels = [c["name"] for c in coco["categories"]]
    p = project(client, alice, template="object_detection", labels=tuple(labels))
    vid = imp(client, alice, p, "coco.json", json.dumps(coco).encode(), "coco",
              extra_files=images).json()["dataset_version_id"]
    client.post(f"{API}/dataset-versions/{vid}/seal", headers=alice.h)
    rep = client.get(f"{API}/dataset-versions/{vid}/validation", headers=alice.h).json()
    acks = sorted({i["code"] for i in rep["items"] if i["acknowledgeable"]})
    if acks:
        client.post(f"{API}/dataset-versions/{vid}/validation/acknowledge", headers=alice.h, json={"codes": acks})
    jid = client.post(f"{API}/projects/{p['id']}/training-jobs", headers=alice.h,
                      json={"dataset_version_id": vid}).json()["job"]["id"]
    job = client.get(f"{API}/training-jobs/{jid}", headers=alice.h).json()
    assert job["status"] == "succeeded", job
    model = load_model(job["model_version_id"])
    gt_counts = {im["file_name"]: 0 for im in coco["images"]}
    for a in coco["annotations"]:
        gt_counts[next(i["file_name"] for i in coco["images"] if i["id"] == a["image_id"])] += 1
    counts = []
    for name, data in images[:10]:
        boxes = model.predict([data])[0]["boxes"]
        counts.append((len(boxes), gt_counts[name.split("/")[-1]]))
    print("predicted vs ground-truth boxes per image:", counts)
    assert all(n <= 20 for n, _ in counts) and sum(n for n, _ in counts) >= sum(g for _, g in counts) * 0.6
