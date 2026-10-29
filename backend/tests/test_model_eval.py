"""Evaluation task + routes, ONNX export + verification, predict guard.
A random-init timm model stands in for a trained one; the artifact contract is what's under test."""

import io
import uuid

import httpx
import pytest
import torch
from sqlalchemy import select

from app import models as m
from app.db import session_factory
from app.ml import artifacts, hooks
from app.services import storage
from tests.test_auth import problem
from tests.test_uploads import folder_files, project, upload

API = "/api/v1"
ARCH = "mobilenetv3_small_100"
SIZE = 64


def _fake_model(client, user) -> str:
    """Sealed 2-class image dataset + a ready model version written per app/ml/artifacts.py."""
    import timm

    p = project(client, user)
    _, r = upload(client, user, p, folder_files(10))
    vid = r.json()["dataset_version_id"]
    assert client.post(f"{API}/dataset-versions/{vid}/seal", headers=user.h).status_code == 202
    labels = ["cat", "dog"]
    torch.manual_seed(0)
    net = timm.create_model(ARCH, pretrained=False, num_classes=2).eval()
    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)
    with session_factory()() as db:
        org = uuid.UUID(user.org_id)
        model = m.Model(organization_id=org, project_id=uuid.UUID(p["id"]), name="P", task_type="image_classification")
        db.add(model)
        db.flush()
        mv = m.ModelVersion(organization_id=org, model_id=model.id, version_no=1, status="ready", framework="timm", input_spec={"modality": "image"},
                            architecture=ARCH, artifact_prefix=artifacts.prefix(org, model.id, 1), label_classes=labels,
                            dataset_version_id=uuid.UUID(vid), metrics={"val": {"macro_f1": 0.99}})
        db.add(mv)
        db.flush()
        test = db.execute(select(m.Sample.id, m.Label.class_name).join(m.Label, m.Label.sample_id == m.Sample.id)
                          .where(m.Sample.dataset_version_id == mv.dataset_version_id)).all()
        assert test
        spec = {"modality": "image", "size": [SIZE, SIZE], "channels": 3, "color": "RGB", "resize": "bilinear",
                "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}, "layout": "NCHW",
                "dtype": "float32", "scale": 255}
        meta = artifacts.model_json(
            model_version_id=mv.id, project_id=p["id"], organization_id=org, model_id=model.id, version_no=1,
            task_type="image_classification", labels=labels, framework="timm", architecture=ARCH,
            base_model={"registry_id": f"timm/{ARCH}", "licence": "Apache-2.0", "commercial_use": True},
            input_spec=spec, output_spec={"type": "probabilities", "activation": "softmax", "classes": labels},
            dataset_version_id=vid, dataset_content_hash=None, spec_version=1,
            training={"budget_mode": "quick", "trials": 1, "seed": 0, "device": "cpu"},
            val_metrics={"macro_f1": 0.99}, primary_metric="macro_f1")
        artifacts.write_json(mv.artifact_prefix + "model.json", meta)
        storage.put_bytes(mv.artifact_prefix + "native/model.pt", buf.getvalue())
        rows = [{"sample_id": str(sid), "y_true": lb, "y_pred": "cat" if i % 3 else "dog",
                 "probs": [0.8, 0.2] if i % 3 else [0.3, 0.7], "confidence": 0.8 if i % 3 else 0.7}
                for i, (sid, lb) in enumerate(test)]
        artifacts.write_predictions(mv.artifact_prefix + "test_predictions.parquet", rows)
        db.commit()
        return str(mv.id)


def test_evaluate_export_and_guards(client, alice, bob):
    mv_id = _fake_model(client, alice)
    assert client.get(f"{API}/model-versions/{mv_id}/evaluation", headers=alice.h).json()["status"] == "pending"
    hooks.on_model_ready(mv_id)  # CELERY_EAGER: runs inline

    ev = client.get(f"{API}/model-versions/{mv_id}/evaluation", headers=alice.h).json()
    assert ev["status"] == "ready", ev
    rep = ev["report"]
    assert rep["n_test"] == 20 and rep["confusion"]["labels"] == ["cat", "dog"] and rep["baseline"]
    assert rep["recommendations"] and set(ev["samples"]) == {w["sample_id"] for w in rep["worst_mistakes"]}
    assert next(iter(ev["samples"].values()))["thumb_url"]
    cell = client.get(f"{API}/model-versions/{mv_id}/evaluation/cell", params={"y_true": "dog", "y_pred": "cat"},
                      headers=alice.h).json()
    assert cell and all(s["url"] for s in cell)
    card = client.get(f"{API}/model-versions/{mv_id}/model-card", headers=alice.h)
    assert card.status_code == 200 and "Test results" in card.text
    with session_factory()() as db:
        mv = db.scalar(select(m.ModelVersion).where(m.ModelVersion.id == uuid.UUID(mv_id)))
        assert artifacts.read_json(mv.artifact_prefix + "model.json")["metrics"]["test"]["accuracy"] is not None
        assert mv.metrics["test"] == rep["metrics"]

    # tenant isolation
    assert client.get(f"{API}/model-versions/{mv_id}/evaluation", headers=bob.h).status_code == 404
    assert client.post(f"{API}/model-versions/{mv_id}/exports", json={}, headers=bob.h).status_code == 404

    # TFLite is Phase 2
    r = client.post(f"{API}/model-versions/{mv_id}/exports", json={"format": "tflite"}, headers=alice.h)
    problem(r, 422, "phase_2")

    r = client.post(f"{API}/model-versions/{mv_id}/exports", json={"format": "onnx"}, headers=alice.h)
    assert r.status_code == 202, r.text
    ex = client.get(f"{API}/exports/{r.json()['id']}", headers=alice.h).json()
    assert ex["status"] == "ready", ex
    assert ex["verified"] and ex["verify_report"]["max_abs_diff"] < 1e-3 and ex["verify_report"]["n"] == 20
    assert ex["verify_report"]["top1_agreement"] == 1.0
    z = httpx.get(ex["download_url"])
    assert z.status_code == 200 and len(z.content) == ex["size_bytes"]
    import zipfile

    names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    assert {"model.onnx", "model.json", "labels.txt", "README.md", "MODEL_CARD.md"} <= set(names)
    again = client.post(f"{API}/model-versions/{mv_id}/exports", json={"format": "onnx"}, headers=alice.h).json()
    assert again["id"] == ex["id"]  # idempotent
    assert len(client.get(f"{API}/model-versions/{mv_id}/exports", headers=alice.h).json()) == 1

    sn = client.get(f"{API}/model-versions/{mv_id}/snippets", headers=alice.h).json()
    assert "onnxruntime" in sn["python"] and "resize((64, 64)" in sn["python"] and mv_id in sn["curl"]
    compile(sn["python"].replace("predict(...)", "predict('x.jpg')"), "snippet", "exec")


def test_predict_input_checks(client, alice):
    mv_id = _fake_model(client, alice)
    r = client.post(f"{API}/model-versions/{mv_id}/predict", json={}, headers=alice.h)
    problem(r, 422, "input_required")
    for wrong in ({"text": "a cat"}, {"row": {"a": 1}}):
        body = problem(client.post(f"{API}/model-versions/{mv_id}/predict", json=wrong, headers=alice.h), 422, "wrong_input_type")
        assert body["expected"] == "image"
    r = client.post(f"{API}/model-versions/{mv_id}/predict", files={"file": ("a.png", folder_files(1)[0][1])},
                    headers=alice.h)
    assert r.status_code == 200, r.text
    assert r.json()["label"] in ("cat", "dog") and "latency_ms" in r.json()
    assert r.json()["model_load_ms"] >= 0


@pytest.mark.parametrize("modality,payload,expected", [
    ("text", {"files": {"file": ("a.png", b"x")}}, "text"),
    ("text", {"json": {"row": {"a": 1}}}, "text"),
    ("tabular", {"json": {"text": "hi"}}, "row"),
    ("tabular", {"files": {"file": ("a.png", b"x")}}, "row"),
])
def test_predict_wrong_input_type(client, alice, modality, payload, expected):
    mv_id = _fake_model(client, alice)
    with session_factory()() as db:
        mv = db.scalar(select(m.ModelVersion).where(m.ModelVersion.id == uuid.UUID(mv_id)))
        mv.input_spec = {"modality": modality}
        db.commit()
    r = client.post(f"{API}/model-versions/{mv_id}/predict", headers=alice.h, **payload)
    assert problem(r, 422, "wrong_input_type")["expected"] == expected


def test_detection_export_is_phase_2(client, alice):
    mv_id = _fake_model(client, alice)
    with session_factory()() as db:
        db.scalar(select(m.ModelVersion).where(m.ModelVersion.id == uuid.UUID(mv_id))).framework = "torchvision"
        db.commit()
    problem(client.post(f"{API}/model-versions/{mv_id}/exports", json={"format": "onnx"}, headers=alice.h), 422, "phase_2")


def test_detection_snippets_never_500(client, alice):
    mv_id = _fake_model(client, alice)
    with session_factory()() as db:
        mv = db.scalar(select(m.ModelVersion).where(m.ModelVersion.id == uuid.UUID(mv_id)))
        mv.framework = "torchvision"
        meta = artifacts.read_json(mv.artifact_prefix + "model.json")
        meta.update(framework="torchvision", task_type="object_detection", input={"modality": "image", "size": None})
        artifacts.write_json(mv.artifact_prefix + "model.json", meta)
        db.commit()
    r = client.get(f"{API}/model-versions/{mv_id}/snippets", headers=alice.h)
    assert r.status_code == 200, r.text
    assert "Phase 2" in r.json()["python"] and f"/model-versions/{mv_id}/predict" in r.json()["curl"]
    compile(r.json()["python"], "snippet", "exec")
