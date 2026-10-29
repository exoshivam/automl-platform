"""Text trainer on tests/fixtures/text/mixed.csv, then the artifact contract end to end: eval, ONNX export (child
process), TextModel inference. Needs the cached multilingual checkpoint; skipped without it."""

import csv
import pathlib
import uuid

import pytest
from sqlalchemy import select

from app import models as m
from app.db import session_factory
from app.ml import artifacts, eval_tasks, text
from app.ml.common import Ctx, budget
from app.services import storage
from tests.test_uploads import project

FX = pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "text" / "mixed.csv"


def _cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(text.CHECKPOINT, "model.safetensors"), str)
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not (FX.exists() and _cached()), reason="text fixture or cached checkpoint missing")


TRAIN_BUDGET_GB = 3.0  # what the text trainer itself may add on top of the process (real jobs run in a fresh child)


def test_text_train_eval_export_predict(client, alice, monkeypatch, settings):
    import psutil

    # The memory guard reads process RSS, which in a full run includes earlier tests' leftovers: budget from here.
    base = psutil.Process().memory_info().rss / 1e9
    monkeypatch.setattr(settings, "train_max_memory_gb", round(base + TRAIN_BUDGET_GB, 1))
    rows = list(csv.DictReader(FX.open(encoding="utf-8")))
    labels = sorted({r["label"] for r in rows})
    ids = [uuid.uuid4() for _ in rows]
    split = lambda i: "test" if i % 5 == 0 else "val" if i % 5 == 1 else "train"
    items = [(ids[i], r["label"], split(i)) for i, r in enumerate(rows)]
    events, trials = [], []
    ctx = Ctx(job_id="t", task_type="text_classification", spec={"primary_metric": "macro_f1"}, labels=labels,
              budget=budget("quick"), device="cpu", publish=events.append, is_cancelled=lambda: False,
              log_trial=lambda *a: trials.append(a), data={"items": items, "texts": {ids[i]: r["text"] for i, r in enumerate(rows)}})
    res = text.train(ctx)
    assert res.framework == "transformers" and len(trials) == max(budget("quick")["trials"], text.MIN_TRIALS)
    assert any(e.get("phase") == "searching" for e in events) and "macro_f1" in res.val_metrics
    assert len(res.predictions) == sum(1 for *_, s in items if s == "test")
    assert {"native/hf/model.safetensors", "native/hf/config.json", "native/hf/tokenizer.json"} <= res.native.keys()

    # write it exactly as train_tasks does, on real sample rows so eval/export can find the texts
    p = project(client, alice, template="text_classification", labels=labels)
    with session_factory()() as db:
        org = uuid.UUID(alice.org_id)
        ds = db.scalar(select(m.Dataset).where(m.Dataset.id == uuid.UUID(p["dataset_id"])))
        v = m.DatasetVersion(organization_id=org, dataset_id=ds.id, version_no=99)
        db.add(v)
        db.flush()
        for i, r in enumerate(rows):
            db.add(m.Sample(id=ids[i], organization_id=org, dataset_version_id=v.id, split=split(i), meta={"text": r["text"]}))
            db.flush()
            db.add(m.Label(organization_id=org, sample_id=ids[i], class_name=r["label"]))
        model = m.Model(organization_id=org, project_id=uuid.UUID(p["id"]), name="T", task_type="text_classification")
        db.add(model)
        db.flush()
        mv = m.ModelVersion(organization_id=org, model_id=model.id, version_no=1, status="ready", framework=res.framework,
                            architecture=res.architecture, artifact_prefix=artifacts.prefix(org, model.id, 1),
                            label_classes=labels, input_spec=res.input_spec, dataset_version_id=v.id)
        db.add(mv)
        db.flush()
        meta = artifacts.model_json(
            model_version_id=mv.id, project_id=p["id"], organization_id=org, model_id=model.id, version_no=1,
            task_type="text_classification", labels=labels, framework=res.framework, architecture=res.architecture,
            base_model=res.base_model, input_spec=res.input_spec, output_spec=res.output_spec, dataset_version_id=v.id,
            dataset_content_hash=None, spec_version=1, training={"budget_mode": "quick"}, val_metrics=res.val_metrics,
            primary_metric="macro_f1")
        artifacts.write_json(mv.artifact_prefix + "model.json", meta)
        artifacts.write_predictions(mv.artifact_prefix + "test_predictions.parquet", res.predictions)
        for rel, data in res.native.items():
            storage.put_bytes(mv.artifact_prefix + rel, data)
        ex = m.Export(organization_id=org, model_version_id=mv.id, format="onnx")
        db.add(ex)
        db.commit()
        mv_id, ex_id, prefix = mv.id, ex.id, mv.artifact_prefix

    eval_tasks.evaluate(str(mv_id))
    eval_tasks.export(str(ex_id))
    with session_factory()() as db:
        mv = db.scalar(select(m.ModelVersion).where(m.ModelVersion.id == mv_id))
        ex = db.scalar(select(m.Export).where(m.Export.id == ex_id))
        assert mv.eval_report["status"] == "ready" and mv.eval_report["confusion"]["labels"][: len(labels)] == labels
        assert ex.status == "ready", ex.error
        assert ex.verified and ex.verify_report["top1_agreement"] == 1.0, ex.verify_report

    tm = text.TextModel(artifacts.read_json(prefix + "model.json"), prefix)
    out = tm.predict(["Delivery was super fast, loved it", "पैसे वापस नहीं मिले"])
    assert all(o["label"] in labels and abs(sum(o["probs"].values()) - 1) < 1e-4 for o in out)
    with pytest.raises(TypeError):
        tm.predict([b"bytes"])
