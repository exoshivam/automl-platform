"""Tabular ONNX export verified on the real test rows against native predictions.
A sklearn model converted with skl2onnx stands in for LightGBM: same served-file contract (native/model.onnx, input
"input" float32 [N, F], outputs label + probabilities), and no lightgbm in the pytest process (macOS OpenMP clash)."""

import io
import uuid

import numpy as np
import pandas as pd
from sqlalchemy import select

from app import models as m
from app.db import session_factory
from app.ml import artifacts, eval_tasks
from app.ml.tabular import _prepare
from app.services import storage
from tests.test_uploads import project


def _setup(client, user, regression: bool):
    from skl2onnx import to_onnx
    from sklearn.linear_model import LinearRegression, LogisticRegression

    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "c": rng.choice(["a", "b"], n)})
    df["y"] = (df.x1 * 3 + df.x2 + (df.c == "a")) if regression else np.where(df.x1 + df.x2 > 0, "yes", "no")
    features = [{"name": "x1", "type": "numeric"}, {"name": "x2", "type": "numeric"},
                {"name": "c", "type": "categorical", "categories": ["a", "b"]}]
    X = _prepare(df, features).to_numpy(np.float32)
    test = df.index[df.index % 4 == 0]
    labels = [] if regression else ["no", "yes"]
    if regression:
        est = LinearRegression().fit(X, df.y)
        rows = [{"sample_id": f"row-{i}", "y_true": str(df.y[i]), "y_pred": str(float(est.predict(X[i:i + 1])[0]))}
                for i in test]
    else:
        est = LogisticRegression().fit(X, df.y)
        rows = []
        for i in test:
            p = est.predict_proba(X[i:i + 1])[0].tolist()
            rows.append({"sample_id": f"row-{i}", "y_true": df.y[i], "y_pred": labels[int(np.argmax(p))], "probs": p,
                         "confidence": max(p)})
    onx = to_onnx(est, X[:1], options={id(est): {"zipmap": False}} if not regression else None, target_opset=15)
    onx.graph.input[0].name = "input"
    for node in onx.graph.node:
        node.input[:] = ["input" if x == "X" else x for x in node.input]
    p = project(client, user, template="tabular", labels=None,
                task_type="tabular_regression" if regression else "tabular_classification")
    with session_factory()() as db:
        org = uuid.UUID(user.org_id)
        buf = io.BytesIO()
        df.to_parquet(buf)
        key = f"{storage.org_prefix(org)}test/manifest-{uuid.uuid4()}.parquet"
        storage.put_bytes(key, buf.getvalue())
        v = m.DatasetVersion(organization_id=org, dataset_id=uuid.UUID(p["dataset_id"]), version_no=7, manifest_uri=key)
        db.add(v)
        model = m.Model(organization_id=org, project_id=uuid.UUID(p["id"]), name="T", task_type=p["task_type"])
        db.add(model)
        db.flush()
        mv = m.ModelVersion(organization_id=org, model_id=model.id, version_no=1, status="ready", framework="lightgbm",
                            architecture="lightgbm_gbdt", artifact_prefix=artifacts.prefix(org, model.id, 1),
                            label_classes=labels, input_spec={"modality": "tabular"}, dataset_version_id=v.id)
        db.add(mv)
        db.flush()
        meta = artifacts.model_json(
            model_version_id=mv.id, project_id=p["id"], organization_id=org, model_id=model.id, version_no=1,
            task_type=p["task_type"], labels=labels, framework="lightgbm", architecture="lightgbm_gbdt", base_model={},
            input_spec={"modality": "tabular", "features": features, "target": "y"},
            output_spec={"type": "value"} if regression else {"type": "probabilities", "classes": labels},
            dataset_version_id=v.id, dataset_content_hash=None, spec_version=1, training={}, val_metrics={},
            primary_metric="rmse" if regression else "accuracy")
        artifacts.write_json(mv.artifact_prefix + "model.json", meta)
        artifacts.write_predictions(mv.artifact_prefix + "test_predictions.parquet", rows)
        storage.put_bytes(mv.artifact_prefix + "native/model.onnx", onx.SerializeToString())
        ex = m.Export(organization_id=org, model_version_id=mv.id, format="onnx")
        db.add(ex)
        db.commit()
        return ex.id


def _run(ex_id):
    eval_tasks.export(str(ex_id))
    with session_factory()() as db:
        return db.scalar(select(m.Export).where(m.Export.id == ex_id))


def test_tabular_classification_export_has_accuracy(client, alice):
    ex = _run(_setup(client, alice, regression=False))
    r = ex.verify_report
    assert ex.status == "ready", ex.error
    assert ex.verified and r["n"] == 30 and r["top1_agreement"] == 1.0, r
    assert r["onnx_accuracy"] == r["native_accuracy"] and ex.accuracy_delta == 0.0


def test_tabular_regression_export_has_rmse(client, alice):
    ex = _run(_setup(client, alice, regression=True))
    r = ex.verify_report
    assert ex.status == "ready", ex.error
    assert ex.verified and abs(r["onnx_rmse"] - r["native_rmse"]) < 1e-3 and ex.accuracy_delta is not None, r
