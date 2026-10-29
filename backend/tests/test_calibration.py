"""EVL-4: temperature scaling fitted on val lowers calibration error on held-out data."""

import os
import pathlib

import numpy as np
import pytest

from app.ml.common import ece, fit_temperature, softmax_t


def test_temperature_fixes_overconfidence_on_held_out_data():
    rng = np.random.default_rng(0)

    def make(n):  # true 3-class logits, then a 4x-overconfident model (logits scaled up)
        y = rng.integers(0, 3, n)
        base = rng.normal(0, 1, (n, 3))
        base[np.arange(n), y] += 1.2
        return base * 4.0, y

    val_logits, val_y = make(400)
    test_logits, test_y = make(400)
    t = fit_temperature(val_logits, val_y)
    assert 2.5 < t < 6.0  # recovers roughly the 4x scaling
    before, after = ece(softmax_t(test_logits), test_y), ece(softmax_t(test_logits, t), test_y)
    assert after < before / 2, (before, after)
    assert fit_temperature(val_logits[:2], val_y[:2]) == 1.0  # too little data: leave probabilities alone
    from app.ml.common import calibrate

    t2, info = calibrate(val_logits, val_y)
    assert info["applied"] and t2 == t
    well = softmax_t(val_logits, t)  # an already-calibrated model: a refit must never make val ECE worse
    _, info2 = calibrate(np.log(well), val_y)
    assert info2["val_ece_after"] <= info2["val_ece_before"]


SHAPES = pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "demo" / "shapes"


@pytest.mark.skipif(not SHAPES.exists() or not os.environ.get("RUN_SLOW"), reason="slow: RUN_SLOW=1 + demo shapes")
def test_demo_shapes_calibration(client, alice, monkeypatch):
    from tests.test_uploads import project, upload

    for k, v in {"TRAIN_FAST_TEST": "false", "TRAIN_PRETRAINED": "true", "TRAIN_DEVICE": "mps"}.items():
        monkeypatch.setenv(k, v)
    files = [(f"{p.parent.name}/{p.name}", p.read_bytes()) for p in sorted(SHAPES.rglob("*.png"))]
    p = project(client, alice, labels=tuple(sorted({f.split("/")[0] for f, _ in files})))
    vid = upload(client, alice, p, files)[1].json()["dataset_version_id"]
    client.post(f"/api/v1/dataset-versions/{vid}/seal", headers=alice.h)
    rep = client.get(f"/api/v1/dataset-versions/{vid}/validation", headers=alice.h).json()
    acks = sorted({i["code"] for i in rep["items"] if i["acknowledgeable"]})
    if acks:
        client.post(f"/api/v1/dataset-versions/{vid}/validation/acknowledge", headers=alice.h, json={"codes": acks})
    jid = client.post(f"/api/v1/projects/{p['id']}/training-jobs", headers=alice.h,
                      json={"dataset_version_id": vid}).json()["job"]["id"]
    job = client.get(f"/api/v1/training-jobs/{jid}", headers=alice.h).json()
    assert job["status"] == "succeeded", job
    meta = client.get(f"/api/v1/model-versions/{job['model_version_id']}", headers=alice.h).json()["model_json"]
    cal = meta["output"]["calibration"]
    print("demo shapes calibration:", cal)
    assert meta["output"]["temperature"] == cal["temperature"] > 0
    assert cal["val_ece_after"] <= cal["val_ece_before"]  # T is kept only when it helps


def test_calibration_threshold_and_clamp():
    from app.ml.common import calibrate

    rng = np.random.default_rng(1)

    def make(n, scale):
        y = rng.integers(0, 3, n)
        base = rng.normal(0, 1, (n, 3))
        base[np.arange(n), y] += 1.2
        return base * scale, y

    logits, y = make(27, 4.0)  # overconfident but only 27 val samples (the fragile real case)
    t, info = calibrate(logits, y)
    assert t == 1.0 and not info["applied"] and info["reason"] == "val_too_small" and info["val_samples"] == 27
    logits, y = make(400, 1.0)  # already calibrated: any gain is below max(0.01, 10%)
    t, info = calibrate(logits, y)
    assert t == 1.0 and info["reason"] == "gain_too_small"
    assert info["val_ece_before"] - ece(softmax_t(logits, info["clamped_temperature"]), y) < info["min_gain"]
    logits, y = make(400, 12.0)  # wildly overconfident: fitted T ~12 is clamped to 5, still applied
    t, info = calibrate(logits, y)
    assert info["fitted_temperature"] > 5 and t == info["clamped_temperature"] == 5.0 and info["applied"]
    assert info["reason"] is None and info["val_ece_before"] - info["val_ece_after"] >= info["min_gain"]
