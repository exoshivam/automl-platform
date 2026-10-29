"""Pure eval/recommend/model-card checks on synthetic predictions."""

import json

import numpy as np
import pandas as pd
import pytest

from app.ml import evaluation, model_card, recommend


def _cls_df():
    # cat: 8 right, 2 -> dog ; dog: 5 right, 5 -> cat (low recall) ; bird: 4 right
    rows = [("cat", "cat")] * 8 + [("cat", "dog")] * 2 + [("dog", "dog")] * 5 + [("dog", "cat")] * 5 + [("bird", "bird")] * 4
    labels = ["bird", "cat", "dog"]
    probs = [np.array([0.1, 0.1, 0.1], dtype=np.float32) + 0.7 * np.eye(3, dtype=np.float32)[labels.index(p)] for _, p in rows]
    return pd.DataFrame({"sample_id": [f"s{i}" for i in range(len(rows))], "y_true": [t for t, _ in rows],
                         "y_pred": [p for _, p in rows], "probs": probs, "confidence": [0.8] * len(rows)}), labels


def test_classification_report():
    df, labels = _cls_df()
    meta = {"task_type": "image_classification", "labels": labels, "metrics": {"primary_metric": "macro_f1", "val": {"macro_f1": 0.95}}}
    r = evaluation.evaluate(df, meta, train_counts={"bird": 40, "cat": 300, "dog": 60})
    assert r["n_test"] == 24 and r["metrics"]["accuracy"] == pytest.approx(17 / 24)
    assert {"roc_auc", "log_loss", "macro_f1"} <= r["metrics"].keys()
    cal = r["calibration"]
    assert sum(b["count"] for b in cal["bins"]) == 24 and 0 <= cal["ece"] <= 1 and r["metrics"]["ece"] == cal["ece"]
    assert r["confusion"]["labels"] == labels
    assert r["confusion"]["counts"][2] == [0, 5, 5]  # dog row
    assert len(r["confusion"]["cells"]["dog|cat"]) == 5
    assert len(r["worst_mistakes"]) == 7 and all(m["y_true"] != m["y_pred"] for m in r["worst_mistakes"])
    assert r["baseline"]["kind"] == "always 'cat'"
    recs = recommend.recommend(r, meta, {"bird": 40, "cat": 300, "dog": 60})
    kinds = [x["kind"] for x in recs]
    assert "class_data" in kinds and "confusion" in kinds and "overfitting" in kinds
    sev = [x["severity"] for x in recs]
    assert sev == sorted(sev, key=["high", "medium", "low"].index)
    r["recommendations"] = recs
    card = model_card.render(meta, r, "Pets")
    assert "# Model card: Pets" in card and "| dog |" in card


def test_regression_report():
    rng = np.random.default_rng(1)
    y = rng.uniform(10, 20, 50)
    p = y + rng.normal(0, 0.5, 50)
    df = pd.DataFrame({"sample_id": [f"r{i}" for i in range(50)], "y_true": [str(v) for v in y], "y_pred": [str(v) for v in p],
                       "probs": [None] * 50, "confidence": [None] * 50})
    r = evaluation.evaluate(df, {"task_type": "tabular_regression", "metrics": {"primary_metric": "rmse"}})
    assert r["metrics"]["rmse"] < 1 and r["metrics"]["r2"] > 0.8 and "mape" in r["metrics"]
    assert r["baseline"]["metrics"]["rmse"] > r["metrics"]["rmse"]
    assert len(r["regression"]["scatter"]) == 50 and sum(r["regression"]["residual_hist"]["counts"]) == 50
    assert r["worst_mistakes"][0]["error"] >= r["worst_mistakes"][-1]["error"]
    json.dumps(r)  # JSON-safe


def _box(c, x, y, w=10, h=10, conf=None):
    b = {"class_name": c, "bbox": {"x": x, "y": y, "w": w, "h": h}}
    return {**b, "confidence": conf} if conf is not None else b


def test_detection_ap():
    gt = [[_box("car", 0, 0), _box("person", 50, 50)], [_box("car", 20, 20)]]
    perfect = [[_box("car", 0, 0, conf=0.9), _box("person", 50, 50, conf=0.8)], [_box("car", 20, 20, conf=0.7)]]
    df = pd.DataFrame({"sample_id": ["a", "b"], "y_true": [json.dumps(g) for g in gt], "y_pred": [json.dumps(p) for p in perfect],
                       "probs": [None, None], "confidence": [0.9, 0.7]})
    meta = {"task_type": "object_detection", "labels": ["car", "person"], "metrics": {"primary_metric": "map_50"}}
    r = evaluation.evaluate(df, meta)
    assert r["metrics"]["map_50"] == pytest.approx(1.0) and r["worst_mistakes"] == []
    # low-score boxes (kept for AP) don't count as visible mistakes
    noisy = [p + [_box("person", 200, 200, conf=0.06)] for p in perfect]
    assert evaluation.evaluate(df.assign(y_pred=[json.dumps(p) for p in noisy]), meta)["worst_mistakes"] == []
    # miss the person, shift one car by half a box: AP@0.5 drops, 0.5:0.95 drops more
    worse = [[_box("car", 0, 0, conf=0.9)], [_box("car", 23, 20, conf=0.7)]]
    r2 = evaluation.evaluate(df.assign(y_pred=[json.dumps(p) for p in worse]), meta)
    pc = {p["class"]: p for p in r2["per_class"]}
    assert pc["person"]["ap50"] == 0 and pc["car"]["ap50"] == pytest.approx(1.0)
    assert pc["car"]["ap50_95"] < 1 and r2["worst_mistakes"][0]["sample_id"] == "a"
    assert any(x["kind"] == "class_data" for x in recommend.recommend(r2, meta, {"car": 30, "person": 5}))


def test_empty_and_single_class():
    meta = {"task_type": "text_classification", "labels": ["neg", "pos"], "metrics": {"primary_metric": "macro_f1"}}
    empty = pd.DataFrame(columns=["sample_id", "y_true", "y_pred", "probs", "confidence"])
    assert evaluation.evaluate(empty, meta)["metrics"] == {}
    one = pd.DataFrame({"sample_id": ["x"], "y_true": ["pos"], "y_pred": ["pos"], "probs": [np.array([0.2, 0.8])], "confidence": [0.8]})
    r = evaluation.evaluate(one, meta)
    assert "roc_auc" not in r["metrics"] and r["metrics"]["accuracy"] == 1.0


def test_suspiciously_perfect_flagged_only_on_low_diversity():
    from types import SimpleNamespace

    from app.ml import eval_tasks

    rep = {"task_type": "image_classification", "primary_metric": "macro_f1", "metrics": {"macro_f1": 1.0},
           "per_class": [], "confusion": None, "baseline": None}
    dupes = SimpleNamespace(sample_count=100, validation_report={"items": [{"code": "VAL-6", "count": 33}]})
    varied = SimpleNamespace(sample_count=100, validation_report={"items": [{"code": "VAL-6", "count": 2}]})
    for v, expected in ((dupes, True), (varied, False)):
        rep["data_diversity"] = eval_tasks.diversity(v)
        rep["suspicious_perfect"] = eval_tasks.suspicious_perfect(rep)
        assert rep["suspicious_perfect"] is expected
        kinds = [r["kind"] for r in recommend.recommend(rep, {})]
        assert ("too_similar" in kinds) is expected
    rep["metrics"]["macro_f1"] = 0.9
    rep["data_diversity"] = eval_tasks.diversity(dupes)
    assert eval_tasks.suspicious_perfect(rep) is False
