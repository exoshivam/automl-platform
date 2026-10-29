"""Test-set evaluation (PRD §6.7 EVL-1/2/3/8).

Pure functions over the predictions DataFrame defined by app.ml.artifacts (one row per TEST sample)
plus model.json. No I/O here: app.ml.eval_tasks reads the artifacts and stores the report.

Report shape (stored in model_versions.eval_report, served by GET /model-versions/{id}/evaluation):
{
  "task_type", "primary_metric", "n_test",
  "metrics": {name: float},                       # test metrics (EVL-1); also copied to model.json metrics.test
  "baseline": {"kind", "metrics": {name: float}}, # trivial predictor (EVL-8)
  "per_class": [{"class", "precision", "recall", "f1", "support"} | {"class", "ap50", "ap50_95", "support"}],
  "confusion": {"labels", "counts", "normalized", "cells": {"<true>|<pred>": [sample_id, ...]}} | null,
  "regression": {"scatter": [{"y_true", "y_pred", "sample_id"}], "residual_hist": {"edges", "counts"}} | null,
  "worst_mistakes": [{"sample_id", "y_true", "y_pred", "confidence"}],   # EVL-3
  "recommendations": [...]                         # filled by app.ml.recommend (EVL-6)
}
"""

from __future__ import annotations

import itertools
import json
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics as skm

WORST_N = 24
CELL_IDS = 50
SCATTER_N = 500


def _f(x: Any) -> float | None:
    """JSON-safe float (NaN/inf -> None)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ---------------------------------------------------------------- classification

def classification_metrics(y_true: list[str], y_pred: list[str], labels: list[str], probs: np.ndarray | None) -> dict:
    out = {
        "accuracy": _f(skm.accuracy_score(y_true, y_pred)),
        "macro_f1": _f(skm.f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "micro_f1": _f(skm.f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)),
        "weighted_f1": _f(skm.f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }
    present = sorted(set(y_true))
    if probs is not None and len(present) >= 2:
        idx = {c: i for i, c in enumerate(labels)}
        y_idx = np.array([idx[c] for c in y_true])
        try:
            if len(labels) == 2:
                out["roc_auc"] = _f(skm.roc_auc_score(y_idx, probs[:, 1]))
                out["pr_auc"] = _f(skm.average_precision_score(y_idx, probs[:, 1]))
            else:
                # one-vs-rest over the classes that actually occur in the test set
                cols = [idx[c] for c in present]
                p = probs[:, cols]
                p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
                remap = {c: i for i, c in enumerate(cols)}
                out["roc_auc"] = _f(skm.roc_auc_score([remap[i] for i in y_idx], p, multi_class="ovr", average="macro"))
            out["log_loss"] = _f(skm.log_loss(y_idx, np.clip(probs, 1e-7, 1), labels=list(range(len(labels)))))
        except ValueError:
            pass  # degenerate test set (e.g. one class); AUCs stay absent
    return out


def per_class_table(y_true: list[str], y_pred: list[str], labels: list[str]) -> list[dict]:
    p, r, f, s = skm.precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return [{"class": c, "precision": _f(p[i]), "recall": _f(r[i]), "f1": _f(f[i]), "support": int(s[i])}
            for i, c in enumerate(labels)]


def confusion(df: pd.DataFrame, labels: list[str]) -> dict:
    """EVL-2: counts + row-normalised matrix (rows = true class), and sample ids per cell for click-through."""
    order = list(labels) + sorted({*df.y_true, *df.y_pred} - set(labels))  # unseen predictions get their own column
    cm = skm.confusion_matrix(df.y_true, df.y_pred, labels=order)
    rows = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros_like(cm, dtype=float), where=rows > 0)
    cells: dict[str, list[str]] = {}
    for (t, p), g in df.groupby(["y_true", "y_pred"]):
        cells[f"{t}|{p}"] = g.sort_values("confidence", ascending=False, na_position="last").sample_id.head(CELL_IDS).tolist()
    return {"labels": order, "counts": cm.tolist(), "normalized": np.round(norm, 4).tolist(), "cells": cells}


def worst_mistakes(df: pd.DataFrame, n: int = WORST_N) -> list[dict]:
    """EVL-3: highest-confidence wrong predictions."""
    wrong = df[df.y_true != df.y_pred].sort_values("confidence", ascending=False, na_position="last").head(n)
    return [{"sample_id": r.sample_id, "y_true": r.y_true, "y_pred": r.y_pred, "confidence": _f(r.confidence)}
            for r in wrong.itertuples()]


def majority_baseline(y_true: list[str], labels: list[str], train_counts: dict[str, int] | None) -> dict:
    """EVL-8: always predict the most common TRAIN class (test majority if train counts are unknown)."""
    source = train_counts or pd.Series(y_true).value_counts().to_dict()
    majority = max(source, key=lambda c: source[c])
    pred = [majority] * len(y_true)
    return {"kind": f"always '{majority}'",
            "metrics": {"accuracy": _f(skm.accuracy_score(y_true, pred)),
                        "macro_f1": _f(skm.f1_score(y_true, pred, labels=labels, average="macro", zero_division=0))}}


def reliability(probs: np.ndarray, y_idx: np.ndarray, bins: int = 10) -> dict:
    """EVL-4: top-1 reliability diagram + ECE on the (calibrated) test probabilities."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for lo, hi in itertools.pairwise(edges):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc, c = float((pred[m] == y_idx[m]).mean()), float(conf[m].mean())
            ece += m.mean() * abs(acc - c)
            rows.append({"lo": _f(lo), "hi": _f(hi), "confidence": _f(c), "accuracy": _f(acc), "count": int(m.sum())})
    return {"ece": _f(ece), "bins": rows}


# ---------------------------------------------------------------- regression

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {
        "rmse": _f(math.sqrt(skm.mean_squared_error(y_true, y_pred))),
        "mae": _f(skm.mean_absolute_error(y_true, y_pred)),
        "r2": _f(skm.r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
    }
    # MAPE only where it's meaningful: no targets at (or near) zero.
    if len(y_true) and np.min(np.abs(y_true)) > 1e-6:
        out["mape"] = _f(skm.mean_absolute_percentage_error(y_true, y_pred))
    return out


def mean_baseline(y_true: np.ndarray, train_mean: float | None) -> dict:
    m = float(np.mean(y_true)) if train_mean is None else train_mean
    pred = np.full_like(y_true, m, dtype=float)
    return {"kind": f"always {m:.4g} (mean)", "metrics": regression_metrics(y_true, pred)}


# ---------------------------------------------------------------- detection (COCO-style AP)

def _iou(a: dict, b: dict) -> float:
    ax2, ay2, bx2, by2 = a["x"] + a["w"], a["y"] + a["h"], b["x"] + b["w"], b["y"] + b["h"]
    iw = max(0.0, min(ax2, bx2) - max(a["x"], b["x"]))
    ih = max(0.0, min(ay2, by2) - max(a["y"], b["y"]))
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def _ap(scores: list[float], matched: list[bool], n_gt: int) -> float:
    """101-point interpolated AP (COCO)."""
    if n_gt == 0:
        return float("nan")
    if not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(matched, dtype=float)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # monotone envelope
    out = 0.0
    for t in np.linspace(0, 1, 101):
        i = np.searchsorted(recall, t, side="left")
        out += precision[i] if i < len(precision) else 0.0
    return out / 101


def detection_ap(df: pd.DataFrame, labels: list[str], iou_thresholds=None) -> tuple[dict, list[dict]]:
    """Per-class AP at IoU 0.5 and averaged over 0.5:0.95; boxes are JSON in y_true / y_pred (artifacts contract)."""
    thresholds = iou_thresholds or [round(0.5 + 0.05 * i, 2) for i in range(10)]
    gt = {r.sample_id: json.loads(r.y_true or "[]") for r in df.itertuples()}
    pr = {r.sample_id: json.loads(r.y_pred or "[]") for r in df.itertuples()}
    per_class = []
    for c in labels:
        n_gt = sum(1 for boxes in gt.values() for b in boxes if b["class_name"] == c)
        aps = []
        for t in thresholds:
            scores, matched = [], []
            for sid, preds in pr.items():
                gts = [b for b in gt.get(sid, []) if b["class_name"] == c]
                used = [False] * len(gts)
                for p in sorted((p for p in preds if p["class_name"] == c), key=lambda p: -p.get("confidence", 0)):
                    best, j = 0.0, -1
                    for k, g in enumerate(gts):
                        if not used[k]:
                            v = _iou(p["bbox"], g["bbox"])
                            if v > best:
                                best, j = v, k
                    hit = best >= t and j >= 0
                    if hit:
                        used[j] = True
                    scores.append(p.get("confidence", 0.0))
                    matched.append(hit)
            aps.append(_ap(scores, matched, n_gt))
        per_class.append({"class": c, "ap50": _f(aps[0]), "ap50_95": _f(np.nanmean(aps) if n_gt else float("nan")),
                          "support": n_gt})
    valid = [p for p in per_class if p["support"]]
    m = {"map_50": _f(np.mean([p["ap50"] for p in valid])) if valid else None,
         "map_50_95": _f(np.mean([p["ap50_95"] for p in valid])) if valid else None}
    return m, per_class


def detection_worst(df: pd.DataFrame, n: int = WORST_N, threshold: float = 0.3) -> list[dict]:
    """Images where the predicted boxes disagree most with the truth (missed + spurious, weighted by confidence).
    Only boxes the model would actually show (score >= threshold) count; low-score ones are kept for AP only."""
    scored = []
    for r in df.itertuples():
        g = json.loads(r.y_true or "[]")
        p = [q for q in json.loads(r.y_pred or "[]") if q.get("confidence", 1.0) >= threshold]
        missed = sum(1 for b in g if not any(q["class_name"] == b["class_name"] and _iou(q["bbox"], b["bbox"]) >= 0.5 for q in p))
        spurious = sum(q.get("confidence", 0.5) for q in p
                       if not any(b["class_name"] == q["class_name"] and _iou(q["bbox"], b["bbox"]) >= 0.5 for b in g))
        if missed or spurious:
            scored.append((missed + spurious, r))
    scored.sort(key=lambda x: -x[0])
    return [{"sample_id": r.sample_id, "y_true": r.y_true, "y_pred": r.y_pred, "confidence": _f(r.confidence),
             "error_score": _f(s)} for s, r in scored[:n]]


# ---------------------------------------------------------------- entry point

def evaluate(df: pd.DataFrame, meta: dict, *, train_counts: dict[str, int] | None = None,
             train_mean: float | None = None) -> dict:
    """Build the eval report from test predictions + model.json (see module docstring for the shape)."""
    task = meta["task_type"]
    labels = list(meta.get("labels") or [])
    primary = (meta.get("metrics") or {}).get("primary_metric")
    report: dict[str, Any] = {"task_type": task, "primary_metric": primary, "n_test": len(df), "metrics": {},
                              "baseline": None, "per_class": [], "confusion": None, "regression": None,
                              "worst_mistakes": [], "recommendations": [], "calibration": None}
    if df.empty:
        return report

    if task == "tabular_regression":
        y_true = df.y_true.astype(float).to_numpy()
        y_pred = df.y_pred.astype(float).to_numpy()
        report["metrics"] = regression_metrics(y_true, y_pred)
        report["baseline"] = mean_baseline(y_true, train_mean)
        idx = np.random.default_rng(0).permutation(len(df))[:SCATTER_N]
        report["regression"] = {
            "scatter": [{"y_true": _f(y_true[i]), "y_pred": _f(y_pred[i]), "sample_id": df.sample_id.iloc[i]} for i in idx],
            "residual_hist": dict(zip(("counts", "edges"), (a.tolist() for a in np.histogram(y_pred - y_true, bins=20)))),
        }
        err = np.abs(y_pred - y_true)
        worst = np.argsort(-err)[:WORST_N]
        report["worst_mistakes"] = [{"sample_id": df.sample_id.iloc[i], "y_true": df.y_true.iloc[i],
                                     "y_pred": df.y_pred.iloc[i], "confidence": None, "error": _f(err[i])} for i in worst]
    elif task == "object_detection":
        report["metrics"], report["per_class"] = detection_ap(df, labels)
        report["worst_mistakes"] = detection_worst(df, threshold=(meta.get("output") or {}).get("score_threshold", 0.3))
    else:
        df = df.assign(y_true=df.y_true.astype(str), y_pred=df.y_pred.astype(str))
        probs = None
        if "probs" in df and df.probs.notna().all():
            arr = np.stack(df.probs.to_numpy()).astype(float)
            if arr.shape[1] == len(labels):
                probs = arr / np.clip(arr.sum(axis=1, keepdims=True), 1e-12, None)  # float32 Parquet: renormalise
        report["metrics"] = classification_metrics(df.y_true.tolist(), df.y_pred.tolist(), labels, probs)
        if probs is not None:
            idx = {c: i for i, c in enumerate(labels)}
            known = df.y_true.isin(idx).to_numpy()
            report["calibration"] = {**reliability(probs[known], np.array([idx[c] for c in df.y_true[known]])),
                                     "training": (meta.get("output") or {}).get("calibration")}
            report["metrics"]["ece"] = report["calibration"]["ece"]
        report["per_class"] = per_class_table(df.y_true.tolist(), df.y_pred.tolist(), labels)
        report["confusion"] = confusion(df, labels)
        report["worst_mistakes"] = worst_mistakes(df)
        report["baseline"] = majority_baseline(df.y_true.tolist(), labels, train_counts)
    return report
