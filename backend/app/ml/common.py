"""Shared pieces for trainers: budgets, the per-job context, the trainer result, metrics."""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.settings import get_settings

# Quick must finish in minutes on a laptop CPU/MPS (PRD TRN-4). "fast_test" is only for unit tests.
BUDGETS = {
    "fast_test": {"trials": 1, "epochs": 1, "max_train": 64, "image_size": 96, "boost_rounds": 50, "patience": 1},
    "low_memory": {"trials": 2, "epochs": 10, "max_train": None, "image_size": 112, "boost_rounds": 300,
                   "patience": 2, "max_batch": 16},
    "quick": {"trials": 2, "epochs": 8, "max_train": None, "image_size": None, "boost_rounds": 300, "patience": 2},
    "standard": {"trials": 6, "epochs": 15, "max_train": None, "image_size": None, "boost_rounds": 1000, "patience": 3},
    "thorough": {"trials": 12, "epochs": 25, "max_train": None, "image_size": None, "boost_rounds": 3000, "patience": 4},
}


class Cancelled(Exception):
    pass


def budget(mode: str) -> dict[str, Any]:
    s = get_settings()
    name = "fast_test" if s.train_fast_test else "low_memory" if s.train_low_memory and mode == "quick" else mode
    b = dict(BUDGETS[name])
    b["image_size"] = s.train_image_size or b["image_size"] or 160  # an explicit TRAIN_IMAGE_SIZE always wins
    b["name"], b["mode"] = name, mode
    return b


@dataclass
class Ctx:
    """What a trainer gets. publish() pushes a live progress event; check() raises Cancelled / NotEnoughMemory."""

    job_id: str
    task_type: str
    spec: dict[str, Any]
    labels: list[str]
    budget: dict[str, Any]
    device: str
    publish: Callable[[dict[str, Any]], None]
    is_cancelled: Callable[[], bool]
    log_trial: Callable[[int, dict, dict, str], None]  # (trial_no, params, metrics, state) -> DB + MLflow
    data: dict[str, Any] = field(default_factory=dict)
    started: float = field(default_factory=time.monotonic)

    def check(self, stage: str) -> None:
        from app.ml import memory

        if self.is_cancelled():
            raise Cancelled(stage)
        memory.check(stage)

    def eta(self, done_fraction: float) -> int | None:
        if done_fraction <= 0:
            return None
        elapsed = time.monotonic() - self.started
        return int(elapsed / done_fraction - elapsed)


@dataclass
class Result:
    framework: str
    architecture: str
    base_model: dict[str, Any]
    labels: list[str]
    input_spec: dict[str, Any]
    output_spec: dict[str, Any]
    val_metrics: dict[str, Any]
    best_params: dict[str, Any]
    trials: list[dict[str, Any]]
    native: dict[str, bytes]  # relative path -> bytes (written under the version prefix)
    predictions: list[dict[str, Any]]  # rows per artifacts.PREDICTIONS_SCHEMA
    cancelled: bool = False
    library_versions: dict[str, str] = field(default_factory=dict)
    budget_name: str | None = None  # effective budget (e.g. low_memory), recorded in model.json


def classification_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))}


def softmax_t(logits, t: float = 1.0):
    import numpy as np

    z = np.asarray(logits, dtype=np.float64) / t
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits, y) -> float:
    """EVL-4 calibration: the single T minimising val NLL of softmax(logits / T) (golden-section on log T)."""
    import numpy as np

    logits, y = np.asarray(logits, dtype=np.float64), np.asarray(y)
    if len(y) < 4:
        return 1.0

    def nll(log_t):
        p = softmax_t(logits, float(np.exp(log_t)))
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean()

    lo, hi, g = np.log(0.05), np.log(20.0), (5 ** 0.5 - 1) / 2
    for _ in range(60):
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        lo, hi = (lo, b) if nll(a) < nll(b) else (a, hi)
    return round(float(np.exp((lo + hi) / 2)), 4)


def ece(probs, y, bins: int = 10) -> float:
    """Expected calibration error (top-1)."""
    import numpy as np

    probs, y = np.asarray(probs), np.asarray(y)
    if not len(y):
        return 0.0
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in itertools.pairwise(edges):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return round(float(total), 4)


CAL_MIN_VAL = 30  # fewer val samples: ECE is too noisy to justify changing the model's probabilities
CAL_T_RANGE = (0.5, 5.0)


def calibrate(val_logits, val_y) -> tuple[float, dict]:
    """Fit T on val (clamped to CAL_T_RANGE); apply it only with >= CAL_MIN_VAL samples and a val ECE gain of at least
    max(0.01 absolute, 10% relative). Otherwise T = 1 and `reason` says why."""
    n = len(val_y)
    fitted = fit_temperature(val_logits, val_y)
    t = round(min(max(fitted, CAL_T_RANGE[0]), CAL_T_RANGE[1]), 4)
    before, after = ece(softmax_t(val_logits), val_y), ece(softmax_t(val_logits, t), val_y)
    need = max(0.01, 0.1 * before)
    reason = ("val_too_small" if n < CAL_MIN_VAL else "gain_too_small" if before - after < need else None)
    applied = reason is None
    return (t if applied else 1.0), {"temperature": t if applied else 1.0, "fitted_temperature": fitted,
                                     "clamped_temperature": t, "applied": applied, "reason": reason,
                                     "val_samples": n, "min_gain": round(need, 6), "val_ece_before": before,
                                     "val_ece_after": after if applied else before, "fitted_on": "val"}


def pick_metric(metrics: dict[str, float], primary: str) -> tuple[str, float]:
    """The search optimises the spec's primary metric when we compute it, else macro_f1."""
    if primary in metrics:
        return primary, metrics[primary]
    return "macro_f1", metrics.get("macro_f1", 0.0)
