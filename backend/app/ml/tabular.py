"""Tabular classification / regression: LightGBM with a small Optuna search and early stopping on val."""

from __future__ import annotations

import time

from app.ml.common import Cancelled, Ctx, Result, classification_metrics, pick_metric

LICENCE = {"registry_id": "lightgbm", "source": "LightGBM (trained from data, no pretrained weights)",
           "licence": "MIT", "commercial_use": True}


def _prepare(df, features: list[dict]):
    """All-float matrix: categoricals become their index in `categories` (unseen -> NaN), booleans 0/1.
    Plain numeric features keep the model ONNX-convertible, so serving never needs LightGBM (macOS OpenMP clash)."""
    import numpy as np
    import pandas as pd

    X = pd.DataFrame(index=df.index)
    for f in features:
        col = df[f["name"]]
        if f["type"] in ("categorical", "text"):
            codes = {c: i for i, c in enumerate(f["categories"])}
            X[f["name"]] = col.map(lambda v, c=codes: np.nan if pd.isna(v) else c.get(str(v), np.nan)).astype("float32")
        elif f["type"] == "boolean":
            X[f["name"]] = col.map(lambda v: np.nan if pd.isna(v) else float(str(v).lower() in ("1", "true", "yes"))
                                   ).astype("float32")
        else:
            X[f["name"]] = pd.to_numeric(col, errors="coerce").astype("float32")
    return X


def to_onnx(booster, n_features: int) -> bytes:
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    onx = convert_lightgbm(booster, initial_types=[("input", FloatTensorType([None, n_features]))], zipmap=False,
                           target_opset=15)
    return onx.SerializeToString()


def train(ctx: Ctx) -> Result:
    import lightgbm as lgb
    import numpy as np
    import optuna
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    df = ctx.data["frame"]
    cm = ctx.data["column_mapping"]
    target = cm["target"]
    regression = ctx.task_type == "tabular_regression"
    feat_cols = [c for c in cm["columns"] if c["role"] == "feature" and c["name"] != target]
    split = df["__split__"]
    train_df = df[split == "train"]
    if ctx.budget["max_train"]:
        train_df = train_df.head(ctx.budget["max_train"] * 8)
    val_df, test_df = df[split == "val"], df[split == "test"]
    if val_df.empty:
        val_df = train_df
    features = []
    for c in feat_cols:
        t = c["type"] if c["type"] in ("numeric", "boolean") else "categorical"
        if t == "categorical":
            cats = sorted(train_df[c["name"]].dropna().astype(str).unique().tolist())[:1000]
            features.append({"name": c["name"], "type": t, "categories": cats})
        else:
            features.append({"name": c["name"], "type": t})
    if not features:
        raise ValueError("no feature columns")

    if regression:
        labels: list[str] = []
        y_tr = pd.to_numeric(train_df[target], errors="coerce")
        y_va = pd.to_numeric(val_df[target], errors="coerce")
        objective = {"objective": "regression", "metric": "rmse"}
    else:
        labels = sorted(df[target].dropna().astype(str).unique().tolist())
        cls = {c: i for i, c in enumerate(labels)}
        y_tr = train_df[target].astype(str).map(cls)
        y_va = val_df[target].astype(str).map(cls)
        objective = {"objective": "binary", "metric": "binary_logloss"} if len(labels) == 2 else \
            {"objective": "multiclass", "num_class": len(labels), "metric": "multi_logloss"}
    X_tr, X_va = _prepare(train_df, features), _prepare(val_df, features)
    primary = ctx.spec.get("primary_metric", "rmse" if regression else "macro_f1")
    n_trials, rounds = ctx.budget["trials"], ctx.budget["boost_rounds"]
    best = {"score": None, "booster": None, "params": {}, "metrics": {}}
    trials_log: list[dict] = []

    def predict(booster, X):
        p = booster.predict(X, num_iteration=booster.best_iteration)
        if regression:
            return p, None
        probs = np.column_stack([1 - p, p]) if len(labels) == 2 else p
        return probs.argmax(1), probs

    def score_of(booster, X, y):
        pred, _ = predict(booster, X)
        if regression:
            mse = float(mean_squared_error(y, pred))
            m = {"rmse": mse ** 0.5, "mae": float(mean_absolute_error(y, pred)), "r2": float(r2_score(y, pred))}
            return m, -m["rmse"] if primary not in ("r2",) else m["r2"]
        m = classification_metrics([labels[i] for i in y], [labels[i] for i in pred], labels)
        return m, pick_metric(m, primary)[1]

    def objective_fn(trial: optuna.Trial) -> float:
        ctx.check(f"trial {trial.number}")
        params = {**objective, "verbosity": -1, "num_threads": 2, "seed": ctx.spec["split"]["seed"],
                  "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                  "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
                  "min_child_samples": trial.suggest_int("min_child_samples", 3, 40),
                  "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                  "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10, log=True)}
        if not regression and len(labels) == 2:
            counts = y_tr.value_counts()
            if counts.max() / max(counts.min(), 1) > 10:
                params["is_unbalance"] = True  # TRN-13 (binary; multiclass imbalance is left to the metric)
        t0 = time.monotonic()
        booster = lgb.train(params, lgb.Dataset(X_tr, y_tr), num_boost_round=rounds,
                            valid_sets=[lgb.Dataset(X_va, y_va)],
                            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        metrics, score = score_of(booster, X_va, y_va)
        if best["score"] is None or score > best["score"]:
            best.update(score=score, booster=booster, params={**params, "best_iteration": booster.best_iteration},
                        metrics=metrics)
        name = (primary if primary in metrics else "rmse") if regression else pick_metric(metrics, primary)[0]
        ctx.publish({"phase": "searching", "trial": trial.number + 1, "trials_total": n_trials, "metric_name": name,
                     "current_metric": round(metrics.get(name, score), 4),
                     "best_metric": round(best["metrics"].get(name, best["score"]), 4),
                     "eta_seconds": ctx.eta((trial.number + 1) / n_trials)})
        trials_log.append({"trial_no": trial.number, "params": params, "metrics": metrics, "state": "complete",
                           "seconds": round(time.monotonic() - t0, 2)})
        ctx.log_trial(trial.number, params, metrics, "complete")
        return score

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=ctx.spec["split"]["seed"]))
    cancelled = False
    try:
        study.optimize(objective_fn, n_trials=n_trials)
    except Cancelled:
        if best["booster"] is None:
            raise
        cancelled = True

    ctx.publish({"phase": "finalizing", "message": "Predicting on the test set"})
    booster = best["booster"]
    temperature, calibration = 1.0, None
    if not regression:
        from app.ml.common import calibrate, softmax_t

        _, vp = predict(booster, X_va)
        temperature, calibration = calibrate(np.log(np.clip(vp, 1e-12, 1)), y_va.to_numpy())
    rows = []
    if not test_df.empty:
        X_te = _prepare(test_df, features)
        pred, probs = predict(booster, X_te)
        if not regression:
            probs = softmax_t(np.log(np.clip(probs, 1e-12, 1)), temperature)
        for i, (row_idx, truth) in enumerate(test_df[target].items()):
            if regression:
                rows.append({"sample_id": f"row-{row_idx}", "y_true": str(truth), "y_pred": str(float(pred[i])),
                             "probs": None, "confidence": None})
            else:
                p = probs[i].tolist()
                k = int(pred[i])
                rows.append({"sample_id": f"row-{row_idx}", "y_true": str(truth), "y_pred": labels[k], "probs": p,
                             "confidence": p[k]})
    return Result(
        framework="lightgbm", architecture="lightgbm_gbdt", base_model=LICENCE, labels=labels,
        input_spec={"modality": "tabular", "features": features, "target": target,
                    "encoding": "float32 matrix in `features` order; categorical = index in categories (unseen NaN),"
                                " boolean = 0/1; served from native/model.onnx (input 'input')"},
        output_spec={"type": "value", "target": target} if regression else
        {"type": "probabilities", "activation": "softmax", "classes": labels, "temperature": temperature,
         "calibration": calibration},
        val_metrics=best["metrics"], best_params=best["params"], trials=trials_log,
        native={"native/model.txt": booster.model_to_string(num_iteration=booster.best_iteration).encode(),
                "native/model.onnx": to_onnx(booster, len(features))},
        predictions=rows, cancelled=cancelled, budget_name=ctx.budget["name"],
        library_versions={"lightgbm": lgb.__version__, "optuna": optuna.__version__, "pandas": pd.__version__},
    )

