"""Improvement recommendations (PRD EVL-6).

Rules over the eval report, model.json and training label counts. Each item:
{"kind", "severity": "high"|"medium"|"low", "title", "detail", "action"} - ordered most useful first.
"""

from __future__ import annotations

from typing import Any

LOW_RECALL = 0.7
FEW_EXAMPLES = 100
CONFUSION_SHARE = 0.15
OVERFIT_GAP = 0.08
NEAR_BASELINE = 0.05
WEAK_AP = 0.5

_HIGHER_IS_BETTER = {"accuracy", "macro_f1", "micro_f1", "weighted_f1", "roc_auc", "pr_auc", "map_50", "map_50_95", "r2"}


def _rec(kind: str, severity: str, title: str, detail: str, action: str) -> dict[str, Any]:
    return {"kind": kind, "severity": severity, "title": title, "detail": detail, "action": action}


def recommend(report: dict, meta: dict, train_counts: dict[str, int] | None = None) -> list[dict]:
    task = report.get("task_type")
    metrics = report.get("metrics") or {}
    primary = report.get("primary_metric") or meta.get("metrics", {}).get("primary_metric")
    val = (meta.get("metrics") or {}).get("val") or {}
    counts = train_counts or {}
    out: list[dict] = []
    detection = task == "object_detection"
    unit = "boxes" if detection else "examples"

    if report.get("suspicious_perfect"):
        div = report.get("data_diversity") or {}
        share = div.get("near_duplicate_share")
        out.append(_rec("too_similar", "high", "Suspiciously perfect: test data looks like the training data",
                        "Test samples are very similar to training samples"
                        + (f" ({share:.0%} of the dataset are near-duplicates)" if share else "")
                        + ", so the score mostly measures memorisation. Real-world accuracy will be lower.",
                        f"Collect varied {unit}: different places, lighting, angles, people and devices, captured in "
                        "separate sessions, then retrain."))

    # Per-class data shortage (classification) / weak classes (detection).
    for row in report.get("per_class") or []:
        c = row["class"]
        n = counts.get(c)
        if detection:
            if row.get("ap50") is not None and row["ap50"] < WEAK_AP and row.get("support"):
                out.append(_rec("class_data", "high", f'"{c}" is detected poorly (AP@0.5 {row["ap50"]:.2f})',
                                f'{n if n is not None else "few"} training {unit}; the model misses or misplaces many "{c}" objects.',
                                f'Label about {max(50, 2 * (n or 25))} more "{c}" boxes, including small, partly hidden and differently lit ones.'))
        elif row.get("recall") is not None and row["recall"] < LOW_RECALL and row.get("support"):
            more = max(50, FEW_EXAMPLES - (n or 0)) if (n or 0) < FEW_EXAMPLES else n // 2
            out.append(_rec("class_data", "high" if row["recall"] < 0.5 else "medium",
                            f'"{c}" has low recall ({row["recall"]:.2f})',
                            f'{f"{n} training {unit}" if n is not None else "Few training examples"}; '
                            f'{row["support"] - round(row["recall"] * row["support"])} of {row["support"]} test examples were missed.',
                            f'Add about {more} more varied "{c}" examples (angles, lighting, backgrounds).'))

    # Confusion between specific pairs.
    cm = report.get("confusion")
    if cm:
        labels, counts_m = cm["labels"], cm["counts"]
        pairs = []
        for i, t in enumerate(labels):
            total = sum(counts_m[i])
            for j, p in enumerate(labels):
                if i != j and total and counts_m[i][j] / total >= CONFUSION_SHARE and counts_m[i][j] >= 2:
                    pairs.append((counts_m[i][j] / total, t, p, counts_m[i][j], total))
        for share, t, p, k, total in sorted(pairs, reverse=True)[:3]:
            out.append(_rec("confusion", "medium", f'"{t}" is often mistaken for "{p}"',
                            f'{k} of {total} "{t}" test examples ({share:.0%}) were predicted as "{p}".',
                            f'Add examples that show what makes "{t}" different from "{p}", and check both classes for mislabels.'))

    # Overfitting gap (validation much better than test).
    if primary and primary in metrics and primary in val and metrics[primary] is not None and val[primary] is not None:
        gap = (val[primary] - metrics[primary]) if primary in _HIGHER_IS_BETTER else (metrics[primary] - val[primary])
        if gap > OVERFIT_GAP:
            out.append(_rec("overfitting", "medium", "Validation score is much higher than test",
                            f"{primary}: {val[primary]:.3f} on validation vs {metrics[primary]:.3f} on test.",
                            "Add more data, especially examples unlike the current ones; check for near-duplicates across splits."))

    # Barely beats the trivial baseline (EVL-8).
    base = (report.get("baseline") or {}).get("metrics") or {}
    for m in ("macro_f1", "r2" if task == "tabular_regression" else ""):
        if m and m in metrics and m in base and metrics[m] is not None and base[m] is not None:
            if metrics[m] - base[m] < NEAR_BASELINE:
                out.append(_rec("baseline", "high", "The model barely beats a trivial guess",
                                f"{m}: model {metrics[m]:.3f} vs baseline {base[m]:.3f} ({report['baseline']['kind']}).",
                                "Check that labels are correct and that the inputs actually contain the signal; add more data."))
            break

    # Class imbalance.
    if len(counts) > 1 and min(counts.values()) > 0 and max(counts.values()) / min(counts.values()) > 10:
        small = min(counts, key=counts.get)
        out.append(_rec("imbalance", "low", "Classes are very imbalanced",
                        f'Largest class has {max(counts.values())} {unit}, "{small}" has {counts[small]}. Class weighting was applied.',
                        f'Collect more "{small}" examples; judge the model on per-class recall, not accuracy.'))

    # Regression: a few huge errors.
    if task == "tabular_regression" and metrics.get("rmse") and metrics.get("mae") and metrics["rmse"] > 2.5 * metrics["mae"]:
        out.append(_rec("outliers", "low", "A few predictions are far off",
                        f'RMSE {metrics["rmse"]:.3g} is much larger than MAE {metrics["mae"]:.3g}: errors are concentrated in a few rows.',
                        "Look at the worst predictions for data-entry errors or a missing feature that explains them."))

    # Budget (only Quick runs in the local build).
    score = metrics.get(primary) if primary else None
    if score is not None and primary in _HIGHER_IS_BETTER and score < 0.85 and (meta.get("training") or {}).get("budget_mode") == "quick":
        out.append(_rec("budget", "low", "A longer search may help",
                        f"This model used the Quick budget ({primary} {score:.3f}).",
                        "When the Standard budget is available, retrain with it; adding data usually helps more."))

    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(out, key=lambda r: order[r["severity"]])
