"""Auto-generated model card (PRD §4.2 deliverables)."""

from __future__ import annotations

from datetime import UTC, datetime

_NAMES = {"map_50": "mAP@0.5", "map_50_95": "mAP@0.5:0.95", "macro_f1": "Macro F1", "roc_auc": "ROC-AUC",
          "pr_auc": "PR-AUC", "rmse": "RMSE", "mae": "MAE", "r2": "R²", "mape": "MAPE"}


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)


def render(meta: dict, report: dict, project_name: str | None = None) -> str:
    m = report.get("metrics") or {}
    base = report.get("baseline") or {}
    bm = meta.get("base_model") or {}
    tr = meta.get("training") or {}
    lines = [
        f"# Model card: {project_name or meta.get('task_type', 'model')} v{meta.get('version_no', '?')}",
        "",
        f"*Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC from the held-out test set. Trained on customer data.*",
        "",
        "## Intended use",
        (f"- Task: **{meta.get('task_type')}** with {len(meta.get('labels') or [])} classes"
         f"{': ' + ', '.join(meta['labels']) if meta.get('labels') else ''}."),
        "- Use it for the inputs it was trained on (same kind of camera, text or table). Re-evaluate before using it elsewhere.",
        "- Not for decisions with legal or similarly significant effects without human review (Acceptable Use Policy).",
        "",
        "## Model",
        f"- Framework: {meta.get('framework')}; architecture: `{meta.get('architecture')}`.",
        (f"- Base model: {bm.get('registry_id') or bm.get('source') or 'n/a'}, licence **{bm.get('licence', 'see notices')}**"
         f"{' (commercial use allowed)' if bm.get('commercial_use') else ''}."),
        f"- Training: budget {tr.get('budget_mode')}, {tr.get('trials')} trials, seed {tr.get('seed')}, device {tr.get('device')}.",
        "",
        "## Data",
        f"- Dataset version `{meta.get('dataset_version_id')}` (content hash `{meta.get('dataset_content_hash')}`), spec v{meta.get('spec_version')}.",
        f"- Test set: {report.get('n_test', 0)} samples, never used for training or model selection.",
        "",
        "## Test results",
        "| Metric | Model | Trivial baseline |",
        "|---|---|---|",
    ]
    for k, v in m.items():
        lines.append(f"| {_NAMES.get(k, k)} | {_fmt(v)} | {_fmt((base.get('metrics') or {}).get(k))} |")
    if base.get("kind"):
        lines.append(f"\nBaseline: {base['kind']}.")
    if report.get("per_class"):
        detection = "ap50" in report["per_class"][0]
        lines += ["", "### Per class", "| Class | " + ("AP@0.5 | AP@0.5:0.95" if detection else "Precision | Recall | F1") + " | Support |",
                  "|---|---|---|" + ("" if detection else "---|") + "---|"]
        for r in report["per_class"]:
            vals = [r["ap50"], r["ap50_95"]] if detection else [r["precision"], r["recall"], r["f1"]]
            lines.append(f"| {r['class']} | " + " | ".join(_fmt(v) for v in vals) + f" | {r['support']} |")
    recs = report.get("recommendations") or []
    lines += ["", "## Known limitations"]
    lines += [f"- {r['title']}: {r['detail']}" for r in recs[:6]] or ["- None flagged by the automatic checks; test it on your own real-world data."]
    lines += ["- Accuracy can drop with different cameras, lighting, languages or populations than the training data.", "",
              "## Licence notices", f"- Base model weights: {bm.get('licence', 'see the registry entry')} ({bm.get('source', 'n/a')}).",
              "- Fine-tuned weights and this card belong to the workspace that trained them."]
    return "\n".join(lines) + "\n"
