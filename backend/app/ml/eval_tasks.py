"""Celery tasks for evaluation and export. Routed to the single-process "train" queue;
heavy libraries (torch, onnx) are imported inside the task bodies only."""

from __future__ import annotations

import uuid
from collections import Counter

import structlog
from sqlalchemy import select

from app import models as m
from app.db import session_factory
from app.ml import artifacts, evaluation, model_card, recommend
from app.worker import celery

log = structlog.get_logger()


def _load(db, model, row_id):
    return db.scalar(select(model).where(model.id == uuid.UUID(str(row_id))))


def train_counts(db, mv: m.ModelVersion) -> dict[str, int] | None:
    """Training-split label counts (boxes for detection). None for tabular (labels live in the table)."""
    if not mv.dataset_version_id:
        return None
    rows = db.execute(select(m.Label.class_name).join(m.Sample, m.Sample.id == m.Label.sample_id)
                      .where(m.Sample.dataset_version_id == mv.dataset_version_id, m.Sample.split == "train",
                             m.Label.organization_id == mv.organization_id)).scalars().all()
    return dict(Counter(rows)) or None


NEAR_DUP_SHARE = 0.2
PERFECT = 0.99


def diversity(v: m.DatasetVersion | None) -> dict:
    """How varied the data is: near/exact duplicate share from validation (VAL-5/6), plus the structured
    validation_report["diversity"] stat when present. low=True means the test set likely resembles training."""
    if v is None or not v.validation_report:
        return {"low": False}
    rep = v.validation_report
    dup = sum(it["count"] for it in rep.get("items", []) if it["code"] in ("VAL-5", "VAL-6"))
    share = dup / max(v.sample_count or 0, 1)
    structured = rep.get("diversity") or {}
    return {"near_duplicate_share": round(share, 3), **structured,
            "low": bool(structured.get("low")) or share >= NEAR_DUP_SHARE}


def suspicious_perfect(report: dict) -> bool:
    from app.ml.recommend import _HIGHER_IS_BETTER

    p = report.get("primary_metric")
    score = (report.get("metrics") or {}).get(p) if p in _HIGHER_IS_BETTER else None
    return score is not None and score >= PERFECT and report["data_diversity"].get("low", False)


def build_report(db, mv: m.ModelVersion) -> tuple[dict, dict]:
    meta = artifacts.read_json(mv.artifact_prefix + "model.json")
    df = artifacts.read_predictions(mv.artifact_prefix + meta.get("files", {}).get("test_predictions",
                                                                                  "test_predictions.parquet"))
    counts = train_counts(db, mv)
    report = evaluation.evaluate(df, meta, train_counts=counts)
    v = db.scalar(select(m.DatasetVersion).where(m.DatasetVersion.id == mv.dataset_version_id,
                                                 m.DatasetVersion.organization_id == mv.organization_id)) \
        if mv.dataset_version_id else None
    report["data_diversity"] = diversity(v)
    report["suspicious_perfect"] = suspicious_perfect(report)
    report["recommendations"] = recommend.recommend(report, meta, counts)
    return report, meta


@celery.task(name="ml.evaluate")
def evaluate(model_version_id: str) -> None:
    with session_factory()() as db:
        mv = _load(db, m.ModelVersion, model_version_id)
        if not mv or mv.status != "ready" or not mv.artifact_prefix:
            return
        try:
            report, meta = build_report(db, mv)
        except Exception as e:
            log.exception("eval_failed", model_version_id=model_version_id)
            mv.eval_report = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
            db.commit()
            return
        report["status"] = "ready"
        project = db.scalar(select(m.Project).join(m.Model, m.Model.project_id == m.Project.id)
                            .where(m.Model.id == mv.model_id, m.Project.organization_id == mv.organization_id))
        card = model_card.render(meta, report, project.name if project else None)
        meta.setdefault("metrics", {})["test"] = report["metrics"]
        meta["eval"] = {"baseline": report["baseline"], "n_test": report["n_test"]}
        artifacts.write_json(mv.artifact_prefix + "model.json", meta)
        from app.services import storage

        storage.put_bytes(mv.artifact_prefix + "MODEL_CARD.md", card.encode(), "text/markdown")
        mv.eval_report = report
        mv.metrics = {**(mv.metrics or {}), "test": report["metrics"]}
        db.commit()
    log.info("eval_ready", model_version_id=model_version_id)


@celery.task(name="ml.export")
def export(export_id: str) -> None:
    """Runs in a child process: torch and lightgbm ship clashing OpenMP runtimes on macOS (segfault/hang in one
    process), and the worker must survive a converter crash."""
    import subprocess
    import sys
    from pathlib import Path

    import app

    r = subprocess.run([sys.executable, "-m", "app.ml.export", export_id], cwd=Path(app.__file__).parents[1],
                       capture_output=True, text=True, timeout=1800, check=False)
    if r.returncode != 0:
        log.error("export_child_failed", export_id=export_id, code=r.returncode, stderr=r.stderr[-2000:])
        with session_factory()() as db:
            ex = _load(db, m.Export, export_id)
            if ex and ex.status != "ready":
                ex.status, ex.error = "failed", f"Export process exited with code {r.returncode}"
                db.commit()
