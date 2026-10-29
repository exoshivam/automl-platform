"""Celery task `train_job` (queue: train, -c 1): data -> trainer -> artifacts (app/ml/artifacts.py) -> model version.

Progress: Redis pub/sub channel job:{id} (SSE fan-out) + a snapshot in training_jobs.progress.
Cancel: redis key job:{id}:cancel, checked every 10 batches / epoch / trial (<= 30 s).
MLflow is best effort: if it's down the job still runs (NFR-R5).
"""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from typing import Any

import structlog
from sqlalchemy import func, select

from app import models as m
from app.db import now, session_factory
from app.ml import artifacts
from app.services import storage
from app.services.project_status import refresh_status
from app.services.ratelimit import client as redis
from app.settings import get_settings
from app.worker import celery

log = structlog.get_logger()
TERMINAL = ("succeeded", "failed", "cancelled")
SUPPORTED = {"image_classification", "tabular_classification", "tabular_regression", "text_classification",
             "object_detection"}


def channel(job_id) -> str:
    return f"job:{job_id}"


def cancel_key(job_id) -> str:
    return f"job:{job_id}:cancel"


def _load(db, model, row_id):
    return db.scalar(select(model).where(model.id == uuid.UUID(str(row_id))))


def publish(job_id, event: dict[str, Any]) -> None:
    event = {**event, "job_id": str(job_id), "ts": time.time()}
    redis().publish(channel(job_id), json.dumps(event, default=str))
    with session_factory()() as db:  # snapshot for GET /training-jobs/{id} and late SSE subscribers
        job = _load(db, m.TrainingJob, job_id)
        job.progress = {**(job.progress or {}), **event}
        job.heartbeat_at = now()
        db.commit()


class _Mlflow:
    """Best-effort MLflow: one experiment per project, one run per trial."""

    def __init__(self, project_id, job_id):
        self.ok, self.experiment_id = False, None
        uri = get_settings().mlflow_tracking_uri
        if not uri:
            return
        try:
            os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
            os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
            os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
            import mlflow

            mlflow.set_tracking_uri(uri)
            exp = mlflow.set_experiment(f"project-{project_id}")
            self.mlflow, self.experiment_id, self.job_id, self.ok = mlflow, exp.experiment_id, job_id, True
        except Exception as e:  # noqa: BLE001
            log.warning("mlflow_unavailable", error=str(e)[:200])

    def log_trial(self, trial_no: int, params: dict, metrics: dict) -> str | None:
        if not self.ok:
            return None
        try:
            with self.mlflow.start_run(run_name=f"job-{self.job_id}-trial-{trial_no}") as run:
                self.mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
                self.mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, int | float)})
                self.mlflow.set_tag("training_job_id", str(self.job_id))
                return run.info.run_id
        except Exception as e:  # noqa: BLE001
            log.warning("mlflow_log_failed", error=str(e)[:200])
            self.ok = False
            return None


def _image_data(db, v: m.DatasetVersion, labels: list[str]):
    org = v.organization_id
    samples = db.scalars(select(m.Sample).where(m.Sample.dataset_version_id == v.id,
                                                m.Sample.organization_id == org)).all()
    first: dict = {}
    for sid, cls in db.execute(select(m.Label.sample_id, m.Label.class_name).join(
            m.Sample, m.Sample.id == m.Label.sample_id).where(m.Sample.dataset_version_id == v.id,
                                                              m.Label.organization_id == org)
                               .order_by(m.Label.created_at)).all():
        first.setdefault(sid, cls)
    keys = {s.id: s.object_key for s in samples}
    items = [(s.id, first[s.id], s.split) for s in sorted(samples, key=lambda s: str(s.id))
             if s.id in first and not (s.meta or {}).get("corrupt") and s.split and first[s.id] in labels]
    cache: dict = {}

    def fetch(sid):
        if sid not in cache:
            cache[sid] = storage.get_bytes(keys[sid])
        return cache[sid]

    return {"items": items, "fetch": fetch}


def _detection_data(db, v: m.DatasetVersion, labels: list[str]):
    org = v.organization_id
    samples = db.scalars(select(m.Sample).where(m.Sample.dataset_version_id == v.id,
                                                m.Sample.organization_id == org)).all()
    boxes: dict = {}
    for sid, cls, bbox in db.execute(select(m.Label.sample_id, m.Label.class_name, m.Label.bbox).join(
            m.Sample, m.Sample.id == m.Label.sample_id).where(m.Sample.dataset_version_id == v.id,
                                                              m.Label.organization_id == org,
                                                              m.Label.bbox.is_not(None))).all():
        if cls in labels:
            boxes.setdefault(sid, []).append((cls, bbox))
    keys = {s.id: s.object_key for s in samples}
    items = [(s.id, boxes.get(s.id, []), s.split, s.no_objects) for s in sorted(samples, key=lambda s: str(s.id))
             if s.split and not (s.meta or {}).get("corrupt")]
    cache: dict = {}

    def fetch(sid):
        if sid not in cache:
            cache[sid] = storage.get_bytes(keys[sid])
        return cache[sid]

    return {"items": items, "fetch": fetch}


def _tabular_data(v: m.DatasetVersion):
    import io

    import pandas as pd

    return {"frame": pd.read_parquet(io.BytesIO(storage.get_bytes(v.manifest_uri))),
            "column_mapping": v.column_mapping}


def _labels_for(project: m.Project, v: m.DatasetVersion) -> list[str]:
    from app.services.specs import train_labels

    spec_labels = train_labels(project.current_spec)
    present = set((v.label_stats or {}).keys())
    if project.current_spec.get("mode") == "pretrained":  # hybrid: ONE model = chosen COCO classes + custom objects
        return list(project.current_spec.get("labels") or []) + [c for c in spec_labels if c in present]
    ordered = [c for c in spec_labels if c in present] + sorted(present - set(spec_labels))
    return ordered


REPLAY_MAX_TRAIN, REPLAY_MAX_EVAL = 200, 60


def _hybrid_data(project: m.Project, labels: list[str]) -> dict:
    """COCO id per ready-made class, the custom classes, and a replay split of COCO images with the chosen classes
    (train: keeps them in view while fine-tuning; eval: held out to measure forgetting)."""
    import random

    from app.ml.pretrained import COCO

    ids = {n: i for i, n in COCO.items()}
    coco = [c for c in labels if c in ids and c in (project.current_spec.get("labels") or [])]
    custom = [c for c in labels if c not in coco]
    from app.services import replay

    index, index_sha = replay.load()
    index_key = replay.index_key()
    wanted = set(coco)
    rows = [("replay:" + r["key"], [(b["class_name"], b["bbox"]) for b in r["boxes"] if b["class_name"] in wanted],
             None, False) for r in index]
    rows = [r for r in rows if r[1]]
    random.Random(13).shuffle(rows)
    n_eval = min(REPLAY_MAX_EVAL, len(rows) // 4)
    ev = [(sid, b, "replay_eval", f) for sid, b, _, f in rows[:n_eval]]
    tr = [(sid, b, "train", f) for sid, b, _, f in rows[n_eval:n_eval + REPLAY_MAX_TRAIN]]
    return {"coco_ids": {c: ids[c] for c in coco}, "custom": custom,
            "replay": {"train": tr, "eval": ev, "index_key": index_key if index else None, "index_sha256": index_sha,
                       "fetch": lambda sid: storage.get_bytes(sid[len("replay:"):])}}


JOB_TIMEOUT_S = {"quick": 3600, "standard": 4 * 3600, "thorough": 12 * 3600}


@celery.task(name="train_job")
def train_job(job_id: str) -> None:
    """Runs the job in a FRESH child process: memory is fully returned afterwards (16 GB laptops), and torch's and
    LightGBM's OpenMP runtimes never meet in one process (on macOS that segfaults)."""
    import subprocess
    import sys

    with session_factory()() as db:
        job = _load(db, m.TrainingJob, job_id)
        timeout = JOB_TIMEOUT_S.get(job.budget_mode, 3600) if job else 3600
    proc = subprocess.run([sys.executable, "-m", "app.ml.run_job", job_id], capture_output=True, text=True,
                          timeout=timeout, env=os.environ.copy(), check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-1500:]
        log.error("training_process_crashed", job_id=job_id, returncode=proc.returncode, stderr=tail)
        with session_factory()() as db:
            job = _load(db, m.TrainingJob, job_id)
            still_open = job and job.status not in TERMINAL
        if still_open:
            _finish(job_id, "failed", "worker_crashed",
                    f"The training process exited with code {proc.returncode}. {tail[-300:]}")


def run(job_id: str) -> None:
    from app.ml import memory
    from app.ml.common import Cancelled, Ctx, budget
    from app.ml.device import pick_device

    with session_factory()() as db:
        job = _load(db, m.TrainingJob, job_id)
        if job is None or job.status in TERMINAL:
            return
        if job.cancel_requested or redis().get(cancel_key(job_id)):
            job.status, job.finished_at = "cancelled", now()
            db.commit()
            return
        project = _load(db, m.Project, job.project_id)
        v = _load(db, m.DatasetVersion, job.dataset_version_id)
        job.status, job.started_at, job.worker_id = "running", now(), f"{socket.gethostname()}:{os.getpid()}"
        refresh_status(db, project)
        db.commit()
        task, spec, org = project.task_type, dict(project.current_spec), project.organization_id
        labels = _labels_for(project, v)
        if task == "image_classification":
            data = _image_data(db, v, labels)
        elif task == "object_detection":
            data = _detection_data(db, v, labels)
            if project.current_spec.get("mode") == "pretrained":
                data["hybrid"] = _hybrid_data(project, labels)
        elif task == "text_classification":
            from app.ml import text

            data = text.load_data(db, v, labels)
        else:
            data = _tabular_data(v)
        project_id, version_id, content_hash = project.id, v.id, v.content_hash
        spec_version, budget_mode = job.spec_version, job.budget_mode

    device = pick_device() if task in ("image_classification", "text_classification", "object_detection") else "cpu"
    mlf = _Mlflow(project_id, job_id)
    started = time.monotonic()

    def log_trial(trial_no, params, metrics, state):
        run_id = mlf.log_trial(trial_no, params, metrics) if state == "complete" else None
        with session_factory()() as db:
            db.add(m.Trial(organization_id=org, training_job_id=uuid.UUID(job_id), trial_no=trial_no, params=params,
                           state=state, metrics=metrics, mlflow_run_id=run_id,
                           val_metric=next((float(metrics[k]) for k in ("macro_f1", "rmse", "r2", "accuracy")
                                            if k in metrics), None),
                           gpu_seconds=float(metrics.get("seconds", 0)) if device != "cpu" else 0))
            db.commit()

    ctx = Ctx(job_id=job_id, task_type=task, spec=spec, labels=labels, budget=budget(budget_mode), device=device,
              publish=lambda e: publish(job_id, e), is_cancelled=lambda: bool(redis().get(cancel_key(job_id))),
              log_trial=log_trial, data=data)
    publish(job_id, {"phase": "preparing", "device": device, "budget": ctx.budget["name"],
                     "trials_total": ctx.budget["trials"], "status": "running"})
    try:
        memory.check("start")
        if task == "image_classification":
            from app.ml import image as trainer
        elif task == "text_classification":
            from app.ml import text as trainer
        elif task == "object_detection":
            from app.ml import detection as trainer
        else:
            from app.ml import tabular as trainer
        result = trainer.train(ctx)
    except Cancelled:
        _finish(job_id, "cancelled", "cancelled", "Cancelled before any trial finished")
        return
    except memory.NotEnoughMemory as e:
        _finish(job_id, "failed", "not_enough_memory", f"Not enough memory: {e}")
        return
    except ImportError as e:
        _finish(job_id, "failed", "dependency_missing", f"A training library is unavailable: {e}")
        return
    except Exception as e:
        log.exception("training_failed", job_id=job_id)
        _finish(job_id, "failed", "training_error", str(e)[:500])
        return

    seconds = time.monotonic() - started
    for sps in (ctx.data.get("speed") or {}).values():  # feeds the next job's ETA (routers/training.estimate)
        redis().set(f"train_speed:{task}", f"{sps:.6f}")
    mv_id = _save(job_id, org, project_id, version_id, content_hash, spec, spec_version, task, budget_mode, device,
                  result, seconds, mlf.experiment_id)
    _finish(job_id, "cancelled" if result.cancelled else "succeeded", None, None, seconds=seconds)
    try:
        from app.ml import hooks  # enqueues evaluation

        hooks.on_model_ready(mv_id)
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001
        log.warning("on_model_ready_failed", error=str(e)[:200])


def _save(job_id, org, project_id, version_id, content_hash, spec, spec_version, task, budget_mode, device, result,
          seconds, experiment_id) -> uuid.UUID:
    with session_factory()() as db:
        model = db.scalar(select(m.Model).where(m.Model.project_id == project_id, m.Model.organization_id == org))
        if model is None:
            project = _load(db, m.Project, project_id)
            model = m.Model(organization_id=org, project_id=project_id, name=project.name, task_type=task)
            db.add(model)
            db.flush()
        n = (db.scalar(select(func.max(m.ModelVersion.version_no)).where(m.ModelVersion.model_id == model.id,
                                                                         m.ModelVersion.organization_id == org)) or 0) + 1
        mv = m.ModelVersion(organization_id=org, model_id=model.id, version_no=n, training_job_id=uuid.UUID(job_id),
                            status="training", framework=result.framework, architecture=result.architecture,
                            base_model=result.base_model, label_classes=result.labels, input_spec=result.input_spec,
                            dataset_version_id=version_id,
                            metrics={"primary_metric": spec.get("primary_metric"), "val": result.val_metrics,
                                     "test": None})
        db.add(mv)
        db.flush()
        prefix = artifacts.prefix(org, model.id, n)
        for rel, data in result.native.items():
            storage.put_bytes(prefix + rel, data)
        training = {"device": device, "budget_mode": budget_mode, "budget": result.budget_name or budget_mode,
                    "trials": len(result.trials),
                    "best_params": result.best_params, "seed": spec["split"]["seed"], "duration_s": round(seconds, 1),
                    "library_versions": result.library_versions, "cancelled": result.cancelled,
                    "mlflow_experiment_id": experiment_id}
        meta = artifacts.model_json(
            model_version_id=mv.id, project_id=project_id, organization_id=org, model_id=model.id, version_no=n,
            task_type=task, labels=result.labels, framework=result.framework, architecture=result.architecture,
            base_model=result.base_model, input_spec=result.input_spec, output_spec=result.output_spec,
            dataset_version_id=version_id, dataset_content_hash=content_hash, spec_version=spec_version,
            training=training, val_metrics=result.val_metrics, primary_metric=spec.get("primary_metric"))
        artifacts.write_json(prefix + "model.json", meta)
        artifacts.write_predictions(prefix + "test_predictions.parquet", result.predictions)
        artifacts.write_json(prefix + "training.json", {"trials": result.trials, **training})
        mv.artifact_prefix, mv.status = prefix, "ready"
        job = _load(db, m.TrainingJob, job_id)
        job.repro_manifest = {"dataset_version_id": str(version_id), "dataset_content_hash": content_hash,
                              "spec_version": spec_version, "split_seed": spec["split"]["seed"],
                              "budget": training["budget"], "framework": result.framework,
                              "architecture": result.architecture, "base_model": result.base_model,
                              "best_params": result.best_params, "library_versions": result.library_versions,
                              "device": device}
        hybrid = (result.output_spec or {}).get("hybrid")
        if hybrid:  # in-memory pseudo-labels are reproducible from the pinned stock model + threshold + replay set
            job.repro_manifest["hybrid"] = {k: hybrid.get(k) for k in ("stock_model", "pseudo_score", "replay_index",
                                                                       "replay_train", "replay_eval", "pseudo_boxes")}
        job.mlflow_experiment_id = experiment_id
        job.progress = {**(job.progress or {}), "model_version_id": str(mv.id)}
        db.commit()
        return mv.id


def _finish(job_id, status: str, error_code: str | None, error_message: str | None, seconds: float = 0) -> None:
    with session_factory()() as db:
        job = _load(db, m.TrainingJob, job_id)
        job.status, job.finished_at = status, now()
        job.error_code, job.error_message = error_code, error_message
        job.used_gpu_seconds = round(seconds, 1)
        project = _load(db, m.Project, job.project_id)
        db.flush()
        refresh_status(db, project)
        db.commit()
        model_version_id = (job.progress or {}).get("model_version_id")
    redis().delete(cancel_key(job_id))
    publish(job_id, {"phase": status, "status": status, "error_code": error_code, "message": error_message,
                     "model_version_id": model_version_id})
