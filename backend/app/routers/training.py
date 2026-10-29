"""TRN: training jobs (gated on a validated, sealed dataset version), cancel, trials, live SSE events, models."""

import json
import time
import uuid
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from app import models as m
from app.deps import DbDep, PrincipalDep, TenantRef, audit, authorize, check_refs, get_by_id, load_project
from app.errors import ApiError, not_found
from app.services import storage
from app.services.project_status import refresh_status
from app.services.ratelimit import client as redis

router = APIRouter(tags=["training"])
Budget = Literal["quick", "standard", "thorough"]
SUPPORTED = {"image_classification", "tabular_classification", "tabular_regression", "text_classification",
             "object_detection"}
TERMINAL = ("succeeded", "failed", "cancelled")
ESTIMATE_MIN = {"quick": 10, "standard": 45, "thorough": 240}


class JobCreate(BaseModel):
    dataset_version_id: uuid.UUID = TenantRef(m.DatasetVersion, "A sealed, validated version of this project's dataset")
    budget_mode: Budget | None = Field(default=None, description="Defaults to the spec's budget_mode")


class JobOut(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    dataset_version_id: uuid.UUID
    spec_version: int
    budget_mode: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    cancel_requested: bool
    progress: dict[str, Any]
    error_code: str | None
    error_message: str | None
    model_version_id: uuid.UUID | None
    used_gpu_seconds: float
    repro_manifest: dict[str, Any]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class JobCreated(BaseModel):
    job: JobOut
    estimate: dict[str, Any]


class TrialOut(BaseModel):
    trial_no: int
    state: str
    params: dict[str, Any]
    metrics: dict[str, Any]
    val_metric: float | None
    gpu_seconds: float
    mlflow_run_id: str | None
    created_at: datetime


class ModelVersionOut(BaseModel):
    id: uuid.UUID
    model_id: uuid.UUID
    project_id: uuid.UUID
    version_no: int
    status: str
    framework: str | None
    architecture: str | None
    base_model: dict[str, Any]
    labels: list[str]
    metrics: dict[str, Any]
    eval_report: dict[str, Any] | None
    training_job_id: uuid.UUID | None
    dataset_version_id: uuid.UUID | None
    model_json: dict[str, Any] | None = None
    created_at: datetime


class ModelOut(BaseModel):
    id: uuid.UUID
    name: str
    task_type: str
    versions: list[ModelVersionOut]


# Seconds per (training sample x epoch) before any job has been measured on this machine.
DEFAULT_SPS = {"image_classification": 0.04, "tabular_classification": 0.0004, "tabular_regression": 0.0004,
               "text_classification": 0.05, "object_detection": 0.4}


def estimate(task: str, mode: str, samples: int) -> dict[str, Any]:
    """ETA = training samples x expected epochs x trials x measured sec/sample (learned from finished jobs)."""
    from app.ml.common import budget

    b = budget(mode)
    measured = redis().get(f"train_speed:{task}")
    sps = float(measured) if measured else DEFAULT_SPS.get(task, 0.05)
    n_train = int(samples * 0.7) if not b["max_train"] else min(int(samples * 0.7), b["max_train"])
    epochs = 1 if task.startswith("tabular") else min(b["epochs"], b["patience"] + 4)  # early stopping, typically
    seconds = n_train * epochs * b["trials"] * sps + 20  # + model download/setup
    return {"budget_mode": mode, "effective_budget": b["name"], "trials": b["trials"], "max_epochs": b["epochs"],
            "eta_minutes": round(seconds / 60, 1), "eta_seconds": int(seconds), "samples": samples,
            "measured": bool(measured), "gpu_minutes": round(seconds / 60, 1)}


def job_out(j: m.TrainingJob) -> JobOut:
    mv = (j.progress or {}).get("model_version_id")
    return JobOut(id=j.id, project_id=j.project_id, dataset_version_id=j.dataset_version_id,
                  spec_version=j.spec_version, budget_mode=j.budget_mode, status=j.status,
                  cancel_requested=j.cancel_requested, progress=j.progress or {}, error_code=j.error_code,
                  error_message=j.error_message, model_version_id=uuid.UUID(mv) if mv else None,
                  used_gpu_seconds=j.used_gpu_seconds or 0, repro_manifest=j.repro_manifest or {},
                  created_at=j.created_at, started_at=j.started_at,
                  finished_at=j.finished_at)


def _job(db, principal, job_id, action="read") -> m.TrainingJob:
    j = get_by_id(db, m.TrainingJob, job_id)
    if not j:
        raise not_found("Training job")
    authorize(db, principal, j.organization_id, action, "Training job")
    return j


@router.post("/projects/{project_id}/training-jobs", status_code=202, response_model=JobCreated)
def start_training(project_id: uuid.UUID, body: JobCreate, principal: PrincipalDep, db: DbDep, request: Request):
    p = load_project(db, principal, project_id, "write")
    check_refs(db, body, p.organization_id)
    v = get_by_id(db, m.DatasetVersion, body.dataset_version_id)
    ds = get_by_id(db, m.Dataset, v.dataset_id)
    if ds.project_id != p.id:
        raise not_found("Dataset version")
    if p.task_type not in SUPPORTED:
        raise ApiError(422, "task_not_supported_yet", f"Training for {p.task_type} isn't available yet")
    if p.current_spec.get("mode") == "pretrained" and not p.current_spec.get("custom_labels"):
        raise ApiError(409, "pretrained_mode", "This project uses the ready-made detector; add a custom object "
                       "(or convert it to custom) first")
    if p.current_spec.get("mode") == "pretrained":  # hybrid: replay keeps the ready-made classes from being forgotten
        from app.services import replay
        from app.settings import get_settings

        need, rows = get_settings().hybrid_min_replay, replay.load()[0]
        have = replay.images_with(rows, set(p.current_spec.get("labels") or []))
        if have < need:
            raise ApiError(409, "replay_unavailable", "Training now would make the ready-made objects be forgotten: "
                           f"{have} COCO replay images contain the chosen ready-made classes, {need} are needed",
                           extra={"replay_images": have, "required": need})
    if not v.sealed_at or v.ingest_status != "ready":
        raise ApiError(409, "version_not_sealed", "Seal the dataset version first")
    if v.validation_status != "passed":
        raise ApiError(409, "validation_not_passed", "Fix validation errors and acknowledge warnings before training",
                       extra={"validation_status": v.validation_status})
    active = db.scalar(select(m.TrainingJob).where(m.TrainingJob.project_id == p.id,
                                                   m.TrainingJob.organization_id == p.organization_id,
                                                   m.TrainingJob.status.in_(("queued", "running"))).limit(1))
    if active:
        raise ApiError(409, "job_running", "A training job is already running for this project",
                       extra={"training_job_id": str(active.id)})
    mode = body.budget_mode or p.current_spec.get("budget_mode", "quick")
    job = m.TrainingJob(organization_id=p.organization_id, project_id=p.id, dataset_version_id=v.id,
                        spec_version=p.spec_version, budget_mode=mode, status="queued", created_by=principal.user_id,
                        progress={"phase": "queued", "status": "queued"},
                        reserved_gpu_seconds=ESTIMATE_MIN[mode] * 60)
    db.add(job)
    db.flush()
    refresh_status(db, p)
    audit(db, request, principal, p.organization_id, "training.start", "training_job", job.id, budget=mode)
    db.commit()  # the worker reads the job from its own session
    from app.ml.train_tasks import train_job

    train_job.delay(str(job.id))
    db.refresh(job)
    return JobCreated(job=job_out(job), estimate=estimate(p.task_type, mode, v.sample_count))


@router.get("/projects/{project_id}/training-jobs", response_model=list[JobOut])
def list_jobs(project_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    p = load_project(db, principal, project_id, "read")
    rows = db.scalars(select(m.TrainingJob).where(m.TrainingJob.project_id == p.id,
                                                  m.TrainingJob.organization_id == p.organization_id)
                      .order_by(m.TrainingJob.created_at.desc()).limit(50)).all()
    return [job_out(j) for j in rows]


@router.get("/training-jobs/{job_id}", response_model=JobOut)
def get_job(job_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    return job_out(_job(db, principal, job_id))


@router.post("/training-jobs/{job_id}/cancel", status_code=202, response_model=JobOut)
def cancel_job(job_id: uuid.UUID, principal: PrincipalDep, db: DbDep, request: Request):
    from app.ml.train_tasks import cancel_key, publish

    j = _job(db, principal, job_id, "write")
    if j.status in TERMINAL:
        return job_out(j)
    j.cancel_requested = True
    redis().set(cancel_key(j.id), "1", ex=24 * 3600)
    if j.status == "queued":
        j.status = "cancelled"
        db.flush()
        refresh_status(db, get_by_id(db, m.Project, j.project_id))
    audit(db, request, principal, j.organization_id, "training.cancel", "training_job", j.id)
    db.commit()
    if j.status == "cancelled":
        publish(j.id, {"phase": "cancelled", "status": "cancelled"})
    return job_out(j)


@router.get("/training-jobs/{job_id}/trials", response_model=list[TrialOut])
def list_trials(job_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    j = _job(db, principal, job_id)
    rows = db.scalars(select(m.Trial).where(m.Trial.training_job_id == j.id, m.Trial.organization_id == j.organization_id)
                      .order_by(m.Trial.trial_no)).all()
    return [TrialOut.model_validate(t, from_attributes=True) for t in rows]


@router.get("/training-jobs/{job_id}/events", responses={200: {"content": {"text/event-stream": {}}}})
def job_events(job_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    """SSE: `progress` events, then one `done` event. EventSource can't send headers: use ?access_token=<jwt>."""
    from app.ml.train_tasks import channel

    j = _job(db, principal, job_id)
    snapshot = {**(j.progress or {}), "status": j.status}
    jid = j.id

    def stream():
        yield f"event: progress\ndata: {json.dumps(snapshot, default=str)}\n\n"
        if snapshot["status"] in TERMINAL:
            yield f"event: done\ndata: {json.dumps(snapshot, default=str)}\n\n"
            return
        ps = redis().pubsub()
        ps.subscribe(channel(jid))
        last = time.monotonic()
        try:
            while True:
                msg = ps.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if msg and msg["type"] == "message":
                    data = json.loads(msg["data"])
                    yield f"event: progress\ndata: {json.dumps(data)}\n\n"
                    last = time.monotonic()
                    if data.get("status") in TERMINAL:
                        yield f"event: done\ndata: {json.dumps(data)}\n\n"
                        return
                elif time.monotonic() - last > 15:
                    yield ": keep-alive\n\n"
                    last = time.monotonic()
        finally:
            ps.close()

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def version_out(db, mv: m.ModelVersion, include_meta: bool = False) -> ModelVersionOut:
    model = get_by_id(db, m.Model, mv.model_id)
    meta = None
    if include_meta and mv.artifact_prefix:
        try:
            meta = json.loads(storage.get_bytes(mv.artifact_prefix + "model.json"))
        except Exception:  # noqa: BLE001
            meta = None
    return ModelVersionOut(id=mv.id, model_id=mv.model_id, project_id=model.project_id, version_no=mv.version_no,
                           status=mv.status, framework=mv.framework, architecture=mv.architecture,
                           base_model=mv.base_model or {}, labels=mv.label_classes or [], metrics=mv.metrics or {},
                           eval_report=mv.eval_report, training_job_id=mv.training_job_id,
                           dataset_version_id=mv.dataset_version_id, model_json=meta, created_at=mv.created_at)


@router.get("/projects/{project_id}/models", response_model=list[ModelOut])
def list_models(project_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    p = load_project(db, principal, project_id, "read")
    out = []
    for model in db.scalars(select(m.Model).where(m.Model.project_id == p.id, m.Model.organization_id == p.organization_id)):
        versions = db.scalars(select(m.ModelVersion).where(m.ModelVersion.model_id == model.id,
                                                           m.ModelVersion.organization_id == p.organization_id)
                              .order_by(m.ModelVersion.version_no.desc())).all()
        out.append(ModelOut(id=model.id, name=model.name, task_type=model.task_type,
                            versions=[version_out(db, v) for v in versions]))
    return out


@router.get("/model-versions/{version_id}", response_model=ModelVersionOut)
def get_model_version(version_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    mv = get_by_id(db, m.ModelVersion, version_id)
    if not mv:
        raise not_found("Model version")
    authorize(db, principal, mv.organization_id, "read", "Model version")
    return version_out(db, mv, include_meta=True)

