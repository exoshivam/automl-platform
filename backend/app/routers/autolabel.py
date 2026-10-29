"""Bulk auto-label for object detection: start / follow / cancel a job, list suggestions, accept / reject / edit them.
Suggestions live in box_suggestions, never in labels, so training / validation / stats / seal can't see them until a
person accepts: then they become ordinary labels (source "auto") exactly as labels:batch writes them."""

import uuid
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select, update

from app import models as m
from app.db import now
from app.deps import DbDep, PrincipalDep, TenantRef, audit, check_refs, get_by_id, load_version, require_current
from app.errors import ApiError, not_found
from app.services import storage
from app.services.specs import train_labels

router = APIRouter(tags=["autolabel"])
Source = Literal["auto", "text_prompt", "own_model", "ready_made"]
ACTIVE = ("queued", "running")


class JobIn(BaseModel):
    source: Source = "auto"
    classes: list[str] | None = Field(default=None, max_length=50, description="Default: every trainable label")
    prompts: dict[str, str] = Field(default_factory=dict, description="Optional English prompt per label "
                                    "(text_prompt), e.g. {'charger': 'white phone charger'}")
    min_confidence: float = Field(default=0.3, ge=0.05, le=0.95)
    scope: Literal["unlabelled", "sample_ids"] = "unlabelled"
    sample_ids: list[uuid.UUID] | None = TenantRef(m.Sample, "Photos of this version (scope=sample_ids)")


class JobOut(BaseModel):
    id: uuid.UUID
    dataset_version_id: uuid.UUID
    status: Literal["queued", "running", "done", "failed", "cancelled"]
    source: str
    classes: list[str]
    min_confidence: float
    progress: dict[str, Any]
    counts: dict[str, int]
    error: str | None
    created_at: datetime
    finished_at: datetime | None


class EditIn(BaseModel):
    id: uuid.UUID
    bbox: dict[str, float] | None = None
    class_name: str | None = Field(default=None, max_length=200)


class ReviewIn(BaseModel):
    accept: list[uuid.UUID] = Field(default_factory=list)
    reject: list[uuid.UUID] = Field(default_factory=list)
    edit: list[EditIn] = Field(default_factory=list)
    accept_samples: list[uuid.UUID] = TenantRef(m.Sample, "Accept every pending box of these photos")
    reject_samples: list[uuid.UUID] = TenantRef(m.Sample, "Reject every pending box of these photos")
    accept_above: float | None = Field(default=None, ge=0, le=1)
    reject_remaining: bool = False


def _job_out(j: m.AutolabelJob) -> JobOut:
    res = j.results or {}
    return JobOut(id=j.id, dataset_version_id=j.dataset_version_id, status=j.status, source=j.source,
                  classes=j.classes, min_confidence=j.min_confidence, progress=j.progress or {},
                  counts={"suggested": sum(res.values()), "samples_with_boxes": sum(1 for n in res.values() if n),
                          "none_found": sum(1 for n in res.values() if not n)},
                  error=j.error, created_at=j.created_at, finished_at=j.finished_at)


def _project(db, v: m.DatasetVersion) -> m.Project:
    ds = get_by_id(db, m.Dataset, v.dataset_id)
    p = get_by_id(db, m.Project, ds.project_id)
    if p.task_type != "object_detection":
        raise ApiError(422, "not_detection", "Auto-labelling boxes is for object detection projects")
    return p


@router.get("/dataset-versions/{version_id}/autolabel-sources")
def sources(version_id: uuid.UUID, principal: PrincipalDep, db: DbDep) -> dict[str, Any]:
    from app.ml import pretrained
    from app.services.detector import OwnModelDetector, detector, text_prompt_available
    from app.settings import get_settings

    v = load_version(db, principal, version_id, "read")
    p = _project(db, v)
    own = detector(db, p)
    own_classes = (db.scalar(select(m.ModelVersion.label_classes).where(m.ModelVersion.id == own.model_version_id))
                   or []) if isinstance(own, OwnModelDetector) else []
    coco = pretrained.canonical(train_labels(p.current_spec))[0]
    tm = get_settings().autolabel_text_model
    return {"text_prompt": {"available": text_prompt_available(), "model_id": tm, "licence": "Apache-2.0",
                            "note": "Finds objects from their name; works with zero boxes drawn"},
            "own_model": {"available": isinstance(own, OwnModelDetector),
                          "model_version_id": own.model_version_id if isinstance(own, OwnModelDetector) else None,
                          "classes": own_classes},
            "ready_made": {"available": bool(coco), "model_id": pretrained.MODEL["id"],
                           "licence": pretrained.MODEL["licence"], "classes": coco}}


@router.post("/dataset-versions/{version_id}/autolabel-jobs", status_code=202, response_model=JobOut)
def start(version_id: uuid.UUID, body: JobIn, principal: PrincipalDep, db: DbDep, request: Request):
    from app.ml import autolabel
    from app.services.detector import text_prompt_available

    v = load_version(db, principal, version_id, "write")
    p = _project(db, v)
    if v.sealed_at is not None:
        raise ApiError(409, "version_sealed", "Auto-label the open draft (add photos or start a new draft first)")
    require_current(db, v, "auto-label")
    check_refs(db, body, v.organization_id, scope={"samples": [m.Sample.dataset_version_id == v.id]})
    known = train_labels(p.current_spec)
    classes = body.classes or known
    unknown = sorted(set(classes) - set(known))
    if unknown or not classes:
        raise ApiError(422, "unknown_class", f"Not a class of this project: {unknown}" if unknown else "No classes")
    if db.scalar(select(m.AutolabelJob.id).where(m.AutolabelJob.dataset_version_id == v.id,
                                                 m.AutolabelJob.status.in_(ACTIVE)).limit(1)):
        raise ApiError(409, "job_running", "An auto-label job is already running for this version")
    routes = autolabel.plan(db, p, classes, body.source)
    if "text_prompt" in routes.values() and not text_prompt_available():
        raise ApiError(409, "source_unavailable", "The text-prompt detector isn't installed yet")
    if body.source == "own_model" and not hasattr(__import__("app.services.detector", fromlist=["x"]).detector(db, p),
                                                   "model_version_id"):
        raise ApiError(409, "source_unavailable", "No trained detector yet")
    j = m.AutolabelJob(organization_id=v.organization_id, dataset_version_id=v.id, created_by=principal.user_id,
                       source=body.source, classes=classes, prompts=body.prompts, min_confidence=body.min_confidence,
                       sample_ids=[str(s) for s in body.sample_ids] if body.scope == "sample_ids" and body.sample_ids
                       else None, progress={"done": 0, "total": None, "plan": routes})
    db.add(j)
    db.flush()
    audit(db, request, principal, v.organization_id, "autolabel.start", "autolabel_job", j.id, source=body.source,
          classes=classes)
    db.commit()  # the worker reads the row
    autolabel.autolabel.delay(str(j.id))
    return _job_out(j)


@router.get("/dataset-versions/{version_id}/autolabel-jobs", response_model=list[JobOut])
def list_jobs(version_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    v = load_version(db, principal, version_id, "read")
    rows = db.scalars(select(m.AutolabelJob).where(m.AutolabelJob.dataset_version_id == v.id,
                                                   m.AutolabelJob.organization_id == v.organization_id)
                      .order_by(m.AutolabelJob.created_at.desc()).limit(20)).all()
    return [_job_out(j) for j in rows]


def _job(db, principal, job_id, action) -> m.AutolabelJob:
    j = get_by_id(db, m.AutolabelJob, job_id)
    if j is None:
        raise not_found("Auto-label job")
    load_version(db, principal, j.dataset_version_id, action)
    return j


@router.get("/autolabel-jobs/{job_id}", response_model=JobOut)
def get_job(job_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    return _job_out(_job(db, principal, job_id, "read"))


@router.post("/autolabel-jobs/{job_id}:cancel", response_model=JobOut)
def cancel(job_id: uuid.UUID, principal: PrincipalDep, db: DbDep, request: Request):
    j = _job(db, principal, job_id, "write")
    if j.status in ACTIVE:
        j.cancel_requested = True
        if j.status == "queued":
            j.status, j.finished_at = "cancelled", now()
        audit(db, request, principal, j.organization_id, "autolabel.cancel", "autolabel_job", j.id)
    return _job_out(j)


@router.get("/dataset-versions/{version_id}/suggestions")
def list_suggestions(version_id: uuid.UUID, principal: PrincipalDep, db: DbDep,
                     state: Literal["needs_review", "accepted", "rejected", "none_found"] | None = None,
                     sample_id: uuid.UUID | None = None, min_confidence: float = 0.0) -> dict[str, Any]:
    v = load_version(db, principal, version_id, "read")
    org = v.organization_id
    sugg = db.scalars(select(m.BoxSuggestion).where(m.BoxSuggestion.dataset_version_id == v.id,
                                                    m.BoxSuggestion.organization_id == org)).all()
    processed = set()
    for j in db.scalars(select(m.AutolabelJob).where(m.AutolabelJob.dataset_version_id == v.id,
                                                     m.AutolabelJob.organization_id == org)):
        processed |= set((j.results or {}).keys())
    by: dict[uuid.UUID, list[m.BoxSuggestion]] = {}
    for s in sugg:
        by.setdefault(s.sample_id, []).append(s)
    ids = set(by) | {uuid.UUID(x) for x in processed}

    def state_of(rows: list) -> str:
        st = {r.status for r in rows}
        return ("none_found" if not rows else "needs_review" if "pending" in st else
                "accepted" if "accepted" in st else "rejected")

    counts = {"needs_review": 0, "accepted": 0, "rejected": 0, "none_found": 0,
              "pending_boxes": sum(s.status == "pending" for s in sugg),
              "confirmed_boxes": db.scalar(select(func.count()).select_from(m.Label).join(
                  m.Sample, m.Sample.id == m.Label.sample_id).where(
                  m.Sample.dataset_version_id == v.id, m.Label.organization_id == org,
                  m.Label.bbox.is_not(None))) or 0}
    samples = {s.id: s for s in db.scalars(select(m.Sample).where(m.Sample.id.in_(ids), m.Sample.organization_id == org,
                                                                   m.Sample.dataset_version_id == v.id))} if ids else {}
    labels: dict = {}
    for lb in db.scalars(select(m.Label).where(m.Label.sample_id.in_(list(samples)), m.Label.organization_id == org,
                                               m.Label.bbox.is_not(None))) if samples else []:
        labels.setdefault(lb.sample_id, []).append({"id": str(lb.id), "class_name": lb.class_name, "bbox": lb.bbox,
                                                    "source": lb.source, "confidence": lb.confidence})
    out = []
    for sid in sorted(samples, key=lambda x: (samples[x].original_name or "", str(x))):
        rows = by.get(sid, [])
        st = state_of(rows)
        counts[st] += 1
        if (state and st != state) or (sample_id and sid != sample_id):
            continue
        smp = samples[sid]
        out.append({"sample_id": str(sid), "name": smp.original_name, "width": smp.width, "height": smp.height,
                    "thumb_url": storage.presign_get(smp.thumbnail_key) if smp.thumbnail_key else None,
                    "url": storage.presign_get(smp.object_key) if smp.object_key else None, "state": st,
                    "labels": labels.get(sid, []),
                    "boxes": [{"id": str(r.id), "class_name": r.class_name, "bbox": r.bbox, "confidence": r.confidence,
                               "source": r.source, "model_id": r.model_id, "status": r.status}
                              for r in sorted(rows, key=lambda r: -r.confidence) if r.confidence >= min_confidence]})
    return {"samples": out, "counts": counts}


@router.post("/dataset-versions/{version_id}/suggestions:review")
def review(version_id: uuid.UUID, body: ReviewIn, principal: PrincipalDep, db: DbDep, request: Request) -> dict:
    from app.routers.labels import writable_version
    from app.services.ingest import refresh_stats
    from app.services.project_status import refresh_status

    v = load_version(db, principal, version_id, "write")
    p = _project(db, v)
    require_current(db, v, "review suggestions on")  # 409 stale_version, as for labels
    check_refs(db, body, v.organization_id, scope={"samples": [m.Sample.dataset_version_id == v.id]})
    org = v.organization_id
    pending = {s.id: s for s in db.scalars(select(m.BoxSuggestion).where(
        m.BoxSuggestion.dataset_version_id == v.id, m.BoxSuggestion.organization_id == org,
        m.BoxSuggestion.status == "pending"))}
    edits = {e.id: e for e in body.edit}
    acc_ids = set(body.accept) | set(edits)
    rej_ids = set(body.reject)
    acc_samples, rej_samples = set(body.accept_samples or []), set(body.reject_samples or [])
    for s in pending.values():
        if s.sample_id in acc_samples or (body.accept_above is not None and s.confidence >= body.accept_above):
            acc_ids.add(s.id)
        if s.sample_id in rej_samples:
            rej_ids.add(s.id)
    unknown = (acc_ids | rej_ids) - set(pending)
    if unknown:
        raise ApiError(404, "not_found", f"{len(unknown)} suggestion(s) aren't pending in this version")
    if body.reject_remaining:
        rej_ids |= set(pending) - acc_ids
    rej_ids -= acc_ids  # accepting wins over a blanket reject
    known = set(train_labels(p.current_spec))
    for e in body.edit:
        if e.class_name and e.class_name not in known:
            raise ApiError(422, "unknown_class", f"'{e.class_name}' isn't a class of this project")
    target, id_map = writable_version(db, v)  # sealed current -> forks a draft (suggestions move with it)
    ls = db.scalar(select(m.LabelSet).where(m.LabelSet.project_id == p.id, m.LabelSet.organization_id == org)
                   .order_by(m.LabelSet.version.desc()).limit(1))
    created, models_used = 0, set()
    for sid in acc_ids:
        s = pending[sid]
        e = edits.get(sid)
        bbox = {**s.bbox, **(e.bbox or {})} if e else s.bbox
        lb = m.Label(organization_id=org, sample_id=id_map.get(s.sample_id, s.sample_id),
                     label_set_id=ls.id if ls else None, class_name=(e.class_name if e and e.class_name else
                                                                      s.class_name),
                     bbox=bbox, source="auto", confidence=s.confidence, labeled_by=principal.user_id,
                     reviewed_by=principal.user_id)
        db.add(lb)
        db.flush()
        s.status, s.label_id, s.reviewed_by = "accepted", lb.id, principal.user_id
        if e:
            s.bbox, s.class_name = bbox, lb.class_name
        created += 1
        models_used.add(s.model_id)
    if rej_ids:
        db.execute(update(m.BoxSuggestion).where(m.BoxSuggestion.id.in_(list(rej_ids)),
                                                 m.BoxSuggestion.organization_id == org)
                   .values(status="rejected", reviewed_by=principal.user_id))
    db.flush()
    refresh_stats(db, target)
    target.updated_at = now()
    refresh_status(db, p)
    audit(db, request, principal, org, "autolabel.review", "dataset_version", target.id, accepted=created,
          rejected=len(rej_ids), edited=len(edits), models=sorted(models_used))
    return {"accepted": created, "rejected": len(rej_ids), "edited": len(edits), "labels_created": created,
            "dataset_version_id": str(target.id), "forked": target.id != v.id}
