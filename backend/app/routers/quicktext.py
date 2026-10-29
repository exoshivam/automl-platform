"""Quick text labels (examples, no training) + ready-made sentiment for text_classification projects. Examples are
ordinary labelled text samples in the project's draft, so seal + train ("Train for better accuracy") works unchanged."""

import copy
import hashlib
import time
import uuid
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app import models as m
from app.deps import DbDep, PrincipalDep, audit, get_by_id, load_project
from app.errors import ApiError, not_found
from app.ml import quicktext
from app.routers.labels import writable_version
from app.routers.uploads import open_draft
from app.services.ingest import refresh_stats
from app.services.project_status import refresh_status

router = APIRouter(tags=["text"])


class ExampleIn(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    label: str = Field(min_length=1, max_length=100)


class ExampleOut(BaseModel):
    id: uuid.UUID
    text: str
    label: str | None
    dataset_version_id: uuid.UUID
    created_at: datetime


class TextSettingsOut(BaseModel):
    mode: Literal["custom", "quick"]
    threshold: float
    default_threshold: float
    ready_made: list[str]
    labels: list[str]
    embedding_model: dict[str, Any]
    sentiment_model: dict[str, Any] | None
    has_trained_model: bool


class TextSettingsPatch(BaseModel):
    threshold: float | None = Field(default=None, ge=0.05, le=0.95)
    ready_made: list[Literal["sentiment"]] | None = None


class TextPredictIn(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


def _text_project(db, principal, project_id, action) -> m.Project:
    p = load_project(db, principal, project_id, action)
    if p.task_type != "text_classification":
        raise ApiError(422, "not_text", "This is not a text classification project")
    return p


def _dataset(db, p: m.Project) -> m.Dataset:
    return db.scalar(select(m.Dataset).where(m.Dataset.project_id == p.id, m.Dataset.organization_id == p.organization_id))


def _current(db, p: m.Project) -> m.DatasetVersion | None:
    ds = _dataset(db, p)
    return get_by_id(db, m.DatasetVersion, ds.current_version_id) if ds and ds.current_version_id else None


def examples_of(db, p: m.Project) -> dict[str, list[str]]:
    """{label: [texts]} from the current version (every spec label present, possibly empty)."""
    out: dict[str, list[str]] = {lbl: [] for lbl in p.current_spec.get("labels") or []}
    v = _current(db, p)
    if v is None:
        return out
    for text, cls in db.execute(select(m.Sample.meta["text"].astext, m.Label.class_name).join(
            m.Label, m.Label.sample_id == m.Sample.id).where(
            m.Sample.dataset_version_id == v.id, m.Sample.organization_id == p.organization_id)).all():
        if cls in out and text and text.strip():
            out[cls].append(text)
    return out


def _trained(db, p: m.Project) -> uuid.UUID | None:
    return db.scalar(select(m.ModelVersion.id).join(m.Model, m.Model.id == m.ModelVersion.model_id).where(
        m.Model.project_id == p.id, m.Model.organization_id == p.organization_id, m.ModelVersion.status == "ready")
        .order_by(m.ModelVersion.created_at.desc()).limit(1))


def _settings(db, p: m.Project) -> TextSettingsOut:
    s = p.current_spec
    sid = quicktext.sentiment_model_id()
    return TextSettingsOut(mode=s.get("mode") if s.get("mode") == "quick" else "custom",
                           threshold=s.get("threshold", quicktext.DEFAULT_THRESHOLD),
                           default_threshold=quicktext.DEFAULT_THRESHOLD, ready_made=s.get("ready_made") or [],
                           labels=s.get("labels") or [], embedding_model=quicktext.EMBED,
                           sentiment_model={"id": sid, "licence": quicktext.SENTIMENT_LICENCES.get(sid, "see model card")}
                           if sid else None, has_trained_model=_trained(db, p) is not None)


@router.get("/projects/{project_id}/text-settings", response_model=TextSettingsOut)
def get_text_settings(project_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    return _settings(db, _text_project(db, principal, project_id, "read"))


@router.patch("/projects/{project_id}/text-settings", response_model=TextSettingsOut)
def patch_text_settings(project_id: uuid.UUID, body: TextSettingsPatch, principal: PrincipalDep, db: DbDep,
                        request: Request):
    p = _text_project(db, principal, project_id, "write")
    spec = copy.deepcopy(p.current_spec)
    spec.update(body.model_dump(exclude_none=True))
    p.spec_version += 1
    p.current_spec = spec
    db.add(m.TaskSpecVersion(organization_id=p.organization_id, project_id=p.id, version=p.spec_version, spec=spec,
                             source="user_edit", created_by=principal.user_id))
    refresh_status(db, p)
    audit(db, request, principal, p.organization_id, "text.settings", "project", p.id,
          **body.model_dump(exclude_none=True))
    return _settings(db, p)


@router.get("/projects/{project_id}/examples", response_model=list[ExampleOut])
def list_examples(project_id: uuid.UUID, principal: PrincipalDep, db: DbDep, label: str | None = None):
    p = _text_project(db, principal, project_id, "read")
    v = _current(db, p)
    if v is None:
        return []
    rows = db.execute(select(m.Sample, m.Label.class_name).outerjoin(m.Label, m.Label.sample_id == m.Sample.id).where(
        m.Sample.dataset_version_id == v.id, m.Sample.organization_id == p.organization_id)
        .order_by(m.Sample.created_at)).all()
    return [ExampleOut(id=s.id, text=(s.meta or {}).get("text") or "", label=cls, dataset_version_id=v.id,
                       created_at=s.created_at) for s, cls in rows if label is None or cls == label]


@router.post("/projects/{project_id}/examples", status_code=201, response_model=ExampleOut)
def add_example(project_id: uuid.UUID, body: ExampleIn, principal: PrincipalDep, db: DbDep, request: Request):
    p = _text_project(db, principal, project_id, "write")
    if body.label not in (p.current_spec.get("labels") or []):
        raise ApiError(422, "unknown_class", f"'{body.label}' is not one of this project's labels",
                       errors=[{"loc": ["body", "label"], "msg": f"use one of {p.current_spec.get('labels')}"}])
    text = body.text.strip()
    if text in examples_of(db, p).get(body.label, []):
        raise ApiError(409, "duplicate_example", "This text is already an example of that label")
    v, _ = open_draft(db, _dataset(db, p), "examples")
    org = p.organization_id
    n = db.scalar(select(func.count()).select_from(m.Sample).where(m.Sample.dataset_version_id == v.id)) or 0
    s = m.Sample(organization_id=org, dataset_version_id=v.id, original_name=f"example-{n + 1}",
                 sha256=hashlib.sha256(text.encode()).hexdigest(), mime="text/plain", bytes=len(text.encode()),
                 meta={"text": text, "source": "example"})
    db.add(s)
    db.flush()
    db.add(m.Label(organization_id=org, sample_id=s.id, class_name=body.label, source="user"))
    v.ingest_status = "ready"
    db.flush()
    refresh_stats(db, v)
    refresh_status(db, p)
    audit(db, request, principal, org, "text.example_added", "sample", s.id, label=body.label)
    return ExampleOut(id=s.id, text=text, label=body.label, dataset_version_id=v.id, created_at=s.created_at)


@router.delete("/projects/{project_id}/examples/{example_id}", status_code=204)
def delete_example(project_id: uuid.UUID, example_id: uuid.UUID, principal: PrincipalDep, db: DbDep,
                   request: Request):
    p = _text_project(db, principal, project_id, "write")
    v = _current(db, p)
    s = get_by_id(db, m.Sample, example_id)
    if v is None or s is None or s.dataset_version_id != v.id:
        raise not_found("Example")
    target, id_map = writable_version(db, v)  # a sealed version is immutable: fork a draft first
    sid = id_map.get(s.id, s.id)
    row = get_by_id(db, m.Sample, sid)
    db.delete(row)
    db.flush()
    refresh_stats(db, target)
    refresh_status(db, p)
    audit(db, request, principal, p.organization_id, "text.example_deleted", "sample", example_id)
    return Response(status_code=204, headers={"X-Dataset-Version-Id": str(target.id)})


def predict_text(db, p: m.Project, text: str) -> dict:
    t0 = time.perf_counter()
    spec, load_ms = p.current_spec, 0
    out: dict[str, Any] = {"label": None, "score": None, "top3": [], "other": False}
    mv = _trained(db, p)
    if mv is not None:  # "Train for better accuracy" happened: the trained model wins
        from app.ml.loader import load_model_timed

        model, load_ms = load_model_timed(mv)
        r = model.predict([text])[0]
        top = sorted(r["probs"].items(), key=lambda x: -x[1])[:3]
        out.update(label=r["label"], score=round(r["confidence"], 4), source="model",
                   top3=[{"label": k, "score": round(v, 4)} for k, v in top])
    elif spec.get("mode") == "quick" and spec.get("labels"):
        r = quicktext.classify(examples_of(db, p), text, spec.get("threshold", quicktext.DEFAULT_THRESHOLD))
        load_ms = r.pop("model_load_ms")
        out.update(r, source="examples")
    elif "sentiment" in (spec.get("ready_made") or []):
        out["source"] = "sentiment_only"
    else:
        raise ApiError(409, "model_not_ready", "Train a model first, or switch to quick (examples) mode")
    if "sentiment" in (spec.get("ready_made") or []):
        out["sentiment"], ms = quicktext.sentiment(text)
        load_ms += ms
    return {**out, "model_load_ms": load_ms, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


@router.get("/projects/{project_id}/text-report")
def text_report(project_id: uuid.UUID, principal: PrincipalDep, db: DbDep) -> dict:
    p = _text_project(db, principal, project_id, "read")
    return quicktext.report(examples_of(db, p), p.current_spec.get("threshold", quicktext.DEFAULT_THRESHOLD))
