"""DAT-5/6/11: dataset versions, sample browser, tabular rows, seal."""

import io
import uuid
from datetime import datetime
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import exists, func, select

from app import models as m
from app import tasks
from app.deps import DbDep, PrincipalDep, audit, load_dataset, load_version, require_current
from app.errors import ApiError
from app.pagination import Page
from app.services import storage
from app.services.project_status import refresh_status

router = APIRouter(tags=["datasets"])
ValidationStatus = Literal["pending", "running", "passed", "warnings", "failed"]


class VersionOut(BaseModel):
    id: uuid.UUID
    dataset_id: uuid.UUID
    version_no: int
    parent_version_id: uuid.UUID | None
    origin: str
    sample_count: int
    label_stats: dict[str, Any]
    split_counts: dict[str, int] | None
    validation_status: ValidationStatus
    ingest_status: Literal["pending", "running", "ready", "failed"]
    ingest_errors: list[dict[str, Any]]
    content_hash: str | None
    sealed_at: datetime | None
    created_at: datetime
    is_draft: bool = Field(default=False, description="True while unsealed: uploads append here; seal freezes it")
    unknown_classes: list[str] = Field(default_factory=list, description="Classes in the data but not in the "
                                       "project's label set (e.g. new folder names); add via POST /label-sets/{id}/classes")


class VersionDetail(VersionOut):
    split_strategy: str | None
    split_seed: int | None
    split_summary: dict[str, Any] | None
    column_mapping: dict[str, Any] | None
    acknowledgements: list[str]
    capture_counts: dict[str, int] = Field(default_factory=dict, description="Photos per capture_target hint; "
                                           "key '__none' = photos without a hint")


class LabelOut(BaseModel):
    id: uuid.UUID
    class_name: str
    bbox: dict[str, float] | None
    source: str
    confidence: float | None


class SampleOut(BaseModel):
    id: uuid.UUID
    name: str | None
    sha256: str | None
    thumb_url: str | None
    url: str | None
    width: int | None
    height: int | None
    mime: str | None
    bytes: int | None
    split: str | None
    labels: list[LabelOut]
    no_objects: bool
    flags: list[str]
    corrupt: str | None = Field(default=None, description="Set at ingest: zero_bytes | unreadable | "
                                "truncated_or_invalid | too_many_pixels | not_utf8 (always flagged VAL-1)")
    text: str | None = None
    capture_target: str | None = Field(default=None, description="Detection hint set at upload: the object the "
                                       "user was photographing (not a label)")


class RowsOut(BaseModel):
    columns: list[dict[str, Any]]
    items: list[dict[str, Any]]
    next_cursor: str | None


def version_out(v: m.DatasetVersion, detail: bool = False, known: list[str] | None = None) -> VersionOut:
    status = {"not_run": "pending"}.get(v.validation_status, v.validation_status)
    unknown = sorted(set(v.label_stats or {}) - set(known)) if known is not None and not v.table_uri else []
    base = {"id": v.id, "dataset_id": v.dataset_id, "version_no": v.version_no, "parent_version_id": v.parent_version_id,
                "origin": v.origin, "sample_count": v.sample_count, "label_stats": v.label_stats or {},
                "split_counts": (v.split_summary or {}).get("counts") if v.sealed_at else None,
                "validation_status": status, "ingest_status": v.ingest_status, "ingest_errors": v.ingest_errors or [],
                "content_hash": v.content_hash, "sealed_at": v.sealed_at, "created_at": v.created_at,
        "unknown_classes": unknown, "is_draft": v.sealed_at is None}
    if not detail:
        return VersionOut(**base)
    return VersionDetail(**base, split_strategy=v.split_strategy, split_seed=v.split_seed,
                         split_summary=v.split_summary, column_mapping=v.column_mapping,
                         acknowledgements=v.acknowledgements or [])


@router.get("/datasets/{dataset_id}/versions", response_model=Page[VersionOut])
def list_versions(dataset_id: uuid.UUID, principal: PrincipalDep, db: DbDep, cursor: str | None = None,
                  limit: int = Query(50, ge=1, le=200)):
    ds = load_dataset(db, principal, dataset_id, "read")
    stmt = select(m.DatasetVersion).where(m.DatasetVersion.dataset_id == ds.id,
                                          m.DatasetVersion.organization_id == ds.organization_id,
                                          m.DatasetVersion.ingest_status != "failed")
    if cursor:  # newest first; cursor = the last version_no seen
        try:
            stmt = stmt.where(m.DatasetVersion.version_no < int(cursor))
        except ValueError as e:
            raise ApiError(422, "invalid_cursor", "Malformed cursor") from e
    rows = db.scalars(stmt.order_by(m.DatasetVersion.version_no.desc()).limit(limit + 1)).all()
    more = len(rows) > limit
    rows = rows[:limit]
    known = _known_classes(db, ds)
    return Page[VersionOut](items=[version_out(v, known=known) for v in rows],
                            next_cursor=str(rows[-1].version_no) if more else None)


@router.get("/dataset-versions/{version_id}", response_model=VersionDetail)
def get_version(version_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    v = load_version(db, principal, version_id, "read")
    ds = db.scalar(select(m.Dataset).where(m.Dataset.id == v.dataset_id, m.Dataset.organization_id == v.organization_id))
    out = version_out(v, detail=True, known=_known_classes(db, ds))
    hint = m.Sample.meta["capture_target"].astext
    out.capture_counts = {k or "__none": n for k, n in db.execute(select(hint, func.count()).where(
        m.Sample.dataset_version_id == v.id, m.Sample.organization_id == v.organization_id).group_by(hint)).all()}
    return out


def _known_classes(db, ds: m.Dataset) -> list[str]:
    ls = db.scalar(select(m.LabelSet).where(m.LabelSet.project_id == ds.project_id,
                                            m.LabelSet.organization_id == ds.organization_id)
                   .order_by(m.LabelSet.version.desc()).limit(1))
    return list(ls.classes) if ls else []


@router.get("/dataset-versions/{version_id}/samples", response_model=Page[SampleOut])
def list_samples(version_id: uuid.UUID, principal: PrincipalDep, db: DbDep, label: str | None = None,
                 split: Literal["train", "val", "test"] | None = None, flag: str | None = None,
                 unlabeled: bool = False, capture_target: str | None = None, cursor: str | None = None,
                 limit: int = Query(100, ge=1, le=200)):
    v = load_version(db, principal, version_id, "read")
    org = v.organization_id
    stmt = select(m.Sample).where(m.Sample.dataset_version_id == v.id, m.Sample.organization_id == org)
    has_label = exists().where(m.Label.sample_id == m.Sample.id, m.Label.organization_id == org)
    if label:
        stmt = stmt.where(exists().where(m.Label.sample_id == m.Sample.id, m.Label.organization_id == org,
                                         m.Label.class_name == label))
    if unlabeled:
        stmt = stmt.where(~has_label, m.Sample.no_objects.is_(False))
    if capture_target == "__none":
        stmt = stmt.where(~m.Sample.meta.has_key("capture_target"))
    elif capture_target:
        stmt = stmt.where(m.Sample.meta["capture_target"].astext == capture_target)
    if split:
        stmt = stmt.where(m.Sample.split == split)
    if flag == "VAL-1":  # visible from ingest on, before validation has run
        stmt = stmt.where(m.Sample.meta.has_key("corrupt") | m.Sample.meta["flags"].contains([flag]))
    elif flag:
        stmt = stmt.where(m.Sample.meta["flags"].contains([flag]))
    if cursor:
        try:
            stmt = stmt.where(m.Sample.id > uuid.UUID(cursor))
        except ValueError as e:
            raise ApiError(422, "invalid_cursor", "Malformed cursor") from e
    samples = list(db.scalars(stmt.order_by(m.Sample.id).limit(limit + 1)).all())
    more = len(samples) > limit
    samples = samples[:limit]
    labels: dict[uuid.UUID, list[m.Label]] = {}
    if samples:
        for lb in db.scalars(select(m.Label).where(m.Label.sample_id.in_([s.id for s in samples]),
                                                   m.Label.organization_id == org).order_by(m.Label.created_at)):
            labels.setdefault(lb.sample_id, []).append(lb)
    items = [SampleOut(
        id=s.id, name=s.original_name, sha256=s.sha256, width=s.width, height=s.height, mime=s.mime, bytes=s.bytes,
        split=s.split, no_objects=s.no_objects, text=(s.meta or {}).get("text"), corrupt=(s.meta or {}).get("corrupt"),
        capture_target=(s.meta or {}).get("capture_target"),
        flags=sorted(set((s.meta or {}).get("flags", [])) | ({"VAL-1"} if (s.meta or {}).get("corrupt") else set())),
        thumb_url=storage.presign_get(s.thumbnail_key) if s.thumbnail_key else None,
        url=storage.presign_get(s.object_key) if s.object_key else None,
        labels=[LabelOut(id=lb.id, class_name=lb.class_name, bbox=lb.bbox, source=lb.source,
                         confidence=lb.confidence) for lb in labels.get(s.id, [])],
    ) for s in samples]
    return Page[SampleOut](items=items, next_cursor=str(samples[-1].id) if more else None)


@router.get("/dataset-versions/{version_id}/rows", response_model=RowsOut)
def list_rows(version_id: uuid.UUID, principal: PrincipalDep, db: DbDep, cursor: str | None = None,
              limit: int = Query(100, ge=1, le=500)):
    v = load_version(db, principal, version_id, "read")
    key = v.manifest_uri if v.sealed_at and v.table_uri else v.table_uri
    if not key:
        raise ApiError(409, "not_tabular", "This version has no table")
    start = int(cursor) if cursor and cursor.isdigit() else 0
    df = pd.read_parquet(io.BytesIO(storage.get_bytes(key)))
    page = df.iloc[start:start + limit]
    split = page.pop("__split__") if "__split__" in page.columns else None
    items = [{"row": int(i), "values": {k: (None if pd.isna(val) else val) for k, val in r.items()},
              "split": None if split is None else split[i]} for i, r in page.iterrows()]
    cols = [{k: c[k] for k in ("name", "type", "role")} for c in (v.column_mapping or {}).get("columns", [])]
    return RowsOut(columns=cols, items=items, next_cursor=str(start + limit) if start + limit < len(df) else None)


@router.post("/dataset-versions/{version_id}/seal", status_code=202)
def seal(version_id: uuid.UUID, principal: PrincipalDep, db: DbDep, request: Request):
    v = load_version(db, principal, version_id, "write")
    db.execute(select(m.DatasetVersion.id).where(m.DatasetVersion.id == v.id,
                                                 m.DatasetVersion.organization_id == v.organization_id)
               .with_for_update())  # serialise concurrent seals
    db.refresh(v)
    if v.sealed_at:  # idempotent (DAT-5c)
        return JSONResponse({"dataset_version_id": str(v.id), "validation_status": version_out(v).validation_status},
                            status_code=200)
    require_current(db, v, "seal")
    if v.ingest_status != "ready" or db.scalar(select(m.Upload.id).where(
            m.Upload.dataset_version_id == v.id, m.Upload.status == "ingesting").limit(1)):
        raise ApiError(409, "not_ready", f"Ingestion is '{v.ingest_status}'; wait until it is ready")
    v.validation_status = "running"
    ds = db.scalar(select(m.Dataset).where(m.Dataset.id == v.dataset_id, m.Dataset.organization_id == v.organization_id))
    project = db.scalar(select(m.Project).where(m.Project.id == ds.project_id,
                                                m.Project.organization_id == v.organization_id))
    db.flush()
    refresh_status(db, project)
    audit(db, request, principal, v.organization_id, "dataset.seal", "dataset_version", v.id)
    db.commit()
    tasks.seal_version.delay(str(v.id))
    return {"dataset_version_id": str(v.id), "validation_status": "running"}
