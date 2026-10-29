"""LBL-1/2/4a: bulk labels (drafts fork from sealed versions), label sets (merge/rename), export, box suggestions."""

import io
import uuid
import zipfile
from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import delete, select

from app import models as m
from app.db import now
from app.deps import (
    DbDep,
    PrincipalDep,
    TenantRef,
    audit,
    authorize,
    check_refs,
    get_by_id,
    load_version,
    require_current,
)
from app.errors import ApiError, not_found
from app.routers.uploads import _new_version
from app.services import annotations, storage
from app.services.detector import detector
from app.services.ingest import VersionBuilder, refresh_stats
from app.services.project_status import refresh_status
from app.services.specs import train_labels

router = APIRouter(tags=["labels"])


class BBox(BaseModel):
    x: float
    y: float
    w: float = Field(ge=0)
    h: float = Field(ge=0)


class LabelIn(BaseModel):
    class_name: str = Field(min_length=1, max_length=200)
    bbox: BBox | None = None
    source: Literal["user", "import", "ai_prelabel"] = "user"
    confidence: float | None = Field(default=None, ge=0, le=1)

    @field_validator("class_name")
    @classmethod
    def strip(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("class_name can't be blank")
        return v.strip()


class Upsert(BaseModel):
    sample_id: uuid.UUID = TenantRef(m.Sample, "A sample of this dataset version")
    labels: list[LabelIn] = Field(default_factory=list, max_length=1000)
    no_objects: bool | None = Field(default=None, description="Detection: the image has no objects")


class LabelsBatch(BaseModel):
    upserts: list[Upsert] = Field(min_length=1, max_length=5000)


class LabelsBatchOut(BaseModel):
    updated: int
    dataset_version_id: uuid.UUID
    forked: bool
    sample_id_map: dict[str, str] = Field(default_factory=dict, description="Old -> new sample ids when forked")


class LabelSetOut(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    version: int
    classes: list[str]


class MergeIn(BaseModel):
    from_: list[str] = Field(alias="from", min_length=1)
    into: str = Field(min_length=1, max_length=200)


class RenameIn(BaseModel):
    from_: str = Field(alias="from", min_length=1)
    to: str = Field(min_length=1, max_length=200)


class AddClassesIn(BaseModel):
    add: list[str] = Field(min_length=1, max_length=500)


class LabelSetChangeOut(BaseModel):
    label_set: LabelSetOut
    dataset_version_id: uuid.UUID | None
    relabeled: int


class SuggestIn(BaseModel):
    sample_ids: list[uuid.UUID] = TenantRef(m.Sample, "Samples of this version (max 50)")
    classes: list[str] | None = Field(default=None, max_length=50)


class SuggestOut(BaseModel):
    model: str
    licence: str
    suggestions: dict[str, list[dict]]
    max_scores: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Own model only: best score per class per sample, including below the "
                                          "suggestion floor (0.3), so a near-miss is visible")


def _project_of(db, v: m.DatasetVersion) -> m.Project:
    ds = get_by_id(db, m.Dataset, v.dataset_id)
    return get_by_id(db, m.Project, ds.project_id)


def writable_version(db, v: m.DatasetVersion) -> tuple[m.DatasetVersion, dict]:
    """Labels go to a draft. A sealed version is immutable (DAT-5), so fork a draft child and return the id map."""
    if not v.sealed_at:
        return v, {}
    ds = get_by_id(db, m.Dataset, v.dataset_id)
    child = _new_version(db, ds, "labels")
    child.parent_version_id = v.id
    b = VersionBuilder(db, child, ds, ds.modality)
    b.copy_parent(v.id)
    child.ingest_status = "ready"
    ds.current_version_id = child.id
    db.flush()
    for s in db.scalars(select(m.BoxSuggestion).where(  # pending auto-label suggestions follow their photos
            m.BoxSuggestion.dataset_version_id == v.id, m.BoxSuggestion.organization_id == v.organization_id,
            m.BoxSuggestion.status == "pending")):
        if s.sample_id in b.id_map:
            s.dataset_version_id, s.sample_id = child.id, b.id_map[s.sample_id]
    db.flush()
    return child, b.id_map


@router.post("/dataset-versions/{version_id}/labels:batch", response_model=LabelsBatchOut)
def labels_batch(version_id: uuid.UUID, body: LabelsBatch, principal: PrincipalDep, db: DbDep, request: Request):
    v = load_version(db, principal, version_id, "write")
    if v.ingest_status != "ready":
        raise ApiError(409, "not_ready", "Wait until ingestion is ready")
    if v.sealed_at:
        require_current(db, v, "label")
    check_refs(db, body, v.organization_id, scope={"samples": [m.Sample.dataset_version_id == v.id]})
    target, id_map = writable_version(db, v)
    org = target.organization_id
    project = _project_of(db, target)
    label_set = db.scalar(select(m.LabelSet).where(m.LabelSet.project_id == project.id,
                                                   m.LabelSet.organization_id == org)
                          .order_by(m.LabelSet.version.desc()).limit(1))
    for up in body.upserts:
        sid = id_map.get(up.sample_id, up.sample_id)
        db.execute(delete(m.Label).where(m.Label.sample_id == sid, m.Label.organization_id == org))
        for lb in up.labels:
            db.add(m.Label(organization_id=org, sample_id=sid, label_set_id=label_set.id if label_set else None,
                           class_name=lb.class_name, bbox=lb.bbox.model_dump() if lb.bbox else None, source=lb.source,
                           confidence=lb.confidence, labeled_by=principal.user_id))
        if up.no_objects is not None:
            sample = get_by_id(db, m.Sample, sid)
            sample.no_objects = up.no_objects and not up.labels
    db.flush()
    refresh_stats(db, target)
    target.updated_at = now()
    db.flush()
    refresh_status(db, project)
    audit(db, request, principal, org, "labels.batch", "dataset_version", target.id, count=len(body.upserts))
    return LabelsBatchOut(updated=len(body.upserts), dataset_version_id=target.id, forked=target.id != v.id,
                          sample_id_map={str(k): str(val) for k, val in id_map.items()
                                         if k in {u.sample_id for u in body.upserts}})


# ---- label sets

def _label_set(db, principal, label_set_id, action) -> m.LabelSet:
    ls = get_by_id(db, m.LabelSet, label_set_id)
    if not ls:
        raise not_found("Label set")
    authorize(db, principal, ls.organization_id, action, "Label set")
    return ls


def _current_label_set(db, principal, label_set_id) -> m.LabelSet:
    """Mutations must name the CURRENT version (optimistic concurrency): a stale id would drop newer changes."""
    ls = _label_set(db, principal, label_set_id, "write")
    latest = db.scalar(select(m.LabelSet).where(m.LabelSet.project_id == ls.project_id,
                                                m.LabelSet.organization_id == ls.organization_id)
                       .order_by(m.LabelSet.version.desc()).limit(1).with_for_update())
    if latest.id != ls.id:
        raise ApiError(409, "stale_label_set", "This label set changed since you loaded it; refetch and retry",
                       extra={"current_label_set_id": str(latest.id), "current_version": latest.version,
                              "current_classes": latest.classes})
    return ls


@router.get("/label-sets/{label_set_id}", response_model=LabelSetOut)
def get_label_set(label_set_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    return LabelSetOut.model_validate(_label_set(db, principal, label_set_id, "read"), from_attributes=True)


def _apply_class_change(db, principal, request, ls: m.LabelSet, mapping: dict[str, str], classes: list[str],
                        action: str) -> LabelSetChangeOut:
    org = ls.organization_id
    latest = db.scalar(select(m.LabelSet).where(m.LabelSet.project_id == ls.project_id, m.LabelSet.organization_id == org)
                       .order_by(m.LabelSet.version.desc()).limit(1))
    new_ls = m.LabelSet(organization_id=org, project_id=ls.project_id, classes=classes, version=latest.version + 1)
    db.add(new_ls)
    project = get_by_id(db, m.Project, ls.project_id)
    if project.current_spec.get("labels") is not None and project.task_type != "tabular_regression":
        key = "custom_labels" if project.current_spec.get("mode") == "pretrained" else "labels"  # hybrid: own objects
        spec = dict(project.current_spec, **{key: classes})
        project.spec_version += 1
        project.current_spec = spec
        db.add(m.TaskSpecVersion(organization_id=org, project_id=project.id, version=project.spec_version, spec=spec,
                                 source="user_edit", created_by=principal.user_id))
    ds = db.scalar(select(m.Dataset).where(m.Dataset.project_id == project.id, m.Dataset.organization_id == org))
    relabeled, target_id = 0, None
    if mapping and ds and ds.current_version_id:  # adding classes relabels nothing
        current = get_by_id(db, m.DatasetVersion, ds.current_version_id)
        target, _ = writable_version(db, current)
        rows = db.scalars(select(m.Label).join(m.Sample, m.Sample.id == m.Label.sample_id).where(
            m.Sample.dataset_version_id == target.id, m.Label.organization_id == org,
            m.Label.class_name.in_(list(mapping)))).all()
        for lb in rows:
            lb.class_name = mapping[lb.class_name]
        relabeled, target_id = len(rows), target.id
        db.flush()
        refresh_stats(db, target)
    db.flush()
    refresh_status(db, project)
    audit(db, request, principal, org, action, "label_set", new_ls.id, mapping=mapping)
    return LabelSetChangeOut(label_set=LabelSetOut.model_validate(new_ls, from_attributes=True),
                             dataset_version_id=target_id, relabeled=relabeled)


@router.post("/label-sets/{label_set_id}/merge", response_model=LabelSetChangeOut)
def merge_classes(label_set_id: uuid.UUID, body: MergeIn, principal: PrincipalDep, db: DbDep, request: Request):
    ls = _current_label_set(db, principal, label_set_id)
    if body.into not in ls.classes:
        raise ApiError(422, "unknown_class", f"'{body.into}' is not in this label set; use rename to create it")
    unknown = sorted(set(body.from_) - set(ls.classes))
    if unknown:
        raise ApiError(422, "unknown_class", f"Not in this label set: {unknown}")
    sources = [c for c in body.from_ if c != body.into]
    classes = [c for c in ls.classes if c not in sources]
    if body.into not in classes:
        classes.append(body.into)
    return _apply_class_change(db, principal, request, ls, {c: body.into for c in sources}, classes, "labels.merge")


@router.post("/label-sets/{label_set_id}/classes", response_model=LabelSetChangeOut)
def add_classes(label_set_id: uuid.UUID, body: AddClassesIn, principal: PrincipalDep, db: DbDep, request: Request):
    """Adds classes, e.g. new folder names reported in a version's unknown_classes."""
    ls = _current_label_set(db, principal, label_set_id)
    new = [c.strip() for c in body.add if c.strip() and c.strip().lower() not in {x.lower() for x in ls.classes}]
    if len(ls.classes) + len(set(new)) > 500:
        raise ApiError(422, "too_many_classes", "At most 500 classes")
    return _apply_class_change(db, principal, request, ls, {}, ls.classes + sorted(set(new)), "labels.add_classes")


@router.post("/label-sets/{label_set_id}/rename", response_model=LabelSetChangeOut)
def rename_class(label_set_id: uuid.UUID, body: RenameIn, principal: PrincipalDep, db: DbDep, request: Request):
    ls = _current_label_set(db, principal, label_set_id)
    if body.from_ not in ls.classes:
        raise ApiError(422, "unknown_class", f"'{body.from_}' is not in this label set")
    if body.to in ls.classes:
        raise ApiError(409, "class_exists", f"'{body.to}' already exists; merge instead")
    classes = [body.to if c == body.from_ else c for c in ls.classes]
    return _apply_class_change(db, principal, request, ls, {body.from_: body.to}, classes, "labels.rename")


# ---- export + suggestions

@router.get("/dataset-versions/{version_id}/labels/export", responses={200: {"content": {"application/json": {},
                                                                                           "application/zip": {}}}})
def export_labels(version_id: uuid.UUID, principal: PrincipalDep, db: DbDep, format: Literal["coco", "yolo"] = "coco"):
    v = load_version(db, principal, version_id, "read")
    org = v.organization_id
    samples = db.scalars(select(m.Sample).where(m.Sample.dataset_version_id == v.id, m.Sample.organization_id == org)
                         .order_by(m.Sample.original_name)).all()
    by_sample: dict = {}
    for lb in db.scalars(select(m.Label).where(m.Label.sample_id.in_([s.id for s in samples]),
                                               m.Label.organization_id == org, m.Label.bbox.is_not(None))):
        by_sample.setdefault(lb.sample_id, []).append({"class_name": lb.class_name, "bbox": lb.bbox})
    project = _project_of(db, v)
    classes = train_labels(project.current_spec)
    rows = [{"file_name": s.original_name, "width": s.width, "height": s.height, "boxes": by_sample.get(s.id, [])}
            for s in samples]
    if format == "coco":
        return JSONResponse(annotations.export_coco(rows, classes))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, text in annotations.export_yolo(rows, classes).items():
            zf.writestr(path, text)
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="labels-v{v.version_no}-yolo.zip"'})


@router.post("/dataset-versions/{version_id}/box-suggestions", response_model=SuggestOut)
def box_suggestions(version_id: uuid.UUID, body: SuggestIn, principal: PrincipalDep, db: DbDep):
    """Suggestions only; nothing is stored until the client sends accepted boxes via labels:batch (ai_prelabel)."""
    v = load_version(db, principal, version_id, "write")
    if not body.sample_ids or len(body.sample_ids) > 50:
        raise ApiError(422, "invalid_sample_ids", "Send 1 to 50 sample_ids")
    check_refs(db, body, v.organization_id, scope={"samples": [m.Sample.dataset_version_id == v.id]})
    project = _project_of(db, v)
    if project.task_type != "object_detection":
        raise ApiError(422, "not_detection", "Box suggestions are for object detection projects")
    classes = body.classes or train_labels(project.current_spec)
    det = detector(db, project)
    out, maxes = {}, {}
    for sid in body.sample_ids:
        s = get_by_id(db, m.Sample, sid)
        if not s.width or not s.height:
            out[str(sid)] = []
            continue
        out[str(sid)] = det.suggest(s.width, s.height, classes, storage.get_bytes(s.object_key))
        if getattr(det, "max_scores", None) is not None:
            maxes[str(sid)] = dict(det.max_scores)
    return SuggestOut(model=det.model_id, licence=det.licence, suggestions=out, max_scores=maxes)


@router.post("/dataset-versions/{version_id}/ready-made-preview", response_model=SuggestOut)
def ready_made_preview(version_id: uuid.UUID, body: SuggestIn, principal: PrincipalDep, db: DbDep):
    """Hybrid projects: the auto (ready-made) boxes training WILL add to these photos. Read-only, never stored: the
    stock COCO model at the pseudo-label score, minus anything overlapping the user's own boxes (same rule as
    training, detection.keep_pseudo)."""
    from PIL import Image

    from app.ml import pretrained
    from app.ml.detection import PSEUDO_SCORE, keep_pseudo

    v = load_version(db, principal, version_id, "read")
    if not body.sample_ids or len(body.sample_ids) > 50:
        raise ApiError(422, "invalid_sample_ids", "Send 1 to 50 sample_ids")
    check_refs(db, body, v.organization_id, scope={"samples": [m.Sample.dataset_version_id == v.id]})
    project = _project_of(db, v)
    if project.current_spec.get("mode") != "pretrained":
        raise ApiError(409, "not_pretrained", "Ready-made previews are for ready-made detection projects")
    chosen, out = project.current_spec.get("labels") or [], {}
    for sid in body.sample_ids:
        s = get_by_id(db, m.Sample, sid)
        mine = [lb.bbox for lb in db.scalars(select(m.Label).where(m.Label.sample_id == s.id,
                m.Label.organization_id == v.organization_id, m.Label.bbox.is_not(None)))]
        found, _ = pretrained.predict(Image.open(io.BytesIO(storage.get_bytes(s.object_key))), chosen, PSEUDO_SCORE)
        out[str(sid)] = [{**f, "source": "auto"} for f in keep_pseudo(found, mine)]
    return SuggestOut(model=pretrained.MODEL["id"], licence=pretrained.MODEL["licence"], suggestions=out)
