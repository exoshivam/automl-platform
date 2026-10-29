"""Storage hygiene: delete/prune dataset versions, delete model versions, storage usage per project.

Blobs are content-addressed and shared across versions, so requests only delete DB rows; a Celery task (gc_objects)
removes an object from S3 only when no remaining row in the org references it.
"""

import uuid
from typing import Any

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select, update

from app import models as m
from app import tasks
from app.deps import (
    DbDep,
    PrincipalDep,
    TenantRef,
    audit,
    authorize,
    check_refs,
    get_by_id,
    load_dataset,
    load_project,
    load_version,
)
from app.errors import ApiError, not_found
from app.services import storage

router = APIRouter(tags=["storage"])


class PruneIn(BaseModel):
    keep_latest: int = Field(default=0, ge=0, description="Also keep this many newest sealed versions")
    dry_run: bool = Field(default=True, description="Preview only (default); set false to delete")


class PruneOut(BaseModel):
    dry_run: bool
    deleted: list[dict[str, Any]]
    kept: list[dict[str, Any]]
    reclaimable_bytes: int


class StorageOut(BaseModel):
    totals: dict[str, int]
    versions: list[dict[str, Any]]


def _models_using(db, v: m.DatasetVersion) -> list[dict]:
    rows = db.execute(select(m.ModelVersion.id, m.ModelVersion.version_no, m.ModelVersion.model_id).where(
        m.ModelVersion.dataset_version_id == v.id, m.ModelVersion.organization_id == v.organization_id)).all()
    return [{"model_version_id": str(i), "version_no": n, "model_id": str(mid)} for i, n, mid in rows]


def _guard(db, v: m.DatasetVersion) -> None:
    ds = get_by_id(db, m.Dataset, v.dataset_id)
    if ds.current_version_id == v.id or v.sealed_at is None:
        raise ApiError(409, "version_in_use", "The current version and the open draft can't be deleted")
    models = _models_using(db, v)
    if models:
        raise ApiError(409, "version_has_models", "Models were trained on this version; delete them first",
                       extra={"models": models})
    if db.scalar(select(m.TrainingJob.id).where(m.TrainingJob.dataset_version_id == v.id,
                                                m.TrainingJob.status.in_(("queued", "running"))).limit(1)):
        raise ApiError(409, "version_in_use", "A training job is using this version")


def _object_keys(db, v: m.DatasetVersion) -> set[str]:
    keys = set()
    for ok, tk in db.execute(select(m.Sample.object_key, m.Sample.thumbnail_key).where(
            m.Sample.dataset_version_id == v.id, m.Sample.organization_id == v.organization_id)).all():
        keys.update(k for k in (ok, tk) if k)
    keys.update(k for k in (v.manifest_uri, v.table_uri) if k)
    return keys


def delete_version(db, v: m.DatasetVersion) -> set[str]:
    """Deletes the rows (samples/labels cascade) and returns candidate object keys for GC."""
    _guard(db, v)
    keys = _object_keys(db, v)
    org = v.organization_id
    db.execute(update(m.DatasetVersion).where(m.DatasetVersion.parent_version_id == v.id,
                                              m.DatasetVersion.organization_id == org)
               .values(parent_version_id=v.parent_version_id))  # keep the lineage connected
    db.execute(update(m.Upload).where(m.Upload.dataset_version_id == v.id, m.Upload.organization_id == org)
               .values(dataset_version_id=None))
    job_ids = list(db.scalars(select(m.TrainingJob.id).where(m.TrainingJob.dataset_version_id == v.id,
                                                             m.TrainingJob.organization_id == org)))
    if job_ids:  # finished jobs without models (failed/cancelled) go with the version
        db.execute(delete(m.Trial).where(m.Trial.training_job_id.in_(job_ids), m.Trial.organization_id == org))
        db.execute(delete(m.TrainingJob).where(m.TrainingJob.id.in_(job_ids), m.TrainingJob.organization_id == org))
    db.delete(v)
    db.flush()
    return keys


def _reclaimable(db, v: m.DatasetVersion) -> int:
    """Bytes of blobs referenced ONLY by this version (within the org)."""
    org = v.organization_id
    mine = select(m.Sample.object_key, func.max(m.Sample.bytes).label("b")).where(
        m.Sample.dataset_version_id == v.id, m.Sample.organization_id == org, m.Sample.object_key.is_not(None)
    ).group_by(m.Sample.object_key).subquery()
    shared = select(m.Sample.object_key).where(m.Sample.organization_id == org, m.Sample.dataset_version_id != v.id,
                                               m.Sample.object_key.is_not(None))
    return int(db.scalar(select(func.coalesce(func.sum(mine.c.b), 0)).where(mine.c.object_key.not_in(shared))) or 0)


def _version_row(db, v: m.DatasetVersion, current_id) -> dict[str, Any]:
    return {"id": str(v.id), "version_no": v.version_no, "sample_count": v.sample_count, "is_draft": v.sealed_at is None,
            "is_current": v.id == current_id, "has_models": bool(_models_using(db, v)),
            "reclaimable_bytes": _reclaimable(db, v), "created_at": v.created_at.isoformat()}


@router.delete("/dataset-versions/{version_id}", status_code=204)
def delete_dataset_version(version_id: uuid.UUID, principal: PrincipalDep, db: DbDep, request: Request):
    v = load_version(db, principal, version_id, "write")
    org, no = v.organization_id, v.version_no
    keys = delete_version(db, v)
    audit(db, request, principal, org, "dataset_version.delete", "dataset_version", version_id, version_no=no,
          objects=len(keys))
    db.commit()
    tasks.gc_objects.delay(str(org), sorted(keys))
    return Response(status_code=204)


@router.post("/datasets/{dataset_id}/versions:prune", response_model=PruneOut)
def prune_versions(dataset_id: uuid.UUID, body: PruneIn, principal: PrincipalDep, db: DbDep, request: Request):
    ds = load_dataset(db, principal, dataset_id, "write")
    versions = db.scalars(select(m.DatasetVersion).where(m.DatasetVersion.dataset_id == ds.id,
                                                         m.DatasetVersion.organization_id == ds.organization_id)
                          .order_by(m.DatasetVersion.version_no.desc())).all()
    sealed = [v for v in versions if v.sealed_at is not None and v.id != ds.current_version_id]
    keep_newest = {v.id for v in sealed[: body.keep_latest]}
    deleted, kept, reclaim = [], [], 0
    for v in versions:
        row = _version_row(db, v, ds.current_version_id)
        reason = ("current" if row["is_current"] else "draft" if row["is_draft"] else "used_by_models"
                  if row["has_models"] else "keep_latest" if v.id in keep_newest else None)
        if reason:
            kept.append({**row, "reason": reason})
        else:
            deleted.append(row)
            reclaim += row["reclaimable_bytes"]
    if not body.dry_run and deleted:
        keys: set[str] = set()
        for row in deleted:
            keys |= delete_version(db, get_by_id(db, m.DatasetVersion, uuid.UUID(row["id"])))
        audit(db, request, principal, ds.organization_id, "dataset_version.prune", "dataset", ds.id,
              deleted=[r["version_no"] for r in deleted])
        db.commit()
        tasks.gc_objects.delay(str(ds.organization_id), sorted(keys))
    return PruneOut(dry_run=body.dry_run, deleted=deleted, kept=kept, reclaimable_bytes=reclaim)


@router.delete("/model-versions/{version_id}", status_code=204)
def delete_model_version(version_id: uuid.UUID, principal: PrincipalDep, db: DbDep, request: Request):
    mv = get_by_id(db, m.ModelVersion, version_id)
    if not mv:
        raise not_found("Model version")
    authorize(db, principal, mv.organization_id, "write", "Model version")
    org = mv.organization_id
    keys = set(storage.existing_sizes(mv.artifact_prefix)) if mv.artifact_prefix else set()
    keys |= {k for k in db.scalars(select(m.Export.artifact_key).where(m.Export.model_version_id == mv.id,
                                                                        m.Export.organization_id == org)) if k}
    db.execute(delete(m.Export).where(m.Export.model_version_id == mv.id, m.Export.organization_id == org))
    model = get_by_id(db, m.Model, mv.model_id)
    if model and model.production_version_id == mv.id:
        model.production_version_id = None
    db.delete(mv)
    db.flush()
    from app.services.project_status import refresh_status

    if model:
        refresh_status(db, get_by_id(db, m.Project, model.project_id))
    audit(db, request, principal, org, "model_version.delete", "model_version", version_id, objects=len(keys))
    db.commit()
    tasks.gc_objects.delay(str(org), sorted(keys), True)  # model artefacts are never shared
    return Response(status_code=204)


@router.get("/projects/{project_id}/storage", response_model=StorageOut)
def project_storage(project_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    p = load_project(db, principal, project_id, "read")
    org = p.organization_id
    ds = db.scalar(select(m.Dataset).where(m.Dataset.project_id == p.id, m.Dataset.organization_id == org))
    versions = db.scalars(select(m.DatasetVersion).where(m.DatasetVersion.dataset_id == ds.id,
                                                         m.DatasetVersion.organization_id == org)
                          .order_by(m.DatasetVersion.version_no.desc())).all() if ds else []
    blobs = select(m.Sample.object_key, func.max(m.Sample.bytes).label("b")).join(
        m.DatasetVersion, m.DatasetVersion.id == m.Sample.dataset_version_id).where(
        m.DatasetVersion.dataset_id == ds.id, m.Sample.organization_id == org, m.Sample.object_key.is_not(None)
    ).group_by(m.Sample.object_key).subquery() if ds else None
    data_bytes = int(db.scalar(select(func.coalesce(func.sum(blobs.c.b), 0))) or 0) if ds else 0
    manifests = sum((storage.head(v.manifest_uri) or {}).get("ContentLength", 0) for v in versions if v.manifest_uri)
    prefixes = db.scalars(select(m.ModelVersion.artifact_prefix).join(m.Model, m.Model.id == m.ModelVersion.model_id)
                          .where(m.Model.project_id == p.id, m.Model.organization_id == org)).all()
    models_bytes = sum(sum(storage.existing_sizes(pre).values()) for pre in prefixes if pre)
    exports_bytes = int(db.scalar(select(func.coalesce(func.sum(m.Export.size_bytes), 0)).join(
        m.ModelVersion, m.ModelVersion.id == m.Export.model_version_id).join(
        m.Model, m.Model.id == m.ModelVersion.model_id).where(m.Model.project_id == p.id,
                                                              m.Export.organization_id == org)) or 0)
    # face / speaker galleries keep their media OUTSIDE datasets (photos, thumbnails, voice clips, export bundles):
    # without this, the delete dialog showed "0 B" for galleries that hold megabytes of biometric files
    g = db.scalar(select(m.FaceGallery.id).where(m.FaceGallery.project_id == p.id, m.FaceGallery.organization_id == org))
    media = storage.existing_sizes(f"{storage.org_prefix(org)}faces/{g}/") if g else {}
    gallery_bytes = sum(media.values())
    # the people's own media only (photos, thumbnails, voice clips), without earlier export bundles (~190 MB each)
    gallery_photos_bytes = sum(v for k, v in media.items() if f"faces/{g}/exports/" not in k)
    return StorageOut(totals={"data_bytes": data_bytes, "manifests_bytes": manifests, "models_bytes": models_bytes,
                              "exports_bytes": exports_bytes, "gallery_bytes": gallery_bytes,
                              "gallery_photos_bytes": gallery_photos_bytes,
                              "total_bytes": data_bytes + manifests + models_bytes + exports_bytes + gallery_bytes},
                      versions=[_version_row(db, v, ds.current_version_id) for v in versions])


# ---- removing samples from the open draft (wrong burst photos etc.)

class RemoveIn(BaseModel):
    sample_ids: list[uuid.UUID] = TenantRef(m.Sample, "Samples of this draft (1-1000)")


class RemoveOut(BaseModel):
    removed: int
    sample_count: int
    dataset_version_id: uuid.UUID


def _remove(db, v: m.DatasetVersion, rows: list[m.Sample]) -> set[str]:
    """Deletes the rows (labels cascade); returns candidate blob keys for the refcounted GC."""
    keys = {k for s in rows for k in (s.object_key, s.thumbnail_key) if k}
    ids = [s.id for s in rows]
    if ids:
        db.execute(delete(m.Label).where(m.Label.sample_id.in_(ids), m.Label.organization_id == v.organization_id))
        db.execute(delete(m.Sample).where(m.Sample.id.in_(ids), m.Sample.organization_id == v.organization_id))
        db.flush()
    from app.services.ingest import refresh_stats
    from app.services.project_status import refresh_status

    refresh_stats(db, v)
    ds = get_by_id(db, m.Dataset, v.dataset_id)
    refresh_status(db, get_by_id(db, m.Project, ds.project_id))
    return keys


def _draft(v: m.DatasetVersion) -> None:
    if v.sealed_at is not None:
        raise ApiError(409, "version_sealed", "Sealed versions are immutable; remove samples from the draft")


@router.post("/dataset-versions/{version_id}/samples:remove", response_model=RemoveOut)
def remove_samples(version_id: uuid.UUID, body: RemoveIn, principal: PrincipalDep, db: DbDep, request: Request):
    v = load_version(db, principal, version_id, "write")  # viewer 403, other tenant 404
    _draft(v)
    if not 1 <= len(body.sample_ids) <= 1000:
        raise ApiError(422, "invalid_sample_ids", "Send 1 to 1000 sample_ids")
    check_refs(db, body, v.organization_id, scope={"samples": [m.Sample.dataset_version_id == v.id]})  # all or 404
    rows = db.scalars(select(m.Sample).where(m.Sample.id.in_(body.sample_ids),
                                             m.Sample.organization_id == v.organization_id)).all()
    keys = _remove(db, v, list(rows))
    audit(db, request, principal, v.organization_id, "samples.remove", "dataset_version", v.id, removed=len(rows))
    db.commit()
    tasks.gc_objects.delay(str(v.organization_id), sorted(keys))  # only blobs no other version references
    return RemoveOut(removed=len(rows), sample_count=v.sample_count, dataset_version_id=v.id)


@router.post("/uploads/{upload_id}:undo", response_model=RemoveOut)
def undo_upload(upload_id: uuid.UUID, principal: PrincipalDep, db: DbDep, request: Request):
    """"Undo last burst": removes every sample that this upload / capture batch added to the draft."""
    up = get_by_id(db, m.Upload, upload_id)
    if up is None:
        raise not_found("Upload")
    authorize(db, principal, up.organization_id, "write", "Upload")
    if up.status in ("open", "completing", "ingesting"):
        raise ApiError(409, "upload_not_finished", "Wait until this upload has finished processing")
    if up.dataset_version_id is None:
        raise not_found("Upload samples")
    v = load_version(db, principal, up.dataset_version_id, "write")
    _draft(v)
    rows = db.scalars(select(m.Sample).where(m.Sample.dataset_version_id == v.id,
                                             m.Sample.organization_id == v.organization_id,
                                             m.Sample.meta["upload_id"].astext == str(up.id))).all()
    keys = _remove(db, v, list(rows))
    audit(db, request, principal, v.organization_id, "upload.undo", "upload", up.id, removed=len(rows))
    db.commit()
    tasks.gc_objects.delay(str(v.organization_id), sorted(keys))
    return RemoveOut(removed=len(rows), sample_count=v.sample_count, dataset_version_id=v.id)
