"""DAT-1..4, DAT-8: upload sessions (Uppy-style signing of every S3 call), completion, imports."""

import uuid
from datetime import datetime, timedelta
from pathlib import PurePosixPath
from typing import Any, Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app import models as m
from app import tasks
from app.db import now
from app.deps import DbDep, PrincipalDep, TenantRef, audit, authorize, check_refs, get_by_id, load_dataset
from app.errors import ApiError, not_found
from app.services import storage, tables
from app.services.ingest import safe_relpath
from app.services.project_status import refresh_status
from app.settings import get_settings

router = APIRouter(tags=["uploads"])
Source = Literal["files", "folder", "zip", "camera", "annotations"]
TABLE_FORMATS = ("csv", "xlsx", "parquet", "jsonl")
DETECTION_FORMATS = ("coco", "yolo", "voc")


class UploadCreate(BaseModel):
    source: Source = "files"
    label_from_folder: bool = False
    file_count: int = Field(ge=1)
    total_bytes: int = Field(ge=0)
    label: str | None = Field(default=None, max_length=200, description="Applies to every file (e.g. camera class)")
    capture_target: str | None = Field(default=None, max_length=100, description="Detection HINT: the custom object "
                                       "being photographed. Stored as sample metadata only: no label, no training "
                                       "effect. Must be one of the project's trainable labels")


class UploadOut(BaseModel):
    upload_id: uuid.UUID
    key_prefix: str
    expires_at: datetime
    max_file_bytes: int
    part_size: int
    status: str


class SignIn(BaseModel):
    method: Literal["PUT", "POST", "GET", "DELETE"]
    key: str = Field(max_length=2048)
    upload_id: str | None = Field(default=None, description="The S3 multipart UploadId (not a platform id)")
    part_number: int | None = Field(default=None, ge=1, le=10_000)
    size: int | None = Field(default=None, ge=0, description="File size; required for PutObject/CreateMultipart")


class SignOut(BaseModel):
    url: str
    headers: dict[str, str] = {}


class UploadedFile(BaseModel):
    key: str
    name: str = Field(max_length=1024)
    relative_path: str | None = Field(default=None, max_length=1024)
    size: int = Field(ge=0)
    mime: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    label: str | None = Field(default=None, max_length=200)


class CompleteIn(BaseModel):
    files: list[UploadedFile] = Field(min_length=1)


class CompleteOut(BaseModel):
    upload_id: uuid.UUID
    dataset_version_id: uuid.UUID | None
    status: str


class UploadState(BaseModel):
    upload_id: uuid.UUID
    status: str
    source: str
    dataset_version_id: uuid.UUID | None
    error: str | None
    objects: dict[str, int] = Field(description="Objects already in storage under key_prefix: {key: size}")


class ImportIn(BaseModel):
    upload_id: uuid.UUID = TenantRef(m.Upload, "An upload with source 'annotations'")
    format: Literal["csv", "xlsx", "parquet", "jsonl", "coco", "yolo", "voc"]
    key: str = Field(description="The table / annotation file inside the upload (any .xml for voc)")
    mapping: dict[str, Any] | None = Field(default=None, description="{columns:[{name, role, type}]}; omit for preview")


class ImportAccepted(BaseModel):
    dataset_version_id: uuid.UUID


def _prefix(up: m.Upload) -> str:
    return f"{storage.org_prefix(up.organization_id)}uploads/{up.id}/"


def _upload(db, principal, upload_id: uuid.UUID, action="write") -> m.Upload:
    up = get_by_id(db, m.Upload, upload_id)
    if not up:
        raise not_found("Upload")
    authorize(db, principal, up.organization_id, action, "Upload")
    return up


def _check_key(up: m.Upload, key: str) -> None:
    prefix = _prefix(up)
    if not key.startswith(prefix) or safe_relpath(key[len(prefix):]) != key[len(prefix):]:
        raise ApiError(422, "invalid_key", f"Keys must be under {prefix} with a plain relative path")


def _dataset_in_project(db, principal, project_id, dataset_id) -> m.Dataset:
    ds = load_dataset(db, principal, dataset_id, "write")
    if ds.project_id != project_id:
        raise not_found("Dataset")
    return ds


def open_draft(db, ds: m.Dataset, origin: str) -> tuple[m.DatasetVersion, bool]:
    """Uploads/captures/imports APPEND to the dataset's open draft (the unsealed current version). Only when the
    current version is sealed (or there is none) is a new draft created; immutable versions come from seal / fixes."""
    db.execute(select(m.Dataset.id).where(m.Dataset.id == ds.id, m.Dataset.organization_id == ds.organization_id)
               .with_for_update())  # one draft even when two uploads complete at once
    db.refresh(ds)
    cur = get_by_id(db, m.DatasetVersion, ds.current_version_id) if ds.current_version_id else None
    if cur is not None and cur.sealed_at is None and cur.ingest_status != "failed":
        return cur, True
    v = _new_version(db, ds, origin)
    ds.current_version_id = v.id
    return v, False


def _new_version(db, ds: m.Dataset, origin: str) -> m.DatasetVersion:
    n = db.scalar(select(func.coalesce(func.max(m.DatasetVersion.version_no), 0)).where(
        m.DatasetVersion.dataset_id == ds.id, m.DatasetVersion.organization_id == ds.organization_id))
    v = m.DatasetVersion(organization_id=ds.organization_id, dataset_id=ds.id, version_no=n + 1,
                         parent_version_id=ds.current_version_id, origin=origin, ingest_status="pending")
    db.add(v)
    db.flush()
    return v


@router.post("/projects/{project_id}/datasets/{dataset_id}/uploads", status_code=201, response_model=UploadOut)
def create_upload(project_id: uuid.UUID, dataset_id: uuid.UUID, body: UploadCreate, principal: PrincipalDep,
                  db: DbDep, request: Request):
    s = get_settings()
    ds = _dataset_in_project(db, principal, project_id, dataset_id)
    if body.capture_target is not None:
        from app.services.specs import train_labels

        proj = get_by_id(db, m.Project, project_id)
        allowed = train_labels(proj.current_spec)
        if proj.task_type != "object_detection":
            raise ApiError(422, "capture_target_detection_only", "capture_target is for object detection; "
                           "classification uploads use `label`")
        if body.capture_target not in allowed:
            raise ApiError(422, "unknown_class", f"'{body.capture_target}' is not one of this project's classes",
                           errors=[{"loc": ["body", "capture_target"], "msg": f"use one of {allowed}"}])
    if body.file_count > s.max_files_per_batch:
        raise ApiError(413, "too_many_files", f"At most {s.max_files_per_batch} files per upload")
    up = m.Upload(organization_id=ds.organization_id, project_id=project_id, dataset_id=ds.id, files=[],
                  kind="import" if body.source == "annotations" else "files", created_by=principal.user_id,
                  options=body.model_dump())
    db.add(up)
    db.flush()
    audit(db, request, principal, ds.organization_id, "upload.create", "upload", up.id, files=body.file_count)
    return UploadOut(upload_id=up.id, key_prefix=_prefix(up), expires_at=up.created_at + timedelta(hours=24),
                     max_file_bytes=s.max_file_bytes, part_size=s.part_size_bytes, status=up.status)


@router.post("/uploads/{upload_id}/sign", response_model=SignOut)
def sign(upload_id: uuid.UUID, body: SignIn, principal: PrincipalDep, db: DbDep):
    up = _upload(db, principal, upload_id)
    if up.status != "open":
        raise ApiError(409, "upload_closed", "This upload is already completed")
    if up.created_at + timedelta(hours=24) < now():
        raise ApiError(410, "upload_expired", "This upload session expired; start a new one")
    _check_key(up, body.key)
    op = storage.SIGNABLE.get((body.method, body.upload_id is not None))
    if op is None or (op == "upload_part") != (body.part_number is not None):
        raise ApiError(422, "unsupported_operation", "Unsupported combination of method/upload_id/part_number")
    if op in ("put_object", "create_multipart_upload"):
        if body.size is None:
            raise ApiError(422, "size_required", "Send the file size with PutObject/CreateMultipartUpload")
        if body.size > get_settings().max_file_bytes:
            raise ApiError(413, "file_too_large", "Files can be at most 20 GB")
    return SignOut(url=storage.presign(op, body.key, body.upload_id, body.part_number))


@router.get("/uploads/{upload_id}", response_model=UploadState)
def upload_state(upload_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    up = _upload(db, principal, upload_id, "read")
    return UploadState(upload_id=up.id, status=up.status, source=(up.options or {}).get("source", "files"),
                       dataset_version_id=up.dataset_version_id, error=up.error,
                       objects=storage.existing_sizes(_prefix(up)) if up.status == "open" else {})


@router.post("/uploads/{upload_id}/complete", status_code=202, response_model=CompleteOut)
def complete(upload_id: uuid.UUID, body: CompleteIn, principal: PrincipalDep, db: DbDep, request: Request):
    s = get_settings()
    up = db.scalar(select(m.Upload).where(m.Upload.id == upload_id).with_for_update())
    if not up:
        raise not_found("Upload")
    authorize(db, principal, up.organization_id, "write", "Upload")
    if up.status != "open":
        raise ApiError(409, "upload_closed", "This upload is already completed")
    if len(body.files) > s.max_files_per_batch:
        raise ApiError(413, "too_many_files", f"At most {s.max_files_per_batch} files per upload")
    for f in body.files:
        _check_key(up, f.key)
    present = storage.existing_sizes(_prefix(up))
    problems = [{"key": f.key, "name": f.name, "expected": f.size, "found": present.get(f.key)}
                for f in body.files if present.get(f.key) != f.size]
    if problems:
        raise ApiError(409, "upload_incomplete", "Some files are missing or incomplete; finish them, then complete",
                       extra={"files": problems[:100]})
    up.files = [f.model_dump() for f in body.files]
    audit(db, request, principal, up.organization_id, "upload.complete", "upload", up.id, files=len(body.files))
    if up.kind == "import":
        up.status = "uploaded"
        return JSONResponse(CompleteOut(upload_id=up.id, dataset_version_id=None, status="uploaded")
                            .model_dump(mode="json"), status_code=200)
    ds = get_by_id(db, m.Dataset, up.dataset_id)
    v, append = open_draft(db, ds, "upload")
    v.ingest_status = "running" if append else v.ingest_status
    up.options = {**(up.options or {}), "append": append}
    up.status, up.dataset_version_id = "ingesting", v.id
    refresh_status(db, get_by_id(db, m.Project, up.project_id))
    db.commit()  # the worker reads these rows from its own session
    tasks.ingest_upload.delay(str(up.id))
    return CompleteOut(upload_id=up.id, dataset_version_id=v.id, status="ingesting")


@router.post("/projects/{project_id}/datasets/{dataset_id}/imports", status_code=202,
             responses={200: {"description": "Preview (no mapping given)"}})
def create_import(project_id: uuid.UUID, dataset_id: uuid.UUID, body: ImportIn, principal: PrincipalDep, db: DbDep,
                  request: Request) -> Any:
    ds = _dataset_in_project(db, principal, project_id, dataset_id)
    check_refs(db, body, ds.organization_id)
    up = get_by_id(db, m.Upload, body.upload_id)
    if up.dataset_id != ds.id or up.kind != "import":
        raise ApiError(422, "invalid_upload", "Use an upload of this dataset with source 'annotations'")
    if up.status != "uploaded":
        raise ApiError(409, "upload_not_ready", f"Upload is '{up.status}'; complete it first")
    files = {f["key"]: f for f in up.files}
    if body.key not in files:
        raise ApiError(422, "invalid_key", "`key` must be one of the files in this upload")
    project = get_by_id(db, m.Project, project_id)

    if body.format in TABLE_FORMATS:
        if project.task_type not in ("tabular_classification", "tabular_regression", "text_classification"):
            raise ApiError(422, "wrong_format_for_task", "Tables are for tabular and text projects")
        if body.mapping is None:
            df = tables.read_table(body.key, body.format)
            return JSONResponse(tables.preview(df.head(1000), len(df)), status_code=200)
        df = tables.read_table(body.key, body.format, nrows=1000)
        columns = tables.check_mapping(df, body.mapping, project.task_type)
        v, append = open_draft(db, ds, "import")
        v.ingest_status = "running" if append else v.ingest_status
        up.status, up.dataset_version_id = "completed", v.id
        refresh_status(db, project)
        db.commit()
        tasks.import_table.delay(str(v.id), body.key, body.format, columns)
    else:
        if project.task_type != "object_detection":
            raise ApiError(422, "wrong_format_for_task", "COCO/YOLO/VOC imports are for object detection projects")
        named = {(safe_relpath(f.get("relative_path") or f["name"]) or f["name"]): f for f in up.files}
        images = {n for n in named if PurePosixPath(n).suffix.lower() in tasks.IMAGE_EXT}
        _, errors, _ = tasks.parse_annotations(body.format, body.key, named, images)
        if errors:
            raise ApiError(422, "import_invalid", f"{len(errors)} annotation record(s) are invalid",
                           errors=[{"loc": "/annotations", "msg": str(e)} for e in errors[:200]])
        v, append = open_draft(db, ds, "import")
        v.ingest_status = "running" if append else v.ingest_status
        up.options = {**(up.options or {}), "append": append}
        up.status, up.dataset_version_id = "ingesting", v.id
        refresh_status(db, project)
        db.commit()
        tasks.import_detection.delay(str(v.id), str(up.id), body.format, body.key)
    audit(db, request, principal, ds.organization_id, "dataset.import", "dataset_version", v.id, format=body.format)
    return ImportAccepted(dataset_version_id=v.id)
