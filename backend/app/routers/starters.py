"""DAT-13: starter dataset catalog + import into a project's open draft through the normal ingest pipeline."""

import uuid
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from sqlalchemy import select

from app import models as m
from app import tasks
from app.deps import DbDep, PrincipalDep, audit, load_project
from app.errors import ApiError, not_found
from app.routers.uploads import open_draft
from app.services import starters, storage
from app.services.project_status import refresh_status
from app.services.specs import NEEDS_LABELS

router = APIRouter(tags=["starters"])
BUSY = ("ingesting", "completing")


class StarterOut(BaseModel):
    id: str
    name: str
    description: str
    task_type: str
    classes: list[str]
    target: str | None = None
    sample_count: int
    size_bytes: int
    source: str
    licence: str
    source_url: str | None = None
    preview_urls: list[str] = []


class StarterImportIn(BaseModel):
    starter_id: str = Field(max_length=100)


class StarterImportOut(BaseModel):
    dataset_version_id: uuid.UUID
    upload_id: uuid.UUID
    status: Literal["ingesting"]
    sample_count: int


def _out(s: dict) -> StarterOut:
    return StarterOut(**{k: v for k, v in s.items() if k in StarterOut.model_fields},
                      preview_urls=[storage.presign_get(k) for k in s.get("previews", [])])


@router.get("/starter-datasets", response_model=list[StarterOut])
def list_starters(principal: PrincipalDep, task_type: str | None = None):
    return [_out(s) for s in starters.catalog() if task_type is None or s["task_type"] == task_type]


@router.post("/projects/{project_id}/starter-imports", status_code=202, response_model=StarterImportOut)
def import_starter(project_id: uuid.UUID, body: StarterImportIn, principal: PrincipalDep, db: DbDep,
                   request: Request):
    p = load_project(db, principal, project_id, "write")  # 404 other tenant, 409 project_deleting
    s = starters.get(body.starter_id)
    if s is None:
        raise not_found("Starter dataset")
    if s["task_type"] != p.task_type:
        raise ApiError(422, "task_mismatch", f"'{s['name']}' is a {s['task_type']} dataset; this project is "
                       f"{p.task_type}", extra={"starter_task_type": s["task_type"], "project_task_type": p.task_type})
    org = p.organization_id
    ds = db.scalar(select(m.Dataset).where(m.Dataset.project_id == p.id, m.Dataset.organization_id == org))
    ups = db.scalars(select(m.Upload).where(m.Upload.dataset_id == ds.id, m.Upload.organization_id == org)
                     .with_for_update()).all()
    if any(u.status in BUSY for u in ups):
        raise ApiError(409, "import_running", "An upload or import is still being processed; wait for it to finish")
    if any((u.options or {}).get("starter_id") == s["id"] and u.status != "failed" for u in ups):
        raise ApiError(409, "already_imported", f"'{s['name']}' is already in this project's dataset")
    if p.task_type in NEEDS_LABELS:
        _add_classes(db, principal, request, p, s["classes"])
    v, append = open_draft(db, ds, "import")
    v.ingest_status = "running" if append else v.ingest_status
    up = m.Upload(organization_id=org, project_id=p.id, dataset_id=ds.id, files=[], status="ingesting",
                  kind="files" if s["kind"] == "files" else "import", created_by=principal.user_id,
                  dataset_version_id=v.id, options={"source": "starter", "starter_id": s["id"], "append": append,
                                                    "label_from_folder": s["kind"] == "files"})
    db.add(up)
    db.flush()
    refresh_status(db, p)
    audit(db, request, principal, org, "dataset.starter_import", "dataset_version", v.id, starter_id=s["id"])
    db.commit()  # the worker reads these rows
    tasks.import_starter.delay(str(up.id))
    return StarterImportOut(dataset_version_id=v.id, upload_id=up.id, status="ingesting", sample_count=s["sample_count"])


def _add_classes(db, principal, request, p: m.Project, classes: list[str]) -> None:
    from app.routers.labels import _apply_class_change

    ls = db.scalar(select(m.LabelSet).where(m.LabelSet.project_id == p.id, m.LabelSet.organization_id ==
                                            p.organization_id).order_by(m.LabelSet.version.desc()).limit(1)
                   .with_for_update())
    have = {c.lower() for c in (ls.classes if ls else [])}
    new = [c for c in classes if c.lower() not in have]
    if not new:
        return
    if ls is None:
        db.add(m.LabelSet(organization_id=p.organization_id, project_id=p.id, classes=new, version=1))
        db.flush()
        return
    _apply_class_change(db, principal, request, ls, {}, ls.classes + new, "labels.add_classes")
