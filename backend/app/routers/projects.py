"""PRJ-1..7: templates, projects, versioned task specs, plain-English drafts."""

import copy
import uuid
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Query, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app import models as m
from app.db import now
from app.deps import DbDep, PrincipalDep, TenantRef, audit, authorize, check_refs, get_by_id, load_project
from app.errors import ApiError, not_found
from app.pagination import Page, paginate
from app.services import llm
from app.services.project_status import refresh_status
from app.services.specs import (
    MODALITY,
    TEMPLATES,
    category_of,
    default_spec_for_task,
    template,
    train_labels,
    validate_spec,
)

router = APIRouter(tags=["projects"])
Status = Literal["Draft", "Data needed", "Validating", "Ready to train", "Training", "Trained", "Deployed", "Deleting", "Ready"]


class TemplateOut(BaseModel):
    id: str
    category: str
    name: str
    description: str
    enabled: bool
    task_types: list[str]
    default_spec: dict[str, Any] | None


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    template: str | None = Field(default=None, description="Template id from GET /templates")
    task_type: str | None = Field(default=None, description="Only for the tabular template: tabular_classification "
                                                            "(default) or tabular_regression")
    labels: list[str] | None = Field(default=None, description="Class names (2-500) for classification/detection")
    mode: Literal["pretrained", "custom", "quick"] | None = Field(
        default=None, description="Detection: 'pretrained' = ready-made COCO detector. Text: 'quick' = labels from a "
                                  "few examples, no training. Default 'custom' (train your own)")
    ready_made: list[Literal["sentiment"]] | None = Field(default=None, description="Text only: add ready-made "
                                                          "sentiment to every prediction")
    spec: dict[str, Any] | None = Field(default=None, description="A full spec, e.g. the reviewed draft from "
                                                                  "from-description; overrides template/labels")
    llm_log_id: uuid.UUID | None = TenantRef(m.LlmLog, "Set when `spec` came from from-description")


class ProjectPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    archived: bool | None = None


class SpecPut(BaseModel):
    spec: dict[str, Any]


class ProjectOut(BaseModel):
    id: uuid.UUID
    organization_id: uuid.UUID
    name: str
    task_type: str
    category: str
    status: Status
    current_spec: dict[str, Any]
    spec_version: int
    dataset_id: uuid.UUID | None
    label_set_id: uuid.UUID | None
    sample_count: int = 0
    current_version_id: uuid.UUID | None = None
    best_metric: float | None = None  # latest ready model version's primary metric (test if evaluated, else val)
    best_metric_name: str | None = None
    custom_status: Literal["none", "not_trained", "training", "ready", "outdated", "needs_attention"] | None = Field(
        default=None, description="Ready-made detection projects only: state of the user's own (custom) objects")
    custom_labels: list[str] = []
    custom_model_version_id: uuid.UUID | None = None
    best_metric_split: Literal["test", "val"] | None = None
    latest_model_version_id: uuid.UUID | None = None
    archived_at: datetime | None
    created_at: datetime
    updated_at: datetime


class SpecVersionOut(BaseModel):
    version: int
    spec: dict[str, Any]
    source: Literal["template", "llm", "user_edit"]
    created_by: uuid.UUID | None
    created_at: datetime


class FromDescriptionIn(BaseModel):
    description: str = Field(min_length=10, max_length=2000)
    include_sample_rows: bool = Field(default=False, description="Opt-in: send column names and up to 5 rows")
    columns: list[str] | None = Field(default=None, max_length=500)
    sample_rows: list[Any] | None = Field(default=None, description="Only the first 5 are ever sent")


class FromDescriptionOut(BaseModel):
    status: Literal["ok", "unsupported", "ambiguous"]
    message: str
    category: str | None
    draft_spec: dict[str, Any] | None
    llm_log_id: uuid.UUID


def _latest_label_set(db, project_id) -> m.LabelSet | None:
    return db.scalar(select(m.LabelSet).where(m.LabelSet.project_id == project_id)
                     .order_by(m.LabelSet.version.desc()).limit(1))


def project_out(db, p: m.Project) -> ProjectOut:
    ds = db.scalar(select(m.Dataset).where(m.Dataset.project_id == p.id,
                                           m.Dataset.organization_id == p.organization_id).limit(1))
    cur = db.scalar(select(m.DatasetVersion).where(m.DatasetVersion.id == ds.current_version_id,
                                                   m.DatasetVersion.organization_id == p.organization_id)) \
        if ds and ds.current_version_id else None
    ls = _latest_label_set(db, p.id)
    best = _best_metric(db, p)
    if p.current_spec.get("mode") == "pretrained":
        best = {**best, **custom_state(db, p)}
    return ProjectOut(**best,id=p.id, organization_id=p.organization_id, name=p.name, task_type=p.task_type,
                      category=category_of(p.task_type), status=p.status, current_spec=p.current_spec,
                      spec_version=p.spec_version, dataset_id=ds.id if ds else None,
                      label_set_id=ls.id if ls else None, sample_count=_face_photos(db, p) if
                      p.task_type in ("face_recognition", "speaker_recognition") else (cur.sample_count if cur else 0),
                      current_version_id=cur.id if cur else None,
                      archived_at=p.archived_at, created_at=p.created_at, updated_at=p.updated_at)


def custom_state(db, p: m.Project) -> dict:
    """Hybrid detector: what the project's own objects can do right now."""
    custom = p.current_spec.get("custom_labels") or []
    org = p.organization_id
    mv = db.scalar(select(m.ModelVersion.id).join(m.Model, m.Model.id == m.ModelVersion.model_id).where(
        m.Model.project_id == p.id, m.Model.organization_id == org, m.ModelVersion.status == "ready",
        m.ModelVersion.framework == "torchvision").order_by(m.ModelVersion.created_at.desc()).limit(1))
    busy = db.scalar(select(m.TrainingJob.id).where(m.TrainingJob.project_id == p.id, m.TrainingJob.organization_id ==
                                                   org, m.TrainingJob.status.in_(("queued", "running"))).limit(1))
    status = "none" if not custom else "training" if busy else "ready" if mv else "not_trained"
    if status == "ready":
        row = db.execute(select(m.ModelVersion.label_classes, m.ModelVersion.metrics).where(
            m.ModelVersion.id == mv)).one()
        hy = {k[7:]: v for k, v in ((row.metrics or {}).get("val") or {}).items() if k.startswith("hybrid/")}
        if hy.get("warning") or hy.get("ready_made_from") == "stock":
            status = "needs_attention"  # the fine-tune forgot (or couldn't show it kept) the ready-made classes
        elif set(row.label_classes or []) != set(p.current_spec.get("labels") or []) | set(custom):
            status = "outdated"  # classes changed since training: it still works, but retrain
    return {"custom_status": status, "custom_labels": custom, "custom_model_version_id": mv if custom else None}


def _face_photos(db, p: m.Project) -> int:
    """Face projects have no dataset versions: their samples are the enrolled photos (templates excluded)."""
    return db.scalar(select(func.count()).select_from(m.FaceEmbedding).join(
        m.FaceGallery, m.FaceGallery.id == m.FaceEmbedding.gallery_id).where(
        m.FaceGallery.project_id == p.id, m.FaceEmbedding.organization_id == p.organization_id,
        m.FaceEmbedding.is_template.is_(False))) or 0


def _best_metric(db, p: m.Project) -> dict:
    mv = db.scalar(select(m.ModelVersion).join(m.Model, m.Model.id == m.ModelVersion.model_id).where(
        m.Model.project_id == p.id, m.Model.organization_id == p.organization_id, m.ModelVersion.status == "ready")
        .order_by(m.ModelVersion.created_at.desc()).limit(1))
    if mv is None:
        return {}
    metrics = mv.metrics or {}
    name = metrics.get("primary_metric") or p.current_spec.get("primary_metric")
    for split in ("test", "val"):
        vals = metrics.get(split) or {}
        key = name if name in vals else next((k for k in ("macro_f1", "map_50", "rmse", "accuracy") if k in vals), None)
        if key is not None and isinstance(vals[key], int | float):
            return {"best_metric": float(vals[key]), "best_metric_name": key, "best_metric_split": split,
                    "latest_model_version_id": mv.id}
    return {"latest_model_version_id": mv.id}


def pretrained_spec(spec: dict) -> dict:
    """Ready-made detector: labels must be COCO classes (stored with the canonical spelling), default threshold."""
    if not isinstance(spec, dict) or spec.get("mode") != "pretrained":
        return spec
    from app.ml import pretrained

    names, unknown = pretrained.canonical(spec.get("labels") or [])
    if unknown:
        raise ApiError(422, "unknown_pretrained_class", f"Not a class of the ready-made detector: {unknown}",
                       errors=[{"loc": ["body", "labels"], "msg": "pick classes from GET /detection/pretrained-classes"}],
                       extra={"unknown": unknown})
    return {**spec, "labels": list(dict.fromkeys(names)), "custom_labels": spec.get("custom_labels") or [],
            "score_threshold": spec.get("score_threshold", pretrained.DEFAULT_THRESHOLD)}


def _invalid(errors) -> ApiError:
    return ApiError(422, "invalid_spec", "The task spec is invalid", errors=errors)


def sync_label_set(db, project: m.Project, labels: list[str]) -> None:
    current = _latest_label_set(db, project.id)
    if current and current.classes == labels:
        return
    db.add(m.LabelSet(organization_id=project.organization_id, project_id=project.id, classes=labels,
                      version=(current.version + 1) if current else 1))


@router.get("/templates", response_model=list[TemplateOut])
def list_templates():
    return TEMPLATES


@router.post("/orgs/{org_id}/projects", status_code=201, response_model=ProjectOut)
def create_project(org_id: uuid.UUID, body: ProjectCreate, principal: PrincipalDep, db: DbDep, request: Request):
    authorize(db, principal, org_id, "write", "Workspace")
    check_refs(db, body, org_id)  # ids in bodies are tenant-checked too (NFR-SEC3)
    if body.spec is not None:
        spec, source = copy.deepcopy(body.spec), ("llm" if body.llm_log_id else "user_edit")
    else:
        tpl = template(body.template or "")
        if not tpl:
            raise ApiError(422, "unknown_template", f"Unknown template '{body.template}'",
                           errors=[{"loc": ["body", "template"], "msg": "Use an id from GET /templates"}])
        if not tpl["enabled"]:
            raise ApiError(422, "template_coming_soon", f"'{tpl['name']}' is coming soon",
                           errors=[{"loc": ["body", "template"], "msg": "Not available yet"}])
        task = body.task_type or tpl["task_types"][0]
        if task not in tpl["task_types"]:
            raise ApiError(422, "invalid_task_type", f"Template '{tpl['id']}' supports {tpl['task_types']}",
                           errors=[{"loc": ["body", "task_type"], "msg": "Not valid for this template"}])
        spec, source = default_spec_for_task(task), "template"
        if body.labels is not None:
            spec["labels"] = body.labels
        if body.mode is not None:
            spec["mode"] = body.mode
        if body.ready_made is not None:
            spec["ready_made"] = body.ready_made
    spec = pretrained_spec(spec)
    if errors := validate_spec(spec):
        raise _invalid(errors)

    project = m.Project(organization_id=org_id, name=body.name, task_type=spec["task_type"], status="Draft",
                        current_spec=spec, spec_version=1, created_by=principal.user_id)
    db.add(project)
    db.flush()
    db.add(m.TaskSpecVersion(organization_id=org_id, project_id=project.id, version=1, spec=spec, source=source,
                             llm_prompt_ref=body.llm_log_id, created_by=principal.user_id))
    db.add(m.Dataset(organization_id=org_id, project_id=project.id, name=body.name,
                     modality=MODALITY[spec["task_type"]]))
    sync_label_set(db, project, train_labels(spec))
    if project.task_type == "face_recognition":
        from app.ml.face import model_info
        from app.settings import get_settings

        db.add(m.FaceGallery(organization_id=org_id, project_id=project.id, name=body.name,
                             threshold=get_settings().face_default_threshold,
                             embedding_model_id=model_info()["embedding_model"]["id"]))
    elif project.task_type == "speaker_recognition":  # same gallery tables, kind=speaker
        from app.ml import speaker
        from app.settings import get_settings

        db.add(m.FaceGallery(organization_id=org_id, project_id=project.id, name=body.name, kind="speaker",
                             threshold=get_settings().speaker_default_threshold, embedding_model_id=speaker.MODEL_ID))
    db.flush()
    refresh_status(db, project)  # e.g. a ready-made detector is Ready right away
    audit(db, request, principal, org_id, "project.create", "project", project.id, source=source)
    return project_out(db, project)


@router.post("/orgs/{org_id}/projects/from-description", response_model=FromDescriptionOut)
def from_description(org_id: uuid.UUID, body: FromDescriptionIn, principal: PrincipalDep, db: DbDep):
    """Drafts a spec with the LLM. Creates nothing; the client reviews the draft and POSTs /orgs/{id}/projects."""
    authorize(db, principal, org_id, "write", "Workspace")
    return llm.spec_from_description(org_id, principal.user_id, body.description, body.columns, body.sample_rows,
                                     body.include_sample_rows)


@router.get("/orgs/{org_id}/projects", response_model=Page[ProjectOut])
def list_projects(org_id: uuid.UUID, principal: PrincipalDep, db: DbDep, archived: bool = False,
                  search: str | None = Query(None, max_length=200), status: Status | None = None,
                  cursor: str | None = None, limit: int = Query(50, ge=1, le=200)):
    authorize(db, principal, org_id, "read", "Workspace")
    stmt = select(m.Project).where(m.Project.organization_id == org_id, m.Project.status != "Deleting",
                                   m.Project.archived_at.is_not(None) if archived else m.Project.archived_at.is_(None))
    if search:
        stmt = stmt.where(func.lower(m.Project.name).contains(search.lower()))
    if status:
        stmt = stmt.where(m.Project.status == status)
    rows, nxt = paginate(db, stmt, m.Project.id, cursor, limit)
    return Page[ProjectOut](items=[project_out(db, p) for p in rows], next_cursor=nxt)


@router.get("/projects/{project_id}", response_model=ProjectOut)
def get_project(project_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    return project_out(db, load_project(db, principal, project_id, "read"))


@router.patch("/projects/{project_id}", response_model=ProjectOut)
def patch_project(project_id: uuid.UUID, body: ProjectPatch, principal: PrincipalDep, db: DbDep, request: Request):
    p = load_project(db, principal, project_id, "write")
    if body.name is not None:
        p.name = body.name
    if body.archived is not None:
        p.archived_at = (p.archived_at or now()) if body.archived else None
    audit(db, request, principal, p.organization_id, "project.update", "project", p.id)
    return project_out(db, p)


@router.delete("/projects/{project_id}", status_code=204)
def archive_project(project_id: uuid.UUID, principal: PrincipalDep, db: DbDep, request: Request):
    """§11: delete = archive (restorable with PATCH {archived: false})."""
    p = load_project(db, principal, project_id, "write")
    p.archived_at = p.archived_at or now()
    audit(db, request, principal, p.organization_id, "project.archive", "project", p.id)
    return Response(status_code=204)


@router.put("/projects/{project_id}/spec", response_model=ProjectOut)
def put_spec(project_id: uuid.UUID, body: SpecPut, principal: PrincipalDep, db: DbDep, request: Request):
    p = db.scalar(select(m.Project).where(m.Project.id == project_id).with_for_update())
    if not p:
        raise not_found("Project")
    authorize(db, principal, p.organization_id, "write", "Project")
    if errors := validate_spec(body.spec):
        raise _invalid(errors)
    if body.spec["task_type"] != p.task_type:
        raise _invalid([{"loc": "/task_type", "msg": "The task type can't change; create a new project instead"}])
    if body.spec.get("mode", "custom") != p.current_spec.get("mode", "custom"):
        raise _invalid([{"loc": "/mode", "msg": "Use POST /projects/{id}:convert-to-custom to change the mode"}])
    body.spec = pretrained_spec(body.spec)
    p.spec_version += 1
    p.current_spec = body.spec
    db.add(m.TaskSpecVersion(organization_id=p.organization_id, project_id=p.id, version=p.spec_version,
                             spec=body.spec, source="user_edit", created_by=principal.user_id))
    sync_label_set(db, p, train_labels(body.spec))
    audit(db, request, principal, p.organization_id, "project.spec_update", "project", p.id, version=p.spec_version)
    db.flush()
    return project_out(db, p)


@router.get("/projects/{project_id}/spec/versions", response_model=list[SpecVersionOut])
def spec_versions(project_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    p = load_project(db, principal, project_id, "read")
    rows = db.scalars(select(m.TaskSpecVersion).where(m.TaskSpecVersion.project_id == p.id,
                                                      m.TaskSpecVersion.organization_id == p.organization_id)
                      .order_by(m.TaskSpecVersion.version)).all()
    return [SpecVersionOut.model_validate(r, from_attributes=True) for r in rows]


@router.get("/projects/{project_id}/spec/versions/{version}", response_model=SpecVersionOut)
def spec_version(project_id: uuid.UUID, version: int, principal: PrincipalDep, db: DbDep):
    p = load_project(db, principal, project_id, "read")
    row = db.scalar(select(m.TaskSpecVersion).where(m.TaskSpecVersion.project_id == p.id,
                                                    m.TaskSpecVersion.version == version))
    if not row:
        raise not_found("Spec version")
    return SpecVersionOut.model_validate(row, from_attributes=True)



class PurgeIn(BaseModel):
    confirm_name: str = Field(description="Must equal the project name exactly")


class PurgeOut(BaseModel):
    project_id: uuid.UUID
    status: Literal["Deleting"]
    task_id: str


@router.post("/projects/{project_id}:purge", status_code=202, response_model=PurgeOut)
def purge_project(project_id: uuid.UUID, body: PurgeIn, principal: PrincipalDep, db: DbDep, request: Request):
    """PERMANENT delete of the project and everything in it (photos, versions, models, exports, MLflow).
    Owner/admin only. Running jobs are cancelled; the purge task waits for them to stop."""
    p = get_by_id(db, m.Project, project_id)
    if not p:
        raise not_found("Project")
    authorize(db, principal, p.organization_id, "manage", "Project")
    if body.confirm_name != p.name:
        raise ApiError(422, "confirmation_mismatch", "Type the project name exactly to confirm",
                       errors=[{"loc": ["body", "confirm_name"], "msg": "does not match the project name"}])
    from app.ml.train_tasks import cancel_key
    from app.services.ratelimit import client as redis

    for j in db.scalars(select(m.TrainingJob).where(m.TrainingJob.project_id == p.id,
                                                    m.TrainingJob.organization_id == p.organization_id,
                                                    m.TrainingJob.status.in_(("queued", "running")))):
        j.cancel_requested = True
        redis().set(cancel_key(j.id), "1", ex=24 * 3600)
    p.status = "Deleting"
    audit(db, request, principal, p.organization_id, "project.purge_requested", "project", p.id, name=p.name)
    db.commit()
    from app import tasks

    res = tasks.purge_project.delay(str(p.id))
    return PurgeOut(project_id=p.id, status="Deleting", task_id=res.id)
