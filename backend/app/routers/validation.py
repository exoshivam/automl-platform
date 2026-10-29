"""VAL: report, acknowledgements, auto-fixes (each fix creates a new version, sealed + re-validated)."""

import copy
import uuid
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm.attributes import flag_modified

from app import models as m
from app import tasks
from app.deps import DbDep, PrincipalDep, audit, get_by_id, load_version, require_current
from app.errors import ApiError
from app.routers.uploads import _new_version
from app.services import validation
from app.services.project_status import refresh_status

router = APIRouter(tags=["validation"])
FixId = Literal["exclude_corrupt", "dedupe_exact", "resplit_groups", "exclude_id_columns", "dedupe_rows"]


class FixOut(BaseModel):
    code: str
    severity: Literal["error", "warning", "info"]
    title: str
    why: str
    count: int
    examples: list[str]
    fix: dict
    acknowledgeable: bool
    acknowledged: bool
    details: dict


class ReportOut(BaseModel):
    dataset_version_id: uuid.UUID
    status: Literal["pending", "running", "passed", "warnings", "failed"]
    summary: dict[str, int]
    items: list[FixOut]
    diversity: dict | None = Field(default=None, description="{samples, effective_samples, near_duplicate_share, "
                                   "sessions_per_class, min_sessions_per_class}; images only")


class FixesIn(BaseModel):
    fix_ids: list[FixId] = Field(min_length=1)


class FixesOut(BaseModel):
    dataset_version_id: uuid.UUID
    status: str = "validating"


class AckIn(BaseModel):
    codes: list[str] = Field(min_length=1)


def report_out(v: m.DatasetVersion) -> ReportOut:
    status = {"not_run": "pending"}.get(v.validation_status, v.validation_status)
    rep = v.validation_report or {"summary": {"errors": 0, "warnings": 0, "info": 0}, "items": []}
    return ReportOut(dataset_version_id=v.id, status=status, summary=rep["summary"], items=rep["items"],
                     diversity=rep.get("diversity"))


@router.get("/dataset-versions/{version_id}/validation", response_model=ReportOut)
def get_report(version_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    return report_out(load_version(db, principal, version_id, "read"))


@router.post("/dataset-versions/{version_id}/validation/acknowledge", response_model=ReportOut)
def acknowledge(version_id: uuid.UUID, body: AckIn, principal: PrincipalDep, db: DbDep, request: Request):
    v = load_version(db, principal, version_id, "write")
    if not v.validation_report:
        raise ApiError(409, "not_validated", "Seal the version first")
    items = v.validation_report["items"]
    ackable = {it["code"] for it in items if it["acknowledgeable"]}
    info_only = {it["code"] for it in items if it["severity"] == "info"} - {it["code"] for it in items
                                                                          if it["severity"] != "info"}
    bad = sorted(set(body.codes) - ackable - info_only)  # info needs no ack: accepted as a no-op
    if bad:
        raise ApiError(422, "not_acknowledgeable", f"These can't be acknowledged here: {bad}",
                       errors=[{"loc": ["body", "codes"], "msg": f"{c} is not an acknowledgeable item"} for c in bad])
    v.acknowledgements = sorted(set(v.acknowledgements or []) | set(body.codes))
    # deep copy + flag_modified: summarise() edits item dicts, and an in-place edit of a JSONB value looks unchanged
    # to SQLAlchemy, so the UPDATE was skipped whenever the overall status didn't change (lost VAL-18 acks).
    current = copy.deepcopy(v.validation_report)
    report = {**validation.summarise(current["items"], v.acknowledgements), "diversity": current.get("diversity")}
    v.validation_report, v.validation_status = report, report["status"]
    flag_modified(v, "validation_report")
    project = get_by_id(db, m.Project, get_by_id(db, m.Dataset, v.dataset_id).project_id)
    db.flush()
    refresh_status(db, project)
    audit(db, request, principal, v.organization_id, "validation.acknowledge", "dataset_version", v.id,
          codes=body.codes)
    return report_out(v)


@router.post("/dataset-versions/{version_id}/validation/fixes", status_code=202, response_model=FixesOut)
def apply_fixes(version_id: uuid.UUID, body: FixesIn, principal: PrincipalDep, db: DbDep, request: Request):
    v = load_version(db, principal, version_id, "write")
    if not v.validation_report:
        raise ApiError(409, "not_validated", "Seal the version first")
    ds = require_current(db, v, "fix")
    offered = {it["fix"]["auto_fix_id"] for it in v.validation_report["items"] if it["fix"]["auto_fix_id"]}
    bad = sorted(set(body.fix_ids) - offered)
    if bad:
        raise ApiError(422, "fix_not_offered", f"These fixes aren't offered for this version: {bad}")
    child = _new_version(db, ds, "fix")
    child.parent_version_id = v.id
    child.options = {**(v.options or {}), "fixes": sorted(body.fix_ids)}
    db.commit()
    tasks.apply_fixes.delay(str(child.id), str(v.id), sorted(body.fix_ids))
    audit(db, request, principal, v.organization_id, "validation.fix", "dataset_version", child.id, fixes=body.fix_ids)
    return FixesOut(dataset_version_id=child.id)
