"""Evaluation, playground prediction and export (PRD §6.7-6.9)."""

import importlib.util
import json
import uuid
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel
from sqlalchemy import select

from app import models as m
from app.deps import DbDep, PrincipalDep, audit, authorize, get_by_id
from app.errors import ApiError, not_found
from app.services import storage

router = APIRouter(tags=["evaluation"])
MAX_PREDICT_BYTES = 20 * 1024 * 1024


def load_mv(db, principal, mv_id: uuid.UUID, action="read") -> m.ModelVersion:
    mv = get_by_id(db, m.ModelVersion, mv_id)
    if not mv:
        raise not_found("Model version")
    authorize(db, principal, mv.organization_id, action, "Model version")
    return mv


class SampleRef(BaseModel):
    id: str
    name: str | None = None
    thumb_url: str | None = None
    url: str | None = None
    text: str | None = None
    width: int | None = None
    height: int | None = None


class EvaluationOut(BaseModel):
    model_version_id: uuid.UUID
    status: Literal["pending", "ready", "failed"]
    error: str | None = None
    report: dict[str, Any] | None = None
    samples: dict[str, SampleRef] = {}  # previews for worst_mistakes (EVL-3)


def sample_refs(db, org_id, ids: list[str]) -> dict[str, SampleRef]:
    uids = []
    for s in ids:
        try:
            uids.append(uuid.UUID(str(s)))
        except ValueError:
            continue  # tabular "row-<i>"
    if not uids:
        return {}
    out = {}
    for s in db.scalars(select(m.Sample).where(m.Sample.id.in_(uids), m.Sample.organization_id == org_id)):
        text = (s.meta or {}).get("text")
        out[str(s.id)] = SampleRef(id=str(s.id), name=s.original_name, width=s.width, height=s.height,
                                   text=text[:500] if text else None,
                                   thumb_url=storage.presign_get(s.thumbnail_key) if s.thumbnail_key else None,
                                   url=storage.presign_get(s.object_key) if s.object_key else None)
    return out


@router.get("/model-versions/{mv_id}/evaluation", response_model=EvaluationOut)
def get_evaluation(mv_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    mv = load_mv(db, principal, mv_id)
    rep = mv.eval_report
    if not rep:
        return EvaluationOut(model_version_id=mv.id, status="pending")
    if rep.get("status") == "failed":
        return EvaluationOut(model_version_id=mv.id, status="failed", error=rep.get("error"))
    return EvaluationOut(model_version_id=mv.id, status="ready", report=rep,
                         samples=sample_refs(db, mv.organization_id, [w["sample_id"] for w in rep["worst_mistakes"]]))


@router.get("/model-versions/{mv_id}/evaluation/cell", response_model=list[SampleRef])
def confusion_cell(mv_id: uuid.UUID, principal: PrincipalDep, db: DbDep, y_true: str = Query(), y_pred: str = Query()):
    """EVL-2 click-through: the test samples in one confusion-matrix cell (most confident first)."""
    mv = load_mv(db, principal, mv_id)
    cells = ((mv.eval_report or {}).get("confusion") or {}).get("cells") or {}
    ids = cells.get(f"{y_true}|{y_pred}", [])
    refs = sample_refs(db, mv.organization_id, ids)
    return [refs.get(i, SampleRef(id=i)) for i in ids]


@router.post("/model-versions/{mv_id}/evaluation/rerun", status_code=202, response_model=EvaluationOut)
def rerun_evaluation(mv_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    mv = load_mv(db, principal, mv_id, "write")
    if mv.status != "ready":
        raise ApiError(409, "model_not_ready", "The model isn't trained yet")
    mv.eval_report = None
    db.commit()
    from app.ml import eval_tasks

    eval_tasks.evaluate.delay(str(mv.id))
    return EvaluationOut(model_version_id=mv.id, status="pending")


@router.get("/model-versions/{mv_id}/model-card")
def model_card(mv_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    from fastapi.responses import PlainTextResponse

    mv = load_mv(db, principal, mv_id)
    key = (mv.artifact_prefix or "") + "MODEL_CARD.md"
    if not mv.artifact_prefix or not storage.head(key):
        raise ApiError(409, "not_evaluated", "The model card is written after evaluation")
    return PlainTextResponse(storage.get_bytes(key).decode(), media_type="text/markdown")


# ---------------------------------------------------------------- playground (PLY-*)

class PredictIn(BaseModel):
    text: str | None = None
    row: dict[str, Any] | None = None


async def read_predict_input(request: Request, mv: m.ModelVersion) -> Any:
    """Parse a predict request for `mv`: multipart `file` (image bytes) or JSON {"text"} / {"row"}; 422 when the
    input kind doesn't match the model's modality. Shared by the playground route and the deployed /v1/predict."""
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("multipart/"):
        form = await request.form()
        f = form.get("file")
        if f is None or isinstance(f, str):
            raise ApiError(422, "file_required", "Send the image as the multipart field 'file'")
        x: Any = await f.read(MAX_PREDICT_BYTES + 1)
        if len(x) > MAX_PREDICT_BYTES:
            raise ApiError(413, "too_large", "Images up to 20 MB")
    else:
        try:
            body = PredictIn.model_validate(json.loads(await request.body() or b"{}"))
        except ValueError as e:
            raise ApiError(422, "invalid_body", 'Send {"text": ...} or {"row": {...}}') from e
        x = body.text if body.text is not None else body.row
        if x is None:
            raise ApiError(422, "input_required", 'Send {"text": ...}, {"row": {...}} or a multipart file')
    kind = {bytes: "image", str: "text", dict: "row"}[type(x)]
    expected = {"image": "image", "text": "text", "tabular": "row"}.get((mv.input_spec or {}).get("modality"))
    if expected and kind != expected:
        how = {"image": "an image as the multipart field 'file'", "text": '{"text": ...}', "row": '{"row": {...}}'}[expected]
        raise ApiError(422, "wrong_input_type", f"This model takes {how}, not {kind} input",
                       extra={"expected": expected, "got": kind})
    return x


def run_model_version(mv: m.ModelVersion, x: Any) -> dict[str, Any]:
    """Load (cached) and run `mv` on one input. Returns the model's output plus latency_ms and model_load_ms
    (>0 when this call loaded the model: first use or after an idle unload)."""
    import time

    if mv.status != "ready":
        raise ApiError(409, "model_not_ready", "The model isn't trained yet")
    if importlib.util.find_spec("app.ml.loader") is None:
        raise ApiError(503, "predict_unavailable", "Model loading isn't available yet")
    from app.ml.loader import load_model_timed

    t0 = time.perf_counter()
    try:
        model, load_ms = load_model_timed(mv.id)
        out = model.predict([x])[0]
    except (ValueError, KeyError, OSError, TypeError, AttributeError) as e:
        raise ApiError(422, "bad_input", f"Couldn't run the model on this input: {e}") from e
    return {**out, "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "model_load_ms": load_ms}


@router.post("/model-versions/{mv_id}/predict")
async def predict(mv_id: uuid.UUID, request: Request, principal: PrincipalDep, db: DbDep) -> dict[str, Any]:
    """multipart `file` (image) or JSON {"text"} / {"row"}. Returns {label, confidence, probs} | {value} | {boxes}
    plus latency_ms and model_load_ms."""
    mv = load_mv(db, principal, mv_id)
    if mv.status != "ready":
        raise ApiError(409, "model_not_ready", "The model isn't trained yet")
    return run_model_version(mv, await read_predict_input(request, mv))


@router.get("/model-versions/{mv_id}/snippets")
def get_snippets(mv_id: uuid.UUID, principal: PrincipalDep, db: DbDep, request: Request) -> dict[str, str]:
    from app.ml import artifacts, export

    mv = load_mv(db, principal, mv_id)
    if not mv.artifact_prefix:
        raise ApiError(409, "model_not_ready", "The model isn't trained yet")
    base = str(request.base_url).rstrip("/") + "/api/v1"
    return export.snippets(artifacts.read_json(mv.artifact_prefix + "model.json"), base)


# ---------------------------------------------------------------- exports (EXP-*)

class ExportIn(BaseModel):
    format: Literal["onnx", "tflite", "coreml"] = "onnx"
    quantization: Literal["none"] = "none"


class ExportOut(BaseModel):
    id: uuid.UUID
    model_version_id: uuid.UUID
    format: str
    quantization: str
    status: Literal["queued", "running", "ready", "failed"]
    size_bytes: int | None
    verified: bool
    verify_report: dict[str, Any] | None
    accuracy_delta: float | None
    error: str | None
    download_url: str | None
    created_at: datetime


def export_out(e: m.Export) -> ExportOut:
    return ExportOut(id=e.id, model_version_id=e.model_version_id, format=e.format, quantization=e.quantization,
                     status=e.status, size_bytes=e.size_bytes, verified=e.verified, verify_report=e.verify_report,
                     accuracy_delta=e.accuracy_delta, error=e.error, created_at=e.created_at,
                     download_url=storage.presign_get(e.artifact_key) if e.status == "ready" and e.artifact_key else None)


@router.get("/model-versions/{mv_id}/exports/preview")
def export_preview(mv_id: uuid.UUID, principal: PrincipalDep, db: DbDep) -> dict[str, Any]:
    """What the ONNX bundle will contain and roughly how big it is, before exporting. Model only: no training data."""
    from app.ml import artifacts, export

    mv = load_mv(db, principal, mv_id)
    if not mv.artifact_prefix or mv.status != "ready":
        raise ApiError(409, "model_not_ready", "The model isn't trained yet")
    return export.bundle_preview(mv, artifacts.read_json(mv.artifact_prefix + "model.json"))


@router.post("/model-versions/{mv_id}/exports", status_code=202, response_model=ExportOut)
def create_export(mv_id: uuid.UUID, body: ExportIn, principal: PrincipalDep, db: DbDep, request: Request):
    mv = load_mv(db, principal, mv_id, "write")
    if body.format != "onnx":
        raise ApiError(422, "phase_2", f"{body.format.upper()} export comes in Phase 2; ONNX is available now")
    if mv.status != "ready":
        raise ApiError(409, "model_not_ready", "The model isn't trained yet")
    from app.ml.export import CONVERTERS

    if mv.framework not in CONVERTERS:
        raise ApiError(422, "phase_2", "ONNX export for this model type (object detection) comes in Phase 2; "
                                       "use the playground or the local API for now")
    ex = db.scalar(select(m.Export).where(m.Export.model_version_id == mv.id, m.Export.format == body.format,
                                          m.Export.quantization == body.quantization,
                                          m.Export.organization_id == mv.organization_id))
    if ex and ex.status in ("queued", "running", "ready"):
        return export_out(ex)  # idempotent; failed ones are retried below
    if not ex:
        ex = m.Export(organization_id=mv.organization_id, model_version_id=mv.id, format=body.format,
                      quantization=body.quantization)
        db.add(ex)
    ex.status, ex.error = "queued", None
    audit(db, request, principal, mv.organization_id, "export.create", "model_version", mv.id, format=body.format)
    db.commit()
    from app.ml import eval_tasks

    eval_tasks.export.delay(str(ex.id))
    db.refresh(ex)
    return export_out(ex)


@router.get("/model-versions/{mv_id}/exports", response_model=list[ExportOut])
def list_exports(mv_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    mv = load_mv(db, principal, mv_id)
    return [export_out(e) for e in db.scalars(select(m.Export).where(
        m.Export.model_version_id == mv.id, m.Export.organization_id == mv.organization_id).order_by(m.Export.created_at))]


@router.get("/exports/{export_id}", response_model=ExportOut)
def get_export(export_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    ex = get_by_id(db, m.Export, export_id)
    if not ex:
        raise not_found("Export")
    authorize(db, principal, ex.organization_id, "read", "Export")
    return export_out(ex)
