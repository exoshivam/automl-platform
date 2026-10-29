"""Ready-made detector mode for object detection (no training): COCO classes, per-project threshold, predict, and
conversion to a custom (trained) project that keeps the classes."""

import copy
import time
import uuid
from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app import models as m
from app.deps import DbDep, PrincipalDep, audit, load_project
from app.errors import ApiError
from app.ml import pretrained
from app.routers.projects import ProjectOut, custom_state, pretrained_spec, project_out, sync_label_set
from app.services.project_status import refresh_status

router = APIRouter(tags=["detection"])
MAX_BYTES = 10 * 1024**2


class ClassesOut(BaseModel):
    model: dict[str, Any]
    classes: list[dict[str, Any]]


class SettingsOut(BaseModel):
    mode: Literal["pretrained", "custom"]
    score_threshold: float
    default_score_threshold: float
    classes: list[str]
    model: dict[str, Any]


class SettingsPatch(BaseModel):
    score_threshold: float | None = Field(default=None, ge=0.05, le=0.95)
    classes: list[str] | None = Field(default=None, min_length=1, max_length=80,
                                      description="Replace the chosen ready-made (COCO) classes")


class CustomObjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class CustomObjectsOut(BaseModel):
    custom_labels: list[str]
    label_set_id: uuid.UUID | None
    project: ProjectOut


class Bbox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class Box(BaseModel):
    class_name: str
    confidence: float
    bbox: Bbox
    source: Literal["ready-made", "custom"] = "ready-made"


class PredictOut(BaseModel):
    boxes: list[Box]
    image: dict[str, int]
    latency_ms: float
    model_load_ms: int
    custom_status: Literal["none", "not_trained", "training", "ready", "outdated", "needs_attention"] = "none"


def _detection(db, principal, project_id, action) -> m.Project:
    p = load_project(db, principal, project_id, action)
    if p.task_type != "object_detection":
        raise ApiError(422, "not_detection", "This is not an object detection project")
    return p


def _new_spec(db, principal, p: m.Project, spec: dict) -> None:
    p.spec_version += 1
    p.current_spec = spec
    db.add(m.TaskSpecVersion(organization_id=p.organization_id, project_id=p.id, version=p.spec_version, spec=spec,
                             source="user_edit", created_by=principal.user_id))


@router.get("/detection/pretrained-classes", response_model=ClassesOut)
def pretrained_classes(principal: PrincipalDep):
    return ClassesOut(model=pretrained.MODEL, classes=[{"id": i, "name": n} for i, n in pretrained.COCO.items()])


@router.get("/projects/{project_id}/detector-settings", response_model=SettingsOut)
def get_detector_settings(project_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    return _settings(_detection(db, principal, project_id, "read"))


@router.patch("/projects/{project_id}/detector-settings", response_model=SettingsOut)
def patch_detector_settings(project_id: uuid.UUID, body: SettingsPatch, principal: PrincipalDep, db: DbDep,
                            request: Request):
    p = _detection(db, principal, project_id, "write")
    spec = copy.deepcopy(p.current_spec)
    if body.score_threshold is not None:
        spec["score_threshold"] = body.score_threshold
    if body.classes is not None:
        if spec.get("mode") != "pretrained":
            raise ApiError(409, "not_pretrained", "Only ready-made detection projects have ready-made classes")
        spec["labels"] = body.classes
        spec = pretrained_spec(spec)  # 422 unknown_pretrained_class, canonical names
        clash = {c.lower() for c in spec["labels"]} & {c.lower() for c in spec.get("custom_labels") or []}
        if clash:
            raise ApiError(409, "duplicate_class", f"Already a custom object: {sorted(clash)}")
    _new_spec(db, principal, p, spec)
    audit(db, request, principal, p.organization_id, "detector.settings", "project", p.id,
          **body.model_dump(exclude_none=True))
    return _settings(p)


def _settings(p: m.Project) -> SettingsOut:
    s = p.current_spec
    return SettingsOut(mode=s.get("mode", "custom"), score_threshold=s.get("score_threshold",
                       pretrained.DEFAULT_THRESHOLD), default_score_threshold=pretrained.DEFAULT_THRESHOLD,
                       classes=s.get("labels") or [], model=pretrained.MODEL)


@router.post("/projects/{project_id}/predict", response_model=PredictOut | dict[str, Any],
             openapi_extra={"requestBody": {"content": {
                 "multipart/form-data": {"schema": {"type": "object", "properties": {"file": {"type": "string",
                                                                                              "format": "binary"}}}},
                 "application/json": {"schema": {"type": "object", "properties": {"text": {"type": "string"}}}}}}})
async def predict(project_id: uuid.UUID, principal: PrincipalDep, db: DbDep, request: Request):
    """Detection: multipart 'file' -> boxes. Text: JSON {"text"} -> label/score/top3/other (+ sentiment)."""
    p = load_project(db, principal, project_id, "read")
    if p.task_type == "text_classification":
        from app.routers.quicktext import TextPredictIn, predict_text

        try:
            body = TextPredictIn.model_validate_json(await request.body())
        except ValueError as e:
            raise ApiError(422, "invalid_body", 'Send {"text": "..."}') from e
        return predict_text(db, p, body.text)
    form = await request.form()
    file = form.get("file")
    if file is None or isinstance(file, str):
        raise ApiError(422, "file_required", "Send the image as the multipart field 'file'")
    return await _predict_image(project_id, principal, db, file)


async def _predict_image(project_id, principal, db, file):
    p = _detection(db, principal, project_id, "read")
    return predict_image_bytes(db, p, await file.read(MAX_BYTES + 1))


def predict_image_bytes(db, p: m.Project, data: bytes) -> PredictOut:
    """Ready-made / hybrid detection on one image. Shared by the playground and deployments (POST /v1/predict)."""
    t0 = time.perf_counter()
    if p.current_spec.get("mode") != "pretrained":
        raise ApiError(409, "not_pretrained", "This project uses its own trained model; predict with "
                       "POST /model-versions/{id}/predict")
    if len(data) > MAX_BYTES:
        raise ApiError(413, "file_too_large", "Images can be at most 10 MB")
    from app.ml.face import load_image  # EXIF-aware decode, shared with face

    try:
        img = load_image(data)
    except Exception:  # noqa: BLE001
        raise ApiError(422, "not_an_image", "Not a readable JPEG/PNG/WebP image") from None
    threshold = p.current_spec.get("score_threshold", pretrained.DEFAULT_THRESHOLD)
    chosen, state = p.current_spec.get("labels") or [], custom_state(db, p)
    custom = set(state["custom_labels"])
    if state["custom_model_version_id"]:  # ONE fine-tuned model (ready-made + custom classes), one pass per frame
        from app.ml.loader import load_model_timed

        model, load_ms = load_model_timed(state["custom_model_version_id"])
        one_pass = ((model.meta.get("output") or {}).get("hybrid") or {}).get("ready_made_from") == "fine-tuned"
        wanted = (set(chosen) if one_pass else set()) | custom
        boxes = [{**b, "source": "custom" if b["class_name"] in custom else "ready-made"}
                 for b in model.predict([img], threshold=threshold)[0]["boxes"] if b["class_name"] in wanted]
        if not one_pass:  # the fine-tune couldn't be shown to keep the ready-made classes: stock model for them
            stock, ms = pretrained.predict(img, chosen, threshold)
            boxes, load_ms = boxes + [{**b, "source": "ready-made"} for b in stock], load_ms + ms
            from app.ml.detection import TOP_K

            boxes = sorted(boxes, key=lambda b: -b["confidence"])[:TOP_K]
    else:  # nothing trained yet: the stock COCO model
        boxes, load_ms = pretrained.predict(img, chosen, threshold)
        boxes = [{**b, "source": "ready-made"} for b in boxes]
    return PredictOut(boxes=boxes, image={"width": img.width, "height": img.height}, model_load_ms=load_ms,
                      custom_status=state["custom_status"], latency_ms=round((time.perf_counter() - t0) * 1000, 1))


@router.post("/projects/{project_id}:convert-to-custom", response_model=ProjectOut)
def convert_to_custom(project_id: uuid.UUID, principal: PrincipalDep, db: DbDep, request: Request):
    p = _detection(db, principal, project_id, "write")
    if p.current_spec.get("mode") != "pretrained":
        raise ApiError(409, "already_custom", "This project already trains its own detector")
    spec = copy.deepcopy(p.current_spec)
    spec["mode"] = "custom"
    spec["labels"] = list(dict.fromkeys((spec.get("labels") or []) + (spec.pop("custom_labels", None) or [])))
    _new_spec(db, principal, p, spec)
    sync_label_set(db, p, spec["labels"])
    db.flush()
    refresh_status(db, p)
    audit(db, request, principal, p.organization_id, "project.convert_to_custom", "project", p.id)
    return project_out(db, p)


def _set_custom(db, principal, request, p: m.Project, custom: list[str], action: str, name: str) -> CustomObjectsOut:
    spec = copy.deepcopy(p.current_spec)
    spec["custom_labels"] = custom
    _new_spec(db, principal, p, spec)
    sync_label_set(db, p, custom)  # the label set of a hybrid project = its custom objects only
    db.flush()
    refresh_status(db, p)
    audit(db, request, principal, p.organization_id, action, "project", p.id, name=name)
    out = project_out(db, p)
    return CustomObjectsOut(custom_labels=custom, label_set_id=out.label_set_id, project=out)


def _pretrained_project(db, principal, project_id) -> m.Project:
    p = _detection(db, principal, project_id, "write")
    if p.current_spec.get("mode") != "pretrained":
        raise ApiError(409, "not_pretrained", "Custom objects are added to ready-made detection projects")
    return p


@router.post("/projects/{project_id}/custom-objects", status_code=201, response_model=CustomObjectsOut)
def add_custom_object(project_id: uuid.UUID, body: CustomObjectIn, principal: PrincipalDep, db: DbDep,
                      request: Request):
    p = _pretrained_project(db, principal, project_id)
    name = body.name.strip()
    coco, _ = pretrained.canonical([name])
    if coco:
        cid = next(i for i, n in pretrained.COCO.items() if n == coco[0])
        raise ApiError(422, "ready_made_class", f"'{name}' is already in the ready-made list as '{coco[0]}'; tick it "
                       "in the detector settings instead", extra={"class_id": cid, "name": coco[0]})
    custom = list(p.current_spec.get("custom_labels") or [])
    if name.lower() in {c.lower() for c in custom}:
        raise ApiError(409, "duplicate_class", f"'{name}' is already a custom object")
    if len(custom) >= 500:
        raise ApiError(422, "too_many_classes", "At most 500 custom objects")
    return _set_custom(db, principal, request, p, [*custom, name], "detector.custom_object_added", name)


@router.delete("/projects/{project_id}/custom-objects/{name}", response_model=CustomObjectsOut)
def delete_custom_object(project_id: uuid.UUID, name: str, principal: PrincipalDep, db: DbDep, request: Request):
    p = _pretrained_project(db, principal, project_id)
    custom = list(p.current_spec.get("custom_labels") or [])
    keep = [c for c in custom if c.lower() != name.strip().lower()]
    if len(keep) == len(custom):
        raise ApiError(404, "not_found", f"'{name}' is not a custom object of this project")
    return _set_custom(db, principal, request, p, keep, "detector.custom_object_deleted", name)
