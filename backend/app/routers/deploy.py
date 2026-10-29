"""Self-serve LOCAL deploy (API bound to 127.0.0.1): one deployment per project, keys bound to it, and ONE fixed
endpoint POST /v1/predict where the key alone picks the model. Prediction reuses the playground code paths exactly:
model_eval.read_predict_input / run_model_version (trained models), pretrained.predict_image_bytes (ready-made and
hybrid detection), quicktext.predict_text (quick text / sentiment), faces.identify_data (face / speaker). Models stay
lazy and idle-unloaded as before: a deployment loads nothing by itself.
"""

import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Literal

from fastapi import APIRouter, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app import models as m
from app.db import now, session_factory
from app.deps import (
    DbDep,
    PrincipalDep,
    TenantRef,
    _api_key_principal,
    audit,
    authorize,
    check_refs,
    get_by_id,
    load_project,
)
from app.errors import ApiError, not_found
from app.security import new_api_key, sha256_hex
from app.services.ratelimit import client as redis

router = APIRouter(tags=["deploy"])
public = APIRouter(tags=["predict"])
PUBLIC_URL = "http://127.0.0.1:8010/v1/predict"
Kind = Literal["model", "ready_made", "quick_text", "sentiment", "face", "speaker"]


class DeploymentIn(BaseModel):
    model_version_id: uuid.UUID | None = TenantRef(m.ModelVersion, "Trained-model projects: the version to pin "
                                                   "(default: the latest ready one)")


class DeploymentPatch(BaseModel):
    status: Literal["active", "paused"] | None = None
    model_version_id: uuid.UUID | None = TenantRef(m.ModelVersion, "Switch / roll back the pinned version")


class DeploymentOut(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    task_type: str
    kind: Kind
    model_version_id: uuid.UUID | None
    model_version_no: int | None
    status: Literal["active", "paused"]
    rate_limit_per_min: int
    key_count: int
    created_by: uuid.UUID | None
    created_at: datetime
    updated_at: datetime


class KeyIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class KeyOut(BaseModel):
    id: uuid.UUID
    name: str
    prefix: str
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None


class KeyCreated(BaseModel):
    id: uuid.UUID
    name: str
    prefix: str
    created_at: datetime
    key: str = Field(description="Shown ONCE; only its hash is stored")


# ---- resolution

def kind_of(p: m.Project) -> Kind:
    spec = p.current_spec
    if p.task_type == "face_recognition":
        return "face"
    if p.task_type == "speaker_recognition":
        return "speaker"
    if p.task_type == "object_detection" and spec.get("mode") == "pretrained":
        return "ready_made"
    if p.task_type == "text_classification" and spec.get("mode") == "quick":
        return "quick_text"
    if p.task_type == "text_classification" and not spec.get("labels") and "sentiment" in (spec.get("ready_made") or []):
        return "sentiment"
    return "model"


def _latest_ready(db, p: m.Project) -> m.ModelVersion | None:
    return db.scalar(select(m.ModelVersion).join(m.Model, m.Model.id == m.ModelVersion.model_id).where(
        m.Model.project_id == p.id, m.Model.organization_id == p.organization_id, m.ModelVersion.status == "ready")
        .order_by(m.ModelVersion.created_at.desc()).limit(1))


def _version_of_project(db, p: m.Project, mv_id: uuid.UUID) -> m.ModelVersion:
    mv = get_by_id(db, m.ModelVersion, mv_id)
    model = get_by_id(db, m.Model, mv.model_id) if mv else None
    if mv is None or model is None or model.project_id != p.id:
        raise ApiError(422, "invalid_model_version", "That model version doesn't belong to this project")
    if mv.status != "ready":
        raise ApiError(422, "model_not_ready", "That model version isn't ready")
    return mv


def _manage(db, principal, p: m.Project) -> None:
    """Member+ manage deployments; face / speaker (biometric) deployments need the workspace OWNER."""
    if kind_of(p) in ("face", "speaker"):
        authorize(db, principal, p.organization_id, "own", "Project")


def _deployment(db, principal, dep_id: uuid.UUID, action="write") -> tuple[m.Deployment, m.Project]:
    d = get_by_id(db, m.Deployment, dep_id)
    if d is None:
        raise not_found("Deployment")
    p = load_project(db, principal, d.project_id, action)  # 404 other tenant, 403 viewer on writes
    if action != "read":
        _manage(db, principal, p)
    return d, p


def _out(db, d: m.Deployment, p: m.Project) -> DeploymentOut:
    mv = get_by_id(db, m.ModelVersion, d.model_version_id) if d.model_version_id else None
    keys = db.scalar(select(func.count()).select_from(m.ApiKey).where(
        m.ApiKey.deployment_id == d.id, m.ApiKey.organization_id == d.organization_id, m.ApiKey.revoked_at.is_(None)))
    return DeploymentOut(id=d.id, project_id=d.project_id, task_type=p.task_type, kind=kind_of(p),
                         model_version_id=d.model_version_id, model_version_no=mv.version_no if mv else None,
                         status=d.status, rate_limit_per_min=d.rate_limit_per_min, key_count=keys or 0,
                         created_by=d.created_by, created_at=d.created_at, updated_at=d.updated_at)


# ---- management (session auth)

@router.get("/projects/{project_id}/deployment", response_model=DeploymentOut)
def get_deployment(project_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    p = load_project(db, principal, project_id, "read")
    d = db.scalar(select(m.Deployment).where(m.Deployment.project_id == p.id,
                                             m.Deployment.organization_id == p.organization_id))
    if d is None:
        raise not_found("Deployment")
    return _out(db, d, p)


@router.post("/projects/{project_id}/deployment", status_code=201, response_model=DeploymentOut)
def create_deployment(project_id: uuid.UUID, body: DeploymentIn, principal: PrincipalDep, db: DbDep,
                      request: Request):
    if principal.api_key:
        raise ApiError(403, "user_required", "Deployments are managed by signed-in users, not API keys")
    p = load_project(db, principal, project_id, "write")
    _manage(db, principal, p)
    check_refs(db, body, p.organization_id)
    if db.scalar(select(m.Deployment.id).where(m.Deployment.project_id == p.id,
                                               m.Deployment.organization_id == p.organization_id)):
        raise ApiError(409, "deployment_exists", "This project is already deployed; switch its version instead")
    mv_id = None
    if kind_of(p) == "model":
        mv = _version_of_project(db, p, body.model_version_id) if body.model_version_id else _latest_ready(db, p)
        if mv is None:
            raise ApiError(409, "no_model", "Train a model before deploying this project")
        mv_id = mv.id
    d = m.Deployment(organization_id=p.organization_id, project_id=p.id, model_version_id=mv_id,
                     created_by=principal.user_id)
    db.add(d)
    db.flush()
    audit(db, request, principal, p.organization_id, "deployment.create", "deployment", d.id,
          model_version_id=str(mv_id) if mv_id else None)
    return _out(db, d, p)


@router.patch("/deployments/{deployment_id}", response_model=DeploymentOut)
def patch_deployment(deployment_id: uuid.UUID, body: DeploymentPatch, principal: PrincipalDep, db: DbDep,
                     request: Request):
    d, p = _deployment(db, principal, deployment_id)
    check_refs(db, body, p.organization_id)
    if body.model_version_id is not None:
        if kind_of(p) != "model":
            raise ApiError(422, "no_versions", "This deployment has no model versions to pin")
        d.model_version_id = _version_of_project(db, p, body.model_version_id).id  # keys stay the same
    if body.status is not None:
        d.status = body.status
    d.updated_at = now()
    audit(db, request, principal, d.organization_id, "deployment.update", "deployment", d.id,
          **{k: str(v) for k, v in body.model_dump(exclude_none=True).items()})
    return _out(db, d, p)


@router.delete("/deployments/{deployment_id}", status_code=204)
def delete_deployment(deployment_id: uuid.UUID, principal: PrincipalDep, db: DbDep, request: Request):
    d, _ = _deployment(db, principal, deployment_id)
    for k in db.scalars(select(m.ApiKey).where(m.ApiKey.deployment_id == d.id,
                                               m.ApiKey.organization_id == d.organization_id)):
        k.revoked_at = k.revoked_at or now()
    org = d.organization_id
    db.delete(d)
    audit(db, request, principal, org, "deployment.delete", "deployment", deployment_id)
    return Response(status_code=204)


@router.post("/deployments/{deployment_id}/keys", status_code=201, response_model=KeyCreated)
def create_key(deployment_id: uuid.UUID, body: KeyIn, principal: PrincipalDep, db: DbDep, request: Request):
    d, _ = _deployment(db, principal, deployment_id)
    full, prefix = new_api_key()  # the same ACC-6 keys: ak_<prefix>_<secret>, only sha256 stored
    k = m.ApiKey(organization_id=d.organization_id, name=body.name, prefix=prefix, key_hash=sha256_hex(full),
                 scopes=["inference"], created_by=principal.user_id, deployment_id=d.id)
    db.add(k)
    db.flush()
    audit(db, request, principal, d.organization_id, "api_key.create", "api_key", k.id, deployment_id=str(d.id))
    return KeyCreated(id=k.id, name=k.name, prefix=k.prefix, created_at=k.created_at, key=full)


@router.get("/deployments/{deployment_id}/keys", response_model=list[KeyOut])
def list_keys(deployment_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    d, _ = _deployment(db, principal, deployment_id, "read")
    rows = db.scalars(select(m.ApiKey).where(m.ApiKey.deployment_id == d.id,
                                             m.ApiKey.organization_id == d.organization_id)
                      .order_by(m.ApiKey.created_at)).all()
    return [KeyOut.model_validate(k, from_attributes=True) for k in rows]


@router.delete("/deployments/{deployment_id}/keys/{key_id}", status_code=204)
def revoke_key(deployment_id: uuid.UUID, key_id: uuid.UUID, principal: PrincipalDep, db: DbDep, request: Request):
    d, _ = _deployment(db, principal, deployment_id)
    k = get_by_id(db, m.ApiKey, key_id)
    if k is None or k.deployment_id != d.id:
        raise not_found("API key")
    k.revoked_at = k.revoked_at or now()
    audit(db, request, principal, d.organization_id, "api_key.revoke", "api_key", key_id)
    return Response(status_code=204)


@router.get("/deployments/{deployment_id}/usage")
def usage(deployment_id: uuid.UUID, principal: PrincipalDep, db: DbDep,
          window: Literal["24h", "7d"] = "24h") -> dict[str, Any]:
    d, _ = _deployment(db, principal, deployment_id, "read")
    since = now() - (timedelta(hours=24) if window == "24h" else timedelta(days=7))
    rows = db.execute(select(m.DeploymentRequest.created_at, m.DeploymentRequest.status_code,
                             m.DeploymentRequest.latency_ms).where(
        m.DeploymentRequest.deployment_id == d.id, m.DeploymentRequest.organization_id == d.organization_id,
        m.DeploymentRequest.created_at >= since)).all()
    lat = sorted(r.latency_ms for r in rows)
    errors = sum(r.status_code >= 400 for r in rows)
    step = timedelta(hours=1) if window == "24h" else timedelta(days=1)
    start = since.replace(minute=0, second=0, microsecond=0) if window == "24h" else since.replace(
        hour=0, minute=0, second=0, microsecond=0)
    buckets: dict[datetime, list] = {}
    t = start
    while t <= now():
        buckets[t] = []
        t += step
    for r in rows:
        key = start + ((r.created_at - start) // step) * step
        buckets.setdefault(key, []).append(r)
    series = [{"t": k.isoformat(), "requests": len(v), "errors": sum(x.status_code >= 400 for x in v),
               "avg_latency_ms": round(sum(x.latency_ms for x in v) / len(v), 1) if v else None}
              for k, v in sorted(buckets.items())]
    return {"window": window, "requests": len(rows), "errors": errors,
            "error_rate": round(errors / len(rows), 4) if rows else 0.0,
            "avg_latency_ms": round(sum(lat) / len(lat), 1) if lat else None,
            "p95_latency_ms": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 1) if lat else None,
            "series": series}


@router.get("/deployments/{deployment_id}/docs")
def docs(deployment_id: uuid.UUID, principal: PrincipalDep, db: DbDep) -> dict[str, Any]:
    d, p = _deployment(db, principal, deployment_id, "read")
    k = kind_of(p)
    mv = get_by_id(db, m.ModelVersion, d.model_version_id) if d.model_version_id else None
    modality = (mv.input_spec or {}).get("modality") if mv else None
    labels = list(mv.label_classes or []) if mv else list(p.current_spec.get("labels") or [])
    if k in ("face", "speaker", "ready_made") or modality == "image":
        field = "file"
        what = "an audio clip (WAV/FLAC/OGG, 16 kHz+ mono)" if k == "speaker" else "an image (JPEG/PNG/WebP)"
        inp = {"content_type": "multipart/form-data", "field": "file", "description": f"Send {what} as 'file'"}
        curl = f'curl -X POST {PUBLIC_URL} -H "Authorization: Bearer $KEY" -F "file=@input.{"wav" if k == "speaker" else "jpg"}"'
    elif modality == "tabular" or p.task_type.startswith("tabular"):
        feats = [f["name"] for f in (mv.input_spec or {}).get("features", [])] if mv else []
        field = None
        inp = {"content_type": "application/json", "schema": {"row": {f: "value" for f in feats}},
               "example": {"row": {f: None for f in feats[:5]}}}
        curl = f'curl -X POST {PUBLIC_URL} -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d \'{{"row": {{...}}}}\''
    else:
        field = None
        inp = {"content_type": "application/json", "schema": {"text": "string"}, "example": {"text": "your text"}}
        curl = f'curl -X POST {PUBLIC_URL} -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d \'{{"text": "your text"}}\''
    out = {
        "face": {"image": {"width": 640, "height": 480}, "threshold": 0.4, "faces": [
            {"box": {"x": 10, "y": 20, "w": 120, "h": 120}, "person_id": "…", "name": "Asha", "score": 0.71,
             "top3": []}], "latency_ms": 180, "model_load_ms": 0},
        "speaker": {"person_id": "…", "name": "Asha", "score": 0.93, "top3": [], "seconds": 3.1, "threshold": 0.91,
                    "latency_ms": 90, "model_load_ms": 0},
        "ready_made": {"boxes": [{"class_name": labels[0] if labels else "person", "confidence": 0.87,
                                  "bbox": {"x": 1, "y": 2, "w": 30, "h": 60}, "source": "ready-made"}],
                       "image": {"width": 640, "height": 480}, "custom_status": "none", "latency_ms": 60,
                       "model_load_ms": 0},
        "quick_text": {"label": labels[0] if labels else None, "score": 0.71, "top3": [], "other": False,
                       "source": "examples", "latency_ms": 20, "model_load_ms": 0},
        "sentiment": {"label": None, "score": None, "top3": [], "other": False, "source": "sentiment_only",
                      "sentiment": {"label": "positive", "score": 0.9, "detail": "positive"}, "latency_ms": 15,
                      "model_load_ms": 0},
    }.get(k) or ({"boxes": [{"class_name": labels[0] if labels else "object", "confidence": 0.8,
                             "bbox": {"x": 1, "y": 2, "w": 30, "h": 60}}], "latency_ms": 50, "model_load_ms": 0}
                 if p.task_type == "object_detection" else {"value": 123.4, "latency_ms": 5, "model_load_ms": 0}
                 if p.task_type == "tabular_regression" else
                 {"label": labels[0] if labels else "class", "confidence": 0.93,
                  "probs": {lb: 0.0 for lb in labels[:5]}, "latency_ms": 30, "model_load_ms": 0})
    return {"endpoint": PUBLIC_URL, "task_type": p.task_type, "kind": k, "status": d.status,
            "model_version_id": d.model_version_id, "input": {**inp, **({"field": field} if field else {})},
            "output_example": out, "labels": labels, "curl": curl, "auth": "Authorization: Bearer <deployment key>",
            "rate_limit_per_min": d.rate_limit_per_min}


# ---- the public endpoint

def _record(d: m.Deployment, key_id, status: int, ms: float) -> None:
    with session_factory()() as s:  # own session: counted even when the request itself fails and rolls back
        s.add(m.DeploymentRequest(organization_id=d.organization_id, deployment_id=d.id, api_key_id=key_id,
                                  status_code=status, latency_ms=round(ms, 1), model_version_id=d.model_version_id))
        s.commit()


def _rate_limit(key: m.ApiKey, per_min: int) -> int | None:
    """Fixed one-minute window per key. Returns seconds to wait when over the limit."""
    t = int(time.time())
    rk = f"rl:deploy:{key.id}:{t // 60}"
    n = redis().incr(rk)
    redis().expire(rk, 70)
    return 60 - t % 60 if n > per_min else None


@public.post("/v1/predict", openapi_extra={"requestBody": {"content": {
    "multipart/form-data": {"schema": {"type": "object", "properties": {"file": {"type": "string",
                                                                               "format": "binary"}}}},
    "application/json": {"schema": {"type": "object", "properties": {"text": {"type": "string"},
                                                                    "row": {"type": "object"}}}}}}})
async def predict(request: Request, db: DbDep):
    """Deployment inference. Authorization: Bearer <deployment key>. The key alone picks the model."""
    auth = request.headers.get("authorization", "")
    raw = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if not raw.startswith("ak_"):
        raise ApiError(401, "unauthorized", "Send the deployment key as 'Authorization: Bearer ak_…'",
                       headers={"WWW-Authenticate": "Bearer"})
    key = _api_key_principal(db, raw).api_key  # 401 bad / revoked / expired
    d = get_by_id(db, m.Deployment, key.deployment_id) if key.deployment_id else None
    if d is None or "inference" not in (key.scopes or []):
        raise ApiError(401, "unauthorized", "This key isn't bound to a deployment", headers={"WWW-Authenticate": "Bearer"})
    t0 = time.perf_counter()
    status = 200
    try:
        if d.status != "active":
            raise ApiError(403, "deployment_paused", "This deployment is paused")
        wait = _rate_limit(key, d.rate_limit_per_min)
        if wait is not None:
            raise ApiError(429, "rate_limited", f"At most {d.rate_limit_per_min} requests per minute per key",
                           headers={"Retry-After": str(wait)})
        p = db.scalar(select(m.Project).where(m.Project.id == d.project_id,
                                              m.Project.organization_id == d.organization_id))
        out = await _run(db, request, d, p)
        db.commit()  # last_used_at
        headers = {"X-Deployment-Id": str(d.id), **({"X-Model-Version": str(d.model_version_id)}
                                                    if d.model_version_id else {})}
        return JSONResponse(jsonable_encoder(out), headers=headers)
    except ApiError as e:
        status = e.status
        raise
    except Exception:
        status = 500
        raise
    finally:
        _record(d, key.id, status, (time.perf_counter() - t0) * 1000)


async def _file(request: Request, limit: int) -> bytes:
    if not request.headers.get("content-type", "").startswith("multipart/"):
        raise ApiError(422, "file_required", "Send the input as the multipart field 'file'")
    form = await request.form()
    f = form.get("file")
    if f is None or isinstance(f, str):
        raise ApiError(422, "file_required", "Send the input as the multipart field 'file'")
    return await f.read(limit + 1)


async def _run(db, request: Request, d: m.Deployment, p: m.Project) -> Any:
    k = kind_of(p)
    if k == "model":
        from app.routers.model_eval import read_predict_input, run_model_version

        mv = get_by_id(db, m.ModelVersion, d.model_version_id) if d.model_version_id else None
        if mv is None:
            raise ApiError(409, "model_not_ready", "The pinned model version no longer exists; pin another one")
        return run_model_version(mv, await read_predict_input(request, mv))
    if k in ("quick_text", "sentiment"):
        from app.routers.quicktext import TextPredictIn, predict_text

        try:
            body = TextPredictIn.model_validate_json(await request.body())
        except ValueError as e:
            raise ApiError(422, "invalid_body", 'Send JSON {"text": "..."}') from e
        return predict_text(db, p, body.text)
    if k == "ready_made":
        from app.routers.pretrained import MAX_BYTES, predict_image_bytes

        return predict_image_bytes(db, p, await _file(request, MAX_BYTES))
    from app.routers.faces import MAX_BYTES as FACE_MAX
    from app.routers.faces import identify_data

    g = db.scalar(select(m.FaceGallery).where(m.FaceGallery.project_id == p.id,
                                              m.FaceGallery.organization_id == p.organization_id))
    return identify_data(db, g, await _file(request, FACE_MAX))
