import importlib
import importlib.util

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app import errors
from app.db import engine
from app.logs import RequestLogMiddleware, configure
from app.routers import (
    auth,
    autolabel,
    datasets,
    deploy,
    dev,
    faces,
    labels,
    orgs,
    pretrained,
    projects,
    quicktext,
    starters,
    storage,
    training,
    uploads,
    validation,
)
from app.services import ratelimit
from app.settings import get_settings


def create_app() -> FastAPI:
    s = get_settings()
    configure(s.log_level, json=s.app_env != "dev")
    if not s.is_prod:
        # Dev/test accounts like acc-a@test.local; prod keeps rejecting special-use domains.
        import email_validator

        email_validator.SPECIAL_USE_DOMAIN_NAMES = [
            d for d in email_validator.SPECIAL_USE_DOMAIN_NAMES if d not in ("local", "test")
        ]
    app = FastAPI(title="AutoML Platform API", version="0.1.0", openapi_url=f"{s.api_prefix}/openapi.json",
                  docs_url=f"{s.api_prefix}/docs", redoc_url=None)
    app.add_middleware(RequestLogMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"], expose_headers=["X-Request-Id", "Retry-After"])
    errors.install(app)

    api = APIRouter(prefix=s.api_prefix)

    @api.get("/health", tags=["health"])
    def health():
        """Readiness: database and Redis reachable."""
        checks = {}
        try:
            with engine().connect() as c:
                c.execute(text("select 1"))
            checks["database"] = "ok"
        except Exception as e:  # noqa: BLE001
            checks["database"] = f"error: {type(e).__name__}"
        try:
            ratelimit.client().ping()
            checks["redis"] = "ok"
        except Exception as e:  # noqa: BLE001
            checks["redis"] = f"error: {type(e).__name__}"
        ok = all(v == "ok" for v in checks.values())
        return {"status": "ok" if ok else "degraded", "checks": checks, "env": s.app_env}

    for r in (auth.router, orgs.router, projects.router, uploads.router, datasets.router, labels.router, validation.router,
              training.router, storage.router, starters.router, faces.router, pretrained.router,
              quicktext.router, deploy.router,
              autolabel.router):
        api.include_router(r)
    if importlib.util.find_spec("app.routers.model_eval"):  # evaluation / predict / export routes
        api.include_router(importlib.import_module("app.routers.model_eval").router)
    if s.app_env in ("dev", "test"):  # explicit allowlist: staging/prod never mount the dev hooks
        api.include_router(dev.router)
    app.include_router(api)
    app.include_router(deploy.public)  # POST /v1/predict: the one public inference endpoint (deployment keys)
    return app


app = create_app()
