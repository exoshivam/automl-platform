"""Dev/test-only hooks. Never mounted when APP_ENV=prod (see main.py)."""

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select

from app import models as m
from app.deps import DbDep
from app.errors import not_found
from app.services import mail
from app.settings import get_settings

LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def local_only(request: Request) -> None:
    """Belt and braces on top of not mounting this router in prod: dev/test env only, loopback clients only
    (TestClient reports host 'testclient'). Anything else gets a plain 404."""
    s = get_settings()
    host = request.client.host if request.client else ""
    allowed = LOOPBACK | ({"testclient"} if s.app_env == "test" else set())
    if s.app_env not in ("dev", "test") or host not in allowed:
        raise not_found()


router = APIRouter(prefix="/_dev", tags=["dev"], include_in_schema=False, dependencies=[Depends(local_only)])


@router.get("/mail")
def dev_mail(limit: int = 50):
    return mail.list_sink(limit)


@router.get("/llm-log")
def dev_llm_log(db: DbDep, limit: int = 50):
    rows = db.scalars(select(m.LlmLog).order_by(m.LlmLog.created_at.desc()).limit(limit)).all()
    return [{"id": r.id, "organization_id": r.organization_id, "provider": r.provider, "attempt": r.attempt,
             "request": r.request, "response": r.response, "outcome": r.outcome, "expires_at": r.expires_at,
             "created_at": r.created_at} for r in rows]
