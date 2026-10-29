"""Authentication (JWT or API key) and the ACC-4 role matrix.

Actions: read < write < manage < own.
  read:   owner, admin, member, viewer   | API key scope projects:read or projects:write
  write:  owner, admin, member           | API key scope projects:write
  manage: owner, admin                   | never via API key (members, keys, org settings, audit)
  own:    owner                          | never via API key (delete org, grant owner)
A principal with no access to the org gets 404, never 403, so ids don't leak existence (NFR-SEC3).
"""

import uuid
from dataclasses import dataclass
from typing import Annotated, Literal

from fastapi import Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import models as m
from app.db import get_db, now
from app.errors import ApiError, forbidden, not_found
from app.security import api_key_matches, decode_access_token, parse_api_key

Action = Literal["read", "write", "manage", "own"]
ROLE_ACTIONS: dict[str, set[str]] = {
    "owner": {"read", "write", "manage", "own"},
    "admin": {"read", "write", "manage"},
    "member": {"read", "write"},
    "viewer": {"read"},
}
SCOPE_ACTIONS: dict[str, set[str]] = {"projects:read": {"read"}, "projects:write": {"read", "write"}}


@dataclass
class Principal:
    user: m.User | None = None
    api_key: m.ApiKey | None = None

    @property
    def user_id(self) -> uuid.UUID | None:
        return self.user.id if self.user else None


# scope="function": commit runs when the endpoint returns, BEFORE the response is sent. With the default
# ("request") a 204 could reach the client before e.g. an API-key revoke was committed.
DbDep = Annotated[Session, Depends(get_db, scope="function")]


def _unauthorized(detail: str = "Authentication required") -> ApiError:
    return ApiError(401, "unauthorized", detail, headers={"WWW-Authenticate": "Bearer"})


def _api_key_principal(db: Session, raw: str) -> Principal:
    prefix = parse_api_key(raw)
    if not prefix:
        raise _unauthorized("Invalid API key")
    keys = db.scalars(select(m.ApiKey).where(m.ApiKey.prefix == prefix, m.ApiKey.revoked_at.is_(None))).all()
    key = next((k for k in keys if api_key_matches(raw, k.key_hash)), None)
    if not key or (key.expires_at and key.expires_at <= now()):
        raise _unauthorized("Invalid or expired API key")
    key.last_used_at = now()
    return Principal(api_key=key)


def get_principal(request: Request, db: DbDep) -> Principal:
    auth = request.headers.get("authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else None
    x_key = request.headers.get("x-api-key")
    if x_key or (token and token.startswith("ak_")):
        return _api_key_principal(db, x_key or token)
    if not token and request.url.path.endswith("/events"):
        token = request.query_params.get("access_token")  # EventSource can't set headers (SSE endpoints only)
    if not token:
        raise _unauthorized()
    user_id = decode_access_token(token)
    user = db.get(m.User, user_id) if user_id else None
    if not user or user.deleted_at:
        raise _unauthorized("Invalid or expired access token")
    return Principal(user=user)


def get_user(principal: Annotated[Principal, Depends(get_principal)]) -> m.User:
    if not principal.user:
        raise forbidden("This endpoint needs a signed-in user, not an API key")
    return principal.user


PrincipalDep = Annotated[Principal, Depends(get_principal)]
UserDep = Annotated[m.User, Depends(get_user)]


def role_in(db: Session, user_id: uuid.UUID, org_id: uuid.UUID) -> str | None:
    return db.scalar(
        select(m.Membership.role)
        .join(m.Organization, m.Organization.id == m.Membership.organization_id)
        .where(
            m.Membership.user_id == user_id,
            m.Membership.organization_id == org_id,
            m.Organization.deleted_at.is_(None),
        )
    )


def authorize(db: Session, principal: Principal, org_id: uuid.UUID, action: Action, what: str = "Resource") -> str | None:
    """Raises 404 when the principal can't see the org at all, 403 when it can but may not do `action`.
    Returns the user's role (None for API keys)."""
    if principal.api_key:
        if principal.api_key.organization_id != org_id:
            raise not_found(what)
        allowed = set().union(*(SCOPE_ACTIONS.get(s, set()) for s in principal.api_key.scopes))
        if action not in allowed:
            raise forbidden(f"API key lacks the scope for '{action}' here")
        return None
    role = role_in(db, principal.user.id, org_id)  # type: ignore[union-attr]
    if role is None:
        raise not_found(what)
    if action not in ROLE_ACTIONS[role]:
        raise forbidden(f"Role '{role}' can't do this")
    return role


def TenantRef(model, description: str = ""):
    """Marks a request-body field as an id of another tenant row. check_refs() resolves every marked field
    against the caller's org; the OpenAPI shows it as x-tenant-ref (tests/test_body_refs.py enforces the marker)."""
    return Field(default=None, description=description, json_schema_extra={"x-tenant-ref": model.__tablename__})


def _ref_table(field) -> str | None:
    extra = field.json_schema_extra
    return extra.get("x-tenant-ref") if isinstance(extra, dict) else None


def _model_for(table: str):
    from app.db import Base

    return next(mp.class_ for mp in Base.registry.mappers if mp.local_table.name == table)


def check_refs(db: Session, body: BaseModel, org_id: uuid.UUID, scope: dict | None = None) -> None:
    """404 when any TenantRef id in `body` (recursively, lists included) isn't a row of `org_id`.
    `scope` adds per-table filters, e.g. {"samples": [m.Sample.dataset_version_id == vid]}."""
    for name, field in type(body).model_fields.items():
        value = getattr(body, name)
        table = _ref_table(field)
        if table is None:
            for item in value if isinstance(value, list) else [value]:
                if isinstance(item, BaseModel):
                    check_refs(db, item, org_id, scope)
            continue
        ids = {v for v in (value if isinstance(value, list) else [value]) if v is not None}
        if not ids:
            continue
        model = _model_for(table)
        found = set(db.scalars(select(model.id).where(model.id.in_(ids), model.organization_id == org_id,
                                                      *((scope or {}).get(table, [])))))
        if found != ids:
            raise not_found(table.rstrip("s").replace("_", " ").capitalize())


def get_by_id(db: Session, model, row_id: uuid.UUID):
    """session.get() for tenant tables (their ORM identity is (id, organization_id))."""
    return db.scalar(select(model).where(model.id == row_id))


def _not_deleting(project: m.Project | None, action: Action) -> None:
    if project is not None and project.status == "Deleting" and action != "read":
        raise ApiError(409, "project_deleting", "This project is being permanently deleted")


def load_project(db: Session, principal: Principal, project_id: uuid.UUID, action: Action) -> m.Project:
    project = get_by_id(db, m.Project, project_id)
    if not project:
        raise not_found("Project")
    authorize(db, principal, project.organization_id, action, "Project")
    _not_deleting(project, action)
    return project


def load_dataset(db: Session, principal: Principal, dataset_id: uuid.UUID, action: Action) -> m.Dataset:
    ds = get_by_id(db, m.Dataset, dataset_id)
    if not ds:
        raise not_found("Dataset")
    authorize(db, principal, ds.organization_id, action, "Dataset")
    _not_deleting(get_by_id(db, m.Project, ds.project_id), action)
    return ds


def load_version(db: Session, principal: Principal, version_id: uuid.UUID, action: Action) -> m.DatasetVersion:
    v = get_by_id(db, m.DatasetVersion, version_id)
    if not v:
        raise not_found("Dataset version")
    authorize(db, principal, v.organization_id, action, "Dataset version")
    if action != "read":
        ds = get_by_id(db, m.Dataset, v.dataset_id)
        _not_deleting(get_by_id(db, m.Project, ds.project_id) if ds else None, action)
    return v


def require_current(db: Session, v: m.DatasetVersion, action: str) -> m.Dataset:
    """Seal / fixes / labels act on the dataset's CURRENT version only; anything else would drop newer data."""
    ds = get_by_id(db, m.Dataset, v.dataset_id)
    if ds.current_version_id != v.id:
        cur = get_by_id(db, m.DatasetVersion, ds.current_version_id) if ds.current_version_id else None
        raise ApiError(409, "stale_version", f"Can't {action} an older version; use the current one",
                       extra={"current_version_id": str(ds.current_version_id) if ds.current_version_id else None,
                              "current_version_no": cur.version_no if cur else None})
    return ds


def audit(db: Session, request: Request | None, principal: Principal | None, org_id, action: str, target_type=None,
          target_id=None, **details) -> None:
    db.add(m.AuditLog(
        organization_id=org_id,
        actor_user_id=principal.user_id if principal else None,
        api_key_id=principal.api_key.id if principal and principal.api_key else None,
        action=action,
        target_type=target_type,
        target_id=str(target_id) if target_id else None,
        ip=request.client.host if request and request.client else None,
        user_agent=request.headers.get("user-agent") if request else None,
        details=details,
    ))
