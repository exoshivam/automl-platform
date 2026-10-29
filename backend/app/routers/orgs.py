"""ACC-3/4/6: workspaces, members, API keys, usage, audit log."""

import uuid
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Query, Request, Response
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy import func, select

from app import models as m
from app.db import now
from app.deps import DbDep, PrincipalDep, UserDep, audit, authorize
from app.errors import ApiError, not_found
from app.pagination import Page, paginate
from app.security import API_KEY_SCOPES, new_api_key, sha256_hex
from app.services.orgs import create_org

router = APIRouter(tags=["orgs"])
Role = Literal["owner", "admin", "member", "viewer"]


class OrgIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class OrgPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)


class OrgOut(BaseModel):
    id: uuid.UUID
    name: str
    slug: str
    plan: str
    region: str
    personal: bool
    role: Role | None = None
    created_at: datetime


class MemberIn(BaseModel):
    email: EmailStr
    role: Role = "member"


class MemberPatch(BaseModel):
    role: Role


class MemberOut(BaseModel):
    user_id: uuid.UUID
    email: str
    name: str
    role: Role
    joined_at: datetime


class ApiKeyIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    scopes: list[str] = Field(min_length=1)
    expires_at: datetime | None = None

    @field_validator("scopes")
    @classmethod
    def known_scopes(cls, v: list[str]) -> list[str]:
        bad = sorted(set(v) - set(API_KEY_SCOPES))
        if bad:
            raise ValueError(f"Unknown scopes {bad}; allowed: {list(API_KEY_SCOPES)}")
        return sorted(set(v))


class ApiKeyOut(BaseModel):
    id: uuid.UUID
    name: str
    prefix: str
    scopes: list[str]
    expires_at: datetime | None
    last_used_at: datetime | None
    revoked_at: datetime | None
    created_at: datetime
    deployment_id: uuid.UUID | None = None  # set for deployment keys (POST /v1/predict); null = workspace key
    project_id: uuid.UUID | None = None
    project_name: str | None = None


class ApiKeyCreated(ApiKeyOut):
    key: str = Field(description="The full key. Shown only in this response.")


def org_out(o: m.Organization, role: str | None = None) -> OrgOut:
    return OrgOut(id=o.id, name=o.name, slug=o.slug, plan=o.plan, region=o.region, personal=o.personal, role=role,
                  created_at=o.created_at)


def _org(db, org_id: uuid.UUID) -> m.Organization:
    org = db.get(m.Organization, org_id)
    if not org or org.deleted_at:
        raise not_found("Workspace")
    return org


def _owner_count(db, org_id) -> int:
    return db.scalar(select(func.count()).select_from(m.Membership).where(
        m.Membership.organization_id == org_id, m.Membership.role == "owner")) or 0


# ---- workspaces

@router.get("/orgs", response_model=list[OrgOut])
def list_orgs(user: UserDep, db: DbDep):
    rows = db.execute(select(m.Organization, m.Membership.role)
                      .join(m.Membership, m.Membership.organization_id == m.Organization.id)
                      .where(m.Membership.user_id == user.id, m.Organization.deleted_at.is_(None))
                      .order_by(m.Organization.created_at)).all()
    return [org_out(o, r) for o, r in rows]


@router.post("/orgs", status_code=201, response_model=OrgOut)
def create(body: OrgIn, user: UserDep, db: DbDep, request: Request, principal: PrincipalDep):
    org = create_org(db, user, body.name)
    audit(db, request, principal, org.id, "org.create", "organization", org.id)
    return org_out(org, "owner")


@router.get("/orgs/{org_id}", response_model=OrgOut)
def get_org(org_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    org = _org(db, org_id)
    role = authorize(db, principal, org_id, "read", "Workspace")
    return org_out(org, role)


@router.patch("/orgs/{org_id}", response_model=OrgOut)
def patch_org(org_id: uuid.UUID, body: OrgPatch, principal: PrincipalDep, db: DbDep, request: Request):
    org = _org(db, org_id)
    role = authorize(db, principal, org_id, "manage", "Workspace")
    if body.name:
        org.name = body.name
    audit(db, request, principal, org_id, "org.update", "organization", org_id)
    return org_out(org, role)


@router.delete("/orgs/{org_id}", status_code=204)
def delete_org(org_id: uuid.UUID, principal: PrincipalDep, db: DbDep, request: Request):
    org = _org(db, org_id)
    authorize(db, principal, org_id, "own", "Workspace")
    if org.personal:
        raise ApiError(409, "personal_workspace", "The personal workspace can't be deleted; delete the account instead")
    org.deleted_at = now()  # ACC-8: 14-day soft delete, hard delete by a scheduled job
    audit(db, request, principal, org_id, "org.delete", "organization", org_id)
    return Response(status_code=204)


# ---- members

@router.get("/orgs/{org_id}/members", response_model=list[MemberOut])
def list_members(org_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    _org(db, org_id)
    authorize(db, principal, org_id, "read", "Workspace")
    rows = db.execute(select(m.Membership, m.User).join(m.User, m.User.id == m.Membership.user_id)
                      .where(m.Membership.organization_id == org_id).order_by(m.Membership.created_at)).all()
    return [MemberOut(user_id=u.id, email=u.email, name=u.name, role=ms.role, joined_at=ms.created_at) for ms, u in rows]


@router.post("/orgs/{org_id}/members", status_code=201, response_model=MemberOut)
def add_member(org_id: uuid.UUID, body: MemberIn, principal: PrincipalDep, db: DbDep, request: Request):
    """Adds an existing, verified user directly (the ACC-5 invite flow is P1)."""
    _org(db, org_id)
    authorize(db, principal, org_id, "own" if body.role == "owner" else "manage", "Workspace")
    user = db.scalar(select(m.User).where(m.User.email == body.email, m.User.deleted_at.is_(None),
                                          m.User.email_verified_at.is_not(None)))
    if not user:
        raise ApiError(404, "user_not_found", "No verified account with this email")
    if db.scalar(select(m.Membership).where(m.Membership.organization_id == org_id, m.Membership.user_id == user.id)):
        raise ApiError(409, "already_member", "This user is already a member")
    ms = m.Membership(user_id=user.id, organization_id=org_id, role=body.role)
    db.add(ms)
    db.flush()
    audit(db, request, principal, org_id, "member.add", "user", user.id, role=body.role)
    return MemberOut(user_id=user.id, email=user.email, name=user.name, role=ms.role, joined_at=ms.created_at)


def _membership(db, org_id, user_id) -> m.Membership:
    ms = db.scalar(select(m.Membership).where(m.Membership.organization_id == org_id, m.Membership.user_id == user_id))
    if not ms:
        raise not_found("Member")
    return ms


@router.patch("/orgs/{org_id}/members/{user_id}", response_model=MemberOut)
def change_role(org_id: uuid.UUID, user_id: uuid.UUID, body: MemberPatch, principal: PrincipalDep, db: DbDep,
                request: Request):
    _org(db, org_id)
    authorize(db, principal, org_id, "manage", "Workspace")
    ms = _membership(db, org_id, user_id)
    if "owner" in (ms.role, body.role):
        authorize(db, principal, org_id, "own", "Workspace")  # only owners touch the owner role
        if ms.role == "owner" and body.role != "owner" and _owner_count(db, org_id) <= 1:
            raise ApiError(409, "last_owner", "A workspace needs at least one owner")
    ms.role = body.role
    user = db.get(m.User, user_id)
    audit(db, request, principal, org_id, "member.role_change", "user", user_id, role=body.role)
    return MemberOut(user_id=user.id, email=user.email, name=user.name, role=ms.role, joined_at=ms.created_at)


@router.delete("/orgs/{org_id}/members/{user_id}", status_code=204)
def remove_member(org_id: uuid.UUID, user_id: uuid.UUID, principal: PrincipalDep, db: DbDep, request: Request):
    _org(db, org_id)
    self_leave = principal.user_id == user_id
    authorize(db, principal, org_id, "read" if self_leave else "manage", "Workspace")
    ms = _membership(db, org_id, user_id)
    if ms.role == "owner":
        if not self_leave:
            authorize(db, principal, org_id, "own", "Workspace")
        if _owner_count(db, org_id) <= 1:
            raise ApiError(409, "last_owner", "A workspace needs at least one owner")
    db.delete(ms)
    audit(db, request, principal, org_id, "member.remove", "user", user_id)
    return Response(status_code=204)


# ---- API keys (ACC-6)

def key_out(k: m.ApiKey) -> ApiKeyOut:
    return ApiKeyOut(id=k.id, name=k.name, prefix=k.prefix, scopes=k.scopes, expires_at=k.expires_at,
                     last_used_at=k.last_used_at, revoked_at=k.revoked_at, created_at=k.created_at)


@router.get("/orgs/{org_id}/api-keys", response_model=list[ApiKeyOut])
def list_keys(org_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    _org(db, org_id)
    authorize(db, principal, org_id, "manage", "Workspace")
    keys = db.scalars(select(m.ApiKey).where(m.ApiKey.organization_id == org_id).order_by(m.ApiKey.created_at)).all()
    deps = dict(db.execute(select(m.Deployment.id, m.Project).join(m.Project, m.Project.id == m.Deployment.project_id)
                           .where(m.Deployment.organization_id == org_id)).all())
    out = []
    for k in keys:
        p = deps.get(k.deployment_id)
        out.append(key_out(k).model_copy(update={"deployment_id": k.deployment_id, "project_id": p.id if p else None,
                                                 "project_name": p.name if p else None}))
    return out


@router.post("/orgs/{org_id}/api-keys", status_code=201, response_model=ApiKeyCreated)
def create_key(org_id: uuid.UUID, body: ApiKeyIn, principal: PrincipalDep, db: DbDep, request: Request):
    _org(db, org_id)
    authorize(db, principal, org_id, "manage", "Workspace")
    full, prefix = new_api_key()
    key = m.ApiKey(organization_id=org_id, name=body.name, prefix=prefix, key_hash=sha256_hex(full),
                   scopes=body.scopes, expires_at=body.expires_at, created_by=principal.user_id)
    db.add(key)
    db.flush()
    audit(db, request, principal, org_id, "api_key.create", "api_key", key.id, scopes=body.scopes)
    return ApiKeyCreated(**key_out(key).model_dump(), key=full)


@router.delete("/orgs/{org_id}/api-keys/{key_id}", status_code=204)
def revoke_key(org_id: uuid.UUID, key_id: uuid.UUID, principal: PrincipalDep, db: DbDep, request: Request):
    _org(db, org_id)
    authorize(db, principal, org_id, "manage", "Workspace")
    key = db.scalar(select(m.ApiKey).where(m.ApiKey.id == key_id, m.ApiKey.organization_id == org_id))
    if not key:
        raise not_found("API key")
    key.revoked_at = key.revoked_at or now()
    audit(db, request, principal, org_id, "api_key.revoke", "api_key", key_id)
    return Response(status_code=204)


# ---- usage + audit

@router.get("/orgs/{org_id}/usage")
def usage(org_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    _org(db, org_id)
    authorize(db, principal, org_id, "read", "Workspace")
    q = db.scalar(select(m.Quota).where(m.Quota.organization_id == org_id).order_by(m.Quota.period_start.desc()))
    totals = dict(db.execute(select(m.UsageEvent.kind, func.sum(m.UsageEvent.quantity))
                             .where(m.UsageEvent.organization_id == org_id).group_by(m.UsageEvent.kind)).all())
    return {
        "quota": None if not q else {
            "period_start": q.period_start, "gpu_seconds_limit": q.gpu_seconds_limit,
            "gpu_seconds_used": q.gpu_seconds_used, "gpu_seconds_reserved": q.gpu_seconds_reserved,
            "storage_bytes_limit": q.storage_bytes_limit, "inference_calls_limit": q.inference_calls_limit,
        },
        "totals": totals,
    }


class AuditOut(BaseModel):
    id: uuid.UUID
    action: str
    actor_user_id: uuid.UUID | None
    api_key_id: uuid.UUID | None
    target_type: str | None
    target_id: str | None
    ip: str | None
    details: dict
    created_at: datetime


@router.get("/orgs/{org_id}/audit-logs", response_model=Page[AuditOut])
def audit_logs(org_id: uuid.UUID, principal: PrincipalDep, db: DbDep, cursor: str | None = None,
               limit: int = Query(50, ge=1, le=200)):
    _org(db, org_id)
    authorize(db, principal, org_id, "manage", "Workspace")
    rows, nxt = paginate(db, select(m.AuditLog).where(m.AuditLog.organization_id == org_id), m.AuditLog.id,
                         cursor, limit)
    return Page[AuditOut](items=[AuditOut.model_validate(r, from_attributes=True) for r in rows], next_cursor=nxt)
