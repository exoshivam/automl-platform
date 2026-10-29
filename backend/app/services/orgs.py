import re
import secrets
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import models as m

# PRD §13 Free tier: 60 GPU minutes, 2 GB storage, 1k inference calls / month.
PLAN_LIMITS = {
    "free": {"gpu_seconds_limit": 60 * 60, "storage_bytes_limit": 2 * 1024**3, "inference_calls_limit": 1_000},
    "pro": {"gpu_seconds_limit": 600 * 60, "storage_bytes_limit": 50 * 1024**3, "inference_calls_limit": 50_000},
}


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50] or "workspace"


def unique_slug(db: Session, name: str) -> str:
    base = slugify(name)
    slug = base
    while db.scalar(select(m.Organization.id).where(m.Organization.slug == slug)):
        slug = f"{base}-{secrets.token_hex(3)}"
    return slug


def create_org(db: Session, owner: m.User, name: str, personal: bool = False) -> m.Organization:
    org = m.Organization(name=name, slug=unique_slug(db, name), personal=personal, created_by=owner.id)
    db.add(org)
    db.flush()
    db.add(m.Membership(user_id=owner.id, organization_id=org.id, role="owner"))
    period = datetime.now(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    db.add(m.Quota(organization_id=org.id, period_start=period, **PLAN_LIMITS["free"]))
    db.flush()
    return org
