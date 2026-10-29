"""Dev-only: a verified demo user with a personal workspace. Idempotent.
Run: uv run python -m app.scripts.seed_dev   (refuses unless APP_ENV is dev or test)

The password is shorter than the API's 10-char minimum on purpose (it matches the old frontend mock); it is set
directly here and can never be registered through the API."""

import sys

from sqlalchemy import select

from app import models as m
from app.db import now, session_factory
from app.security import hash_password
from app.services.orgs import create_org
from app.settings import get_settings

EMAIL, PASSWORD, NAME = "demo@automl.dev", "demo1234", "Demo User"


def seed() -> str:
    s = get_settings()
    if s.app_env not in ("dev", "test"):
        raise SystemExit(f"refusing to seed: APP_ENV={s.app_env!r} (dev or test only)")
    with session_factory()() as db:
        user = db.scalar(select(m.User).where(m.User.email == EMAIL))
        if user is None:
            user = m.User(email=EMAIL, name=NAME, password_hash=hash_password(PASSWORD), email_verified_at=now())
            db.add(user)
            db.flush()
            create_org(db, user, f"{NAME}'s workspace", personal=True)
            result = "created"
        else:
            user.password_hash = hash_password(PASSWORD)
            user.email_verified_at = user.email_verified_at or now()
            user.deleted_at = None
            has_org = db.scalar(select(m.Membership.id).where(m.Membership.user_id == user.id))
            if not has_org:
                create_org(db, user, f"{NAME}'s workspace", personal=True)
            result = "updated"
        db.commit()
    return result


if __name__ == "__main__":
    print(f"demo user {EMAIL} {seed()}", file=sys.stderr)
