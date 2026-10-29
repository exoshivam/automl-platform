import hashlib
import hmac
import secrets
import uuid
from datetime import UTC, datetime, timedelta

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from app.settings import get_settings

_ph = PasswordHasher()
# Verified against when the email is unknown, so both paths cost one argon2 verify (no timing oracle).
_DUMMY_HASH = _ph.hash("not-a-real-password")

API_KEY_SCOPES = ("inference", "projects:read", "projects:write", "training")


def hash_password(password: str) -> str:
    return _ph.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    try:
        return _ph.verify(password_hash or _DUMMY_HASH, password) and password_hash is not None
    except (VerificationError, InvalidHashError):
        return False


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def create_access_token(user_id: uuid.UUID) -> tuple[str, int]:
    s = get_settings()
    iat = datetime.now(UTC)
    payload = {"sub": str(user_id), "iat": iat, "exp": iat + timedelta(seconds=s.access_token_ttl_seconds), "typ": "access"}
    return jwt.encode(payload, s.jwt_secret, algorithm="HS256"), s.access_token_ttl_seconds


def decode_access_token(token: str) -> uuid.UUID | None:
    try:
        payload = jwt.decode(token, get_settings().jwt_secret, algorithms=["HS256"], options={"require": ["exp", "sub"]})
    except jwt.PyJWTError:
        return None
    if payload.get("typ") != "access":
        return None
    try:
        return uuid.UUID(payload["sub"])
    except ValueError:
        return None


def new_refresh_token() -> str:
    return secrets.token_urlsafe(48)


def new_email_token() -> str:
    return secrets.token_urlsafe(32)


def new_api_key() -> tuple[str, str]:
    """Returns (full_key, prefix). Format: ak_<8-char prefix>_<secret>. Only sha256(full_key) is stored."""
    prefix = secrets.token_hex(4)
    return f"ak_{prefix}_{secrets.token_urlsafe(32)}", prefix


def parse_api_key(key: str) -> str | None:
    parts = key.split("_", 2)
    if len(parts) != 3 or parts[0] != "ak" or len(parts[1]) != 8:
        return None
    return parts[1]


def api_key_matches(key: str, key_hash: str) -> bool:
    return hmac.compare_digest(sha256_hex(key), key_hash)
