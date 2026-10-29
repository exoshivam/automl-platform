"""ACC-1/2: email+password (argon2), JWT access tokens, rotating refresh cookies with reuse detection, Google OIDC."""

import base64
import hashlib
import secrets
import uuid
from datetime import timedelta
from typing import Annotated
from urllib.parse import urlencode

import httpx
import jwt
from fastapi import APIRouter, Cookie, Header, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app import models as m
from app.db import now, uuid7
from app.deps import DbDep, UserDep, audit
from app.errors import ApiError
from app.security import (
    create_access_token,
    hash_password,
    new_email_token,
    new_refresh_token,
    sha256_hex,
    verify_password,
)
from app.services import mail, ratelimit
from app.services.orgs import create_org
from app.settings import get_settings

router = APIRouter(tags=["auth"])
COOKIE = "refresh_token"
GOOGLE_AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN = "https://oauth2.googleapis.com/token"
GOOGLE_JWKS = "https://www.googleapis.com/oauth2/v3/certs"


class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=10, max_length=256, description="At least 10 characters")
    name: str = Field(default="", max_length=200)


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(max_length=256)


class VerifyIn(BaseModel):
    token: str


class ResendIn(BaseModel):
    email: EmailStr


class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    name: str
    email_verified: bool


class RegisterOut(BaseModel):
    user: UserOut
    verification_required: bool = True


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserOut


class AuthConfigOut(BaseModel):
    google_enabled: bool


class MembershipOut(BaseModel):
    organization_id: uuid.UUID
    organization_name: str
    slug: str
    personal: bool
    plan: str
    role: str


class MeOut(BaseModel):
    user: UserOut
    memberships: list[MembershipOut]


def user_out(u: m.User) -> UserOut:
    return UserOut(id=u.id, email=u.email, name=u.name, email_verified=u.email_verified_at is not None)


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _send_verification(db: Session, user: m.User) -> None:
    token = new_email_token()
    db.add(m.EmailToken(user_id=user.id, purpose="verify", token_hash=sha256_hex(token),
                        expires_at=now() + timedelta(hours=48)))
    link = f"{get_settings().public_app_url}/verify-email?token={token}"
    mail.send_mail(user.email, "Verify your email", f"Welcome! Verify your email:\n\n{link}\n\nToken: {token}\n")


def _require_csrf(x_csrf: str | None) -> None:
    if x_csrf != "1":
        raise ApiError(403, "csrf_header_missing", "Send the X-CSRF: 1 header with cookie-authenticated requests")


def _issue_session(db: Session, response: Response, request: Request, user: m.User, family_id: uuid.UUID) -> TokenOut:
    s = get_settings()
    raw = new_refresh_token()
    db.add(m.RefreshToken(user_id=user.id, family_id=family_id, token_hash=sha256_hex(raw),
                          expires_at=now() + timedelta(seconds=s.refresh_token_ttl_seconds),
                          ip=_client_ip(request), user_agent=request.headers.get("user-agent")))
    response.set_cookie(COOKIE, raw, max_age=s.refresh_token_ttl_seconds, httponly=True, secure=s.cookie_secure,
                        samesite="lax", path=f"{s.api_prefix}/auth")
    access, ttl = create_access_token(user.id)
    return TokenOut(access_token=access, expires_in=ttl, user=user_out(user))


@router.get("/auth/config", response_model=AuthConfigOut)
def auth_config():
    return AuthConfigOut(google_enabled=get_settings().google_enabled)


@router.post("/auth/register", status_code=201, response_model=RegisterOut)
def register(body: RegisterIn, db: DbDep, request: Request):
    if db.scalar(select(m.User).where(m.User.email == body.email)):
        raise ApiError(409, "email_taken", "An account with this email already exists")
    user = m.User(email=body.email, password_hash=hash_password(body.password), name=body.name)
    db.add(user)
    db.flush()
    create_org(db, user, f"{body.name or body.email.split('@')[0]}'s workspace", personal=True)
    _send_verification(db, user)
    audit(db, request, None, None, "user.register", "user", user.id)
    return RegisterOut(user=user_out(user))


@router.post("/auth/verify-email", response_model=UserOut)
def verify_email(body: VerifyIn, db: DbDep):
    tok = db.scalar(select(m.EmailToken).where(m.EmailToken.token_hash == sha256_hex(body.token),
                                               m.EmailToken.purpose == "verify").with_for_update())
    if not tok or tok.used_at or tok.expires_at <= now():
        raise ApiError(400, "invalid_token", "This verification link is invalid or expired")
    tok.used_at = now()
    user = db.get(m.User, tok.user_id)
    user.email_verified_at = user.email_verified_at or now()
    return user_out(user)


@router.post("/auth/resend-verification", status_code=202)
def resend_verification(body: ResendIn, db: DbDep):
    user = db.scalar(select(m.User).where(m.User.email == body.email))
    if user and not user.email_verified_at:
        _send_verification(db, user)
    return {"status": "accepted"}  # same answer either way (no enumeration)


@router.post("/auth/login", response_model=TokenOut)
def login(body: LoginIn, db: DbDep, request: Request, response: Response):
    ip = _client_ip(request)
    if wait := ratelimit.login_blocked(body.email, ip):
        raise ApiError(429, "too_many_attempts", "Too many failed sign-in attempts. Try again later.",
                       headers={"Retry-After": str(wait)})
    user = db.scalar(select(m.User).where(m.User.email == body.email, m.User.deleted_at.is_(None)))
    if not verify_password(body.password, user.password_hash if user else None):
        ratelimit.record_login_failure(body.email, ip)
        raise ApiError(401, "invalid_credentials", "Email or password is incorrect")
    ratelimit.clear_login_failures(body.email, ip)
    if not user.email_verified_at:
        raise ApiError(403, "email_not_verified", "Verify your email before signing in")
    user.last_login_at = now()
    audit(db, request, None, None, "user.login", "user", user.id)
    return _issue_session(db, response, request, user, uuid7())


@router.post("/auth/refresh", response_model=TokenOut)
def refresh(db: DbDep, request: Request, response: Response,
            refresh_token: Annotated[str | None, Cookie()] = None,
            x_csrf: Annotated[str | None, Header()] = None):
    _require_csrf(x_csrf)
    if not refresh_token:
        raise ApiError(401, "no_session", "No refresh token")
    tok = db.scalar(select(m.RefreshToken).where(m.RefreshToken.token_hash == sha256_hex(refresh_token))
                    .with_for_update())
    if not tok or tok.revoked_at or tok.expires_at <= now():
        raise ApiError(401, "session_expired", "Session expired; sign in again")
    if tok.used_at:
        # Reuse of a rotated token: someone holds a stolen copy. Kill the whole family (ACC-2).
        db.execute(update(m.RefreshToken).where(m.RefreshToken.family_id == tok.family_id,
                                                m.RefreshToken.revoked_at.is_(None)).values(revoked_at=now()))
        audit(db, request, None, None, "session.reuse_detected", "user", tok.user_id)
        db.commit()
        raise ApiError(401, "refresh_reuse", "Session revoked; sign in again")
    tok.used_at = now()
    user = db.get(m.User, tok.user_id)
    if not user or user.deleted_at:
        raise ApiError(401, "session_expired", "Session expired; sign in again")
    return _issue_session(db, response, request, user, tok.family_id)


@router.post("/auth/logout", status_code=204)
def logout(db: DbDep, response: Response, refresh_token: Annotated[str | None, Cookie()] = None,
           x_csrf: Annotated[str | None, Header()] = None):
    _require_csrf(x_csrf)
    if refresh_token:
        tok = db.scalar(select(m.RefreshToken).where(m.RefreshToken.token_hash == sha256_hex(refresh_token)))
        if tok:
            db.execute(update(m.RefreshToken).where(m.RefreshToken.family_id == tok.family_id,
                                                    m.RefreshToken.revoked_at.is_(None)).values(revoked_at=now()))
    s = get_settings()
    response.delete_cookie(COOKIE, path=f"{s.api_prefix}/auth")
    response.status_code = 204
    return response


# ---- Google OIDC (only when GOOGLE_CLIENT_ID/SECRET/REDIRECT_URI are set)

def _google_or_404():
    if not get_settings().google_enabled:
        raise ApiError(404, "google_disabled", "Google sign-in is not configured")


@router.get("/auth/google/start")
def google_start():
    _google_or_404()
    s = get_settings()
    state, nonce, verifier = secrets.token_urlsafe(24), secrets.token_urlsafe(24), secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    ratelimit.client().setex(f"oidc:{state}", 600, f"{nonce}|{verifier}")
    params = {"client_id": s.google_client_id, "redirect_uri": s.google_redirect_uri, "response_type": "code",
              "scope": "openid email profile", "state": state, "nonce": nonce,
              "code_challenge": challenge, "code_challenge_method": "S256", "prompt": "select_account"}
    return RedirectResponse(f"{GOOGLE_AUTH}?{urlencode(params)}", status_code=302)


@router.get("/auth/google/callback")
def google_callback(db: DbDep, request: Request, code: str = "", state: str = ""):
    _google_or_404()
    s = get_settings()
    stored = ratelimit.client().getdel(f"oidc:{state}") if state else None
    if not stored or not code:
        raise ApiError(400, "invalid_oauth_state", "Sign-in link expired or was tampered with")
    nonce, verifier = stored.split("|", 1)
    try:
        tok = httpx.post(GOOGLE_TOKEN, data={"code": code, "client_id": s.google_client_id,
                                             "client_secret": s.google_client_secret,
                                             "redirect_uri": s.google_redirect_uri, "grant_type": "authorization_code",
                                             "code_verifier": verifier}, timeout=10).raise_for_status().json()
        key = jwt.PyJWKClient(GOOGLE_JWKS).get_signing_key_from_jwt(tok["id_token"])
        claims = jwt.decode(tok["id_token"], key.key, algorithms=["RS256"], audience=s.google_client_id,
                            issuer=["https://accounts.google.com", "accounts.google.com"])
    except (httpx.HTTPError, jwt.PyJWTError, KeyError) as e:
        raise ApiError(401, "oauth_failed", "Google sign-in failed") from e
    if claims.get("nonce") != nonce or not claims.get("email_verified"):
        raise ApiError(401, "oauth_failed", "Google sign-in failed")

    user = db.scalar(select(m.User).where(m.User.google_sub == claims["sub"])) or db.scalar(
        select(m.User).where(m.User.email == claims["email"]))
    if not user:
        user = m.User(email=claims["email"], name=claims.get("name", ""), google_sub=claims["sub"],
                      email_verified_at=now())
        db.add(user)
        db.flush()
        create_org(db, user, f"{user.name or user.email.split('@')[0]}'s workspace", personal=True)
    user.google_sub = user.google_sub or claims["sub"]
    user.email_verified_at = user.email_verified_at or now()
    user.last_login_at = now()
    audit(db, request, None, None, "user.login_google", "user", user.id)
    # The SPA lands on its app route and calls POST /auth/refresh (cookie) to get an access token.
    response = RedirectResponse(f"{s.public_app_url}{s.google_success_path}", status_code=302)
    _issue_session(db, response, request, user, uuid7())
    return response


@router.get("/me", response_model=MeOut)
def me(user: UserDep, db: DbDep):
    rows = db.execute(
        select(m.Membership.role, m.Organization)
        .join(m.Organization, m.Organization.id == m.Membership.organization_id)
        .where(m.Membership.user_id == user.id, m.Organization.deleted_at.is_(None))
        .order_by(m.Organization.created_at)
    ).all()
    return {
        "user": user_out(user),
        "memberships": [{"organization_id": o.id, "organization_name": o.name, "slug": o.slug,
                         "personal": o.personal, "plan": o.plan, "role": role} for role, o in rows],
    }
