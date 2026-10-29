"""Tests run against the dev compose services, in a separate database (automl_test) and Redis db 15."""

import os
import re
import tempfile

# Every run gets its OWN database (created from the migrations, dropped at the end) and Redis db index, so parallel
# runs from different sessions never truncate each other. S3 keys are already unique per run (fresh org ids).
# Override with AUTOML_TEST_DB / AUTOML_TEST_REDIS_DB (an explicit DB is kept after the run, for debugging).
_RUN = os.environ.get("AUTOML_TEST_DB")
TEST_DB = _RUN or f"automl_test_{os.getpid()}"
DROP_DB_AT_END = _RUN is None
TEST_REDIS_DB = os.environ.get("AUTOML_TEST_REDIS_DB", str(1 + os.getpid() % 15))
os.environ.update({
    "APP_ENV": "test",
    "DATABASE_URL": f"postgresql+psycopg://automl:automl@127.0.0.1:5442/{TEST_DB}",
    "REDIS_URL": f"redis://127.0.0.1:6381/{TEST_REDIS_DB}",
    "MAIL_SINK_DIR": tempfile.mkdtemp(prefix="automl-mail-"),
    "S3_BUCKET": "automl-test",
    "CELERY_EAGER": "true",
    "LLM_PROVIDER": "stub",
    "LOG_LEVEL": "WARNING",
    "MLFLOW_TRACKING_URI": "",
    "TRAIN_FAST_TEST": "true",
    "TRAIN_PRETRAINED": "false",
    "TRAIN_DEVICE": "cpu",
})

# macOS: torch must be the FIRST OpenMP user in a process; loading it after other OpenMP-linked wheels (sklearn,
# lightgbm) segfaults. Tests load models in-process, so import it up front.
try:
    import torch  # noqa: F401
except ImportError:
    pass

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.db import Base, engine
from app.services import ratelimit
from app.settings import get_settings


@pytest.fixture(scope="session", autouse=True)
def _database():
    with psycopg.connect("postgresql://automl:automl@127.0.0.1:5442/automl", autocommit=True) as c:
        if not c.execute("select 1 from pg_database where datname = %s", (TEST_DB,)).fetchone():
            c.execute(f"create database {TEST_DB}")
    cfg = Config(os.path.join(os.path.dirname(__file__), "..", "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(os.path.dirname(__file__), "..", "migrations"))
    command.upgrade(cfg, "head")
    from app.services import storage

    storage.ensure_bucket()
    yield
    if DROP_DB_AT_END:
        engine().dispose()
        with psycopg.connect("postgresql://automl:automl@127.0.0.1:5442/automl", autocommit=True) as c:
            c.execute(f"drop database if exists {TEST_DB} with (force)")


@pytest.fixture(autouse=True)
def _clean():
    tables = ", ".join(t.name for t in Base.metadata.sorted_tables)
    with engine().begin() as c:
        c.execute(text(f"TRUNCATE {tables} CASCADE"))
    ratelimit.client().flushdb()
    yield


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def client():
    from app.main import app

    with TestClient(app, base_url="http://localhost") as c:
        yield c


def last_mail_token(client: TestClient, to: str) -> str:
    for msg in client.get("/api/v1/_dev/mail").json():
        if msg["to"].lower() == to.lower():
            return re.search(r"Token: (\S+)", msg["body"]).group(1)
    raise AssertionError(f"no mail to {to}")


class User:
    def __init__(self, client: TestClient, email: str, token: str, org_id: str):
        self.client, self.email, self.token, self.org_id = client, email, token, org_id

    @property
    def h(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}


def make_user(client: TestClient, email: str, password: str = "Str0ng!pass-123") -> User:
    r = client.post("/api/v1/auth/register", json={"email": email, "password": password, "name": email.split("@")[0]})
    assert r.status_code == 201, r.text
    client.post("/api/v1/auth/verify-email", json={"token": last_mail_token(client, email)}).raise_for_status()
    r = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    tok = r.json()["access_token"]
    me = client.get("/api/v1/me", headers={"Authorization": f"Bearer {tok}"}).json()
    return User(client, email, tok, me["memberships"][0]["organization_id"])


@pytest.fixture
def alice(client):
    return make_user(client, "alice@test.local")


@pytest.fixture
def bob(client):
    return make_user(client, "bob@test.local")
