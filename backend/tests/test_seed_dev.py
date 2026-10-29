import pytest

from app.scripts import seed_dev
from app.settings import get_settings


def test_seed_is_idempotent_and_login_works(client):
    assert seed_dev.seed() == "created"
    assert seed_dev.seed() == "updated"
    r = client.post("/api/v1/auth/login", json={"email": "demo@automl.dev", "password": "demo1234"})
    assert r.status_code == 200
    me = client.get("/api/v1/me", headers={"Authorization": f"Bearer {r.json()['access_token']}"}).json()
    assert len(me["memberships"]) == 1 and me["memberships"][0]["personal"]


def test_seed_refuses_outside_dev(monkeypatch):
    monkeypatch.setattr(get_settings(), "app_env", "staging")
    with pytest.raises(SystemExit, match="refusing"):
        seed_dev.seed()
