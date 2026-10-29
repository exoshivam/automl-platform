from fastapi.testclient import TestClient

from app import main as app_main
from app.settings import Settings
from tests.test_settings import PROD_OK


def _paths(app):
    with TestClient(app) as c:
        return set(c.get("/api/v1/openapi.json").json()["paths"]), c


def test_dev_hooks_not_mounted_outside_dev_test(monkeypatch):
    for env in ({**PROD_OK}, {"app_env": "staging"}):
        monkeypatch.setattr(app_main, "get_settings", lambda env=env: Settings(**env))
        app = app_main.create_app()
        with TestClient(app) as c:
            assert c.get("/api/v1/_dev/mail").status_code == 404
            assert c.get("/api/v1/_dev/llm-log").status_code == 404


def test_dev_hooks_loopback_only(client):
    assert client.get("/api/v1/_dev/mail").status_code == 200  # TestClient host in APP_ENV=test
    from app.main import app

    with TestClient(app, client=("10.1.2.3", 5000)) as remote:
        assert remote.get("/api/v1/_dev/mail").status_code == 404
        assert remote.get("/api/v1/_dev/llm-log").status_code == 404
