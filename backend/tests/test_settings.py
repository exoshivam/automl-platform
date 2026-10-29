import pytest
from pydantic import ValidationError

from app.settings import Settings

PROD_OK = {"app_env": "prod", "jwt_secret": "x" * 48, "s3_access_key": "AKIAREAL", "s3_secret_key": "real-secret",
           "llm_provider": "anthropic", "anthropic_api_key": "sk-test", "cookie_secure": True}


def test_dev_defaults_to_stub_llm():
    assert Settings(app_env="dev", llm_provider=None).llm_provider == "stub"


def test_prod_with_real_secrets_boots():
    assert Settings(**PROD_OK).is_prod


@pytest.mark.parametrize("override, msg", [
    ({"jwt_secret": "dev-only-change-me-dev-only-change-me"}, "JWT_SECRET"),
    ({"s3_access_key": "automl", "s3_secret_key": "automl-dev-secret"}, "S3_ACCESS_KEY"),
    ({"s3_secret_key": ""}, "S3_ACCESS_KEY"),
    ({"llm_provider": None}, "LLM_PROVIDER"),
    ({"llm_provider": "stub"}, "LLM_PROVIDER"),
    ({"anthropic_api_key": None}, "ANTHROPIC_API_KEY"),
    ({"cookie_secure": False}, "COOKIE_SECURE"),
])
def test_prod_refuses_dev_secrets(override, msg):
    with pytest.raises(ValidationError, match=msg):
        Settings(**{**PROD_OK, **override})
