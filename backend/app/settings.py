from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEV_JWT_SECRET = "dev-only-change-me-dev-only-change-me"
DEV_S3_KEYS = {("automl", "automl-dev-secret")}


class Settings(BaseSettings):
    """All config comes from env (or backend/.env). Defaults match docker-compose.dev.yml."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "dev"  # dev | test | prod
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:5180"]
    public_app_url: str = "http://localhost:5180"
    log_level: str = "INFO"

    database_url: str = "postgresql+psycopg://automl:automl@localhost:5442/automl"
    redis_url: str = "redis://localhost:6381/0"

    s3_endpoint_url: str | None = "http://localhost:9010"
    s3_region: str = "us-east-1"
    s3_access_key: str = "automl"
    s3_secret_key: str = "automl-dev-secret"
    s3_bucket: str = "automl-data"
    presign_ttl_seconds: int = 900  # 15 min (EXP-6, SEC3)

    jwt_secret: str = Field(default=DEV_JWT_SECRET, min_length=32)
    access_token_ttl_seconds: int = 900  # ACC-2: 15 min
    refresh_token_ttl_seconds: int = 30 * 24 * 3600
    cookie_secure: bool = False  # True in prod (TLS)
    login_rate_limit: int = 5  # failures per email+IP per window
    login_rate_window_seconds: int = 900

    google_client_id: str | None = None
    google_client_secret: str | None = None
    # Must be on the APP origin (dev: the Vite proxy) so the refresh cookie set on the callback is first-party:
    #   dev  http://127.0.0.1:5180/api/v1/auth/google/callback    prod https://<app host>/api/v1/auth/google/callback
    google_redirect_uri: str | None = None
    google_success_path: str = "/projects"  # SPA route after Google sign-in; it calls POST /auth/refresh

    llm_provider: str | None = None  # "anthropic" | "stub"; defaults to "stub" outside prod, required in prod
    llm_stub_mode: str = "valid"  # valid | invalid | down
    anthropic_api_key: str | None = None
    llm_model: str = "claude-opus-5-5"

    mail_sink_dir: Path = Path("var/mail")

    # Upload limits (DAT-3, SEC6)
    max_file_bytes: int = 20 * 1024**3
    max_files_per_batch: int = 50_000
    part_size_bytes: int = 64 * 1024**2
    max_zip_uncompressed_bytes: int = 2 * 1024**3
    max_zip_ratio: int = 100
    max_zip_entries: int = 60_000
    max_image_pixels: int = 89_478_485  # Pillow's default bomb threshold

    celery_eager: bool = False  # tests run tasks inline

    # Training (lean defaults for a 16 GB laptop; all overridable by env)
    mlflow_tracking_uri: str | None = "http://127.0.0.1:5005"
    train_batch_size: int = 16
    train_num_workers: int = 0
    train_image_size: int | None = None  # explicit TRAIN_IMAGE_SIZE wins over the budget preset (default 160)
    train_max_memory_gb: float = 4.0  # job fails with not_enough_memory if its RSS exceeds this (text peaks ~2.7 GB over baseline)
    train_min_free_gb: float = 0.4  # ...or if free RAM drops below this
    # Memory-safe mode (dev default on a 16 GB laptop): caps only what drives peak memory (small backbone, 112 px,
    # batch <= 16, 0 loader workers, one model at a time) but trains on ALL samples, 2 trials, <= 10 epochs + early stop.
    train_low_memory: bool = True
    train_fast_test: bool = False  # unit tests only: 1 trial, 1 epoch, <= 64 samples, 96 px
    train_pretrained: bool = True  # tests may set false to avoid downloading weights (never in product use)
    train_device: str | None = None  # force cpu|mps|cuda; default auto cuda -> mps -> cpu

    # Face recognition (PRD §12): pretrained InsightFace ONNX pack, never trained. NON-COMMERCIAL weights (local build).
    face_model_dir: str = str(Path(__file__).resolve().parents[2] / ".model-cache" / "insightface" / "buffalo_l")
    face_default_threshold: float = 0.40  # cosine; conservative (low FAR) for ArcFace w600k
    face_min_px: int = 80  # shorter side of the detected box
    speaker_default_threshold: float = 0.86  # WavLM-SV x-vector cosine (the model card's operating point)
    face_min_blur: float = 20.0  # variance of the Laplacian on the aligned 112 px crop
    # In-process models are freed after this many seconds without a request (RAM-tight laptop); reload on next use
    face_idle_unload_s: float = 300
    predict_idle_unload_s: float = 300
    # comma list of "hf_id@revision" (pinned; works offline without refs/main); the first one in the local cache wins
    # chosen on a 239-text eval (macro-F1 0.52, Devanagari Hindi 0.90, 9 ms/text); 5 classes collapsed to 3
    sentiment_models: str = "tabularisai/multilingual-sentiment-analysis@eea032081f8d247b4303ef3565e7cec1b6f201c9"
    # benchmarked 2026-10-25 (15 COCO photos, no people, 84 boxes): OWLv2 mAP@0.5 0.761 / recall@0.3 0.583, 1.1 s per
    # photo on mps; OWL-ViT-B/32 0.605 / 0.333, 0.1 s. Accuracy wins for a background bulk job.
    autolabel_text_model: str = "google/owlv2-base-patch16-ensemble"  # see services/detector.OWL_MODELS
    autolabel_device: str = "mps"  # cpu when mps isn't available (2.2 s per photo there)
    hybrid_min_replay: int = 50  # COCO replay images with the chosen classes needed before a hybrid fine-tune

    @model_validator(mode="after")
    def _env_rules(self) -> "Settings":
        if not self.is_prod:
            if self.llm_provider is None:
                self.llm_provider = "stub"
            return self
        problems = []
        if self.jwt_secret == DEV_JWT_SECRET:
            problems.append("JWT_SECRET is the dev default")
        if not self.s3_access_key or not self.s3_secret_key or (self.s3_access_key, self.s3_secret_key) in DEV_S3_KEYS:
            problems.append("S3_ACCESS_KEY/S3_SECRET_KEY are empty or the dev defaults")
        if self.llm_provider not in ("anthropic",):
            problems.append("LLM_PROVIDER must be set explicitly (stub is not allowed in prod)")
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            problems.append("ANTHROPIC_API_KEY is empty")
        if self.google_client_id and not (self.google_client_secret and self.google_redirect_uri):
            problems.append("Google OIDC is half-configured")
        if not self.cookie_secure:
            problems.append("COOKIE_SECURE must be true in prod")
        if problems:
            raise ValueError("Refusing to start in prod: " + "; ".join(problems))
        return self

    @property
    def is_prod(self) -> bool:
        return self.app_env == "prod"

    @property
    def google_enabled(self) -> bool:
        return bool(self.google_client_id and self.google_client_secret and self.google_redirect_uri)


@lru_cache
def get_settings() -> Settings:
    return Settings()
