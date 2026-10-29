"""PRD §10 tables for the first slice. Every tenant-owned row carries organization_id (indexed)."""

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, CITEXT, JSONB
from sqlalchemy.orm import Mapped, declared_attr, mapped_column

from app.db import Base, IdMixin

ROLES = ("owner", "admin", "member", "viewer")
TS = DateTime(timezone=True)


def org_fk() -> Mapped[uuid.UUID]:
    return mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"), index=True)


class TenantKey:
    """ORM-level identity (id, organization_id) for tenant tables (the DB primary key stays `id`).
    Every UPDATE/DELETE the ORM flushes then carries `AND organization_id = ?` (NFR-SEC3, defense in depth).
    Load rows with app.deps.get_by_id / select(), not session.get()."""

    @declared_attr.directive
    def __mapper_args__(cls):
        return {"primary_key": [cls.__table__.c.id, cls.__table__.c.organization_id]}


class User(IdMixin, Base):
    __tablename__ = "users"
    email: Mapped[str] = mapped_column(CITEXT, unique=True)
    password_hash: Mapped[str | None] = mapped_column(Text)
    google_sub: Mapped[str | None] = mapped_column(String(255), unique=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    email_verified_at: Mapped[datetime | None] = mapped_column(TS)
    totp_secret_enc: Mapped[str | None] = mapped_column(Text)
    last_login_at: Mapped[datetime | None] = mapped_column(TS)
    deleted_at: Mapped[datetime | None] = mapped_column(TS)


class Organization(IdMixin, Base):
    __tablename__ = "organizations"
    name: Mapped[str] = mapped_column(String(200))
    slug: Mapped[str] = mapped_column(String(120), unique=True)
    plan: Mapped[str] = mapped_column(String(20), default="free")
    region: Mapped[str] = mapped_column(String(40), default="ap-south-1")
    data_residency: Mapped[str] = mapped_column(String(20), default="india")
    personal: Mapped[bool] = mapped_column(Boolean, default=False)
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    deleted_at: Mapped[datetime | None] = mapped_column(TS)


class Membership(TenantKey, IdMixin, Base):
    __tablename__ = "memberships"
    __table_args__ = (UniqueConstraint("user_id", "organization_id"),)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    organization_id: Mapped[uuid.UUID] = org_fk()
    role: Mapped[str] = mapped_column(String(10))


class RefreshToken(IdMixin, Base):
    """One row per issued refresh token. Rotation links tokens into a family; reuse revokes the family (ACC-2)."""

    __tablename__ = "refresh_tokens"
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    family_id: Mapped[uuid.UUID] = mapped_column(index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    expires_at: Mapped[datetime] = mapped_column(TS)
    used_at: Mapped[datetime | None] = mapped_column(TS)
    revoked_at: Mapped[datetime | None] = mapped_column(TS)
    ip: Mapped[str | None] = mapped_column(String(64))
    user_agent: Mapped[str | None] = mapped_column(Text)


class EmailToken(IdMixin, Base):
    __tablename__ = "email_tokens"
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    purpose: Mapped[str] = mapped_column(String(20))  # verify
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    expires_at: Mapped[datetime] = mapped_column(TS)
    used_at: Mapped[datetime | None] = mapped_column(TS)


class ApiKey(TenantKey, IdMixin, Base):
    __tablename__ = "api_keys"
    organization_id: Mapped[uuid.UUID] = org_fk()
    name: Mapped[str] = mapped_column(String(100))
    prefix: Mapped[str] = mapped_column(String(16), index=True)
    key_hash: Mapped[str] = mapped_column(Text)
    scopes: Mapped[list[str]] = mapped_column(ARRAY(String(40)))
    expires_at: Mapped[datetime | None] = mapped_column(TS)
    last_used_at: Mapped[datetime | None] = mapped_column(TS)
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    revoked_at: Mapped[datetime | None] = mapped_column(TS)
    # set = a deployment key (scope "inference"): POST /v1/predict resolves the model from the key alone
    deployment_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("deployments.id", ondelete="SET NULL"),
                                                            index=True)


class Project(TenantKey, IdMixin, Base):
    __tablename__ = "projects"
    __table_args__ = (Index("ix_projects_current_spec", "current_spec", postgresql_using="gin"),)
    organization_id: Mapped[uuid.UUID] = org_fk()
    name: Mapped[str] = mapped_column(String(200))
    task_type: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(20), default="Draft")
    current_spec: Mapped[dict[str, Any]] = mapped_column(JSONB)
    spec_version: Mapped[int] = mapped_column(Integer, default=1)
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    archived_at: Mapped[datetime | None] = mapped_column(TS)


class TaskSpecVersion(TenantKey, IdMixin, Base):
    __tablename__ = "task_spec_versions"
    __table_args__ = (UniqueConstraint("project_id", "version"),)
    organization_id: Mapped[uuid.UUID] = org_fk()
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    spec: Mapped[dict[str, Any]] = mapped_column(JSONB)
    source: Mapped[str] = mapped_column(String(20))  # template | llm | user_edit
    llm_prompt_ref: Mapped[uuid.UUID | None] = mapped_column()
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))


class Dataset(TenantKey, IdMixin, Base):
    __tablename__ = "datasets"
    organization_id: Mapped[uuid.UUID] = org_fk()
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    modality: Mapped[str] = mapped_column(String(20))  # image | text | tabular | audio
    current_version_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="SET NULL", use_alter=True)
    )


class DatasetVersion(TenantKey, IdMixin, Base):
    """Draft until sealed_at; immutable afterwards (DAT-5)."""

    __tablename__ = "dataset_versions"
    __table_args__ = (UniqueConstraint("dataset_id", "version_no"),)
    organization_id: Mapped[uuid.UUID] = org_fk()
    dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    version_no: Mapped[int] = mapped_column(Integer)
    parent_version_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("dataset_versions.id"))
    origin: Mapped[str] = mapped_column(String(20), default="upload")  # upload | import | labels | fix
    manifest_uri: Mapped[str | None] = mapped_column(Text)
    table_uri: Mapped[str | None] = mapped_column(Text)  # tabular/text source table (Parquet) in S3
    column_mapping: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    label_stats: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    split_seed: Mapped[int | None] = mapped_column(Integer)
    split_strategy: Mapped[str | None] = mapped_column(String(20))  # stratified | grouped | random
    split_summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    validation_status: Mapped[str] = mapped_column(String(20), default="not_run")  # not_run|running|passed|failed
    validation_report: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    acknowledgements: Mapped[list[str]] = mapped_column(JSONB, default=list)
    options: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")  # e.g. group_near_duplicates
    content_hash: Mapped[str | None] = mapped_column(String(64))
    sealed_at: Mapped[datetime | None] = mapped_column(TS)
    ingest_status: Mapped[str] = mapped_column(String(20), default="ready")  # pending|running|ready|failed
    ingest_errors: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)


class Sample(TenantKey, IdMixin, Base):
    __tablename__ = "samples"
    __table_args__ = (Index("ix_samples_version_split", "dataset_version_id", "split"),)
    organization_id: Mapped[uuid.UUID] = org_fk()
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="CASCADE"), index=True
    )
    object_key: Mapped[str | None] = mapped_column(Text)
    original_name: Mapped[str | None] = mapped_column(Text)
    sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    phash: Mapped[str | None] = mapped_column(String(32))
    mime: Mapped[str | None] = mapped_column(String(100))
    bytes: Mapped[int | None] = mapped_column(BigInteger)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    duration: Mapped[float | None] = mapped_column(Float)
    split: Mapped[str | None] = mapped_column(String(5))  # train | val | test
    no_objects: Mapped[bool] = mapped_column(Boolean, default=False)  # detection: "image has no objects"
    thumbnail_key: Mapped[str | None] = mapped_column(Text)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)


class LabelSet(TenantKey, IdMixin, Base):
    __tablename__ = "label_sets"
    __table_args__ = (UniqueConstraint("project_id", "version"),)
    organization_id: Mapped[uuid.UUID] = org_fk()
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    classes: Mapped[list[str]] = mapped_column(JSONB)
    version: Mapped[int] = mapped_column(Integer)


class Label(TenantKey, IdMixin, Base):
    __tablename__ = "labels"
    organization_id: Mapped[uuid.UUID] = org_fk()
    sample_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("samples.id", ondelete="CASCADE"), index=True)
    label_set_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("label_sets.id", ondelete="SET NULL"))
    class_name: Mapped[str] = mapped_column(String(200))
    bbox: Mapped[dict[str, float] | None] = mapped_column(JSONB)  # pixels {x, y, w, h}
    source: Mapped[str] = mapped_column(String(20), default="user")  # user | import | ai_prelabel
    confidence: Mapped[float | None] = mapped_column(Float)
    labeled_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    reviewed_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))


class Upload(TenantKey, IdMixin, Base):
    """An upload session: one S3 multipart upload per file (DAT-3)."""

    __tablename__ = "uploads"
    organization_id: Mapped[uuid.UUID] = org_fk()
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(20), default="open")  # open | completing | completed | failed
    kind: Mapped[str] = mapped_column(String(20), default="files")  # files | archive | table
    files: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    options: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    dataset_version_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("dataset_versions.id"))
    error: Mapped[str | None] = mapped_column(Text)
    completed_at: Mapped[datetime | None] = mapped_column(TS)


class AuditLog(IdMixin, Base):
    __tablename__ = "audit_logs"
    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    api_key_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("api_keys.id", ondelete="SET NULL"))
    action: Mapped[str] = mapped_column(String(60))
    target_type: Mapped[str | None] = mapped_column(String(40))
    target_id: Mapped[str | None] = mapped_column(String(64))
    ip: Mapped[str | None] = mapped_column(String(64))
    user_agent: Mapped[str | None] = mapped_column(Text)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)


class Quota(TenantKey, IdMixin, Base):
    __tablename__ = "quotas"
    __table_args__ = (UniqueConstraint("organization_id", "period_start"),)
    organization_id: Mapped[uuid.UUID] = org_fk()
    period_start: Mapped[datetime] = mapped_column(TS)
    gpu_seconds_limit: Mapped[int] = mapped_column(BigInteger)
    gpu_seconds_used: Mapped[int] = mapped_column(BigInteger, default=0)
    gpu_seconds_reserved: Mapped[int] = mapped_column(BigInteger, default=0)
    storage_bytes_limit: Mapped[int] = mapped_column(BigInteger)
    inference_calls_limit: Mapped[int] = mapped_column(BigInteger)


class UsageEvent(TenantKey, IdMixin, Base):
    __tablename__ = "usage_events"
    organization_id: Mapped[uuid.UUID] = org_fk()
    project_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("projects.id", ondelete="SET NULL"))
    kind: Mapped[str] = mapped_column(String(30))
    quantity: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(20))
    ref_type: Mapped[str | None] = mapped_column(String(40))
    ref_id: Mapped[str | None] = mapped_column(String(64))
    occurred_at: Mapped[datetime] = mapped_column(TS)
    idempotency_key: Mapped[str] = mapped_column(String(120), unique=True)


class TrainingJob(TenantKey, IdMixin, Base):
    __tablename__ = "training_jobs"
    __table_args__ = (Index("ix_training_jobs_status_priority", "status", "priority", "created_at"),)
    organization_id: Mapped[uuid.UUID] = org_fk()
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("dataset_versions.id"))
    spec_version: Mapped[int] = mapped_column(Integer)
    budget_mode: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued|running|succeeded|failed|cancelled
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    reserved_gpu_seconds: Mapped[int] = mapped_column(Integer, default=0)
    used_gpu_seconds: Mapped[float] = mapped_column(Float, default=0)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    worker_id: Mapped[str | None] = mapped_column(String(200))
    heartbeat_at: Mapped[datetime | None] = mapped_column(TS)
    progress: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    error_code: Mapped[str | None] = mapped_column(String(60))
    error_message: Mapped[str | None] = mapped_column(Text)
    mlflow_experiment_id: Mapped[str | None] = mapped_column(String(100))
    repro_manifest: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    started_at: Mapped[datetime | None] = mapped_column(TS)
    finished_at: Mapped[datetime | None] = mapped_column(TS)


class Trial(TenantKey, IdMixin, Base):
    __tablename__ = "trials"
    __table_args__ = (UniqueConstraint("training_job_id", "trial_no"),)
    organization_id: Mapped[uuid.UUID] = org_fk()
    training_job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("training_jobs.id", ondelete="CASCADE"), index=True)
    trial_no: Mapped[int] = mapped_column(Integer)
    params: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    state: Mapped[str] = mapped_column(String(20), default="running")  # running|complete|pruned|failed
    val_metric: Mapped[float | None] = mapped_column(Float)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    gpu_seconds: Mapped[float] = mapped_column(Float, default=0)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(100))
    checkpoint_uri: Mapped[str | None] = mapped_column(Text)


class Model(TenantKey, IdMixin, Base):
    __tablename__ = "models"
    organization_id: Mapped[uuid.UUID] = org_fk()
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    task_type: Mapped[str] = mapped_column(String(40))
    production_version_id: Mapped[uuid.UUID | None] = mapped_column()


class ModelVersion(TenantKey, IdMixin, Base):
    """Immutable once ready. Artifacts follow app/ml/artifacts.py."""

    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("model_id", "version_no"),)
    organization_id: Mapped[uuid.UUID] = org_fk()
    model_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("models.id", ondelete="CASCADE"), index=True)
    version_no: Mapped[int] = mapped_column(Integer)
    training_job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("training_jobs.id", ondelete="SET NULL"))
    best_trial_id: Mapped[uuid.UUID | None] = mapped_column()
    status: Mapped[str] = mapped_column(String(20), default="training")  # training|ready|failed
    framework: Mapped[str | None] = mapped_column(String(30))
    architecture: Mapped[str | None] = mapped_column(String(100))
    base_model: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    artifact_prefix: Mapped[str | None] = mapped_column(Text)
    label_classes: Mapped[list[str]] = mapped_column(JSONB, default=list)
    input_spec: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    eval_report: Mapped[dict[str, Any] | None] = mapped_column(JSONB)  # evaluation
    dataset_version_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("dataset_versions.id"))


class Export(TenantKey, IdMixin, Base):
    """EXP-*: one row per (model version, format, quantization). Owned by export code."""

    __tablename__ = "exports"
    __table_args__ = (UniqueConstraint("model_version_id", "format", "quantization"),)
    organization_id: Mapped[uuid.UUID] = org_fk()
    model_version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("model_versions.id", ondelete="CASCADE"), index=True)
    format: Mapped[str] = mapped_column(String(20))  # onnx|tflite|coreml|tfjs|native
    quantization: Mapped[str] = mapped_column(String(20), default="none")
    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued|running|ready|failed
    artifact_key: Mapped[str | None] = mapped_column(Text)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    verify_report: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    accuracy_delta: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)


class LlmLog(TenantKey, IdMixin, Base):
    """PRJ-4: what was sent to the LLM, kept 30 days."""

    __tablename__ = "llm_logs"
    organization_id: Mapped[uuid.UUID] = org_fk()
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    provider: Mapped[str] = mapped_column(String(30))
    attempt: Mapped[int] = mapped_column(Integer)
    request: Mapped[dict[str, Any]] = mapped_column(JSONB)
    response: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    outcome: Mapped[str] = mapped_column(String(20))  # ok | invalid | error | flagged
    expires_at: Mapped[datetime] = mapped_column(TS, index=True)


# ---- face recognition (PRD §10, §12): detection -> alignment -> pretrained embedding -> gallery match. Never trained.

class FaceGallery(TenantKey, IdMixin, Base):
    __tablename__ = "face_galleries"  # also speaker galleries (kind): same consent / embeddings / report / purge
    organization_id: Mapped[uuid.UUID] = org_fk()
    kind: Mapped[str] = mapped_column(String(20), default="face", server_default="face")  # face | speaker
    phrase: Mapped[str | None] = mapped_column(Text)  # speaker: the sentence the UI asks people to say
    # speaker: until the user sets a threshold, use the report's (FAR-first) suggestion once the gallery is ready
    threshold_auto: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), unique=True)
    name: Mapped[str] = mapped_column(String(200))
    threshold: Mapped[float] = mapped_column(Float)
    embedding_model_id: Mapped[str] = mapped_column(String(100))
    purpose: Mapped[str | None] = mapped_column(Text)
    retention_until: Mapped[datetime | None] = mapped_column(TS)
    region: Mapped[str | None] = mapped_column(String(40))


class ConsentRecord(TenantKey, IdMixin, Base):
    """FACE-6 / §12.3: one per enrolled person. v1 local build: checkbox_attested only."""

    __tablename__ = "consent_records"
    organization_id: Mapped[uuid.UUID] = org_fk()
    gallery_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("face_galleries.id", ondelete="CASCADE"), index=True)
    subject_ref: Mapped[str | None] = mapped_column(String(200))
    consent_method: Mapped[str] = mapped_column(String(30))  # written | e-sign | checkbox_attested
    document_object_key: Mapped[str | None] = mapped_column(Text)
    purpose: Mapped[str | None] = mapped_column(Text)
    jurisdiction: Mapped[str | None] = mapped_column(String(60))
    granted_at: Mapped[datetime] = mapped_column(TS)
    expires_at: Mapped[datetime | None] = mapped_column(TS)
    withdrawn_at: Mapped[datetime | None] = mapped_column(TS)
    attested_by_user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))


class FaceIdentity(TenantKey, IdMixin, Base):
    __tablename__ = "face_identities"
    organization_id: Mapped[uuid.UUID] = org_fk()
    gallery_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("face_galleries.id", ondelete="CASCADE"), index=True)
    display_name: Mapped[str] = mapped_column(String(200))  # ponytail: plaintext; FACE-7 per-workspace encryption later
    external_ref: Mapped[str | None] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(20), default="active")
    consent_record_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("consent_records.id"))  # NO ACTION: checked at end of a cascade


class FaceEmbedding(TenantKey, IdMixin, Base):
    """One row per enrolled photo, plus one is_template row (the mean, re-normalised) per identity."""

    __tablename__ = "face_embeddings"
    __table_args__ = (Index("ix_face_embeddings_hnsw", "embedding", postgresql_using="hnsw",
                            postgresql_ops={"embedding": "vector_cosine_ops"}),)
    organization_id: Mapped[uuid.UUID] = org_fk()
    identity_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("face_identities.id", ondelete="CASCADE"), index=True)
    gallery_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("face_galleries.id", ondelete="CASCADE"), index=True)
    embedding: Mapped[Any] = mapped_column(Vector(512))
    quality_score: Mapped[float | None] = mapped_column(Float)
    face_px: Mapped[int | None] = mapped_column(Integer)
    duration_s: Mapped[float | None] = mapped_column(Float)  # speaker clips: seconds of speech after silence trim
    photo_object_key: Mapped[str | None] = mapped_column(Text)
    thumbnail_key: Mapped[str | None] = mapped_column(Text)
    is_template: Mapped[bool] = mapped_column(Boolean, default=False)


class Deployment(TenantKey, IdMixin, Base):
    """Self-serve local deploy: one per project; its keys call POST /v1/predict. model_version_id is pinned for
    trained-model projects; null for ready-made / quick text / face / speaker (they resolve like the playground)."""

    __tablename__ = "deployments"
    organization_id: Mapped[uuid.UUID] = org_fk()
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), unique=True)
    model_version_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("model_versions.id", ondelete="SET NULL"))
    status: Mapped[str] = mapped_column(String(20), default="active")  # active | paused
    rate_limit_per_min: Mapped[int] = mapped_column(Integer, default=60)
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))


class DeploymentRequest(TenantKey, IdMixin, Base):
    """One row per /v1/predict call (usage dashboard: counts, latency, errors)."""

    __tablename__ = "deployment_requests"
    __table_args__ = (Index("ix_deployment_requests_dep_created", "deployment_id", "created_at"),)
    organization_id: Mapped[uuid.UUID] = org_fk()
    deployment_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("deployments.id", ondelete="CASCADE"))
    api_key_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("api_keys.id", ondelete="SET NULL"))
    status_code: Mapped[int] = mapped_column(Integer)
    latency_ms: Mapped[float] = mapped_column(Float)
    model_version_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("model_versions.id", ondelete="SET NULL"))


class AutolabelJob(TenantKey, IdMixin, Base):
    """Bulk auto-label over a draft's photos (train queue). Produces BoxSuggestion rows, never labels."""

    __tablename__ = "autolabel_jobs"
    organization_id: Mapped[uuid.UUID] = org_fk()
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("dataset_versions.id", ondelete="CASCADE"),
                                                          index=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued | running | done | failed | cancelled
    source: Mapped[str] = mapped_column(String(20))  # auto | text_prompt | own_model | ready_made
    classes: Mapped[list[str]] = mapped_column(JSONB)
    prompts: Mapped[dict[str, str]] = mapped_column(JSONB, default=dict)
    min_confidence: Mapped[float] = mapped_column(Float)
    sample_ids: Mapped[list[str] | None] = mapped_column(JSONB)  # None = every unlabelled photo of the version
    progress: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)  # {done, total}
    results: Mapped[dict[str, int]] = mapped_column(JSONB, default=dict)  # {sample_id: boxes suggested}
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[str | None] = mapped_column(Text)
    finished_at: Mapped[datetime | None] = mapped_column(TS)


class BoxSuggestion(TenantKey, IdMixin, Base):
    """A model-proposed box awaiting review. Deliberately NOT a Label: training, validation, stats and seal read
    labels only, so nothing here can reach a model until a person accepts it (then a Label with source='auto')."""

    __tablename__ = "box_suggestions"
    organization_id: Mapped[uuid.UUID] = org_fk()
    job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("autolabel_jobs.id", ondelete="SET NULL"))
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("dataset_versions.id", ondelete="CASCADE"),
                                                          index=True)
    sample_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("samples.id", ondelete="CASCADE"), index=True)
    class_name: Mapped[str] = mapped_column(String(200))
    bbox: Mapped[dict[str, float]] = mapped_column(JSONB)
    confidence: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(20))  # text_prompt | own_model | ready_made
    model_id: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending | accepted | rejected
    label_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("labels.id", ondelete="SET NULL"))
    reviewed_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
