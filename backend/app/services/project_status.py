"""PRJ-7 status, derived from state in ONE place. Call refresh_status() after any state change; reads use the column.

Draft          no data added yet
Data needed    data exists but isn't trainable: 0 samples, ingest pending/failed, not sealed, validation failed,
               or warnings not acknowledged
Validating     the current version is sealed and validation is running
Ready to train sealed + validation passed (warnings acknowledged)
Training       a training job is queued or running (takes precedence)
Trained        a model version is ready (and no job is running)
Face recognition projects: Draft (nobody enrolled) -> Data needed (people, no photos) -> Ready (gallery usable)
"""

from sqlalchemy import exists, select

from app import models as m


def derive_status(db, project: m.Project) -> str:
    if project.status == "Deleting":  # set by a permanent purge; nothing may move it back
        return "Deleting"
    org = project.organization_id
    if project.task_type in ("face_recognition", "speaker_recognition"):  # a gallery, never trained (PRD §12.2)
        g = select(m.FaceGallery.id).where(m.FaceGallery.project_id == project.id, m.FaceGallery.organization_id == org)
        if not db.scalar(select(exists().where(m.FaceIdentity.gallery_id.in_(g), m.FaceIdentity.organization_id == org))):
            return "Draft"
        photos = exists().where(m.FaceEmbedding.gallery_id.in_(g), m.FaceEmbedding.organization_id == org,
                                m.FaceEmbedding.is_template.is_(False))
        return "Ready" if db.scalar(select(photos)) else "Data needed"
    if project.current_spec.get("mode") == "pretrained":  # ready-made COCO detector: nothing to upload or train
        return "Ready"
    if db.scalar(select(exists().where(m.TrainingJob.project_id == project.id, m.TrainingJob.organization_id == org,
                                       m.TrainingJob.status.in_(("queued", "running"))))):
        return "Training"
    if db.scalar(select(exists().where(m.Model.project_id == project.id, m.Model.organization_id == org,
                                       m.ModelVersion.model_id == m.Model.id, m.ModelVersion.status == "ready",
                                       m.ModelVersion.organization_id == org))):
        return "Trained"
    spec = project.current_spec
    if spec.get("mode") == "quick" or (not spec.get("labels") and "sentiment" in (spec.get("ready_made") or [])):
        return "Ready"  # quick text (examples / label names) and sentiment-only work without training
    ds = db.scalar(select(m.Dataset).where(m.Dataset.project_id == project.id, m.Dataset.organization_id == org))
    if ds is None:
        return "Draft"
    any_version = db.scalar(select(exists().where(m.DatasetVersion.dataset_id == ds.id,
                                                  m.DatasetVersion.organization_id == org)))
    if not any_version:
        return "Draft"
    v = db.scalar(select(m.DatasetVersion).where(m.DatasetVersion.id == ds.current_version_id,
                                                 m.DatasetVersion.organization_id == org)) \
        if ds.current_version_id else None
    if v is None or v.sample_count == 0 or v.ingest_status != "ready":
        return "Data needed"
    if v.sealed_at and v.validation_status == "running":
        return "Validating"
    if v.sealed_at and v.validation_status == "passed":
        return "Ready to train"
    return "Data needed"


def refresh_status(db, project: m.Project) -> str:
    project.status = derive_status(db, project)
    return project.status
