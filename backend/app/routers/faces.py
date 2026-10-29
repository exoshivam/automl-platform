"""FACE-1..8 (PRD §12): consented people, photo enrolment with quality checks, 1:N identify with Unknown, threshold,
and an automatic leave-one-out report. Adding or removing a person only touches gallery rows: nothing is trained."""

import io
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from fastapi import APIRouter, File, Request, Response, UploadFile
from PIL import Image
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select

from app import models as m
from app.db import now
from app.deps import DbDep, PrincipalDep, audit, get_by_id, load_project
from app.errors import ApiError, not_found
from app.ml import face
from app.services import storage
from app.services.project_status import refresh_status
from app.settings import get_settings

router = APIRouter(tags=["faces"])
MAX_PHOTOS, MAX_BYTES = 10, 10 * 1024**2
FARS = (0.001, 0.01, 0.1)


class ConsentIn(BaseModel):
    attested: bool = Field(description="The user attests the person gave consent (checkbox_attested, PRD §12.3)")
    purpose: str | None = Field(default=None, max_length=1000)
    jurisdiction: str | None = Field(default=None, max_length=60)


class PersonIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    external_ref: str | None = Field(default=None, max_length=200)
    consent: ConsentIn | None = None


class ConsentOut(BaseModel):
    id: uuid.UUID
    method: str
    attested_by: uuid.UUID | None
    granted_at: datetime
    purpose: str | None


class Warning_(BaseModel):
    code: str
    detail: str
    other_person_id: uuid.UUID | None = None


class PhotoOut(BaseModel):
    id: uuid.UUID
    thumb_url: str | None
    quality: float | None
    face_px: int | None
    created_at: datetime
    audio_url: str | None = None  # speaker clips
    seconds: float | None = None  # speaker clips: speech after silence trim


class PersonOut(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    name: str
    external_ref: str | None
    status: str
    photo_count: int
    consent: ConsentOut
    warnings: list[Warning_]
    created_at: datetime


class PersonDetail(PersonOut):
    photos: list[PhotoOut]


class Accepted(BaseModel):
    photo_id: uuid.UUID
    filename: str
    quality: float
    face_px: int
    thumb_url: str


class Rejected(BaseModel):
    filename: str
    reason: Literal["no_face", "multiple_faces", "face_too_small", "too_blurry", "not_an_image", "too_many_photos",
                    "file_too_large"]
    detail: str


class EnrolOut(BaseModel):
    accepted: list[Accepted]
    rejected: list[Rejected]
    person: PersonOut


class SettingsOut(BaseModel):
    kind: Literal["face", "speaker"] = "face"
    threshold: float = Field(description="The threshold in effect (speaker: the report's suggestion while auto)")
    threshold_source: Literal["default", "auto", "user"] = "default"
    score_hint: str | None = None
    default_threshold: float
    embedding_model: dict[str, Any]
    detector: dict[str, Any] | None
    phrase: str | None = None
    min_enrol_seconds: float | None = None
    min_identify_seconds: float | None = None
    purpose: str | None
    retention_until: datetime | None
    people_count: int
    photo_count: int


class SettingsPatch(BaseModel):
    threshold: float | None = Field(default=None, ge=0.05, le=0.99)
    threshold_auto: bool | None = Field(default=None, description="Speaker: true = follow the report's suggestion")
    phrase: str | None = Field(default=None, max_length=300, description="Speaker: the sentence shown for enrol/test")
    purpose: str | None = Field(default=None, max_length=1000)
    retention_until: datetime | None = None


class Box(BaseModel):
    x: float
    y: float
    w: float
    h: float


class Candidate(BaseModel):
    person_id: uuid.UUID
    name: str
    score: float


class FaceMatch(BaseModel):
    box: Box
    person_id: uuid.UUID | None
    name: str | None = Field(description="null = Unknown (best score below the threshold)")
    score: float | None
    top3: list[Candidate]


class IdentifyOut(BaseModel):
    image: dict[str, int]
    threshold: float
    faces: list[FaceMatch]
    latency_ms: int
    model_load_ms: int = Field(0, description="> 0 when this call (re)loaded the face models after an idle unload")


# ---- loaders (tenant-scoped; another tenant's ids are 404)

def _gallery(db, principal, project_id: uuid.UUID, action) -> tuple[m.Project, m.FaceGallery]:
    p = load_project(db, principal, project_id, action)
    g = db.scalar(select(m.FaceGallery).where(m.FaceGallery.project_id == p.id,
                                              m.FaceGallery.organization_id == p.organization_id))
    if p.task_type not in ("face_recognition", "speaker_recognition") or g is None:
        raise ApiError(422, "not_face_project", "This project is not a face or speaker recognition project")
    return p, g


def _person(db, principal, person_id: uuid.UUID, action) -> tuple[m.Project, m.FaceGallery, m.FaceIdentity]:
    ident = get_by_id(db, m.FaceIdentity, person_id)
    if ident is None:
        raise not_found("Person")
    g = get_by_id(db, m.FaceGallery, ident.gallery_id)
    p, _ = _gallery(db, principal, g.project_id, action)  # authorizes (404 for another tenant) + deleting guard
    return p, g, ident


def _photos(db, ident: m.FaceIdentity) -> list[m.FaceEmbedding]:
    return list(db.scalars(select(m.FaceEmbedding).where(
        m.FaceEmbedding.identity_id == ident.id, m.FaceEmbedding.organization_id == ident.organization_id,
        m.FaceEmbedding.is_template.is_(False)).order_by(m.FaceEmbedding.created_at)))


def _person_out(db, g: m.FaceGallery, ident: m.FaceIdentity, n: int | None = None) -> PersonOut:
    n = len(_photos(db, ident)) if n is None else n
    c = get_by_id(db, m.ConsentRecord, ident.consent_record_id)
    warnings = ([Warning_(code="no_photos", detail="Add 3-10 photos to enrol this person")] if n == 0 else
                [Warning_(code="few_photos", detail=f"Only {n} photo(s); 3-10 recommended")] if n < 3 else [])
    return PersonOut(id=ident.id, project_id=g.project_id, name=ident.display_name, external_ref=ident.external_ref,
                     status=ident.status, photo_count=n, warnings=warnings, created_at=ident.created_at,
                     consent=ConsentOut(id=c.id, method=c.consent_method, attested_by=c.attested_by_user_id,
                                        granted_at=c.granted_at, purpose=c.purpose))


def _photo_prefix(ident: m.FaceIdentity) -> str:
    return f"{storage.org_prefix(ident.organization_id)}faces/{ident.gallery_id}/{ident.id}/"


def _refresh_template(db, ident: m.FaceIdentity) -> None:
    """The mean template (re-normalised) is rewritten after every enrol/remove: instant, no training (FACE-3)."""
    db.execute(delete(m.FaceEmbedding).where(m.FaceEmbedding.identity_id == ident.id,
                                             m.FaceEmbedding.organization_id == ident.organization_id,
                                             m.FaceEmbedding.is_template.is_(True)))
    vecs = [np.asarray(e.embedding) for e in _photos(db, ident)]
    if vecs:
        mean = np.mean(vecs, axis=0)
        db.add(m.FaceEmbedding(organization_id=ident.organization_id, identity_id=ident.id, gallery_id=ident.gallery_id,
                               embedding=(mean / np.linalg.norm(mean)).tolist(), is_template=True))
    db.flush()


# ---- settings

@router.get("/projects/{project_id}/face-settings", response_model=SettingsOut)
@router.get("/projects/{project_id}/speaker-settings", response_model=SettingsOut)
def get_settings_(project_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    _, g = _gallery(db, principal, project_id, "read")
    return _settings_out(db, g)


@router.patch("/projects/{project_id}/face-settings", response_model=SettingsOut)
@router.patch("/projects/{project_id}/speaker-settings", response_model=SettingsOut)
def patch_settings(project_id: uuid.UUID, body: SettingsPatch, principal: PrincipalDep, db: DbDep, request: Request):
    _, g = _gallery(db, principal, project_id, "write")
    changes = body.model_dump(exclude_unset=True)
    if "threshold" in changes and "threshold_auto" not in changes:
        changes["threshold_auto"] = False  # a hand-set threshold sticks
    for k, v in changes.items():
        setattr(g, k, v)
    audit(db, request, principal, g.organization_id, "face.settings", "face_gallery", g.id,
          **{k: str(v) for k, v in changes.items()})
    return _settings_out(db, g)


def _settings_out(db, g: m.FaceGallery) -> SettingsOut:
    org = g.organization_id
    people = db.scalar(select(func.count()).select_from(m.FaceIdentity).where(
        m.FaceIdentity.gallery_id == g.id, m.FaceIdentity.organization_id == org))
    photos = db.scalar(select(func.count()).select_from(m.FaceEmbedding).where(
        m.FaceEmbedding.gallery_id == g.id, m.FaceEmbedding.organization_id == org,
        m.FaceEmbedding.is_template.is_(False)))
    if g.kind == "speaker":
        from app.ml import speaker

        threshold, source = effective_threshold(db, g)
        return SettingsOut(kind="speaker", threshold=threshold, threshold_source=source,
                           score_hint="Cosine similarity of voice embeddings: the same speaker usually scores 0.9+, "
                                      "different speakers 0.5-0.85; similar voices can exceed 0.85",
                           default_threshold=get_settings().speaker_default_threshold,
                           embedding_model=speaker.MODEL, detector=None, phrase=g.phrase,
                           min_enrol_seconds=speaker.MIN_ENROL_S, min_identify_seconds=speaker.MIN_IDENTIFY_S,
                           purpose=g.purpose, retention_until=g.retention_until, people_count=people,
                           photo_count=photos)
    return SettingsOut(threshold=g.threshold, default_threshold=get_settings().face_default_threshold,
                       purpose=g.purpose, retention_until=g.retention_until, people_count=people, photo_count=photos,
                       **face.model_info())


# ---- people

@router.post("/projects/{project_id}/people", status_code=201, response_model=PersonOut)
def create_person(project_id: uuid.UUID, body: PersonIn, principal: PrincipalDep, db: DbDep, request: Request):
    p, g = _gallery(db, principal, project_id, "write")
    if body.consent is None or not body.consent.attested:  # FACE-6: no consent record, no enrolment
        raise ApiError(422, "consent_required", "Confirm that this person has consented before enrolling them",
                       errors=[{"loc": ["body", "consent", "attested"], "msg": "must be true"}])
    org = g.organization_id
    consent = m.ConsentRecord(organization_id=org, gallery_id=g.id, subject_ref=body.external_ref,
                              consent_method="checkbox_attested", purpose=body.consent.purpose or g.purpose,
                              jurisdiction=body.consent.jurisdiction, granted_at=now(),
                              attested_by_user_id=principal.user_id)
    db.add(consent)
    db.flush()
    ident = m.FaceIdentity(organization_id=org, gallery_id=g.id, display_name=body.name.strip(),
                           external_ref=body.external_ref, consent_record_id=consent.id)
    db.add(ident)
    db.flush()
    refresh_status(db, p)
    audit(db, request, principal, org, "face.identity_created", "face_identity", ident.id, consent_id=str(consent.id),
          consent_method=consent.consent_method)
    return _person_out(db, g, ident, 0)


@router.get("/projects/{project_id}/people", response_model=list[PersonOut])
def list_people(project_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    _, g = _gallery(db, principal, project_id, "read")
    people = db.scalars(select(m.FaceIdentity).where(m.FaceIdentity.gallery_id == g.id,
                                                      m.FaceIdentity.organization_id == g.organization_id)
                        .order_by(func.lower(m.FaceIdentity.display_name))).all()
    counts = dict(db.execute(select(m.FaceEmbedding.identity_id, func.count()).where(
        m.FaceEmbedding.gallery_id == g.id, m.FaceEmbedding.organization_id == g.organization_id,
        m.FaceEmbedding.is_template.is_(False)).group_by(m.FaceEmbedding.identity_id)).all())
    return [_person_out(db, g, i, counts.get(i.id, 0)) for i in people]


@router.get("/people/{person_id}", response_model=PersonDetail)
def get_person(person_id: uuid.UUID, principal: PrincipalDep, db: DbDep):
    _, g, ident = _person(db, principal, person_id, "read")
    photos = _photos(db, ident)
    return PersonDetail(**_person_out(db, g, ident, len(photos)).model_dump(), photos=[
        PhotoOut(id=e.id, thumb_url=storage.presign_get(e.thumbnail_key) if e.thumbnail_key else None,
                 quality=e.quality_score, face_px=e.face_px, created_at=e.created_at, seconds=e.duration_s,
                 audio_url=storage.presign_get(e.photo_object_key) if g.kind == "speaker" and e.photo_object_key
                 else None) for e in photos])


@router.post("/people/{person_id}/photos", response_model=EnrolOut)
async def add_photos(person_id: uuid.UUID, principal: PrincipalDep, db: DbDep, request: Request,
                     files: Annotated[list[UploadFile], File()]):
    p, g, ident = _person(db, principal, person_id, "write")
    if g.kind != "face":
        raise ApiError(422, "not_face_project", "This is a speaker gallery: upload voice clips to /people/{id}/clips")
    if not 1 <= len(files) <= MAX_PHOTOS:
        raise ApiError(422, "invalid_photo_count", f"Send 1-{MAX_PHOTOS} photos per request")
    s = get_settings()
    have = len(_photos(db, ident))
    accepted, rejected = [], []
    for f in files:
        name = (f.filename or "photo")[:200]
        data = await f.read(MAX_BYTES + 1)

        def reject(reason, detail, name=name):
            rejected.append(Rejected(filename=name, reason=reason, detail=detail))

        if len(data) > MAX_BYTES:
            reject("file_too_large", "Photos can be at most 10 MB")
            continue
        if have + len(accepted) >= MAX_PHOTOS:
            reject("too_many_photos", f"A person can have at most {MAX_PHOTOS} photos; delete one first")
            continue
        try:
            img = face.load_image(data)
        except Exception:  # noqa: BLE001
            reject("not_an_image", "Not a readable JPEG/PNG/WebP image")
            continue
        faces = face.analyse(img)
        if not faces:
            reject("no_face", "No face found; use a clear, front-facing photo")
        elif len(faces) > 1:
            reject("multiple_faces", f"{len(faces)} faces found; the photo must show only this person")
        elif faces[0].px < s.face_min_px:
            reject("face_too_small", f"The face is {faces[0].px} px; it must be at least {s.face_min_px} px")
        elif faces[0].blur < s.face_min_blur:
            reject("too_blurry", "The face is too blurry; use a sharper photo")
        else:
            fc = faces[0]
            row = m.FaceEmbedding(organization_id=ident.organization_id, identity_id=ident.id, gallery_id=g.id,
                                  embedding=fc.embedding.tolist(), quality_score=fc.quality, face_px=fc.px)
            db.add(row)
            db.flush()  # assigns the id used in the object keys
            row.photo_object_key = f"{_photo_prefix(ident)}{row.id}.jpg"
            row.thumbnail_key = f"{_photo_prefix(ident)}{row.id}.thumb.webp"
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=92)  # re-encoded: no EXIF (location etc.) is kept
            storage.put_bytes(row.photo_object_key, buf.getvalue(), "image/jpeg")
            buf = io.BytesIO()
            fc.crop.resize((160, 160), Image.BILINEAR).save(buf, "WEBP", quality=85)
            storage.put_bytes(row.thumbnail_key, buf.getvalue(), "image/webp")
            accepted.append(Accepted(photo_id=row.id, filename=name, quality=fc.quality, face_px=fc.px,
                                     thumb_url=storage.presign_get(row.thumbnail_key)))
    db.flush()
    if accepted:
        _refresh_template(db, ident)
        refresh_status(db, p)
        audit(db, request, principal, ident.organization_id, "face.photos_enrolled", "face_identity", ident.id,
              accepted=len(accepted), rejected=len(rejected))
    return EnrolOut(accepted=accepted, rejected=rejected, person=_person_out(db, g, ident))


@router.delete("/people/{person_id}/photos/{photo_id}", status_code=204)
def delete_photo(person_id: uuid.UUID, photo_id: uuid.UUID, principal: PrincipalDep, db: DbDep, request: Request):
    p, _, ident = _person(db, principal, person_id, "write")
    row = get_by_id(db, m.FaceEmbedding, photo_id)
    if row is None or row.identity_id != ident.id or row.is_template:
        raise not_found("Photo")
    for k in (row.photo_object_key, row.thumbnail_key):
        if k:
            storage.delete(k)
    db.delete(row)
    db.flush()
    _refresh_template(db, ident)
    refresh_status(db, p)
    audit(db, request, principal, ident.organization_id, "face.photo_deleted", "face_identity", ident.id,
          photo_id=str(photo_id))
    return Response(status_code=204)


@router.delete("/people/{person_id}", status_code=204)
def delete_person(person_id: uuid.UUID, principal: PrincipalDep, db: DbDep, request: Request):
    """FACE-8: hard delete of photos, embeddings, template and consent link, with a deletion certificate."""
    p, _, ident = _person(db, principal, person_id, "write")
    org = ident.organization_id
    rows = db.scalars(select(m.FaceEmbedding).where(m.FaceEmbedding.identity_id == ident.id,
                                                    m.FaceEmbedding.organization_id == org)).all()
    prefix = _photo_prefix(ident)
    objects = storage.delete_prefix(prefix)
    objects += storage.delete_prefix(_export_prefix(ident))  # earlier exports hold this person's template
    consent_id, n = ident.consent_record_id, len(rows)
    db.execute(delete(m.FaceEmbedding).where(m.FaceEmbedding.identity_id == ident.id,
                                             m.FaceEmbedding.organization_id == org))
    db.delete(ident)
    db.flush()
    db.execute(delete(m.ConsentRecord).where(m.ConsentRecord.id == consent_id, m.ConsentRecord.organization_id == org))
    refresh_status(db, p)
    audit(db, request, principal, org, "face.identity_deleted", "face_identity", person_id,
          certificate={"identity_id": str(person_id), "embeddings_deleted": n, "objects_deleted": objects,
                       "consent_record_deleted": str(consent_id), "prefix": prefix, "deleted_at": now().isoformat()})
    return Response(status_code=204)


# ---- identify (1:N, every face in the image)

def effective_threshold(db, g: m.FaceGallery) -> tuple[float, str]:
    """(threshold, source). Speaker galleries follow the report's suggestion until the user sets one by hand."""
    if g.kind == "speaker" and g.threshold_auto:
        rep = face_report(db, g)
        if rep["ready"] and rep["suggested_threshold"] is not None:
            return rep["suggested_threshold"], "auto"
    return g.threshold, "user" if not g.threshold_auto else "default"


def _candidates(db, g: m.FaceGallery, vec: np.ndarray, k: int = 3) -> list[tuple[uuid.UUID, float]]:
    dist = m.FaceEmbedding.embedding.cosine_distance(vec.tolist())
    rows = db.execute(select(m.FaceEmbedding.identity_id, dist).where(
        m.FaceEmbedding.gallery_id == g.id, m.FaceEmbedding.organization_id == g.organization_id)
        .order_by(dist).limit(50)).all()  # HNSW top-50 rows (photos + templates), then best per person
    best: dict[uuid.UUID, float] = {}
    for iid, d in rows:
        best[iid] = max(best.get(iid, -1.0), 1.0 - float(d))
    return sorted(best.items(), key=lambda x: -x[1])[:k]


class SpeakerMatch(BaseModel):
    person_id: uuid.UUID | None
    name: str | None = Field(description="null = Unknown (best score below the threshold)")
    score: float | None
    top3: list[Candidate]
    seconds: float
    threshold: float
    latency_ms: int
    model_load_ms: int = 0


@router.post("/projects/{project_id}/identify", response_model=IdentifyOut | SpeakerMatch)
async def identify(project_id: uuid.UUID, principal: PrincipalDep, db: DbDep, file: Annotated[UploadFile, File()]):
    """Face: every face in an image. Speaker: one voice clip -> one match."""
    _, g = _gallery(db, principal, project_id, "read")
    return identify_data(db, g, await file.read(MAX_BYTES + 1))


def identify_data(db, g: m.FaceGallery, data: bytes) -> IdentifyOut | SpeakerMatch:
    """Shared by the playground (POST /projects/{id}/identify) and deployments (POST /v1/predict)."""
    t0 = time.perf_counter()
    if g.kind == "speaker":
        return _identify_speaker(db, g, data, t0)
    if len(data) > MAX_BYTES:
        raise ApiError(413, "file_too_large", "Images can be at most 10 MB")
    try:
        img = face.load_image(data)
    except Exception:  # noqa: BLE001
        raise ApiError(422, "not_an_image", "Not a readable JPEG/PNG/WebP image") from None
    faces, load_ms = face.analyse_timed(img)
    names = dict(db.execute(select(m.FaceIdentity.id, m.FaceIdentity.display_name).where(
        m.FaceIdentity.gallery_id == g.id, m.FaceIdentity.organization_id == g.organization_id)).all())
    out = []
    for fc in faces:
        top = [Candidate(person_id=i, name=names[i], score=round(s, 4)) for i, s in _candidates(db, g, fc.embedding)
               if i in names]
        hit = top[0] if top and top[0].score >= g.threshold else None
        out.append(FaceMatch(box=Box(x=fc.box[0], y=fc.box[1], w=fc.box[2], h=fc.box[3]),
                             person_id=hit.person_id if hit else None, name=hit.name if hit else None,
                             score=top[0].score if top else None, top3=top))
    return IdentifyOut(image={"width": img.width, "height": img.height}, threshold=g.threshold, faces=out,
                       latency_ms=round((time.perf_counter() - t0) * 1000), model_load_ms=load_ms)


# ---- report (leave-one-out over the enrolled photos)

def _hist(scores: np.ndarray) -> dict:
    edges = np.round(np.arange(-0.2, 1.0001, 0.05), 2)
    counts, _ = np.histogram(np.clip(scores, -0.2, 1.0), bins=edges)
    return {"count": len(scores), "mean": round(float(scores.mean()), 4) if len(scores) else None,
            "hist": [{"lo": float(lo), "hi": float(hi), "count": int(c)}
                     for lo, hi, c in zip(edges[:-1], edges[1:], counts, strict=True)]}


def face_report(db, g: m.FaceGallery) -> dict:
    org = g.organization_id
    people = {i.id: i.display_name for i in db.scalars(select(m.FaceIdentity).where(
        m.FaceIdentity.gallery_id == g.id, m.FaceIdentity.organization_id == org))}
    rows = db.execute(select(m.FaceEmbedding.identity_id, m.FaceEmbedding.embedding, m.FaceEmbedding.quality_score)
                      .where(m.FaceEmbedding.gallery_id == g.id, m.FaceEmbedding.organization_id == org,
                             m.FaceEmbedding.is_template.is_(False))).all()
    by: dict[uuid.UUID, list] = {}
    for iid, emb, q in rows:
        by.setdefault(iid, []).append((np.asarray(emb, dtype=np.float64), q or 0.0))
    usable = [i for i, v in by.items() if len(v) >= 2]
    base = {"people": len(people), "photos": len(rows), "current_threshold": g.threshold}
    if len(usable) < 2:
        return {**base, "ready": False, "suggested_threshold": None, "genuine": None, "impostor": None,
                "tar_at_far": [], "rank1_accuracy": None, "per_person": [],
                "warnings": [{"code": "not_enough_people",
                              "detail": "Enrol at least 2 people with at least 2 photos each to see the report"}]}

    def unit(v):
        return v / np.linalg.norm(v)

    templates = {i: unit(np.mean([e for e, _ in v], axis=0)) for i, v in by.items()}
    genuine, impostor, per = [], [], []
    for iid, photos in by.items():
        hits, worst_other, worst_score, n = 0, None, -1.0, len(photos)
        for k, (e, _) in enumerate(photos):
            others = [templates[j] for j in templates if j != iid]
            imp = [float(e @ t) for t in others]
            impostor += imp
            if n >= 2:
                loo = unit(np.mean([x for j, (x, _) in enumerate(photos) if j != k], axis=0))
                gen = float(e @ loo)
                genuine.append(gen)
                hits += int(gen > max(imp, default=-1.0))
            for j, t in templates.items():
                if j != iid and float(e @ t) > worst_score:
                    worst_other, worst_score = j, float(e @ t)
        warns = []
        if n < 3:
            warns.append({"code": "few_photos", "detail": f"Only {n} photo(s); 3-10 recommended"})
        if np.mean([q for _, q in photos]) < 0.5:
            warns.append({"code": "low_quality", "detail": "Photos are blurry or low-confidence; add sharper ones"})
        if n >= 2 and hits < n:
            warns.append({"code": "not_matched", "detail": f"{n - hits} of {n} photo(s) matched someone else better"})
        if worst_other is not None and worst_score >= g.threshold - 0.1:
            warns.append({"code": "similar_to", "detail": f"Looks similar to {people.get(worst_other, '?')} "
                          f"(score {worst_score:.2f})", "other_person_id": str(worst_other)})
        per.append({"person_id": str(iid), "name": people.get(iid, "?"), "photos": n,
                    "rank1_hits": hits if n >= 2 else None, "warnings": warns})
    gen, imp = np.array(genuine), np.array(impostor)
    tar_at_far = []
    for far in FARS:
        t = float(np.quantile(imp, 1 - far, method="higher")) if len(imp) else 1.0
        tar_at_far.append({"far": far, "threshold": round(t, 4), "tar": round(float((gen > t).mean()), 4)})
    if g.kind == "speaker":
        # x-vector cosines sit high for everyone (enrolled impostors ~0.6-0.8, an UNENROLLED similar voice reached
        # 0.90 at a gallery whose max impostor was 0.79), so enrolled impostors alone can't bound strangers: be
        # FAR-first - above the default, clear of every impostor, and only as low as 95% of genuine clips allow
        gen_p5 = float(np.quantile(gen, 0.05)) if len(gen) else 1.0
        suggested = round(min(0.97, max(get_settings().speaker_default_threshold, float(imp.max()) + 0.05,
                                        gen_p5 - 0.02)), 2)
    else:
        suggested = round(min(0.9, max(0.25, tar_at_far[1]["threshold"] + 0.02)), 2)  # FAR 1% with a margin
    warnings = [] if len(imp) >= 100 else [
        {"code": "small_gallery", "detail": f"Only {len(imp)} impostor comparisons; the threshold estimate is rough. "
                                            "Enrol more people (50+ recommended before deployment, PRD §12.5)"}]
    tried = [x for x in per if x["rank1_hits"] is not None]
    rank1 = sum(x["rank1_hits"] for x in tried) / max(1, sum(x["photos"] for x in tried))
    return {**base, "ready": True, "suggested_threshold": suggested, "genuine": _hist(gen), "impostor": _hist(imp),
            "tar_at_far": tar_at_far, "rank1_accuracy": round(rank1, 4),
            "per_person": sorted(per, key=lambda x: x["name"].lower()), "warnings": warnings}


@router.get("/projects/{project_id}/face-report")
@router.get("/projects/{project_id}/speaker-report")
def get_face_report(project_id: uuid.UUID, principal: PrincipalDep, db: DbDep) -> dict:
    _, g = _gallery(db, principal, project_id, "read")
    return face_report(db, g)


# ---- local export (owner/admin): models + templates + the standalone script

class ExportIn(BaseModel):
    pack: Literal["small", "accurate"] = Field(default="accurate", description="small = buffalo_s (~16 MB), "
                                               "accurate = buffalo_l (~182 MB, the app's own models)")
    acknowledge_biometric: bool = Field(default=False, description="The user acknowledged that the bundle contains "
                                        "biometric templates of the enrolled people")


class ExportOut(BaseModel):
    pack: str = "accurate"
    download_url: str
    filename: str
    size_bytes: int
    sha256: str
    people: int
    created_at: datetime


def _export_prefix(row) -> str:
    gallery_id = row.gallery_id if hasattr(row, "gallery_id") else row.id
    return f"{storage.org_prefix(row.organization_id)}faces/{gallery_id}/exports/"


README = """# Face gallery export: {name}

Exported {created} by the AutoML platform. **This bundle contains biometric templates** of {people} enrolled
people (one mean 512-d ArcFace embedding per person; no photos and no per-photo embeddings). Store it securely,
use it only for the purpose the people consented to, and delete it when that purpose ends.

## Files
- `{det}`: SCRFD face detector (5 landmarks), InsightFace {pack}
- `{rec}`: face recognition model (112x112 -> 512-d), InsightFace {pack}
- `gallery.json`: threshold, embedding model id/version and `people: [{{name, template}}]`
- `identify.py`: the same detect -> align -> embed -> cosine match code the platform runs
- `LICENSE-NOTICE.txt`: the model weights are for NON-COMMERCIAL research only

## Run
```
pip install onnxruntime numpy pillow        # + opencv-python for --webcam
python identify.py photo.jpg                # prints  name<TAB>score<TAB>box  per face; "Other" below the threshold
python identify.py --webcam
```
Threshold: {threshold} (cosine similarity). A face whose best score is below it is reported as `Other`.
"""

NOTICE = """The model files {det} and {rec} are from the InsightFace {pack} model pack.
InsightFace: "ALL models are available for non-commercial research purposes only."
(https://github.com/deepinsight/insightface/tree/master/model_zoo). Commercial use requires a licence from the
vendor or different, licence-cleared weights. The InsightFace code is MIT-licensed; the weights are not.
"""


def _quality(by: dict) -> dict | None:
    """Leave-one-out on the enrolled photos: each photo vs its own person's template WITHOUT it and every other
    person's template. Cheap (numpy over embeddings already computed)."""
    def unit(v):
        return v / np.linalg.norm(v)

    if sum(len(v) >= 2 for v in by.values()) < 2:
        return None
    tpl = {i: unit(np.mean(v, axis=0)) for i, v in by.items()}
    hits = n = 0
    gaps, gen, imp = [], [], []
    for i, vs in by.items():
        if len(vs) < 2:
            continue
        for k, e in enumerate(vs):
            own = float(e @ unit(np.mean([x for j, x in enumerate(vs) if j != k], axis=0)))
            other = max(float(e @ t) for j, t in tpl.items() if j != i)
            hits, n = hits + int(own > other), n + 1
            gaps.append(own - other)
            gen.append(own)
            imp.append(other)
    return {"photos": n, "recognised": hits, "rate": round(hits / n, 4), "mean_gap": round(float(np.mean(gaps)), 4),
            "people_checked": sum(len(v) >= 2 for v in by.values()),
            "_gen_mean": float(np.mean(gen)), "_imp_mean": float(np.mean(imp))}


def _pack_vectors(db, g: m.FaceGallery, pack: str) -> tuple[dict, int]:
    """{identity_id: [embedding, ...]} for the pack, and how many photos the pack failed to detect. Accurate = the
    stored embeddings; Small = RE-EMBEDDED from the stored photos with buffalo_s (embeddings aren't compatible across
    packs), cached in Redis per photo set so a preview followed by an export embeds once."""
    import json

    from app.services.ratelimit import client as redis

    rows = db.execute(select(m.FaceEmbedding.id, m.FaceEmbedding.identity_id, m.FaceEmbedding.embedding,
                             m.FaceEmbedding.photo_object_key).where(
        m.FaceEmbedding.gallery_id == g.id, m.FaceEmbedding.organization_id == g.organization_id,
        m.FaceEmbedding.is_template.is_(False)).order_by(m.FaceEmbedding.id)).all()
    if pack == "accurate":
        by: dict = {}
        for _, iid, emb, _ in rows:
            by.setdefault(iid, []).append(np.asarray(emb, dtype=np.float64))
        return by, 0
    import hashlib

    ck = f"faceexp:{pack}:{g.id}:" + hashlib.sha256("|".join(str(r[0]) for r in rows).encode()).hexdigest()[:16]
    hit = redis().get(ck)
    if hit:
        raw = json.loads(hit)
        return {uuid.UUID(k): [np.asarray(x) for x in v] for k, v in raw["by"].items()}, raw["missed"]
    usable = [r for r in rows if r[3]]
    faces_per = face.analyse_with(pack, [storage.get_bytes(r[3]) for r in usable])
    by, missed = {}, len(rows) - len(usable)
    for (_, iid, _, _), fs in zip(usable, faces_per, strict=True):
        if not fs:
            missed += 1
            continue
        best = max(fs, key=lambda f: f.box[2] * f.box[3])  # enrolment guaranteed one face; take the largest
        by.setdefault(iid, []).append(np.asarray(best.embedding, dtype=np.float64))
    redis().set(ck, json.dumps({"by": {str(k): [v.tolist() for v in vs] for k, vs in by.items()}, "missed": missed}),
                ex=3600)
    return by, missed


def _pack_threshold(db, g: m.FaceGallery, pack: str, q_pack: dict | None) -> tuple[float, str]:
    """Accurate: the gallery's own threshold. Small: the same RELATIVE operating point on MobileFaceNet's score
    scale (its genuine/impostor means sit elsewhere than ArcFace-R50's), measured on this gallery's photos."""
    if pack == "accurate":
        return g.threshold, "gallery"
    q_acc = _quality(_pack_vectors(db, g, "accurate")[0])
    if not q_acc or not q_pack or q_acc["_gen_mean"] <= q_acc["_imp_mean"]:
        return 0.40, "default"
    pos = (g.threshold - q_acc["_imp_mean"]) / (q_acc["_gen_mean"] - q_acc["_imp_mean"])
    t = q_pack["_imp_mean"] + pos * (q_pack["_gen_mean"] - q_pack["_imp_mean"])
    return round(min(0.9, max(0.2, t)), 3), "calibrated"


def _export_bundle(db, principal, project_id, pack: str = "accurate") -> dict:
    """The ONE definition of the export (preview + POST): checks, gallery.json and every file with its description.
    Only the models and ONE mean template per person go in: no photos and no per-photo embeddings."""
    import json
    import re

    from app.deps import authorize

    p, g = _gallery(db, principal, project_id, "read")
    authorize(db, principal, p.organization_id, "manage", "Project")  # owner/admin only
    if g.kind != "face":
        raise ApiError(409, "export_unavailable", "Speaker galleries can't be exported in this build")
    org = g.organization_id
    people = db.scalars(select(m.FaceIdentity).where(m.FaceIdentity.gallery_id == g.id,
                                                      m.FaceIdentity.organization_id == org)
                        .order_by(func.lower(m.FaceIdentity.display_name))).all()
    try:
        face.model_files(pack)
    except FileNotFoundError:
        raise ApiError(503, "pack_unavailable", f"The {face.PACK_LABELS[pack]} face models aren't installed") from None
    by, missed = _pack_vectors(db, g, pack)
    quality = _quality(by)
    threshold, threshold_source = _pack_threshold(db, g, pack, quality)
    templates = {i: (np.mean(v, axis=0) / np.linalg.norm(np.mean(v, axis=0))) for i, v in by.items()}
    photos = db.scalar(select(func.count()).select_from(m.FaceEmbedding).where(
        m.FaceEmbedding.gallery_id == g.id, m.FaceEmbedding.organization_id == org,
        m.FaceEmbedding.is_template.is_(False)))
    entries = [{"name": i.display_name, "template": [round(float(x), 6) for x in templates[i.id]]}
               for i in people if i.id in templates]
    if not entries:
        raise ApiError(409, "gallery_empty", "Enrol at least one person with photos before exporting")
    det, rec, pack_name = face.model_files(pack)
    info = face.model_info(pack)
    created = now().replace(microsecond=0)  # fixed-length timestamps: the preview size matches the real file
    gallery = {"format": "automl-face-gallery/1", "project": p.name, "created_at": created.isoformat(),
               "threshold": threshold, "metric": "cosine", "pack": pack,
               "embedding_model": {**info["embedding_model"], "version": f"{pack_name}/{rec.name}"},
               "detector": {**info["detector"], "version": f"{pack_name}/{det.name}"}, "people": entries}
    readme = README.format(name=p.name, created=created.isoformat(timespec="minutes"), people=len(entries),
                           threshold=threshold, det=det.name, rec=rec.name, pack=pack_name)
    members = [  # (name in zip, path or bytes, stored?, description)
        (det.name, det, True, f"Face detector (SCRFD, InsightFace {pack_name})"),
        (rec.name, rec, True, "Face recognition model (" + ("MobileFaceNet" if pack == "small" else "ArcFace R50")
         + ", 512-d)"),
        ("gallery.json", json.dumps(gallery, indent=1).encode(), False,
         f"Threshold + {len(entries)} people: name and ONE mean face template each (no photos)"),
        ("identify.py", Path(face.face_core.__file__), False, "The same detect/align/embed/match code the app runs"),
        ("README.md", readme.encode(), False, "How to run it"),
        ("LICENSE-NOTICE.txt", NOTICE.format(pack=pack_name, det=det.name, rec=rec.name).encode(), False,
         "The model weights are for non-commercial research only"),
    ]
    slug = re.sub(r"[^a-z0-9]+", "-", p.name.lower()).strip("-")[:40] or "gallery"
    public_q = {k: v for k, v in quality.items() if not k.startswith("_")} if quality else None
    return {"project": p, "gallery": g, "people": len(entries), "photos": photos or 0, "created": created,
            "pack": pack, "threshold": threshold, "threshold_source": threshold_source, "quality": public_q,
            "missed": missed,
            "members": members, "filename": f"face-gallery-{slug}-{created:%Y%m%d-%H%M}.zip"}


def _zip_size(members) -> tuple[int, list[dict]]:
    """Exact size of the zip _write_zip produces (local header + data + central directory per member + end record),
    without building it: STORED members count their file size, the small DEFLATED ones are compressed in memory."""
    import zlib

    total, files = 22, []
    for name, src, stored, desc in members:
        raw = src.stat().st_size if isinstance(src, Path) else len(src)
        if stored:
            data = raw
        else:
            body = src.read_bytes() if isinstance(src, Path) else src
            co = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -15)
            data = len(co.compress(body) + co.flush())
        n = len(name.encode())
        total += (30 + n + data) + (46 + n)
        files.append({"name": name, "size_bytes": raw, "description": desc})
    return total, files


def _write_zip(path: str, members) -> None:
    import zipfile

    with zipfile.ZipFile(path, "w") as z:
        for name, src, stored, _ in members:
            kind = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED  # the ONNX files are already dense
            if isinstance(src, Path):
                z.write(src, name, compress_type=kind)
            else:
                z.writestr(name, src, compress_type=kind)


class ExportFile(BaseModel):
    name: str
    size_bytes: int
    description: str


class ExportPreview(BaseModel):
    pack: Literal["small", "accurate"] = "accurate"
    pack_label: str = ""
    available: bool = True
    threshold: float | None = None
    threshold_source: Literal["gallery", "calibrated", "default"] | None = None
    quality: dict[str, Any] | None = Field(default=None, description="Leave-one-out on the enrolled photos with this "
                                           "pack: {photos, recognised, rate, mean_gap, people_checked}")
    not_detected: int = Field(default=0, description="Enrolled photos this pack's detector found no face in")
    people: int
    photos_excluded: int = Field(description="Enrolled photos that are NOT in the bundle")
    files: list[ExportFile]
    total_bytes: int = Field(description="The size of the zip the export would produce")
    filename: str


@router.get("/projects/{project_id}/face-export/preview", response_model=ExportPreview)
def face_export_preview(project_id: uuid.UUID, principal: PrincipalDep, db: DbDep,
                        pack: Literal["small", "accurate"] = "accurate"):
    try:
        b = _export_bundle(db, principal, project_id, pack)
    except ApiError as e:
        if e.code != "pack_unavailable":
            raise
        return ExportPreview(pack=pack, pack_label=face.PACK_LABELS[pack], available=False, people=0,
                             photos_excluded=0, files=[], total_bytes=0, filename="")
    total, files = _zip_size(b["members"])
    return ExportPreview(pack=pack, pack_label=face.PACK_LABELS[pack], threshold=b["threshold"],
                         threshold_source=b["threshold_source"], quality=b["quality"], not_detected=b["missed"],
                         people=b["people"], photos_excluded=b["photos"], files=files, total_bytes=total,
                         filename=b["filename"])


@router.post("/projects/{project_id}/face-export", response_model=ExportOut)
def face_export(project_id: uuid.UUID, body: ExportIn, principal: PrincipalDep, db: DbDep, request: Request):
    import hashlib
    import tempfile

    if not body.acknowledge_biometric:
        _export_bundle(db, principal, project_id, body.pack)  # same 404 / 403 / 409 order as before
        raise ApiError(422, "acknowledgement_required", "Acknowledge that the export contains biometric templates",
                       errors=[{"loc": ["body", "acknowledge_biometric"], "msg": "must be true"}])
    b = _export_bundle(db, principal, project_id, body.pack)
    g, created, filename, org = b["gallery"], b["created"], b["filename"], b["gallery"].organization_id
    with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
        _write_zip(tmp.name, b["members"])
        size = Path(tmp.name).stat().st_size
        sha = hashlib.sha256()
        with open(tmp.name, "rb") as f:
            while chunk := f.read(1 << 20):
                sha.update(chunk)
        key = f"{_export_prefix(g)}{created:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}.zip"
        storage.put_file(key, tmp.name, "application/zip")
    audit(db, request, principal, org, "face_export", "face_gallery", g.id, people=b["people"], size_bytes=size,
          sha256=sha.hexdigest())
    url = storage.client().generate_presigned_url(
        "get_object", ExpiresIn=get_settings().presign_ttl_seconds,
        Params={"Bucket": storage.bucket(), "Key": key, "ResponseContentDisposition": f'attachment; filename="{filename}"'})
    return ExportOut(pack=b["pack"], download_url=url, filename=filename, size_bytes=size, sha256=sha.hexdigest(), people=b["people"],
                     created_at=created)


# ---- speaker recognition (same gallery tables, kind=speaker): voice clips instead of photos

def _identify_speaker(db, g: m.FaceGallery, data: bytes, t0: float) -> SpeakerMatch:
    from app.ml import speaker

    if len(data) > MAX_BYTES:
        raise ApiError(413, "file_too_large", "Clips can be at most 10 MB")
    try:
        clip = speaker.analyse(data, speaker.MIN_IDENTIFY_S)
    except speaker.AudioError as e:
        raise ApiError(422, e.reason, e.detail) from None
    vec, load_ms = speaker.embed(clip.speech)
    names = dict(db.execute(select(m.FaceIdentity.id, m.FaceIdentity.display_name).where(
        m.FaceIdentity.gallery_id == g.id, m.FaceIdentity.organization_id == g.organization_id)).all())
    top = [Candidate(person_id=i, name=names[i], score=round(s, 4)) for i, s in _candidates(db, g, vec) if i in names]
    threshold, _ = effective_threshold(db, g)
    hit = top[0] if top and top[0].score >= threshold else None
    return SpeakerMatch(person_id=hit.person_id if hit else None, name=hit.name if hit else None,
                        score=top[0].score if top else None, top3=top, seconds=clip.seconds, threshold=threshold,
                        latency_ms=round((time.perf_counter() - t0) * 1000), model_load_ms=load_ms)


class ClipAccepted(BaseModel):
    photo_id: uuid.UUID
    filename: str
    quality: float
    seconds: float


class ClipRejected(BaseModel):
    filename: str
    reason: Literal["not_audio", "unsupported_format", "too_short", "too_quiet", "clipping", "too_many_photos",
                    "file_too_large"]
    detail: str


class ClipsOut(BaseModel):
    accepted: list[ClipAccepted]
    rejected: list[ClipRejected]
    person: PersonOut


@router.post("/people/{person_id}/clips", response_model=ClipsOut)
async def add_clips(person_id: uuid.UUID, principal: PrincipalDep, db: DbDep, request: Request,
                    files: Annotated[list[UploadFile], File()]):
    """Speaker enrolment: 1..10 clips, >= 3 s of speech each. Stored as WAV (16 kHz mono, silence kept for review)."""
    from app.ml import speaker

    p, g, ident = _person(db, principal, person_id, "write")
    if g.kind != "speaker":
        raise ApiError(422, "not_speaker_project", "Voice clips are for speaker recognition projects; upload photos")
    if not 1 <= len(files) <= MAX_PHOTOS:
        raise ApiError(422, "invalid_photo_count", f"Send 1-{MAX_PHOTOS} clips per request")
    have = len(_photos(db, ident))
    accepted, rejected = [], []
    for f in files:
        name = (f.filename or "clip")[:200]
        data = await f.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            rejected.append(ClipRejected(filename=name, reason="file_too_large", detail="Clips can be at most 10 MB"))
            continue
        if have + len(accepted) >= MAX_PHOTOS:
            rejected.append(ClipRejected(filename=name, reason="too_many_photos",
                                         detail=f"A person can have at most {MAX_PHOTOS} clips; delete one first"))
            continue
        try:
            clip = speaker.analyse(data, speaker.MIN_ENROL_S)
        except speaker.AudioError as e:
            rejected.append(ClipRejected(filename=name, reason=e.reason, detail=e.detail))
            continue
        vec, _ = speaker.embed(clip.speech)
        row = m.FaceEmbedding(organization_id=ident.organization_id, identity_id=ident.id, gallery_id=g.id,
                              embedding=vec.tolist(), quality_score=clip.quality, duration_s=clip.seconds)
        db.add(row)
        db.flush()
        row.photo_object_key = f"{_photo_prefix(ident)}{row.id}.wav"
        storage.put_bytes(row.photo_object_key, _wav(speaker.decode(data)), "audio/wav")
        accepted.append(ClipAccepted(photo_id=row.id, filename=name, quality=clip.quality, seconds=clip.seconds))
    db.flush()
    if accepted:
        _refresh_template(db, ident)
        refresh_status(db, p)
        audit(db, request, principal, ident.organization_id, "speaker.clips_enrolled", "face_identity", ident.id,
              accepted=len(accepted), rejected=len(rejected))
    return ClipsOut(accepted=accepted, rejected=rejected, person=_person_out(db, g, ident))


def _wav(x) -> bytes:
    """16 kHz mono 16-bit PCM WAV (re-encoded: no metadata of the original file is kept)."""
    import wave

    import numpy as np

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()
