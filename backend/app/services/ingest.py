"""Ingestion (DAT-1/2/4/8/10/11, NFR-SEC6). Runs in Celery workers; never extracts archives to disk.

Every object: streamed once -> SHA-256 (verified against the client's value) -> stored content-addressed
(dedup per dataset) -> sniffed by content, not extension -> image size checked BEFORE any decode -> thumbnail + pHash.
"""

import hashlib
import io
import json
import stat
import tempfile
import warnings
import zipfile
from collections import Counter
from pathlib import PurePosixPath
from typing import Any, BinaryIO

import imagehash
import structlog
from PIL import Image, UnidentifiedImageError
from pillow_heif import register_heif_opener
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import models as m
from app.db import now
from app.services import storage
from app.services.project_status import refresh_status
from app.settings import get_settings

log = structlog.get_logger()
register_heif_opener()
IGNORED = ("__MACOSX/", ".DS_Store", "Thumbs.db", "desktop.ini")
CHUNK = 1024 * 1024
TEXT_EXT = {".txt"}


class IngestFatal(Exception):
    """The whole upload is rejected (bad checksum, zip bomb)."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code, self.detail = code, detail


def safe_relpath(name: str) -> str | None:
    """Normalised relative POSIX path, or None for anything that could escape (absolute, .., drive, NUL)."""
    if not name or "\x00" in name or len(name) > 1024:
        return None
    name = name.replace("\\", "/")
    if name.startswith("/") or (len(name) > 1 and name[1] == ":"):
        return None
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return None
    return "/".join(parts)


def _spool_with_hash(src: BinaryIO, limit: int) -> tuple[tempfile.SpooledTemporaryFile, str, int]:
    """Copies src into a temp file (memory up to 32 MB) while hashing; refuses more than `limit` bytes."""
    h, n = hashlib.sha256(), 0
    out = tempfile.SpooledTemporaryFile(max_size=32 * CHUNK)  # noqa: SIM115  (caller owns and closes it)
    while chunk := src.read(CHUNK):
        n += len(chunk)
        if n > limit:
            out.close()
            raise ValueError("size_limit")
        h.update(chunk)
        out.write(chunk)
    out.seek(0)
    return out, h.hexdigest(), n


def _sniff_image(fp, sample_meta: dict[str, Any]) -> tuple[Image.Image | None, dict[str, Any]]:
    """Header-only open first: a 20000x20000 PNG is rejected without allocating 1.6 GB (S6-4)."""
    s = get_settings()
    Image.MAX_IMAGE_PIXELS = None  # our own check below; Pillow's would raise before we can record it
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = Image.open(fp)
            w, h = img.size
            info = {"format": img.format, "width": w, "height": h, "mode": img.mode}
            if w * h > s.max_image_pixels:
                sample_meta["corrupt"] = "too_many_pixels"
                return None, info
            img.load()  # full decode: catches truncation
            return img, info
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        sample_meta["corrupt"] = "unreadable" if isinstance(e, UnidentifiedImageError) else "truncated_or_invalid"
        return None, {}


MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "BMP": "image/bmp", "TIFF": "image/tiff",
        "HEIF": "image/heic", "GIF": "image/gif"}
EXT = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp", "TIFF": ".tif", "HEIF": ".heic", "GIF": ".gif"}


class VersionBuilder:
    """Accumulates samples for one new dataset version."""

    def __init__(self, db: Session, version: m.DatasetVersion, dataset: m.Dataset, modality: str, upload_id=None,
                 source: str | None = None, capture_target: str | None = None):
        self.db, self.version, self.dataset, self.modality = db, version, dataset, modality
        self.upload_id = str(upload_id) if upload_id else None  # capture/upload batch -> diversity check
        self.source = source  # files | folder | zip | camera | annotations
        self.capture_target = capture_target  # detection hint: which object the user was photographing
        self.org = version.organization_id
        self.errors: list[dict[str, Any]] = []
        self.samples_by_name: dict[str, m.Sample] = {}
        self.id_map: dict[Any, Any] = {}  # parent sample id -> new sample id

    def copy_parent(self, parent_id) -> None:
        """New versions are cumulative: parent samples + labels are carried over (rows only; objects are shared)."""
        parents = self.db.scalars(select(m.Sample).where(m.Sample.dataset_version_id == parent_id,
                                                         m.Sample.organization_id == self.org)).all()
        labels = self.db.scalars(select(m.Label).where(m.Label.sample_id.in_([p.id for p in parents]),
                                                       m.Label.organization_id == self.org)).all() if parents else []
        by_sample: dict[Any, list[m.Label]] = {}
        for lb in labels:
            by_sample.setdefault(lb.sample_id, []).append(lb)
        for p in parents:
            s = clone_sample(p, self.version.id)
            self.db.add(s)
            self.db.flush()
            self.id_map[p.id] = s.id
            for lb in by_sample.get(p.id, []):
                self.db.add(clone_label(lb, s.id))
            if p.original_name:
                self.samples_by_name[p.original_name] = s

    def add_object(self, src: BinaryIO, rel: str, label: str | None, declared_sha: str | None = None,
                   raw_key: str | None = None) -> m.Sample | None:
        s = get_settings()
        try:
            spool, sha, size = _spool_with_hash(src, s.max_file_bytes)
        except ValueError:
            self.errors.append({"file": rel, "code": "file_too_large"})
            return None
        if declared_sha and declared_sha.lower() != sha:
            spool.close()
            raise IngestFatal("checksum_mismatch", f"SHA-256 of '{rel}' doesn't match the declared value")
        with spool:
            meta: dict[str, Any] = {k: v for k, v in (("upload_id", self.upload_id), ("source", self.source),
                                                      ("capture_target", self.capture_target)) if v}
            sample = m.Sample(organization_id=self.org, dataset_version_id=self.version.id, original_name=rel,
                              sha256=sha, bytes=size, meta=meta)
            suffix = PurePosixPath(rel).suffix.lower()
            if self.modality == "text" and suffix in TEXT_EXT:
                raw = spool.read(1_000_001)
                try:
                    meta["text"] = raw[:1_000_000].decode("utf-8")
                except UnicodeDecodeError:
                    meta["corrupt"] = "not_utf8"
                sample.mime = "text/plain"
                ext = ".txt"
            else:
                img, info = _sniff_image(spool, meta)
                fmt = info.get("format")
                sample.mime = MIME.get(fmt, "application/octet-stream")
                sample.width, sample.height = info.get("width"), info.get("height")
                if info.get("mode"):
                    meta["mode"] = info["mode"]
                ext = EXT.get(fmt, suffix if len(suffix) <= 6 else "")
                if size == 0:
                    meta["corrupt"] = "zero_bytes"
                if img is not None:
                    try:
                        sample.phash = str(imagehash.phash(img))
                        thumb = img.convert("RGB")
                        thumb.thumbnail((256, 256))
                        buf = io.BytesIO()
                        thumb.save(buf, "WEBP", quality=80)
                        sample.thumbnail_key = storage.thumb_key(self.org, self.dataset.id, sha)
                        storage.put_bytes(sample.thumbnail_key, buf.getvalue(), "image/webp")
                    except Exception:  # noqa: BLE001
                        meta["corrupt"] = "truncated_or_invalid"
            key = storage.object_key(self.org, self.dataset.id, sha, ext)
            if storage.head(key) is None:  # dedup: identical bytes are stored once per dataset
                spool.seek(0)
                storage.client().upload_fileobj(spool, storage.bucket(), key,
                                                ExtraArgs={"ContentType": sample.mime})
            sample.object_key = key
        if raw_key:
            storage.delete(raw_key)
        self.db.add(sample)
        self.db.flush()
        if label:
            self.db.add(m.Label(organization_id=self.org, sample_id=sample.id, class_name=label, source="import"))
        self.samples_by_name[rel] = sample
        return sample

    def add_zip(self, raw_key: str, label_from_folder: bool, fixed_label: str | None) -> None:
        s = get_settings()
        with tempfile.TemporaryFile() as tmp:
            body = storage.get_stream(raw_key)
            n = 0
            while chunk := body.read(CHUNK):
                n += len(chunk)
                if n > s.max_file_bytes:
                    raise IngestFatal("file_too_large", "Archive is larger than the upload limit")
                tmp.write(chunk)
            tmp.seek(0)
            try:
                zf = zipfile.ZipFile(tmp)
            except zipfile.BadZipFile as e:
                raise IngestFatal("bad_archive", "The file is not a valid zip archive") from e
            with zf:
                infos = zf.infolist()
                if len(infos) > s.max_zip_entries:
                    raise IngestFatal("zip_too_many_entries", f"Archive has more than {s.max_zip_entries} entries")
                declared = sum(i.file_size for i in infos)
                compressed = max(sum(i.compress_size for i in infos), 1)
                if declared > s.max_zip_uncompressed_bytes or declared / compressed > s.max_zip_ratio:
                    raise IngestFatal("zip_bomb", "Archive expands too much (zip bomb protection)")
                budget = s.max_zip_uncompressed_bytes
                for info in infos:
                    name = info.filename
                    if info.is_dir() or name.startswith(IGNORED[0]) or PurePosixPath(name).name in IGNORED:
                        continue
                    rel = safe_relpath(name)
                    mode = info.external_attr >> 16
                    if rel is None:
                        self.errors.append({"file": name, "code": "path_traversal"})
                        continue
                    if stat.S_ISLNK(mode):
                        self.errors.append({"file": name, "code": "symlink"})
                        continue
                    if info.flag_bits & 0x1:
                        self.errors.append({"file": name, "code": "encrypted"})
                        continue
                    if stat.S_IFMT(mode) not in (0, stat.S_IFREG):  # type bits set and not a regular file
                        self.errors.append({"file": name, "code": "special_file"})
                        continue
                    label = fixed_label or (folder_label(rel) if label_from_folder else None)
                    try:
                        with zf.open(info) as member:
                            self.add_object(_Capped(member, min(info.file_size, budget)), rel, label)
                    except (zipfile.BadZipFile, ValueError, OSError) as e:
                        self.errors.append({"file": name, "code": "bad_entry", "detail": str(e)[:200]})
                        continue
                    budget -= info.file_size
        storage.delete(raw_key)


class _Capped(io.RawIOBase):
    """Reads at most `cap` bytes from a zip member; a member that lies about its size can't inflate past it."""

    def __init__(self, inner, cap: int):
        self.inner, self.left = inner, cap

    def read(self, n: int = -1) -> bytes:
        if self.left <= 0:
            if self.inner.read(1):
                raise ValueError("entry larger than declared")
            return b""
        data = self.inner.read(min(n if n > 0 else CHUNK, self.left))
        self.left -= len(data)
        return data


def folder_label(rel: str) -> str | None:
    parent = PurePosixPath(rel).parent.name
    return parent or None


def clone_sample(p: m.Sample, version_id) -> m.Sample:
    return m.Sample(organization_id=p.organization_id, dataset_version_id=version_id, object_key=p.object_key,
                    original_name=p.original_name, sha256=p.sha256, phash=p.phash, mime=p.mime, bytes=p.bytes,
                    width=p.width, height=p.height, duration=p.duration, split=None, no_objects=p.no_objects,
                    thumbnail_key=p.thumbnail_key, meta=dict(p.meta or {}))


def clone_label(lb: m.Label, sample_id) -> m.Label:
    return m.Label(organization_id=lb.organization_id, sample_id=sample_id, label_set_id=lb.label_set_id,
                   class_name=lb.class_name, bbox=lb.bbox, source=lb.source, confidence=lb.confidence,
                   labeled_by=lb.labeled_by, reviewed_by=lb.reviewed_by)


def refresh_stats(db: Session, version: m.DatasetVersion) -> None:
    """label_stats = {class: {images, boxes}}; sample_count."""
    org = version.organization_id
    samples = db.scalars(select(m.Sample.id).where(m.Sample.dataset_version_id == version.id,
                                                   m.Sample.organization_id == org)).all()
    if version.table_uri and not samples:
        return  # tabular: counts set by the import
    rows = db.execute(select(m.Label.sample_id, m.Label.class_name, m.Label.bbox).join(
        m.Sample, m.Sample.id == m.Label.sample_id).where(m.Sample.dataset_version_id == version.id,
                                                          m.Label.organization_id == org)).all()
    images: dict[str, set] = {}
    boxes: Counter = Counter()
    for sid, cls, bbox in rows:
        images.setdefault(cls, set()).add(sid)
        if bbox:
            boxes[cls] += 1
    version.label_stats = {c: {"images": len(ids), **({"boxes": boxes[c]} if boxes[c] else {})}
                           for c, ids in sorted(images.items())}
    version.sample_count = len(samples)


def content_hash(db: Session, version: m.DatasetVersion) -> str:
    rows = db.execute(select(m.Sample.sha256, m.Sample.split, m.Label.class_name, m.Label.bbox)
                      .outerjoin(m.Label, m.Label.sample_id == m.Sample.id)
                      .where(m.Sample.dataset_version_id == version.id,
                             m.Sample.organization_id == version.organization_id)).all()
    h = hashlib.sha256()
    for r in sorted(str(x) for x in rows):
        h.update(r.encode())
    if version.table_uri:
        h.update(version.table_uri.encode())
        h.update(json.dumps(version.column_mapping, sort_keys=True).encode())
    return h.hexdigest()


def finish(db: Session, version: m.DatasetVersion, dataset: m.Dataset, errors: list[dict], upload_id=None) -> None:
    refresh_stats(db, version)
    version.ingest_errors = [*(version.ingest_errors or []), *errors]
    others = db.scalar(select(m.Upload.id).where(m.Upload.dataset_version_id == version.id,
                                                 m.Upload.status == "ingesting", m.Upload.id != upload_id).limit(1)) \
        if upload_id else None
    version.ingest_status = "running" if others else "ready"  # a draft is ready when no upload is still ingesting
    dataset.current_version_id = version.id
    project = db.scalar(select(m.Project).where(m.Project.id == dataset.project_id,
                                                m.Project.organization_id == dataset.organization_id))
    db.flush()
    refresh_status(db, project)
    version.updated_at = now()
