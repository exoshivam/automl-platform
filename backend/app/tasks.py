"""Celery tasks: ingestion, imports, seal (split + manifest + validation)."""

import hashlib
import io
import uuid
from pathlib import PurePosixPath

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import structlog
from sqlalchemy import select

from app import models as m
from app.db import now, session_factory
from app.services import annotations, splits, storage, tables
from app.services.ingest import (
    IngestFatal,
    VersionBuilder,
    clone_label,
    clone_sample,
    content_hash,
    finish,
    folder_label,
    refresh_stats,
    safe_relpath,
)
from app.services.project_status import refresh_status
from app.worker import celery

log = structlog.get_logger()
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif", ".gif"}


def _load(db, model, row_id):
    return db.scalar(select(model).where(model.id == uuid.UUID(str(row_id))))


def _fail(version_id, upload_id, code: str, detail: str) -> None:
    """A failed upload never destroys a draft that already holds data: only a brand-new version is marked failed."""
    with session_factory()() as db:
        v = _load(db, m.DatasetVersion, version_id)
        up = _load(db, m.Upload, upload_id) if upload_id else None
        if up is not None:
            up.status, up.error = "failed", f"{code}: {detail}"
        if up is not None and (up.options or {}).get("append"):
            v.ingest_errors = [*(v.ingest_errors or []), {"code": code, "detail": detail, "upload_id": str(up.id)}]
            db.flush()
            if not _other_ingesting(db, v.id, up.id):
                v.ingest_status = "ready"
        else:
            v.ingest_status, v.ingest_errors = "failed", [{"code": code, "detail": detail}]
        db.commit()
    log.warning("ingest_failed", version_id=str(version_id), code=code, detail=detail)


def _other_ingesting(db, version_id, upload_id) -> bool:
    return bool(db.scalar(select(m.Upload.id).where(m.Upload.dataset_version_id == version_id,
                                                    m.Upload.status == "ingesting", m.Upload.id != upload_id).limit(1)))


def _context(db, version_id):
    v = _load(db, m.DatasetVersion, version_id)
    ds = db.scalar(select(m.Dataset).where(m.Dataset.id == v.dataset_id, m.Dataset.organization_id == v.organization_id))
    project = db.scalar(select(m.Project).where(m.Project.id == ds.project_id,
                                                m.Project.organization_id == v.organization_id))
    return v, ds, project


@celery.task(name="ingest_upload")
def ingest_upload(upload_id: str) -> None:
    with session_factory()() as db:
        up = _load(db, m.Upload, upload_id)
        v, ds, project = _context(db, up.dataset_version_id)
        v.ingest_status = "running"
        db.commit()
        try:
            opts = up.options or {}
            b = VersionBuilder(db, v, ds, ds.modality, upload_id=up.id, source=opts.get("source"),
                               capture_target=opts.get("capture_target"))
            if v.parent_version_id and not opts.get("append"):
                b.copy_parent(v.parent_version_id)
            detection = project.task_type == "object_detection"
            for f in up.files:
                rel = safe_relpath(f.get("relative_path") or f["name"]) or safe_relpath(f["name"]) or f"file_{f['key'][-6:]}"
                from_folder = opts.get("label_from_folder") and not detection
                label = None if detection else (f.get("label") or opts.get("label") or
                                                (folder_label(rel) if from_folder else None))
                if rel.lower().endswith(".zip"):
                    b.add_zip(f["key"], bool(from_folder), None if detection else (f.get("label") or opts.get("label")))
                else:
                    b.add_object(storage.get_stream(f["key"]), rel, label, f.get("sha256"), raw_key=f["key"])
            finish(db, v, ds, b.errors, upload_id=up.id)
            up.status, up.completed_at = "completed", now()
            db.commit()
        except IngestFatal as e:
            db.rollback()
            _fail(v.id, up.id, e.code, e.detail)


@celery.task(name="import_table")
def import_table(version_id: str, key: str, fmt: str, columns: list[dict]) -> None:
    with session_factory()() as db:
        v, ds, project = _context(db, version_id)
        v.ingest_status = "running"
        db.commit()
        try:
            df = tables.read_table(key, fmt)
        except Exception as e:  # noqa: BLE001
            _fail(version_id, None, "unreadable_table", str(e)[:300])
            return
        data = tables.to_parquet(df, columns)
        sha = hashlib.sha256(data).hexdigest()
        v.table_uri = storage.table_key(v.organization_id, ds.id, sha)
        storage.put_bytes(v.table_uri, data, "application/vnd.apache.parquet")
        target = next(c["name"] for c in columns if c["role"] == "target")
        v.column_mapping = {"format": fmt, "columns": columns, "target": target,
                            "text_column": next((c["name"] for c in columns if c["role"] == "text"), None),
                            "id_columns": [c["name"] for c in columns if c["role"] == "id"],
                            "group_column": next((c["name"] for c in columns if c["role"] == "group"), None)}
        v.sample_count = len(df)
        if project.task_type == "text_classification":
            text_col = v.column_mapping["text_column"]
            for i, row in df.iterrows():
                text = None if pd.isna(row[text_col]) else str(row[text_col])
                s = m.Sample(organization_id=v.organization_id, dataset_version_id=v.id, original_name=f"row-{i}",
                             sha256=hashlib.sha256((text or "").encode()).hexdigest(), mime="text/plain",
                             bytes=len((text or "").encode()), meta={"text": text, "row": int(i)})
                db.add(s)
                db.flush()
                if not pd.isna(row[target]):
                    db.add(m.Label(organization_id=v.organization_id, sample_id=s.id, class_name=str(row[target]),
                                   source="import"))
            refresh_stats(db, v)
        else:
            counts = df[target].astype(str).value_counts().to_dict() if project.task_type == "tabular_classification" \
                else {}
            v.label_stats = {str(k): {"rows": int(n)} for k, n in sorted(counts.items())}
        finish(db, v, ds, [])
        v.sample_count = len(df)
        db.commit()


@celery.task(name="import_detection")
def import_detection(version_id: str, upload_id: str, fmt: str, key: str) -> None:
    with session_factory()() as db:
        up = _load(db, m.Upload, upload_id)
        v, ds, _project = _context(db, version_id)
        v.ingest_status = "running"
        db.commit()
        files = {(safe_relpath(f.get("relative_path") or f["name"]) or f["name"]): f for f in up.files}
        images = {n for n in files if PurePosixPath(n).suffix.lower() in IMAGE_EXT}
        boxes, errors, _ = parse_annotations(fmt, key, files, images)
        try:
            b = VersionBuilder(db, v, ds, "image", upload_id=up.id, source=(up.options or {}).get("source"))
            if v.parent_version_id and not (up.options or {}).get("append"):
                b.copy_parent(v.parent_version_id)
            for name in sorted(images):
                f = files[name]
                sample = b.add_object(storage.get_stream(f["key"]), name, None, f.get("sha256"), raw_key=f["key"])
                for box in boxes.get(name, []) if sample else []:
                    db.add(m.Label(organization_id=v.organization_id, sample_id=sample.id, class_name=box["class_name"],
                                   bbox=annotations.to_pixels(box, sample.width, sample.height), source="import"))
            for name, f in files.items():
                if name not in images:
                    storage.delete(f["key"])
            finish(db, v, ds, b.errors + errors, upload_id=up.id)
            up.status, up.completed_at = "completed", now()
            db.commit()
        except IngestFatal as e:
            db.rollback()
            _fail(v.id, up.id, e.code, e.detail)


def parse_annotations(fmt: str, key: str, files: dict[str, dict], images: set[str]):
    """Used by the import endpoint (to 422 early) and by the task."""
    if fmt == "coco":
        return annotations.parse_coco(storage.get_bytes(key), images)
    if fmt == "yolo":
        labels = {n: storage.get_bytes(f["key"]) for n, f in files.items()
                  if n.lower().endswith(".txt") and f["key"] != key}
        return annotations.parse_yolo(storage.get_bytes(key), labels, images)
    xmls = {n: storage.get_bytes(f["key"]) for n, f in files.items() if n.lower().endswith(".xml")}
    return annotations.parse_voc(xmls, images)


def _near_dup_clusters(samples) -> dict:
    """Union-find over near-duplicate pairs (VAL-7 auto-fix keeps them in one split)."""
    from app.services.validation import _phash_pairs

    parent = {s.id: s.id for s in samples}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_sha: dict = {}
    for s in samples:
        if s.sha256 in by_sha:
            parent[find(s.id)] = find(by_sha[s.sha256])
        by_sha.setdefault(s.sha256, s.id)
    for a, b in _phash_pairs(samples):
        parent[find(a)] = find(b)
    return {s.id: f"c:{find(s.id)}" for s in samples}


@celery.task(name="seal_version")
def seal_version(version_id: str) -> None:
    """Split (DAT-6), write the manifest (DAT-5), hash, then validate (VAL)."""
    from app.services import validation

    with session_factory()() as db:
        v, ds, project = _context(db, version_id)
        sp = project.current_spec["split"]
        stratify = sp.get("stratify", True) and project.task_type != "tabular_regression"
        if v.table_uri and project.task_type != "text_classification":
            df = pd.read_parquet(io.BytesIO(storage.get_bytes(v.table_uri)))
            cm = v.column_mapping or {}
            group_col = sp.get("group_by") or cm.get("group_column")
            target = cm.get("target")
            items = [(i, str(df.at[i, target]) if stratify and target else None,
                      f"g:{df.at[i, group_col]}" if group_col else f"r:{i}") for i in range(len(df))]
            assignment, summary = splits.assign(items, sp["train"], sp["val"], sp["test"], sp["seed"], stratify)
            df["__split__"] = [assignment[i] for i in range(len(df))]
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            v.manifest_uri = storage.manifest_key(v.organization_id, ds.id, v.version_no)
            storage.put_bytes(v.manifest_uri, buf.getvalue(), "application/vnd.apache.parquet")
            v.split_strategy = "grouped" if group_col else ("stratified" if stratify else "random")
        else:
            samples = db.scalars(select(m.Sample).where(m.Sample.dataset_version_id == v.id,
                                                        m.Sample.organization_id == v.organization_id)).all()
            labels = db.execute(select(m.Label.sample_id, m.Label.class_name).join(
                m.Sample, m.Sample.id == m.Label.sample_id).where(m.Sample.dataset_version_id == v.id,
                                                                  m.Label.organization_id == v.organization_id)
                                .order_by(m.Label.class_name)).all()
            first: dict = {}
            for sid, cls in labels:
                first.setdefault(sid, cls)
            group_by = sp.get("group_by")
            cluster = _near_dup_clusters(samples) if (v.options or {}).get("group_near_duplicates") else {}
            items = [(s.id, first.get(s.id) if stratify else None,
                      f"g:{(s.meta or {}).get(group_by)}" if group_by and (s.meta or {}).get(group_by) is not None
                      else cluster.get(s.id) or f"h:{s.sha256}") for s in samples]
            assignment, summary = splits.assign(items, sp["train"], sp["val"], sp["test"], sp["seed"], stratify)
            for s in samples:
                s.split = assignment.get(s.id)
            table = pa.table({"sample_id": [str(s.id) for s in samples], "object_key": [s.object_key for s in samples],
                              "sha256": [s.sha256 for s in samples], "split": [s.split for s in samples],
                              "label": [first.get(s.id) for s in samples]})
            buf = io.BytesIO()
            pq.write_table(table, buf)
            v.manifest_uri = storage.manifest_key(v.organization_id, ds.id, v.version_no)
            storage.put_bytes(v.manifest_uri, buf.getvalue(), "application/vnd.apache.parquet")
            v.split_strategy = "grouped" if group_by else ("stratified" if stratify else "random")
        v.split_seed = sp["seed"]
        v.split_summary = summary
        db.flush()
        v.content_hash = content_hash(db, v)
        v.sealed_at = now()
        v.validation_status = "running"
        db.flush()
        refresh_status(db, project)
        db.commit()
        validation.run(db, v, project)
        db.commit()


@celery.task(name="apply_fixes")
def apply_fixes(child_id: str, parent_id: str, fix_ids: list[str]) -> None:
    """Builds the fixed child version from the parent, then seals and validates it."""
    with session_factory()() as db:
        child, ds, project = _context(db, child_id)
        parent = _load(db, m.DatasetVersion, parent_id)
        org = child.organization_id
        if parent.table_uri and project.task_type != "text_classification":
            df = pd.read_parquet(io.BytesIO(storage.get_bytes(parent.table_uri)))
            cm = dict(parent.column_mapping or {})
            cols = [dict(c) for c in cm.get("columns", [])]
            if "exclude_id_columns" in fix_ids:
                id_cols = next((it["details"].get("columns", []) for it in parent.validation_report["items"]
                                if it["code"] == "VAL-10"), [])
                for c in cols:
                    if c["name"] in id_cols:
                        c["role"] = "id"
            if "dedupe_rows" in fix_ids:
                df = df.drop_duplicates(subset=[c["name"] for c in cols if c["role"] != "id"] or None)
            cm["columns"] = cols
            cm["id_columns"] = [c["name"] for c in cols if c["role"] == "id"]
            data = tables.to_parquet(df.reset_index(drop=True), cols)
            child.table_uri = storage.table_key(org, ds.id, hashlib.sha256(data).hexdigest())
            storage.put_bytes(child.table_uri, data, "application/vnd.apache.parquet")
            child.column_mapping, child.sample_count, child.label_stats = cm, len(df), parent.label_stats
        else:
            samples = db.scalars(select(m.Sample).where(m.Sample.dataset_version_id == parent.id,
                                                        m.Sample.organization_id == org).order_by(m.Sample.created_at)).all()
            seen: set = set()
            keep = []
            for s in samples:
                if "exclude_corrupt" in fix_ids and (s.meta or {}).get("corrupt"):
                    continue
                if "dedupe_exact" in fix_ids:
                    if s.sha256 in seen:
                        continue
                    seen.add(s.sha256)
                keep.append(s)
            labels = db.scalars(select(m.Label).where(m.Label.sample_id.in_([s.id for s in keep]),
                                                      m.Label.organization_id == org)).all() if keep else []
            by_sample: dict = {}
            for lb in labels:
                by_sample.setdefault(lb.sample_id, []).append(lb)
            for s in keep:
                c = clone_sample(s, child.id)
                c.meta = {k: v for k, v in (c.meta or {}).items() if k != "flags"}
                db.add(c)
                db.flush()
                for lb in by_sample.get(s.id, []):
                    db.add(clone_label(lb, c.id))
            db.flush()
            refresh_stats(db, child)
        if "resplit_groups" in fix_ids:
            child.options = {**(child.options or {}), "group_near_duplicates": True}
        child.ingest_status = "ready"
        ds.current_version_id = child.id
        db.commit()
    seal_version(child_id)


@celery.task(name="gc_objects")
def gc_objects(org_id: str, keys: list[str], unconditional: bool = False) -> int:
    """Deletes S3 objects no remaining row in the org references (content-addressed blobs are shared)."""
    from sqlalchemy import or_

    removed = 0
    prefix = storage.org_prefix(org_id)
    with session_factory()() as db:
        for key in keys:
            if not key.startswith(prefix):  # never touch another tenant's objects
                continue
            if not unconditional:
                in_use = db.scalar(select(m.Sample.id).where(
                    m.Sample.organization_id == uuid.UUID(org_id),
                    or_(m.Sample.object_key == key, m.Sample.thumbnail_key == key)).limit(1)) or db.scalar(
                    select(m.DatasetVersion.id).where(m.DatasetVersion.organization_id == uuid.UUID(org_id),
                                                      or_(m.DatasetVersion.manifest_uri == key,
                                                          m.DatasetVersion.table_uri == key)).limit(1))
                if in_use:
                    continue
            storage.delete(key)
            removed += 1
    log.info("gc_objects", org_id=org_id, candidates=len(keys), removed=removed)
    return removed


PURGE_WAIT_RETRIES = 60  # x 10 s: running jobs get ~10 min to notice the cancel key


def _purge_mlflow(project_id: str) -> None:
    """Best effort: deletes the project's MLflow experiment (it only holds params/metrics, never images)."""
    from app.settings import get_settings

    uri = get_settings().mlflow_tracking_uri
    if not uri:
        return
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(uri)
        exp = client.get_experiment_by_name(f"project-{project_id}")
        if exp is not None and exp.lifecycle_stage != "deleted":
            client.delete_experiment(exp.experiment_id)
    except Exception as e:  # noqa: BLE001
        log.warning("purge_mlflow_failed", project_id=project_id, error=str(e)[:200])


@celery.task(name="purge_project")
def purge_project(project_id: str, waited: int = 0) -> dict:
    """PERMANENT delete of a project: S3 prefixes, MLflow experiment, then one DELETE that cascades every row.

    Idempotent/resumable: everything is recomputed from rows that are deleted last, so a re-run after a crash
    finishes the job; a project that is already gone is a no-op. Blobs are content-addressed per dataset, so the
    dataset/model/upload prefixes belong to this project alone.
    """
    from sqlalchemy import delete

    from app.ml.train_tasks import cancel_key
    from app.services.ratelimit import client as redis

    pid = uuid.UUID(project_id)
    with session_factory()() as db:
        p = db.scalar(select(m.Project).where(m.Project.id == pid))
        if p is None:
            return {"project_id": project_id, "status": "gone"}
        org = p.organization_id
        active = db.scalars(select(m.TrainingJob).where(m.TrainingJob.project_id == pid,
                                                        m.TrainingJob.organization_id == org,
                                                        m.TrainingJob.status.in_(("queued", "running")))).all()
        wait = bool(active) and waited < PURGE_WAIT_RETRIES
        for j in active:
            redis().set(cancel_key(j.id), "1", ex=24 * 3600)
    if wait:  # re-enqueue (after the session is closed) rather than block a worker slot
        purge_project.apply_async(args=[project_id], kwargs={"waited": waited + 1}, countdown=10)
        return {"project_id": project_id, "status": "waiting_for_jobs"}
    with session_factory()() as db:
        p = db.scalar(select(m.Project).where(m.Project.id == pid, m.Project.organization_id == org))
        if p is None:
            return {"project_id": project_id, "status": "gone"}
        for j in db.scalars(select(m.TrainingJob).where(  # the worker never confirmed (crashed?): stop waiting
                m.TrainingJob.project_id == pid, m.TrainingJob.organization_id == org,
                m.TrainingJob.status.in_(("queued", "running")))):
            j.status, j.finished_at, j.error_code = "cancelled", now(), "project_deleted"
        db.flush()
        root = storage.org_prefix(org)
        prefixes = [f"{root}datasets/{i}/" for i in db.scalars(select(m.Dataset.id).where(
                        m.Dataset.project_id == pid, m.Dataset.organization_id == org))]
        prefixes += [f"{root}models/{i}/" for i in db.scalars(select(m.Model.id).where(
                        m.Model.project_id == pid, m.Model.organization_id == org))]
        prefixes += [f"{root}uploads/{i}/" for i in db.scalars(select(m.Upload.id).where(
                        m.Upload.project_id == pid, m.Upload.organization_id == org))]
        prefixes += [f"{root}faces/{i}/" for i in db.scalars(select(m.FaceGallery.id).where(  # biometric photos
                        m.FaceGallery.project_id == pid, m.FaceGallery.organization_id == org))]
        # keys a row holds outside those prefixes (should be none): GC'd afterwards only if nothing else uses them
        stray = {k for k in db.scalars(select(m.Sample.object_key).join(
            m.DatasetVersion, m.DatasetVersion.id == m.Sample.dataset_version_id).join(
            m.Dataset, m.Dataset.id == m.DatasetVersion.dataset_id).where(
            m.Dataset.project_id == pid, m.Sample.organization_id == org)) if k and not k.startswith(tuple(prefixes))}
        objects = sum(storage.delete_prefix(pre) for pre in prefixes)
        _purge_mlflow(project_id)
        name = p.name
        # deployment keys would survive the cascade (api_keys.deployment_id -> SET NULL): revoke them explicitly
        revoked = 0
        for k in db.scalars(select(m.ApiKey).join(m.Deployment, m.Deployment.id == m.ApiKey.deployment_id).where(
                m.Deployment.project_id == pid, m.ApiKey.organization_id == org, m.ApiKey.revoked_at.is_(None))):
            k.revoked_at = now()
            revoked += 1
        db.flush()
        db.execute(delete(m.Project).where(m.Project.id == pid, m.Project.organization_id == org))  # rows cascade
        from app.deps import audit

        audit(db, None, None, org, "project.purged", "project", pid, name=name, objects=objects,
              prefixes=len(prefixes), keys_revoked=revoked)
        db.commit()
    if stray:
        gc_objects(str(org), sorted(stray))
    log.info("purge_project", project_id=project_id, objects=objects)
    return {"project_id": project_id, "status": "purged", "objects": objects}


@celery.task(name="import_starter")
def import_starter(upload_id: str) -> None:
    """DAT-13: server-side copy of a starter into the upload's prefix, then the SAME ingest as a user upload/import."""
    from app.services import starters

    with session_factory()() as db:
        up = _load(db, m.Upload, upload_id)
        sid, prefix = up.options["starter_id"], storage.org_prefix(up.organization_id) + f"uploads/{up.id}/"
        try:
            s, man = starters.get(sid), starters.manifest(sid)
            if s is None:
                raise IngestFatal("starter_missing", f"Starter '{sid}' is no longer in the catalog")
            if s["kind"] == "table":
                key = f"{prefix}data.{man['format']}"
                starters.copy(man["key"], key)
                up.files = [{"key": key, "name": f"{sid}.{man['format']}", "size": s["size_bytes"]}]
            else:
                files = []
                for i, f in enumerate(man["files"]):
                    key = f"{prefix}{i:06d}"
                    starters.copy(f["key"], key)
                    files.append({"key": key, "name": f["relative_path"].rsplit("/", 1)[-1], "size": f["size"],
                                  "relative_path": f["relative_path"], "sha256": f.get("sha256"),
                                  "label": f.get("label")})
                if s["kind"] == "detection":
                    key = f"{prefix}annotations.json"
                    starters.copy(man["annotations"], key)
                    files.append({"key": key, "name": "annotations.json", "relative_path": "annotations.json",
                                  "size": 0})
                up.files = files
        except IngestFatal as e:
            db.rollback()
            _fail(up.dataset_version_id, up.id, e.code, e.detail)
            return
        except Exception as e:  # noqa: BLE001
            db.rollback()
            _fail(up.dataset_version_id, up.id, "starter_copy_failed", str(e)[:300])
            return
        vid, kind, first = str(up.dataset_version_id), s["kind"], up.files[0]["key"]
        db.commit()
    if kind == "files":
        ingest_upload(upload_id)
    elif kind == "detection":
        import_detection(vid, upload_id, "coco", f"{prefix}annotations.json")
    else:
        import_table(vid, first, man["format"], man["columns"])
        with session_factory()() as db:
            up = _load(db, m.Upload, upload_id)
            v = _load(db, m.DatasetVersion, vid)
            up.status, up.completed_at = ("failed", None) if v.ingest_status == "failed" else ("completed", now())
            db.commit()
