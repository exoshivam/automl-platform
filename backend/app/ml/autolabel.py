"""Bulk auto-label (detection): one Celery job on the TRAIN queue (-c 1, never the API process) proposes boxes for a
draft's photos and stores them as BoxSuggestion rows, NOT labels. Detectors per class, cheapest that knows it:
own trained model -> ready-made COCO -> text prompt (OWL). All reuse services/detector.py + detection.tidy, and
keep_pseudo drops anything overlapping a box the user already drew. Models load lazily and are freed at the end.
"""

import time
import uuid

import structlog
from sqlalchemy import exists, select

from app import models as m
from app.db import now, session_factory
from app.worker import celery

log = structlog.get_logger()


def plan(db, project: m.Project, classes: list[str], source: str) -> dict[str, str]:
    """{class: source}. "auto" picks per class: own_model if the latest trained detector knows it, else ready_made for
    COCO classes, else text_prompt."""
    from app.ml.pretrained import canonical
    from app.services.detector import detector

    own = detector(db, project)
    own_classes = set()
    if hasattr(own, "model_version_id"):
        own_classes = set(db.scalar(select(m.ModelVersion.label_classes).where(
            m.ModelVersion.id == own.model_version_id)) or [])
    out = {}
    for c in classes:
        if source != "auto":
            out[c] = source
        elif c in own_classes:
            out[c] = "own_model"
        elif canonical([c])[0]:
            out[c] = "ready_made"
        else:
            out[c] = "text_prompt"
    return out


def targets(db, v: m.DatasetVersion, sample_ids: list[str] | None) -> list[m.Sample]:
    q = select(m.Sample).where(m.Sample.dataset_version_id == v.id, m.Sample.organization_id == v.organization_id,
                               m.Sample.object_key.is_not(None))
    if sample_ids is not None:
        q = q.where(m.Sample.id.in_([uuid.UUID(s) for s in sample_ids]))
    else:  # "unlabelled": no box yet and not marked 'no objects'
        q = q.where(m.Sample.no_objects.is_(False), ~exists().where(
            m.Label.sample_id == m.Sample.id, m.Label.organization_id == v.organization_id, m.Label.bbox.is_not(None)))
    return list(db.scalars(q.order_by(m.Sample.id)))


def detectors_for(db, project: m.Project, by_source: dict[str, list[str]], prompts: dict, floor: float) -> dict:
    from app.services.detector import OwnModelDetector, PretrainedDetector, TextPromptDetector, detector

    out = {}
    if by_source.get("own_model"):
        own = detector(db, project)
        out["own_model"] = OwnModelDetector(own.model_version_id, floor=floor)
    if by_source.get("ready_made"):
        out["ready_made"] = PretrainedDetector(threshold=floor)
    if by_source.get("text_prompt"):
        import torch

        from app.settings import get_settings

        dev = get_settings().autolabel_device
        dev = dev if dev == "cpu" or torch.backends.mps.is_available() else "cpu"
        out["text_prompt"] = TextPromptDetector(prompts=prompts, floor=floor, device=dev)
    return out


@celery.task(name="ml.autolabel")
def autolabel(job_id: str) -> None:
    from app.ml.detection import keep_pseudo
    from app.services import storage

    with session_factory()() as db:
        job = db.scalar(select(m.AutolabelJob).where(m.AutolabelJob.id == uuid.UUID(job_id)))
        if job is None or job.status not in ("queued", "running"):
            return
        v = db.scalar(select(m.DatasetVersion).where(m.DatasetVersion.id == job.dataset_version_id))
        ds = db.scalar(select(m.Dataset).where(m.Dataset.id == v.dataset_id))
        project = db.scalar(select(m.Project).where(m.Project.id == ds.project_id))
        routes = plan(db, project, job.classes, job.source)
        by_source: dict[str, list[str]] = {}
        for c, src in routes.items():
            by_source.setdefault(src, []).append(c)
        samples = targets(db, v, job.sample_ids)
        job.status, job.progress = "running", {"done": 0, "total": len(samples), "plan": routes}
        db.commit()
        dets = {}
        t0 = time.perf_counter()
        try:
            dets = detectors_for(db, project, by_source, job.prompts or {}, job.min_confidence)
            results = dict(job.results or {})
            for i, s in enumerate(samples):
                db.refresh(job)
                if job.cancel_requested:
                    job.status = "cancelled"
                    break
                data = storage.get_bytes(s.object_key)
                found = []
                for src, det in dets.items():
                    for b in det.suggest(s.width or 0, s.height or 0, by_source[src], data):
                        found.append((src, det.model_id, b))
                mine = [lb.bbox for lb in db.scalars(select(m.Label).where(
                    m.Label.sample_id == s.id, m.Label.organization_id == s.organization_id,
                    m.Label.bbox.is_not(None)))]
                kept = keep_pseudo([b for _, _, b in found], mine)  # never duplicate a box the user drew
                n = 0
                for src, model_id, b in found:
                    if b in kept and b["confidence"] >= job.min_confidence:
                        db.add(m.BoxSuggestion(organization_id=s.organization_id, job_id=job.id,
                                               dataset_version_id=v.id, sample_id=s.id, class_name=b["class_name"],
                                               bbox=b["bbox"], confidence=round(float(b["confidence"]), 4),
                                               source=src, model_id=model_id))
                        n += 1
                results[str(s.id)] = n
                job.results = results
                job.progress = {**job.progress, "done": i + 1}
                db.commit()  # progress + suggestions visible as we go (and kept if cancelled)
            else:
                job.status = "done"
        except Exception as e:  # noqa: BLE001
            log.warning("autolabel_failed", job_id=job_id, error=str(e)[:300])
            db.rollback()
            job = db.scalar(select(m.AutolabelJob).where(m.AutolabelJob.id == uuid.UUID(job_id)))
            job.status, job.error = "failed", "Auto-labelling stopped: " + (str(e)[:200] or type(e).__name__)
        finally:
            for det in dets.values():  # free the heavy ones now, not at the idle timeout
                if hasattr(det, "unload"):
                    det.unload()
        job.finished_at = now()
        job.progress = {**(job.progress or {}), "seconds": round(time.perf_counter() - t0, 1)}
        db.commit()
        log.info("autolabel_done", job_id=job_id, status=job.status, progress=job.progress)


@celery.task(name="ml.owl_benchmark")
def owl_benchmark(n_images: int = 15, devices: tuple[str, ...] = ("cpu", "mps")) -> dict:
    """One-off model choice (train queue): both OWL models on COCO replay images WITHOUT people, prompted with the
    images' own non-person classes. Recall@IoU0.5 at floor 0.3, mAP@0.5, latency/img, RSS growth. Result -> Redis."""
    import gc
    import json
    import random

    import psutil

    from app.services import replay, storage
    from app.services.detector import OWL_MODELS, TextPromptDetector
    from app.services.ratelimit import client as redis

    rows, _ = replay.load()
    rows = [r for r in rows if not any(b["class_name"] == "person" for b in r["boxes"])]
    random.Random(5).shuffle(rows)
    rows = rows[:n_images]
    imgs = [(r, storage.get_bytes(r["key"])) for r in rows]
    proc = psutil.Process()
    out = {"images": len(imgs), "classes": sorted({b["class_name"] for r in rows for b in r["boxes"]}), "runs": []}
    for model_id in OWL_MODELS:
        for dev in devices:
            gc.collect()
            rss0 = proc.memory_info().rss
            det = TextPromptDetector(model_id=model_id, floor=0.05, device=dev)
            try:
                run = _bench_one(det, imgs, out["classes"], proc, rss0)
            except Exception as e:  # noqa: BLE001  (one bad run must not lose the others)
                run = {"error": f"{type(e).__name__}: {e}"[:300]}
            out["runs"].append({"model": model_id, "device": dev, **run})
            det.unload()
            del det
            gc.collect()
            if dev == "mps":
                import torch

                torch.mps.empty_cache()
            log.info("owl_benchmark_run", **out["runs"][-1])
    redis().set("bench:owl", json.dumps(out), ex=7 * 86400)
    return out


def _bench_one(det, imgs, classes_all, proc, rss0) -> dict:
    from app.ml.detection import _iou, map50

    t0 = time.perf_counter()
    det._load()
    load_s = time.perf_counter() - t0
    peak, lat, gts, preds = proc.memory_info().rss, [], {}, {}
    for r, data in imgs:
        classes = sorted({b["class_name"] for b in r["boxes"]})
        t1 = time.perf_counter()
        boxes = det.suggest(r["width"], r["height"], classes, data)
        lat.append(time.perf_counter() - t1)
        peak = max(peak, proc.memory_info().rss)
        gts[r["key"]] = [(b["class_name"], (b["bbox"]["x"], b["bbox"]["y"], b["bbox"]["w"], b["bbox"]["h"]))
                         for b in r["boxes"]]
        preds[r["key"]] = [(b["class_name"], (b["bbox"]["x"], b["bbox"]["y"], b["bbox"]["w"], b["bbox"]["h"]),
                            b["confidence"]) for b in boxes]
    ap = map50(gts, preds, classes_all)["map_50"]
    hits = total = 0
    for k, g in gts.items():
        for c, gb in g:
            total += 1
            hits += any(pc == c and sc >= 0.3 and _iou(gb, pb) >= 0.5 for pc, pb, sc in preds[k])
    return {"load_s": round(load_s, 1), "latency_ms_per_img": round(1000 * sum(lat[1:]) / max(1, len(lat) - 1)),
            "first_img_ms": round(1000 * lat[0]), "map50": round(ap, 3),
            "recall_at_0.3": round(hits / max(1, total), 3), "gt_boxes": total,
            "rss_growth_mb": round((peak - rss0) / 1e6)}
