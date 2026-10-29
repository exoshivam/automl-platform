"""Face pipeline for the API (PRD §12.2). The pure pipeline lives in app/ml/face_core.py (standalone, shipped verbatim
in the face export); this module binds it to the configured InsightFace pack. Nothing is ever trained per customer.

Licence: InsightFace model packs (buffalo_l / buffalo_s) are NON-COMMERCIAL research only (PRD §12.4).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import structlog
from PIL import Image

from app.ml import face_core, idle
from app.ml.face_core import ARCFACE_112, LICENCE, Face, _similarity, align, blur_score, load_image  # noqa: F401
from app.settings import get_settings

_LOCK = threading.Lock()
_STATE: dict = {"sessions": None, "last_used": 0.0, "load_ms": 0}
log = structlog.get_logger()


PACKS = {"accurate": "buffalo_l", "small": "buffalo_s"}  # the app itself always runs "accurate"
PACK_LABELS = {"accurate": "Accurate (~182 MB)", "small": "Small (~16 MB)"}


def pack_dir(pack: str = "accurate") -> Path:
    d = Path(get_settings().face_model_dir).expanduser()
    return d if pack == "accurate" else d.parent / PACKS[pack]


def model_files(pack: str = "accurate") -> tuple[Path, Path, str]:
    d = pack_dir(pack)
    det = next(iter(sorted(d.glob("det_*.onnx"))), None)
    rec = next(iter(sorted(d.glob("w600k_*.onnx"))), None)
    if det is None or rec is None:
        raise FileNotFoundError(f"InsightFace detector/recognizer ONNX files not found in {d}")
    return det, rec, d.name


def model_info(pack_name: str = "accurate") -> dict:
    try:
        det, rec, pack = model_files(pack_name)
    except FileNotFoundError:
        det = rec = None
        pack = pack_dir(pack_name).name
    return {"embedding_model": {"id": f"insightface-{pack}-arcface", "name": f"ArcFace ({rec.name if rec else '?'})",
                                "licence": LICENCE, "commercial_use": False},
            "detector": {"id": f"insightface-{pack}-scrfd", "name": f"SCRFD ({det.name if det else '?'})",
                         "licence": LICENCE, "commercial_use": False}}


def _sessions():
    """Loaded on first use (call with _LOCK held); freed by the idle reaper after FACE_IDLE_UNLOAD_S."""
    if _STATE["sessions"] is None:
        import onnxruntime as ort

        t0 = time.perf_counter()
        det, rec, _ = model_files()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        _STATE["sessions"] = (ort.InferenceSession(str(det), opts, providers=["CPUExecutionProvider"]),
                              ort.InferenceSession(str(rec), opts, providers=["CPUExecutionProvider"]))
        _STATE["load_ms"] = round((time.perf_counter() - t0) * 1000)
        log.info("face_models_loaded", load_ms=_STATE["load_ms"])
        idle.register("face", _unload_if_idle)
    _STATE["last_used"] = time.monotonic()
    return _STATE["sessions"]


def _unload_if_idle(now: float) -> int:
    with _LOCK:
        idle_for = now - _STATE["last_used"]
        if _STATE["sessions"] is None or idle_for < get_settings().face_idle_unload_s:
            return 0
        _STATE["sessions"] = None
    log.info("face_models_unloaded", idle_s=round(idle_for))
    return 1


def loaded() -> bool:
    return _STATE["sessions"] is not None


def analyse(img: Image.Image) -> list[Face]:
    return analyse_timed(img)[0]


def analyse_timed(img: Image.Image) -> tuple[list[Face], int]:
    """(faces, model_load_ms): load_ms > 0 only when this call had to (re)load the models."""
    with _LOCK:  # one inference at a time per process (memory-safe on a laptop)
        cold = _STATE["sessions"] is None
        faces = face_core.analyse(*_sessions(), img)
        return faces, (_STATE["load_ms"] if cold else 0)


def analyse_with(pack: str, images: list[bytes]) -> list[list[Face]]:
    """Export-only: run another pack (e.g. buffalo_s) over some images, then FREE it. Never cached: the app keeps
    serving with the accurate pack, so the small one must not stay resident (16 GB laptop)."""
    import gc

    import onnxruntime as ort

    det, rec, _ = model_files(pack)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    d = ort.InferenceSession(str(det), opts, providers=["CPUExecutionProvider"])
    r = ort.InferenceSession(str(rec), opts, providers=["CPUExecutionProvider"])
    try:
        out = []
        for data in images:
            try:
                out.append(face_core.analyse(d, r, load_image(data)))
            except Exception:  # noqa: BLE001  (an unreadable stored photo just counts as not detected)
                out.append([])
        return out
    finally:
        del d, r
        gc.collect()
        log.info("face_pack_unloaded", pack=pack)
