"""Ready-made detector (no training): torchvision's COCO-pretrained SSDlite320 with its stock 91-way head, filtered to
the project's chosen COCO classes. Same postprocess as custom detection (detection.tidy). Lazy + idle unload."""

from __future__ import annotations

import threading
import time

import structlog

from app.ml import idle

log = structlog.get_logger()
COCO = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train", 8: "truck",
        9: "boat", 10: "traffic light", 11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
        16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear",
        24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
        34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat",
        40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle", 46: "wine glass",
        47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple", 54: "sandwich",
        55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair",
        63: "couch", 64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet", 72: "tv", 73: "laptop",
        74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster",
        81: "sink", 82: "refrigerator", 84: "book", 85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
        89: "hair drier", 90: "toothbrush"}  # torchvision COCO_V1 label ids (91-way head, gaps are "N/A")
BY_NAME = {n.lower(): n for n in COCO.values()}
MODEL = {"id": "torchvision/ssdlite320_mobilenet_v3_large.coco_v1", "name": "SSDlite320 MobileNetV3 (COCO)",
         "licence": "BSD-3-Clause weights (torchvision); COCO annotations CC BY 4.0", "commercial_use": True}
DEFAULT_THRESHOLD = 0.5
_LOCK = threading.Lock()
_STATE: dict = {"model": None, "last_used": 0.0, "load_ms": 0}


def canonical(labels: list[str]) -> tuple[list[str], list[str]]:
    """(canonical names, unknown names): COCO names match case-insensitively."""
    names = [BY_NAME.get(str(x).strip().lower()) for x in labels]
    return [n for n in names if n], [x for x, n in zip(labels, names, strict=True) if not n]


def _model():
    """Call with _LOCK held."""
    if _STATE["model"] is None:
        t0 = time.perf_counter()
        from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large

        from app.ml.detection import configure_postprocess

        model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=91)
        model.load_state_dict(SSDLite320_MobileNet_V3_Large_Weights.COCO_V1.get_state_dict(progress=False))
        configure_postprocess(model)
        _STATE["model"], _STATE["load_ms"] = model.eval(), round((time.perf_counter() - t0) * 1000)
        log.info("pretrained_detector_loaded", load_ms=_STATE["load_ms"])
        idle.register("pretrained_detector", _unload_if_idle)
    _STATE["last_used"] = time.monotonic()
    return _STATE["model"]


def _unload_if_idle(now: float) -> int:
    from app.settings import get_settings

    with _LOCK:
        idle_for = now - _STATE["last_used"]
        if _STATE["model"] is None or idle_for < get_settings().predict_idle_unload_s:
            return 0
        _STATE["model"] = None
    log.info("pretrained_detector_unloaded", idle_s=round(idle_for))
    return 1


def weights_identity() -> dict:
    """The pinned stock weights (torchvision COCO_V1 URL + sha256 of the cached file): pseudo-labels are a
    deterministic function of these, so recording them makes a hybrid training run reproducible."""
    import hashlib
    from pathlib import Path

    from torch.hub import get_dir
    from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights

    url = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1.url
    path = Path(get_dir()) / "checkpoints" / url.rsplit("/", 1)[-1]
    sha = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return {"id": MODEL["id"], "url": url, "sha256": sha}


def loaded() -> bool:
    return _STATE["model"] is not None


def unload() -> None:
    with _LOCK:
        _STATE["model"] = None


def predict(img, classes: list[str], threshold: float) -> tuple[list[dict], int]:
    """(boxes, model_load_ms) for a PIL image; only `classes` (COCO names), at >= threshold."""
    import torch
    from torchvision.transforms.functional import to_tensor

    from app.ml.detection import tidy

    allowed = [i for i, n in COCO.items() if n in set(classes)]
    names = [COCO.get(i) for i in range(1, 91)]  # tidy maps label id k -> names[k - 1]
    with _LOCK:
        cold = _STATE["model"] is None
        model = _model()
        with torch.no_grad():
            det = model([to_tensor(img.convert("RGB"))])[0]
        load_ms = _STATE["load_ms"] if cold else 0
    keep = torch.isin(det["labels"], torch.tensor(allowed))  # filter BEFORE NMS: other classes can't suppress ours
    det = {k: v[keep] for k, v in det.items()}
    return tidy(det, names, threshold), load_ms
