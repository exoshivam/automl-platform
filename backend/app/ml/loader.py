"""load_model(model_version_id) -> LoadedModel with .meta (model.json) and .predict(batch).

Inputs per task: image -> bytes | PIL.Image; text -> str; tabular -> dict {column: value}.
Outputs per task: classification {label, confidence, probs: {label: p}}; regression {value};
detection {boxes: [{class_name, bbox, confidence}]}.
Inference runs on CPU (the API process must not grab MPS memory). Heavy libraries are imported lazily.
"""

from __future__ import annotations

import io
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any

import structlog
from sqlalchemy import select

from app import models as m
from app.db import session_factory
from app.ml import artifacts, idle
from app.services import storage

log = structlog.get_logger()
_CACHE: OrderedDict[str, LoadedModel] = OrderedDict()
_USED: dict[str, float] = {}
_LOCK = threading.Lock()
CACHE_SIZE = 2  # models held in memory per process


class ModelNotReady(Exception):
    pass


class LoadedModel:
    def __init__(self, meta: dict[str, Any], prefix: str):
        self.meta, self.prefix = meta, prefix
        self.labels: list[str] = meta["labels"]
        self.temperature: float = float((meta.get("output") or {}).get("temperature") or 1.0)  # EVL-4 calibration

    def predict(self, batch: list[Any]) -> list[dict[str, Any]]:
        raise NotImplementedError

    def _classify(self, probs) -> list[dict[str, Any]]:
        out = []
        for p in probs:
            p = [float(x) for x in p]
            k = max(range(len(p)), key=p.__getitem__)
            out.append({"label": self.labels[k], "confidence": p[k], "probs": dict(zip(self.labels, p, strict=True))})
        return out


class TimmModel(LoadedModel):
    def __init__(self, meta, prefix):
        super().__init__(meta, prefix)
        import timm
        import torch

        self.torch = torch
        self.model = timm.create_model(meta["architecture"], pretrained=False, num_classes=len(self.labels))
        state = torch.load(io.BytesIO(storage.get_bytes(prefix + "native/model.pt")), map_location="cpu",
                           weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        inp = meta["input"]
        from torchvision import transforms as T

        self.tf = T.Compose([T.Resize(tuple(inp["size"])), T.ToTensor(),
                             T.Normalize(inp["normalize"]["mean"], inp["normalize"]["std"])])

    def predict(self, batch):
        from PIL import Image

        imgs = [(Image.open(io.BytesIO(x)) if isinstance(x, bytes | bytearray) else x).convert("RGB") for x in batch]
        from app.ml.common import softmax_t

        with self.torch.no_grad():
            logits = self.model(self.torch.stack([self.tf(i) for i in imgs])).numpy()
        return self._classify(softmax_t(logits, self.temperature))


class LightGBMModel(LoadedModel):
    """Served from native/model.onnx with onnxruntime: LightGBM itself is never imported in long-lived processes
    (its OpenMP runtime clashes with torch's on macOS)."""

    def __init__(self, meta, prefix):
        super().__init__(meta, prefix)
        import onnxruntime as ort

        self.session = ort.InferenceSession(storage.get_bytes(prefix + "native/model.onnx"),
                                            providers=["CPUExecutionProvider"])
        self.features = meta["input"]["features"]
        self.regression = meta["output"]["type"] == "value"

    def predict(self, batch):
        import pandas as pd

        from app.ml.tabular import _prepare

        X = _prepare(pd.DataFrame([{f["name"]: row.get(f["name"]) for f in self.features} for row in batch]),
                     self.features).to_numpy(dtype="float32")
        outputs = self.session.run(None, {"input": X})
        if self.regression:
            return [{"value": float(v)} for v in outputs[0].reshape(-1)]
        import numpy as np

        from app.ml.common import softmax_t

        probs = softmax_t(np.log(np.clip(outputs[1], 1e-12, 1)), self.temperature)  # [label, probabilities]
        return self._classify(probs)



class TorchvisionDetector(LoadedModel):
    """SSDlite320-MobileNetV3 (see app/ml/detection.py for the native layout). Boxes above score_threshold."""

    def __init__(self, meta, prefix):
        super().__init__(meta, prefix)
        import torch

        from app.ml.detection import build

        self.torch = torch
        self.model = build(len(self.labels) + 1, False)
        self.model.load_state_dict(torch.load(io.BytesIO(storage.get_bytes(prefix + "native/model.pt")),
                                              map_location="cpu", weights_only=True))
        self.model.eval()
        self.threshold = meta["output"].get("score_threshold", 0.3)

    def predict(self, batch, threshold: float | None = None):
        from PIL import Image
        from torchvision.transforms.functional import to_tensor

        from app.ml.detection import SCORE_FLOOR, configure_postprocess, tidy

        configure_postprocess(self.model)  # also for models saved before the postprocess fix
        floor = SCORE_FLOOR if threshold is None else threshold
        imgs = [(Image.open(io.BytesIO(x)) if isinstance(x, bytes | bytearray) else x).convert("RGB") for x in batch]
        with self.torch.no_grad():
            dets = self.model([to_tensor(i) for i in imgs])
        return [{"boxes": tidy(det, self.labels, floor)} for det in dets]


LOADERS = {"timm": TimmModel, "lightgbm": LightGBMModel, "torchvision": TorchvisionDetector}


def register(framework: str, cls) -> None:
    """Text / detection loaders register here when their trainers land."""
    LOADERS[framework] = cls


def load_model(model_version_id: uuid.UUID | str) -> LoadedModel:
    return load_model_timed(model_version_id)[0]


def _unload_idle(now: float) -> int:
    from app.settings import get_settings

    limit = get_settings().predict_idle_unload_s
    with _LOCK:
        stale = [k for k in _CACHE if now - _USED.get(k, 0) >= limit]
        for k in stale:
            _CACHE.pop(k)
            _USED.pop(k, None)
    for k in stale:
        log.info("predict_model_unloaded", model_version_id=k)
    return len(stale)


def load_model_timed(model_version_id: uuid.UUID | str) -> tuple[LoadedModel, int]:
    """(model, load_ms): load_ms is 0 on a cache hit, else how long this call spent loading (after an idle unload or
    on first use). Models unused for PREDICT_IDLE_UNLOAD_S are freed by app.ml.idle."""
    key = str(model_version_id)
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            _USED[key] = time.monotonic()
            return _CACHE[key], 0
    t0 = time.perf_counter()
    with session_factory()() as db:
        mv = db.scalar(select(m.ModelVersion).where(m.ModelVersion.id == uuid.UUID(key)))
        if mv is None or mv.status != "ready":
            raise ModelNotReady(f"model version {key} is not ready")
        prefix = mv.artifact_prefix
    meta = artifacts.read_json(prefix + "model.json")
    if meta["framework"] == "transformers":
        import app.ml.text  # noqa: F401  (registers TextModel)
    model = LOADERS[meta["framework"]](meta, prefix)
    load_ms = round((time.perf_counter() - t0) * 1000)
    with _LOCK:
        _CACHE[key] = model
        _USED[key] = time.monotonic()
        while len(_CACHE) > CACHE_SIZE:
            _USED.pop(_CACHE.popitem(last=False)[0], None)
    log.info("predict_model_loaded", model_version_id=key, load_ms=load_ms)
    idle.register("predict", _unload_idle)
    return model, load_ms
