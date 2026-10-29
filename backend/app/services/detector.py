"""AI box suggestions (LBL-4a) behind an interface.

This slice ships a deterministic stub. The real backend (next slice) is an open-vocabulary detector with a commercial-
friendly licence (OWLv2, Apache-2.0) from the approved registry, and later the user's own trained model.
"""

from typing import Protocol


class BoxDetector(Protocol):
    model_id: str
    licence: str

    def suggest(self, width: int, height: int, classes: list[str], image_bytes: bytes) -> list[dict]:
        """[{class_name, bbox: {x, y, w, h} in pixels, confidence}]"""


class StubDetector:
    model_id = "stub-detector-v0"
    licence = "internal (no weights)"

    def suggest(self, width: int, height: int, classes: list[str], image_bytes: bytes) -> list[dict]:
        out = []
        for i, cls in enumerate(classes[:5]):
            w, h = max(width // 4, 1), max(height // 4, 1)
            x = min(i * w, max(width - w, 0))
            out.append({"class_name": cls, "bbox": {"x": float(x), "y": float(height // 3), "w": float(w),
                                                    "h": float(h)}, "confidence": round(0.5 - i * 0.05, 2)})
        return out


class OwnModelDetector:
    """LBL-4a: after the first trained model exists, suggestions come from the user's own detector."""

    licence = "project model"

    FLOOR = 0.3  # lower than the playground's 0.5: suggestions are proposals the user confirms

    def __init__(self, model_version_id, floor: float | None = None):
        self.floor = self.FLOOR if floor is None else floor
        self.model_version_id = model_version_id
        self.model_id = f"own:{model_version_id}"
        self.max_scores: dict = {}  # per suggest() call: {class: best score seen, even below the floor} (diagnosis)

    def suggest(self, width: int, height: int, classes: list[str], image_bytes: bytes) -> list[dict]:
        from app.ml.detection import SCORE_KEEP
        from app.ml.loader import load_model

        boxes = load_model(self.model_version_id).predict([image_bytes], threshold=SCORE_KEEP)[0]["boxes"]
        boxes = [b for b in boxes if not classes or b["class_name"] in classes]
        self.max_scores = {}
        for b in boxes:
            self.max_scores[b["class_name"]] = round(max(self.max_scores.get(b["class_name"], 0.0), b["confidence"]), 4)
        return [b for b in boxes if b["confidence"] >= self.floor]


class PretrainedDetector:
    """Before a project has its own detector, COCO classes get suggestions from the ready-made COCO SSDlite."""

    model_id = "torchvision/ssdlite320_mobilenet_v3_large.coco_v1"
    licence = "BSD-3-Clause (torchvision); COCO annotations CC BY 4.0"

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold

    def suggest(self, width: int, height: int, classes: list[str], image_bytes: bytes) -> list[dict]:
        import io

        from PIL import Image

        from app.ml import pretrained

        coco, _ = pretrained.canonical(classes)
        if not coco:
            return []
        boxes, _ = pretrained.predict(Image.open(io.BytesIO(image_bytes)), coco, self.threshold)
        wanted = {c.lower(): c for c in classes}  # report the project's own spelling of the class
        return [{**b, "class_name": wanted.get(b["class_name"].lower(), b["class_name"])} for b in boxes]


def detector(db=None, project=None) -> BoxDetector:
    if db is not None and project is not None:
        from sqlalchemy import select

        from app import models as m

        mv_id = db.scalar(select(m.ModelVersion.id).join(m.Model, m.Model.id == m.ModelVersion.model_id).where(
            m.Model.project_id == project.id, m.Model.organization_id == project.organization_id,
            m.ModelVersion.status == "ready", m.ModelVersion.framework == "torchvision")
            .order_by(m.ModelVersion.created_at.desc()).limit(1))
        if mv_id:
            return OwnModelDetector(mv_id)
        from app.ml.pretrained import canonical
        from app.services.specs import train_labels

        if canonical(train_labels(project.current_spec))[0]:
            return PretrainedDetector()
    return StubDetector()


OWL_MODELS = {  # text-prompt (open-vocabulary) detectors, both Apache-2.0: (transformers class prefix, pinned revision)
    "google/owlv2-base-patch16-ensemble": ("Owlv2", "cfd3195ba4ea9592eec887ded089f4c08eff231d"),
    "google/owlvit-base-patch32": ("OwlViT", "cbc355fb364588351c5d51c7f74465e8e7ec6f72"),
}


class TextPromptDetector:
    """Zero-shot boxes from a class NAME (or an English prompt per class): custom objects with no boxes drawn yet.
    Loaded only inside an auto-label job (train queue) and dropped with unload(): never resident in the API."""

    licence = "Apache-2.0"

    def __init__(self, model_id: str | None = None, prompts: dict[str, str] | None = None, floor: float = 0.3,
                 device: str = "cpu"):
        from app.settings import get_settings

        self.model_id = model_id or get_settings().autolabel_text_model
        self.prompts, self.floor, self.device, self._m = prompts or {}, floor, device, None

    def _load(self):
        if self._m is None:
            import transformers

            kind, rev = OWL_MODELS[self.model_id]
            proc = getattr(transformers, f"{kind}Processor").from_pretrained(self.model_id, revision=rev,
                                                                             local_files_only=True)
            model = getattr(transformers, f"{kind}ForObjectDetection").from_pretrained(
                self.model_id, revision=rev, local_files_only=True, use_safetensors=True)
            self._m = (proc, model.to(self.device).eval())
        return self._m

    def unload(self) -> None:
        import gc

        self._m = None
        gc.collect()
        if self.device == "mps":
            import torch

            torch.mps.empty_cache()  # give the GPU memory back now, not at process exit

    def suggest(self, width: int, height: int, classes: list[str], image_bytes: bytes) -> list[dict]:
        import io

        import torch
        from PIL import Image, ImageOps

        from app.ml.detection import tidy

        if not classes:
            return []
        proc, model = self._load()
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
        queries = [[f"a photo of a {self.prompts.get(c) or c}" for c in classes]]
        inputs = proc(text=queries, images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = model(**inputs)
        out = type(out)(**{k: (v.cpu() if hasattr(v, "cpu") else v) for k, v in out.items()})
        size = max(img.size)  # OWLv2 pads to a square: boxes come back relative to the padded side
        target = torch.tensor([[size, size]] if "Owlv2" in type(model).__name__ else [[img.height, img.width]])
        # transformers 5: the processors expose post_process_GROUNDED_object_detection (same scores / 0-based labels /
        # xyxy boxes); the old post_process_object_detection lives only on the image processor now
        r = proc.post_process_grounded_object_detection(out, threshold=0.01, target_sizes=target)[0]
        det = {"boxes": r["boxes"].clamp(min=0), "scores": r["scores"], "labels": r["labels"] + 1}  # tidy is 1-based
        boxes = tidy(det, classes, self.floor)  # the shared postprocess: class-wise NMS, floor, top-k
        for b in boxes:  # clip to the real image
            bb = b["bbox"]
            bb["w"], bb["h"] = min(bb["w"], img.width - bb["x"]), min(bb["h"], img.height - bb["y"])
        return boxes


def text_prompt_available(model_id: str | None = None) -> bool:
    from huggingface_hub import try_to_load_from_cache

    from app.settings import get_settings

    mid = model_id or get_settings().autolabel_text_model
    return isinstance(try_to_load_from_cache(mid, "config.json", revision=OWL_MODELS[mid][1]), str)
