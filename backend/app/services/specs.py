"""Task specs (PRD Appendix A): JSON Schema + semantic rules, and the per-category templates (PRJ-1)."""

import copy
from typing import Any

from jsonschema import Draft202012Validator

ENABLED_TASKS = ("image_classification", "object_detection", "text_classification",
                 "tabular_classification", "tabular_regression", "face_recognition", "speaker_recognition")
CLASSIFICATION_METRICS = ["accuracy", "macro_f1", "micro_f1", "weighted_f1", "roc_auc", "pr_auc",
                          "per_class_recall", "per_class_precision", "log_loss"]
METRICS: dict[str, list[str]] = {
    "image_classification": CLASSIFICATION_METRICS,
    "text_classification": CLASSIFICATION_METRICS,
    "tabular_classification": CLASSIFICATION_METRICS,
    "object_detection": ["map_50", "map_50_95", "per_class_ap"],
    "tabular_regression": ["rmse", "mae", "r2", "mape"],
    "face_recognition": ["tar_at_far", "rank1_accuracy"],  # from the gallery's leave-one-out report; never trained
    "speaker_recognition": ["tar_at_far", "rank1_accuracy"],
}
MODALITY = {"image_classification": "image", "object_detection": "image", "text_classification": "text",
            "tabular_classification": "tabular", "tabular_regression": "tabular", "face_recognition": "image",
            "speaker_recognition": "audio"}
NEEDS_LABELS = {"image_classification", "object_detection", "text_classification"}
PLATFORMS = ["android", "ios", "web", "server", "edge_linux"]
FORMATS = ["onnx", "tflite", "coreml", "tfjs", "native"]
QUANT = ["none", "fp16", "int8_dynamic", "int8_full"]
BUDGETS = ["quick", "standard", "thorough"]

SPEC_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["spec_version", "task_type", "input", "primary_metric", "constraints", "budget_mode", "split"],
    "properties": {
        "spec_version": {"const": "1.0"},
        "task_type": {"type": "string"},
        "multi_label": {"type": "boolean"},
        "labels": {"type": "array", "maxItems": 500, "items": {"type": "string", "minLength": 1, "maxLength": 100}},
        "input": {
            "type": "object",
            "additionalProperties": False,
            "required": ["modality"],
            "properties": {
                "modality": {"enum": ["image", "text", "tabular", "audio"]},
                "min_resolution": {"type": "integer", "minimum": 16, "maximum": 2048},
                "text_column": {"type": ["string", "null"]},
            },
        },
        "primary_metric": {"type": "string"},
        "secondary_metrics": {"type": "array", "items": {"type": "string"}},
        "constraints": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "target_platforms": {"type": "array", "items": {"enum": PLATFORMS}, "uniqueItems": True},
                "export_formats": {"type": "array", "items": {"enum": FORMATS}, "uniqueItems": True},
                "max_model_size_mb": {"type": ["number", "null"], "exclusiveMinimum": 0},
                "max_latency_ms": {"type": ["number", "null"], "exclusiveMinimum": 0},
                "latency_device": {"type": ["string", "null"]},
                "quantization": {"enum": QUANT},
            },
        },
        "budget_mode": {"enum": BUDGETS},
        "split": {
            "type": "object",
            "additionalProperties": False,
            "required": ["train", "val", "test", "seed"],
            "properties": {
                "train": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
                "val": {"type": "number", "minimum": 0, "exclusiveMaximum": 1},
                "test": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
                "stratify": {"type": "boolean"},
                "group_by": {"type": ["string", "null"]},
                "seed": {"type": "integer", "minimum": 0},
            },
        },
        "language": {"type": ["array", "null"], "items": {"type": "string"}},
        "notes": {"type": ["string", "null"], "maxLength": 2000},
        "target_column": {"type": ["string", "null"]},
        "feature_columns": {"type": ["array", "null"], "items": {"type": "string"}},
        "id_columns": {"type": ["array", "null"], "items": {"type": "string"}},
        "group_column": {"type": ["string", "null"]},
        "mode": {"enum": ["pretrained", "custom", "quick"]},  # detection: ready-made COCO; text: quick = examples
        "ready_made": {"type": "array", "items": {"enum": ["sentiment"]}, "uniqueItems": True},  # text add-ons
        "threshold": {"type": "number", "minimum": 0.05, "maximum": 0.95},  # quick text: "Other" below this
        "custom_labels": {"type": "array", "maxItems": 500,  # hybrid: the user's own objects on a pretrained project
                          "items": {"type": "string", "minLength": 1, "maxLength": 100}},
        "score_threshold": {"type": "number", "minimum": 0.05, "maximum": 0.95},
    },
}
_validator = Draft202012Validator(SPEC_SCHEMA)


def train_labels(spec: dict) -> list[str]:
    """The classes the project's data, label set, validation and training are about. A ready-made (pretrained)
    detection project trains only its custom objects; spec.labels there are the chosen COCO classes."""
    if spec.get("mode") == "pretrained":
        return list(spec.get("custom_labels") or [])
    return list(spec.get("labels") or [])


def _pointer(path) -> str:
    return "/" + "/".join(str(p) for p in path) if path else "/"


def validate_spec(spec: Any) -> list[dict[str, str]]:
    """Returns [] when valid, else [{loc: JSON pointer, msg}] (schema errors first, then semantic rules)."""
    if not isinstance(spec, dict):
        return [{"loc": "/", "msg": "Spec must be a JSON object"}]
    errors = [{"loc": _pointer(e.absolute_path), "msg": e.message}
              for e in sorted(_validator.iter_errors(spec), key=lambda e: list(e.absolute_path))]
    if errors:
        return errors
    task = spec["task_type"]
    if task not in ENABLED_TASKS:
        return [{"loc": "/task_type", "msg": f"'{task}' is not available; enabled: {list(ENABLED_TASKS)}"}]
    if spec["input"]["modality"] != MODALITY[task]:
        errors.append({"loc": "/input/modality", "msg": f"{task} uses modality '{MODALITY[task]}'"})
    if spec["primary_metric"] not in METRICS[task]:
        errors.append({"loc": "/primary_metric", "msg": f"Not valid for {task}; choose from {METRICS[task]}"})
    for i, sm in enumerate(spec.get("secondary_metrics", [])):
        if sm not in METRICS[task]:
            errors.append({"loc": f"/secondary_metrics/{i}", "msg": f"Not valid for {task}"})
    labels = spec.get("labels") or []
    if spec.get("mode", "custom") == "pretrained" and task != "object_detection":
        errors.append({"loc": "/mode", "msg": "Only object detection has a ready-made (pretrained) mode"})
    if spec.get("mode") == "quick" and task != "text_classification":
        errors.append({"loc": "/mode", "msg": "Quick (example-based) mode is for text classification"})
    if spec.get("ready_made") and task != "text_classification":
        errors.append({"loc": "/ready_made", "msg": "Ready-made add-ons are for text classification"})
    if spec.get("custom_labels") and spec.get("mode") != "pretrained":
        errors.append({"loc": "/custom_labels", "msg": "Custom objects are for ready-made detection projects"})
    if task in NEEDS_LABELS or labels:
        low = 1 if task == "object_detection" else 2  # a one-class detector (e.g. just "person") is valid
        sentiment_only = task == "text_classification" and not labels and "sentiment" in (spec.get("ready_made") or [])
        if task in NEEDS_LABELS and not sentiment_only and not low <= len(labels) <= 500:
            errors.append({"loc": "/labels", "msg": f"Between {low} and 500 labels are required"})
        if any(not lbl.strip() for lbl in labels):
            errors.append({"loc": "/labels", "msg": "Labels can't be blank"})
        if len({lbl.strip().lower() for lbl in labels}) != len(labels):
            errors.append({"loc": "/labels", "msg": "Labels must be unique (case-insensitive)"})
    if task == "tabular_regression" and labels:
        errors.append({"loc": "/labels", "msg": "Regression has no class labels"})
    split = spec["split"]
    if abs(split["train"] + split["val"] + split["test"] - 1.0) > 1e-6:
        errors.append({"loc": "/split", "msg": "train + val + test must equal 1.0"})
    return errors


def _base(task: str, primary: str, secondary: list[str], **extra) -> dict[str, Any]:
    spec = {
        "spec_version": "1.0",
        "task_type": task,
        "multi_label": False,
        "labels": [],
        "input": {"modality": MODALITY[task]},
        "primary_metric": primary,
        "secondary_metrics": secondary,
        "constraints": {"target_platforms": ["server"], "export_formats": ["onnx"], "max_model_size_mb": None,
                        "max_latency_ms": None, "quantization": "none"},
        "budget_mode": "quick",
        "split": {"train": 0.7, "val": 0.15, "test": 0.15, "stratify": True, "group_by": None, "seed": 42},
        "language": None,
        "notes": None,
    }
    spec.update(extra)
    return spec


TEMPLATES: list[dict[str, Any]] = [
    {"id": "image_classification", "category": "image", "name": "Image classification",
     "description": "Sort photos into classes (defect / no defect, plant disease, product type).", "enabled": True,
     "task_types": ["image_classification"],
     "default_spec": _base("image_classification", "macro_f1", ["accuracy", "per_class_recall"],
                           input={"modality": "image", "min_resolution": 224})},
    {"id": "object_detection", "category": "image", "name": "Object detection",
     "description": "Draw boxes around objects (helmets, vehicles, shelf items).", "enabled": True,
     "task_types": ["object_detection"],
     "default_spec": _base("object_detection", "map_50", ["map_50_95", "per_class_ap"],
                           input={"modality": "image", "min_resolution": 640})},
    {"id": "text_classification", "category": "text", "name": "Text / Sentiment",
     "description": "Route tickets, detect spam, or score sentiment (Indic and Hinglish supported).", "enabled": True,
     "task_types": ["text_classification"],
     "default_spec": _base("text_classification", "macro_f1", ["accuracy"],
                           input={"modality": "text", "text_column": None}, language=["en"])},
    {"id": "tabular", "category": "tabular", "name": "Tabular (classification / regression)",
     "description": "Predict a column from the others: churn, fraud, price, ETA.", "enabled": True,
     "task_types": ["tabular_classification", "tabular_regression"],
     "default_spec": _base("tabular_classification", "macro_f1", ["accuracy", "roc_auc"],
                           target_column=None, feature_columns=None, id_columns=None, group_column=None)},
    {"id": "face_recognition", "category": "image", "name": "Face recognition",
     "description": "Recognise enrolled, consenting people; strangers come back as Unknown. No training: add or "
                    "remove a person instantly.", "enabled": True, "task_types": ["face_recognition"],
     "default_spec": _base("face_recognition", "tar_at_far", ["rank1_accuracy"])},
    {"id": "speaker_recognition", "category": "audio", "name": "Speaker recognition",
     "description": "Recognise enrolled, consenting people by voice; strangers come back as Unknown. No training.",
     "enabled": True, "task_types": ["speaker_recognition"],
     "default_spec": _base("speaker_recognition", "tar_at_far", ["rank1_accuracy"])},
    *[{"id": tid, "category": cat, "name": name, "description": "Coming soon", "enabled": False,
       "task_types": [tid], "default_spec": None}
      for tid, cat, name in [
          ("audio_classification", "audio", "Audio classification"),
          ("keyword_spotting", "audio", "Keyword spotting"),
          ("named_entity_extraction", "text", "Named entity extraction"),
          ("aspect_sentiment", "text", "Aspect-based sentiment"),
          ("forecasting", "time_series", "Forecasting"),
          ("anomaly_detection", "tabular", "Anomaly detection"),
          ("segmentation", "image", "Segmentation"),
      ]],
]
_BY_ID = {t["id"]: t for t in TEMPLATES}

REGRESSION_DEFAULT = _base("tabular_regression", "rmse", ["mae", "r2"], target_column=None, feature_columns=None,
                           id_columns=None, group_column=None)
REGRESSION_DEFAULT["split"]["stratify"] = False


def template(template_id: str) -> dict[str, Any] | None:
    return _BY_ID.get(template_id)


def default_spec_for_task(task_type: str) -> dict[str, Any]:
    if task_type == "tabular_regression":
        return copy.deepcopy(REGRESSION_DEFAULT)
    for t in TEMPLATES:
        if t["enabled"] and task_type in t["task_types"]:
            spec = copy.deepcopy(t["default_spec"])
            spec["task_type"] = task_type
            return spec
    raise KeyError(task_type)


def category_of(task_type: str) -> str:
    return {"image_classification": "image_classification", "object_detection": "object_detection",
            "text_classification": "text_classification"}.get(task_type, "tabular")
