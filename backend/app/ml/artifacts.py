"""ARTIFACT CONTRACT between training and evaluation / predict / export.

Every trained model version writes, under one org-scoped prefix in S3/MinIO:

  org/{org_id}/models/{model_id}/v{version_no}/
    model.json                 metadata below (training fills everything except metrics.test / eval)
    native/...                 framework checkpoint (see NATIVE)
    test_predictions.parquet   one row per TEST sample (schema below), written once after the final retrain
    training.json              trials, curves, best params, timings (informational)

Readers must use these helpers, never hand-built keys.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from app.services import storage

SCHEMA_VERSION = "1.0"

# native checkpoint layout per framework (paths relative to the version prefix)
NATIVE = {
    "lightgbm": ["native/model.txt"],               # lightgbm.Booster.save_model; categorical codes in model.json
    "timm": ["native/model.pt"],                    # torch state_dict; timm.create_model(architecture, num_classes)
    "transformers": ["native/hf/"],                 # save_pretrained() dir: config.json, model.safetensors, tokenizer
    "torchvision": ["native/model.pt"],             # torch state_dict of torchvision.models.detection.<architecture>
}


def prefix(org_id, model_id, version_no: int) -> str:
    return f"{storage.org_prefix(org_id)}models/{model_id}/v{version_no}/"


def key(org_id, model_id, version_no: int, rel: str) -> str:
    return prefix(org_id, model_id, version_no) + rel


# ---- model.json

def model_json(*, model_version_id, project_id, organization_id, model_id, version_no, task_type: str,
               labels: list[str], framework: str, architecture: str, base_model: dict, input_spec: dict,
               output_spec: dict, dataset_version_id, dataset_content_hash: str | None, spec_version: int,
               training: dict, val_metrics: dict, primary_metric: str) -> dict[str, Any]:
    """
    input_spec by modality:
      image:   {"modality": "image", "size": [H, W], "channels": 3, "color": "RGB", "resize": "bilinear",
                "normalize": {"mean": [..3], "std": [..3]}, "layout": "NCHW", "dtype": "float32", "scale": 255}
      text:    {"modality": "text", "tokenizer": "native/hf/", "max_length": 128}
      tabular: {"modality": "tabular", "features": [{"name", "type": numeric|categorical|boolean,
                "categories": [...] (categorical: code = index; unseen -> null)}], "target": str}
    output_spec:
      classification: {"type": "probabilities", "activation": "softmax", "classes": labels}
      regression:     {"type": "value", "target": str}
      detection:      {"type": "boxes", "format": "xywh_pixels", "score_threshold": 0.3, "classes": labels}
    metrics.test and eval_* are filled by evaluation - training leaves them null.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version_id": str(model_version_id),
        "model_id": str(model_id),
        "version_no": version_no,
        "project_id": str(project_id),
        "organization_id": str(organization_id),
        "task_type": task_type,
        "labels": labels,  # model output order; index i of probs == labels[i]
        "framework": framework,
        "architecture": architecture,
        "base_model": base_model,  # {registry_id, source, licence, commercial_use}
        "input": input_spec,
        "output": output_spec,
        "dataset_version_id": str(dataset_version_id),
        "dataset_content_hash": dataset_content_hash,
        "spec_version": spec_version,
        "training": training,  # {device, budget_mode, trials, best_params, seed, duration_s, library_versions}
        "metrics": {"primary_metric": primary_metric, "val": val_metrics, "test": None},
        "files": {"native": NATIVE[framework], "test_predictions": "test_predictions.parquet",
                  "training": "training.json"},
    }


def write_json(k: str, data: dict) -> None:
    storage.put_bytes(k, json.dumps(data, indent=2, default=str).encode(), "application/json")


def read_json(k: str) -> dict:
    return json.loads(storage.get_bytes(k))


# ---- test_predictions.parquet
# Columns (all tasks):
#   sample_id  string   samples.id; tabular: "row-<i>" (row index in the sealed manifest)
#   y_true     string   class name | str(float) for regression | JSON boxes [{class_name, bbox:{x,y,w,h}}]
#   y_pred     string   same encoding as y_true; detection boxes add "confidence"
#   probs      list<float> | null   classification: per class, ordered as model.json labels
#   confidence float | null         classification: max prob; detection: mean box confidence
PREDICTIONS_SCHEMA = pa.schema([
    ("sample_id", pa.string()), ("y_true", pa.string()), ("y_pred", pa.string()),
    ("probs", pa.list_(pa.float32())), ("confidence", pa.float32()),
])


def write_predictions(k: str, rows: list[dict]) -> None:
    table = pa.Table.from_pylist([{c: r.get(c) for c in PREDICTIONS_SCHEMA.names} for r in rows],
                                 schema=PREDICTIONS_SCHEMA)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    storage.put_bytes(k, buf.getvalue(), "application/vnd.apache.parquet")


def read_predictions(k: str):
    return pq.read_table(io.BytesIO(storage.get_bytes(k))).to_pandas()
