"""ONNX export + verification (PRD EXP-1/2/3) and integration snippets.

Bundle (zip) at <version prefix>exports/onnx-<quantization>.zip:
  model.onnx, model.json, labels.txt, MODEL_CARD.md (when evaluated), README.md (snippets)
Verification (EXP-3) runs the native model and the ONNX model on the same inputs (up to VERIFY_N test samples):
max |prob diff|, top-1 agreement, and accuracy on those samples for both. TFLite / Core ML: Phase 2.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import uuid
import zipfile

import numpy as np
import structlog
from sqlalchemy import select

from app import models as m
from app.db import session_factory
from app.ml import artifacts
from app.services import storage

log = structlog.get_logger()
VERIFY_N = 32
MAX_DIFF = 1e-3
MIN_AGREEMENT = 0.99


class ExportError(Exception):
    pass


# ---------------------------------------------------------------- inputs

def image_tensor(data: bytes, spec: dict) -> np.ndarray:
    """Preprocess one image exactly as input_spec says -> float32 [3, H, W]."""
    from PIL import Image

    h, w = spec["size"]
    img = Image.open(io.BytesIO(data)).convert("RGB").resize((w, h), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / spec.get("scale", 255)
    norm = spec.get("normalize") or {}
    if norm:
        x = (x - np.array(norm["mean"], dtype=np.float32)) / np.array(norm["std"], dtype=np.float32)
    return x.transpose(2, 0, 1).astype(np.float32)


def _test_inputs(db, mv: m.ModelVersion, meta: dict) -> tuple[list, list[str | None]]:
    """Up to VERIFY_N test-set inputs (image bytes / text) and their true labels."""
    try:
        df = artifacts.read_predictions(mv.artifact_prefix + "test_predictions.parquet").head(VERIFY_N)
    except Exception:  # noqa: BLE001
        return [], []
    ids = [uuid.UUID(s) for s in df.sample_id if not str(s).startswith("row-")]
    samples = {str(s.id): s for s in db.scalars(select(m.Sample).where(
        m.Sample.id.in_(ids), m.Sample.organization_id == mv.organization_id))}
    xs, ys = [], []
    for r in df.itertuples():
        s = samples.get(str(r.sample_id))
        if not s:
            continue
        if meta["input"]["modality"] == "text":
            xs.append((s.meta or {}).get("text", ""))
        elif s.object_key:
            xs.append(storage.get_bytes(s.object_key))
        else:
            continue
        ys.append(r.y_true)
    return xs, ys


def _download_file(mv: m.ModelVersion, rel: str, into: str) -> None:
    dst = os.path.join(into, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if storage.head(mv.artifact_prefix + rel):
        with open(dst, "wb") as f:
            f.write(storage.get_bytes(mv.artifact_prefix + rel))


def _download_native(mv: m.ModelVersion, meta: dict, into: str) -> None:
    for rel in meta.get("files", {}).get("native") or artifacts.NATIVE[meta["framework"]]:
        keys = storage.existing_sizes(mv.artifact_prefix + rel) if rel.endswith("/") else {mv.artifact_prefix + rel: 0}
        for k in keys:
            if not rel.endswith("/") and not storage.head(k):
                continue  # e.g. lightgbm serves native/model.onnx; model.txt may be absent
            dst = os.path.join(into, k[len(mv.artifact_prefix):])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(storage.get_bytes(k))


# ---------------------------------------------------------------- converters
# Each returns (native_fn, onnx_feed_fn, output_is_probs, export_fn). The native reference runs BEFORE export_fn:
# tracing can leave a model altered, which would make a broken export "verify" against itself.

def _timm(meta: dict, d: str, out: str):
    import timm
    import torch

    model = timm.create_model(meta["architecture"], pretrained=False, num_classes=len(meta["labels"]))
    model.load_state_dict(torch.load(os.path.join(d, "native/model.pt"), map_location="cpu", weights_only=True))
    model.eval()
    h, w = meta["input"]["size"]
    def do_export():
        _torch_export(model, (torch.zeros(1, 3, h, w),), out, ["input"], {"input": {0: "batch"}, "probs": {0: "batch"}})

    def batch(xs):
        return np.stack([image_tensor(x, meta["input"]) for x in xs])

    def native(xs):
        with torch.no_grad():
            return torch.softmax(model(torch.from_numpy(batch(xs))), dim=1).numpy()

    return native, lambda xs: {"input": batch(xs)}, False, do_export  # ONNX output is logits; softmax after


def _transformers(meta: dict, d: str, out: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    hf = os.path.join(d, "native/hf")
    tok = AutoTokenizer.from_pretrained(hf)
    model = AutoModelForSequenceClassification.from_pretrained(hf).eval()
    max_len = meta["input"].get("max_length", 128)

    class Wrap(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            return self.inner(input_ids=input_ids, attention_mask=attention_mask).logits

    def do_export():
        # dynamo exporter: the legacy tracer freezes transformers 5's padding-mask logic (the graph then ignores
        # attention_mask and padded inputs drift by ~1e-2).
        enc = tok(["example", "a longer example sentence"], padding="max_length", truncation=True, max_length=max_len,
                  return_tensors="pt")
        batch = torch.export.Dim("batch", min=1, max=4096)
        torch.onnx.export(Wrap(model), (enc["input_ids"], enc["attention_mask"]), out,
                          input_names=["input_ids", "attention_mask"], output_names=["probs"], opset_version=18,
                          dynamo=True, dynamic_shapes={"input_ids": {0: batch}, "attention_mask": {0: batch}})

    def feed(xs):
        e = tok(list(xs), padding="max_length", truncation=True, max_length=max_len, return_tensors="np")
        return {"input_ids": e["input_ids"].astype(np.int64), "attention_mask": e["attention_mask"].astype(np.int64)}

    def native(xs):
        f = feed(xs)
        with torch.no_grad():
            logits = model(input_ids=torch.from_numpy(f["input_ids"]), attention_mask=torch.from_numpy(f["attention_mask"])).logits
        return torch.softmax(logits, dim=1).numpy()

    return native, feed, False, do_export  # ONNX output is logits; softmax applied after


def _lightgbm(meta: dict, d: str, out: str):
    """The tabular trainer writes native/model.onnx and the loader serves exactly that file (lightgbm never loads next
    to torch), so the export is that file. Verification runs it on the real test rows (sealed manifest, row-<i>)
    against native LightGBM's own test predictions from test_predictions.parquet."""
    import shutil

    import pandas as pd

    from app.ml.tabular import _prepare

    src = os.path.join(d, "native/model.onnx")
    if not os.path.exists(src):
        raise ExportError("This model has no native/model.onnx; retrain it to export")
    preds = pd.read_parquet(os.path.join(d, "test_predictions.parquet"))
    frame = pd.read_parquet(os.path.join(d, "manifest.parquet"))
    X = _prepare(frame.loc[[int(str(s)[4:]) for s in preds.sample_id]], meta["input"]["features"]).to_numpy(np.float32)
    regression = meta["task_type"] == "tabular_regression"

    def native(_xs):
        if regression:
            return preds.y_pred.astype(float).to_numpy().reshape(-1, 1)
        return np.stack(preds.probs.to_numpy()).astype(np.float64)

    return native, lambda _xs: {"input": X}, None, lambda: shutil.copyfile(src, out)


def _torch_export(model, args, out, names, axes) -> None:
    import torch

    try:
        torch.onnx.export(model, args, out, input_names=names, output_names=["probs"], dynamic_axes=axes,
                          opset_version=17, dynamo=False)
    except Exception:  # noqa: BLE001 - legacy exporter missing or failing: try the dynamo exporter
        batch = torch.export.Dim("batch", min=1, max=4096)
        torch.onnx.export(model, args, out, input_names=names, output_names=["probs"], opset_version=18, dynamo=True,
                          dynamic_shapes=[{0: batch} for _ in args])


CONVERTERS = {"timm": _timm, "transformers": _transformers, "lightgbm": _lightgbm}


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def build_and_verify(db, mv: m.ModelVersion, meta: dict, workdir: str) -> tuple[str, dict]:
    import onnx
    import onnxruntime as ort

    conv = CONVERTERS.get(meta["framework"])
    if conv is None:
        raise ExportError(f"ONNX export for {meta['framework']} models isn't available yet")
    _download_native(mv, meta, workdir)
    out = os.path.join(workdir, "model.onnx")
    if meta["framework"] == "lightgbm":
        _download_file(mv, "native/model.onnx", workdir)
        _download_file(mv, "test_predictions.parquet", workdir)
        v = db.scalar(select(m.DatasetVersion).where(m.DatasetVersion.id == mv.dataset_version_id,
                                                     m.DatasetVersion.organization_id == mv.organization_id))
        if not v or not v.manifest_uri:
            raise ExportError("The dataset version this model was trained on is gone")
        with open(os.path.join(workdir, "manifest.parquet"), "wb") as f:
            f.write(storage.get_bytes(v.manifest_uri))
    native, feed, output_is_probs, do_export = conv(meta, workdir, out)
    if meta["framework"] == "lightgbm":
        ys = artifacts.read_predictions(mv.artifact_prefix + "test_predictions.parquet").y_true.tolist()
        xs = [None] * len(ys)
    else:
        xs, ys = _test_inputs(db, mv, meta)
    if not xs:
        raise ExportError("No test samples to verify the export against")
    a = np.asarray(native(xs), dtype=np.float64)  # reference first (see converters note)
    do_export()
    onnx.checker.check_model(out)
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    outs = sess.run(None, feed(xs))
    b = np.asarray(outs[-1] if meta["framework"] == "lightgbm" and meta["task_type"] != "tabular_regression" else outs[0],
                   dtype=np.float64)
    if b.ndim == 1:
        b = b.reshape(-1, 1) if meta["task_type"] == "tabular_regression" else b
    if output_is_probs is False:
        b = _softmax(b)
    diff = float(np.max(np.abs(a - b))) if a.size else 0.0
    report = {"n": len(xs), "max_abs_diff": diff, "tolerance": MAX_DIFF}
    if a.ndim == 2 and a.shape[1] > 1:
        # Near-ties (top-2 within tolerance) may flip on float noise; they don't count as disagreements.
        top2 = np.sort(a, axis=1)[:, -2:]
        decisive = (top2[:, 1] - top2[:, 0]) > 2 * MAX_DIFF
        agree = float(np.mean((a.argmax(1) == b.argmax(1))[decisive])) if decisive.any() else 1.0
        report["top1_agreement"], report["near_ties"] = agree, int((~decisive).sum())
        if ys:
            labels = meta["labels"]
            acc = lambda p: float(np.mean([labels[i] == y for i, y in zip(p.argmax(1), ys, strict=True)]))
            report["native_accuracy"], report["onnx_accuracy"] = acc(a), acc(b)
        report["verified"] = diff <= MAX_DIFF and agree >= MIN_AGREEMENT
    else:
        if ys:
            y = np.asarray(ys, dtype=np.float64)
            rmse = lambda p: float(np.sqrt(np.mean((p.ravel() - y) ** 2)))
            report["native_rmse"], report["onnx_rmse"] = rmse(a), rmse(b)
        report["verified"] = diff <= MAX_DIFF * max(1.0, float(np.max(np.abs(a))))
    return out, report


# ---------------------------------------------------------------- snippets (EXP-4)

def snippets(meta: dict, api_base: str = "http://127.0.0.1:8010/api/v1") -> dict[str, str]:
    """Copy-paste integration code generated from model.json (input spec + labels)."""
    labels, spec, mvid = meta.get("labels") or [], meta.get("input") or {}, meta.get("model_version_id")
    mod = spec.get("modality")
    if meta.get("framework") not in CONVERTERS or (mod == "image" and not spec.get("size")):
        # No ONNX for this model (e.g. object detection): the local predict API is the integration path.
        url = f"{api_base}/model-versions/{mvid}/predict"
        python = ("# On-device (ONNX) export for this model type comes in Phase 2; call the local API instead.\n"
                  "import requests\n\n"
                  f"URL = '{url}'\nTOKEN = '...'  # access token or API key\n\n"
                  "with open('photo.jpg', 'rb') as f:\n"
                  "    r = requests.post(URL, headers={'Authorization': f'Bearer {TOKEN}'}, files={'file': f})\n"
                  "r.raise_for_status()\n"
                  + ("for b in r.json()['boxes']:\n    print(b['class_name'], round(b['confidence'], 2), b['bbox'])\n"
                     if meta.get("task_type") == "object_detection" else "print(r.json())\n"))
        return {"python": python, "curl": f"curl -X POST {url} -H 'Authorization: Bearer $TOKEN' -F file=@photo.jpg"}
    t = float((meta.get("output") or {}).get("temperature") or 1.0)  # EVL-4 calibration, applied like the server does
    soft = f"z = out[0][0] / {t:g}  # logits / temperature\nprobs = np.exp(z - z.max()) / np.exp(z - z.max()).sum()\n"
    if mod == "image":
        h, w = spec["size"]
        n = spec.get("normalize") or {"mean": [0, 0, 0], "std": [1, 1, 1]}
        pre = (f"img = Image.open(path).convert('RGB').resize(({w}, {h}), Image.BILINEAR)\n"
               f"x = (np.asarray(img, np.float32) / {spec.get('scale', 255)} - np.array({n['mean']}, np.float32))"
               f" / np.array({n['std']}, np.float32)\n"
               "feed = {'input': x.transpose(2, 0, 1)[None].astype(np.float32)}\n")
        imports, post, arg = "from PIL import Image\n", soft, "path"
        curl = f"curl -X POST {api_base}/model-versions/{mvid}/predict -H 'Authorization: Bearer $TOKEN' -F file=@photo.jpg"
    elif mod == "text":
        pre = (f"tok = AutoTokenizer.from_pretrained('tokenizer')  # bundled tokenizer dir\n"
               f"e = tok([text], padding='max_length', truncation=True, max_length={spec.get('max_length', 128)},"
               " return_tensors='np')\n"
               "feed = {'input_ids': e['input_ids'].astype(np.int64), 'attention_mask': e['attention_mask'].astype(np.int64)}\n")
        imports, post, arg = "from transformers import AutoTokenizer\n", soft, "text"
        curl = (f"curl -X POST {api_base}/model-versions/{mvid}/predict -H 'Authorization: Bearer $TOKEN' "
                "-H 'Content-Type: application/json' -d '{\"text\": \"great product\"}'")
    else:
        names = [f["name"] for f in spec.get("features") or []]
        pre = (f"FEATURES = {names}\n# categorical columns: use the index into model.json input.features[i].categories\n"
               "feed = {'input': np.array([[float(row[f]) for f in FEATURES]], np.float32)}\n")
        imports, arg = "", "row"
        post = (("p = out[1][0] if len(out) > 1 else out[0][0]\n"
                 + (f"z = np.log(np.clip(p, 1e-12, 1)) / {t:g}  # temperature\n"
                    "probs = np.exp(z - z.max()) / np.exp(z - z.max()).sum()\n" if t != 1 else "probs = p\n"))
                if labels else "value = float(out[0].ravel()[0])\n")
        example = json.dumps({n: 0 for n in names[:3]})
        curl = (f"curl -X POST {api_base}/model-versions/{mvid}/predict -H 'Authorization: Bearer $TOKEN' "
                f"-H 'Content-Type: application/json' -d '{{\"row\": {example}}}'")
    body = "".join(f"    {ln}\n" for ln in (pre + "out = sess.run(None, feed)\n" + post).splitlines())
    ret = "probs" if labels else "value"
    usage = (f"LABELS = {labels}\nprobs = predict(...)\nprint(LABELS[int(np.argmax(probs))], float(np.max(probs)))\n"
             if labels else "print(predict(...))\n")
    python = ("import numpy as np\nimport onnxruntime as ort\n" + imports + "\nsess = ort.InferenceSession('model.onnx')\n\n"
              f"def predict({arg}):\n{body}    return {ret}\n\n{usage}")
    return {"python": python, "curl": curl}


# ---------------------------------------------------------------- task body

WEIGHT_EXT = (".safetensors", ".bin", ".pt", ".pth", ".onnx", ".txt", ".ubj", ".pkl", ".joblib")
DESCRIBE = {"model.onnx": "the model (ONNX)", "model.json": "input spec, labels, preprocessing", "labels.txt": "class names",
            "MODEL_CARD.md": "evaluation summary", "README.md": "how to run it", "tokenizer/": "text tokenizer + config"}


def _group(name: str) -> str:
    return "tokenizer/" if name.startswith("tokenizer/") else name


def bundle_files(z: zipfile.ZipFile) -> list[dict]:
    """What a built bundle contains (uncompressed bytes), tokenizer files grouped."""
    out: dict[str, dict] = {}
    for i in z.infolist():
        g = out.setdefault(_group(i.filename), {"name": _group(i.filename), "size_bytes": 0, "files": 0,
                                                 "description": DESCRIBE.get(_group(i.filename), "")})
        g["size_bytes"] += i.file_size
        g["files"] += 1
    return list(out.values())


def bundle_preview(mv: m.ModelVersion, meta: dict) -> dict:
    """Cheap size estimate before exporting: model.onnx ~= the native weights (fp32 both ways); the small files are exact.
    Never includes training data: the bundle is the model only."""
    native = storage.existing_sizes(mv.artifact_prefix + "native/")
    hf = lambda k: "/native/hf/" in k or k.startswith("native/hf/")
    is_weight = lambda k: k.endswith((".safetensors", ".bin")) if hf(k) else k.endswith(WEIGHT_EXT)
    weights = sum(v for k, v in native.items() if is_weight(k))
    tok = sum(v for k, v in native.items() if hf(k) and not is_weight(k))  # tokenizer + config ship as tokenizer/
    sn = snippets(meta)
    readme = len(f"# {meta['task_type']} model (ONNX)\n\n```python\n{sn['python']}```\n\n```sh\n{sn['curl']}\n```\n".encode())
    card = storage.head(mv.artifact_prefix + "MODEL_CARD.md")
    files = [{"name": "model.onnx", "size_bytes": weights, "estimated": True},
             {"name": "model.json", "size_bytes": len(json.dumps(meta, indent=2).encode()), "estimated": False},
             {"name": "labels.txt", "size_bytes": len(("\n".join(meta.get("labels") or []) + "\n").encode()), "estimated": False},
             {"name": "README.md", "size_bytes": readme, "estimated": False}]
    if tok:
        files.append({"name": "tokenizer/", "size_bytes": tok, "estimated": False})
    if card:
        files.append({"name": "MODEL_CARD.md", "size_bytes": card["ContentLength"], "estimated": False})
    for f in files:
        f["description"] = DESCRIBE.get(f["name"], "")
    return {"files": files, "total_bytes": sum(f["size_bytes"] for f in files), "estimated": True,
            "includes_training_data": False}


def run(export_id: str) -> None:
    with session_factory()() as db:
        ex = db.scalar(select(m.Export).where(m.Export.id == uuid.UUID(export_id)))
        mv = db.scalar(select(m.ModelVersion).where(m.ModelVersion.id == ex.model_version_id,
                                                    m.ModelVersion.organization_id == ex.organization_id))
        ex.status, ex.error = "running", None
        db.commit()
        try:
            meta = artifacts.read_json(mv.artifact_prefix + "model.json")
            with tempfile.TemporaryDirectory(prefix="automl-export-") as d:
                path, report = build_and_verify(db, mv, meta, d)
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                    z.write(path, "model.onnx")
                    z.writestr("model.json", json.dumps(meta, indent=2))
                    z.writestr("labels.txt", "\n".join(meta.get("labels") or []) + "\n")
                    sn = snippets(meta)
                    z.writestr("README.md", f"# {meta['task_type']} model (ONNX)\n\n```python\n{sn['python']}```\n\n"
                                            f"Hosted API (local server):\n\n```sh\n{sn['curl']}\n```\n")
                    tok = os.path.join(d, "native/hf")
                    if os.path.isdir(tok):
                        for f in os.listdir(tok):
                            if not f.endswith((".safetensors", ".bin")):  # tokenizer + config, not the weights
                                z.write(os.path.join(tok, f), f"tokenizer/{f}")
                    if storage.head(mv.artifact_prefix + "MODEL_CARD.md"):
                        z.writestr("MODEL_CARD.md", storage.get_bytes(mv.artifact_prefix + "MODEL_CARD.md"))
            key = f"{mv.artifact_prefix}exports/onnx-{ex.quantization}.zip"
            storage.put_bytes(key, buf.getvalue(), "application/zip")
            with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as z:
                report["files"] = bundle_files(z)  # exact contents, shown before download
            ex.artifact_key, ex.size_bytes, ex.verify_report = key, len(buf.getvalue()), report
            ex.verified = bool(report["verified"])
            if "onnx_accuracy" in report:
                ex.accuracy_delta = report["onnx_accuracy"] - report["native_accuracy"]
            elif "onnx_rmse" in report:  # regression: positive = ONNX is worse
                ex.accuracy_delta = report["onnx_rmse"] - report["native_rmse"]
            ex.status = "ready"
        except Exception as e:
            log.exception("export_failed", export_id=export_id)
            db.rollback()
            ex = db.scalar(select(m.Export).where(m.Export.id == uuid.UUID(export_id)))
            ex.status, ex.error = "failed", str(e) if isinstance(e, ExportError) else f"{type(e).__name__}: {e}"
        db.commit()


if __name__ == "__main__":  # child process entry (app.ml.eval_tasks.export)
    import sys

    run(sys.argv[1])
