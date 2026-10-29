"""DAT-13 starter datasets: a read-only catalog in S3 under starters/ (built by scripts/build_starters.py).

starters/catalog.json        [{id, name, description, task_type, classes, target, sample_count, size_bytes, source,
                               licence, source_url, previews: [key], kind: files|table|detection}]
starters/<id>/manifest.json  files/detection: {files: [{key, relative_path, size, sha256, label?}], annotations?: key}
                             table: {key, format, columns: [{name, role, type}]}
Not tenant data: it lives outside org/ so purge / GC never touch it, and imports COPY it into the org's prefix.
"""

import hashlib
import io
import json

from app.services import storage

PREFIX = "starters/"  # tests point this at a per-run prefix


def catalog() -> list[dict]:
    try:
        return json.loads(storage.get_bytes(PREFIX + "catalog.json"))
    except Exception:  # noqa: BLE001  (not built yet)
        return []


def get(starter_id: str) -> dict | None:
    return next((s for s in catalog() if s["id"] == starter_id), None)


def manifest(starter_id: str) -> dict:
    return json.loads(storage.get_bytes(f"{PREFIX}{starter_id}/manifest.json"))


def copy(src: str, dst: str) -> None:
    storage.client().copy_object(Bucket=storage.bucket(), Key=dst, CopySource={"Bucket": storage.bucket(), "Key": src})


# ---- publishing (scripts/build_starters.py and tests)

def _previews(sid: str, images: list[bytes], n: int = 6) -> list[str]:
    from PIL import Image

    keys = []
    for i, data in enumerate(images[:n]):
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.thumbnail((256, 256))
        buf = io.BytesIO()
        img.save(buf, "WEBP", quality=80)
        keys.append(f"{PREFIX}{sid}/previews/{i}.webp")
        storage.put_bytes(keys[-1], buf.getvalue(), "image/webp")
    return keys


def publish_files(entry: dict, files: list[tuple[str, bytes, str | None]], annotations: bytes | None = None) -> dict:
    """files: [(relative_path, bytes, label)]. With annotations (COCO json) it is a detection starter."""
    sid, rows, total = entry["id"], [], 0
    for rel, data, label in files:
        key = f"{PREFIX}{sid}/files/{rel}"
        storage.put_bytes(key, data)
        rows.append({"key": key, "relative_path": rel, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                     **({"label": label} if label else {})})
        total += len(data)
    man: dict = {"files": rows}
    if annotations is not None:
        man["annotations"] = f"{PREFIX}{sid}/annotations.json"
        storage.put_bytes(man["annotations"], annotations, "application/json")
        total += len(annotations)
    storage.put_bytes(f"{PREFIX}{sid}/manifest.json", json.dumps(man).encode(), "application/json")
    # spread the previews over classes: one image per label first
    by_label: dict = {}
    for _, data, label in files:
        by_label.setdefault(label, []).append(data)
    firsts = [v[0] for v in by_label.values()]
    picks = firsts + [d for _, d, _ in files if not any(d is f for f in firsts)]
    return {**entry, "kind": "detection" if annotations is not None else "files", "sample_count": len(files),
            "size_bytes": total, "previews": _previews(sid, picks)}


def publish_table(entry: dict, csv_bytes: bytes, columns: list[dict], rows: int) -> dict:
    sid = entry["id"]
    key = f"{PREFIX}{sid}/data.csv"
    storage.put_bytes(key, csv_bytes, "text/csv")
    storage.put_bytes(f"{PREFIX}{sid}/manifest.json",
                      json.dumps({"key": key, "format": "csv", "columns": columns}).encode(), "application/json")
    return {**entry, "kind": "table", "sample_count": rows, "size_bytes": len(csv_bytes), "previews": []}


def write_catalog(entries: list[dict]) -> None:
    storage.put_bytes(PREFIX + "catalog.json", json.dumps(entries, indent=1).encode(), "application/json")
