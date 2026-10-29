"""COCO replay set for hybrid detectors (built by scripts/build_replay.py; read-only, outside org/). No torch here:
the API's training gate reads it too."""

import hashlib
import json
import os

from app.services import storage

INDEX_KEY = "replay/coco_val2017/index.json"


def index_key() -> str:
    return os.environ.get("REPLAY_INDEX_KEY", INDEX_KEY)  # tests point this at a per-run index


def load() -> tuple[list[dict], str | None]:
    """(rows, sha256 of the index) or ([], None) when it isn't built."""
    try:
        raw = storage.get_bytes(index_key())
    except Exception:  # noqa: BLE001
        return [], None
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def images_with(rows: list[dict], classes: set[str]) -> int:
    return sum(1 for r in rows if any(b["class_name"] in classes for b in r["boxes"]))
