"""Builds the COCO replay set for hybrid detectors into S3 replay/coco_val2017/ (read-only, outside org/).

    uv run python scripts/build_replay.py

Reads .starter-cache/coco_val2017_subset/{annotations_trainval2017.zip, val2017/*.jpg}. Every annotated (non-crowd)
object of the 80 COCO classes is kept, so any chosen subset of classes can be replayed. COCO: annotations
CC BY 4.0, images under their Flickr terms.
"""

import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.pretrained import COCO
from app.services import storage

SRC = Path(__file__).resolve().parents[2] / ".starter-cache" / "coco_val2017_subset"
PREFIX = "replay/coco_val2017/"


def main() -> None:
    storage.ensure_bucket()
    with zipfile.ZipFile(SRC / "annotations_trainval2017.zip") as z:
        doc = json.loads(z.read("annotations/instances_val2017.json"))
    present = {p.name for p in (SRC / "val2017").glob("*.jpg")}
    boxes = defaultdict(list)
    for a in doc["annotations"]:
        if not a.get("iscrowd") and a["category_id"] in COCO:
            x, y, w, h = a["bbox"]
            boxes[a["image_id"]].append({"class_name": COCO[a["category_id"]], "bbox": {"x": x, "y": y, "w": w, "h": h}})
    index = []
    for im in doc["images"]:
        if im["file_name"] in present and boxes[im["id"]]:
            key = f"{PREFIX}images/{im['file_name']}"
            if storage.head(key) is None:
                storage.put_file(key, str(SRC / "val2017" / im["file_name"]), "image/jpeg")
            index.append({"key": key, "file_name": im["file_name"], "width": im["width"], "height": im["height"],
                          "boxes": boxes[im["id"]]})
    storage.put_bytes(PREFIX + "index.json", json.dumps(index).encode(), "application/json")
    per = defaultdict(int)
    for r in index:
        for c in {b["class_name"] for b in r["boxes"]}:
            per[c] += 1
    print(f"{len(index)} images; classes with images: {len(per)}/80; min per class: {min(per.values(), default=0)}")


if __name__ == "__main__":
    main()
