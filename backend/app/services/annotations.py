"""COCO / YOLO / VOC detection annotations (DAT-8). Boxes come out in pixels {x, y, w, h}, top-left origin.

parse_* return (boxes_by_image, errors). Bad records are reported (DAT-8c), never silently dropped.
YOLO boxes are normalised, so they're converted to pixels once the image size is known (to_pixels)."""

import json
import xml.etree.ElementTree as ET
from pathlib import PurePosixPath
from typing import Any

Box = dict[str, Any]  # {"class_name", "bbox": {x,y,w,h}, "normalised"?: True}


def parse_coco(raw: bytes, image_names: set[str]) -> tuple[dict[str, list[Box]], list[dict], set[str]]:
    errors: list[dict] = []
    try:
        doc = json.loads(raw)
        cats = {c["id"]: c["name"] for c in doc.get("categories", [])}
        images = {im["id"]: im["file_name"] for im in doc.get("images", [])}
    except (ValueError, KeyError, TypeError) as e:
        return {}, [{"code": "invalid_coco", "detail": str(e)[:200]}], set()
    out: dict[str, list[Box]] = {}
    by_base = _basename_index(image_names)
    for img_id, fname in images.items():
        resolved = _resolve(fname, image_names, by_base)
        if resolved is None:
            errors.append({"code": "image_missing", "image": fname})
            continue
        out.setdefault(resolved, [])
        images[img_id] = resolved
    for i, a in enumerate(doc.get("annotations", [])):
        try:
            img, cat, (x, y, w, h) = a["image_id"], a["category_id"], a["bbox"]
        except (KeyError, TypeError, ValueError):
            errors.append({"code": "invalid_annotation", "index": i})
            continue
        if cat not in cats:
            errors.append({"code": "unknown_category", "index": i, "category_id": cat})
            continue
        if img not in images or images[img] not in out:
            errors.append({"code": "unknown_image", "index": i, "image_id": img})
            continue
        out[images[img]].append({"class_name": cats[cat], "bbox": {"x": x, "y": y, "w": w, "h": h}})
    return out, errors, set(cats.values())


def parse_yolo(classes_txt: bytes, label_files: dict[str, bytes], image_names: set[str]):
    classes = [c.strip() for c in classes_txt.decode("utf-8", "replace").splitlines() if c.strip()]
    errors: list[dict] = []
    stems = {PurePosixPath(n).stem: n for n in image_names}
    out: dict[str, list[Box]] = {n: [] for n in image_names}
    for lname, raw in label_files.items():
        stem = PurePosixPath(lname).stem
        img = stems.get(stem)
        if img is None:
            errors.append({"code": "image_missing", "label_file": lname})
            continue
        for ln, line in enumerate(raw.decode("utf-8", "replace").splitlines(), 1):
            parts = line.split()
            if not parts:
                continue
            try:
                ci, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
            except (ValueError, IndexError):
                errors.append({"code": "invalid_annotation", "label_file": lname, "line": ln})
                continue
            if not 0 <= ci < len(classes):
                errors.append({"code": "unknown_category", "label_file": lname, "line": ln, "category_id": ci})
                continue
            out[img].append({"class_name": classes[ci], "normalised": True,
                             "bbox": {"x": cx - w / 2, "y": cy - h / 2, "w": w, "h": h}})
    return out, errors, set(classes)


def parse_voc(xml_files: dict[str, bytes], image_names: set[str]):
    errors: list[dict] = []
    out: dict[str, list[Box]] = {}
    classes: set[str] = set()
    by_base = _basename_index(image_names)
    for xname, raw in xml_files.items():
        try:
            root = ET.fromstring(raw)
        except ET.ParseError as e:
            errors.append({"code": "invalid_voc", "file": xname, "detail": str(e)[:200]})
            continue
        fname = (root.findtext("filename") or PurePosixPath(xname).stem).strip()
        img = _resolve(fname, image_names, by_base) or next(
            (n for n in image_names if PurePosixPath(n).stem == PurePosixPath(fname).stem), None)
        if img is None:
            errors.append({"code": "image_missing", "file": xname, "image": fname})
            continue
        out.setdefault(img, [])
        for obj in root.findall("object"):
            try:
                name = obj.findtext("name").strip()
                bb = obj.find("bndbox")
                x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            except (AttributeError, TypeError, ValueError):
                errors.append({"code": "invalid_annotation", "file": xname})
                continue
            classes.add(name)
            out[img].append({"class_name": name, "bbox": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}})
    return out, errors, classes


def to_pixels(box: Box, width: int | None, height: int | None) -> dict[str, float]:
    b = box["bbox"]
    if not box.get("normalised") or not width or not height:
        return {k: round(float(v), 2) for k, v in b.items()}
    return {"x": round(b["x"] * width, 2), "y": round(b["y"] * height, 2),
            "w": round(b["w"] * width, 2), "h": round(b["h"] * height, 2)}


def _basename_index(names: set[str]) -> dict[str, str]:
    idx: dict[str, str] = {}
    for n in names:
        idx.setdefault(PurePosixPath(n).name, n)
    return idx


def _resolve(fname: str, names: set[str], by_base: dict[str, str]) -> str | None:
    fname = fname.replace("\\", "/")
    return fname if fname in names else by_base.get(PurePosixPath(fname).name)


def export_coco(samples: list[dict], classes: list[str]) -> dict:
    cats = {c: i + 1 for i, c in enumerate(classes)}
    images, anns = [], []
    for i, s in enumerate(samples, 1):
        images.append({"id": i, "file_name": s["file_name"], "width": s["width"], "height": s["height"]})
        for b in s["boxes"]:
            if b["class_name"] not in cats:
                cats[b["class_name"]] = len(cats) + 1
            bb = b["bbox"]
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": cats[b["class_name"]],
                         "bbox": [bb["x"], bb["y"], bb["w"], bb["h"]], "area": bb["w"] * bb["h"], "iscrowd": 0})
    return {"images": images, "annotations": anns,
            "categories": [{"id": i, "name": c} for c, i in sorted(cats.items(), key=lambda kv: kv[1])]}


def export_yolo(samples: list[dict], classes: list[str]) -> dict[str, str]:
    """{path: text}: classes.txt plus one labels/<stem>.txt per image (normalised cx cy w h)."""
    order = list(classes)
    files: dict[str, str] = {}
    for s in samples:
        lines = []
        for b in s["boxes"]:
            if b["class_name"] not in order:
                order.append(b["class_name"])
            bb, W, H = b["bbox"], s["width"] or 1, s["height"] or 1
            lines.append(f"{order.index(b['class_name'])} {(bb['x'] + bb['w'] / 2) / W:.6f} "
                         f"{(bb['y'] + bb['h'] / 2) / H:.6f} {bb['w'] / W:.6f} {bb['h'] / H:.6f}")
        files[f"labels/{PurePosixPath(s['file_name']).stem}.txt"] = "\n".join(lines) + ("\n" if lines else "")
    files["classes.txt"] = "\n".join(order) + "\n"
    return files
