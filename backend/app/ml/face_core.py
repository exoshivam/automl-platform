"""Standalone face pipeline: SCRFD detect (5 landmarks) -> similarity-align to the ArcFace 112x112 template ->
ArcFace embed (512-d, L2-normalised) -> cosine match against a gallery. Needs only onnxruntime + numpy + Pillow
(opencv-python is optional, for --webcam). The platform's API runs THIS file (via app/ml/face.py) and the face export
ships it verbatim as identify.py, so the exported bundle gives the same answers as POST /identify.

    python identify.py photo.jpg            # uses det_*.onnx, w600k_*.onnx and gallery.json next to this file
    python identify.py --webcam             # needs: pip install opencv-python

Licence: InsightFace model packs (buffalo_l / buffalo_s) are NON-COMMERCIAL research only.
"""

from __future__ import annotations

import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

LICENCE = "InsightFace model pack: non-commercial research use only"
ARCFACE_112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655],
                        [70.7299, 92.2041]], dtype=np.float64)
DET_SIZE, DET_THRESHOLD, NMS_IOU = 640, 0.5, 0.4


@dataclass
class Face:
    box: tuple[float, float, float, float]  # x, y, w, h in source pixels
    score: float
    landmarks: np.ndarray  # (5, 2)
    crop: Image.Image | None = None  # aligned 112x112 RGB
    embedding: np.ndarray | None = None
    blur: float | None = None

    @property
    def px(self) -> int:
        return int(min(self.box[2], self.box[3]))

    @property
    def quality(self) -> float:
        """0..1: detector confidence scaled down for blurry crops."""
        return round(float(self.score) * min(1.0, (self.blur or 0) / 100.0), 3)


def load_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


# ---- detection (SCRFD decode, as in insightface/model_zoo/scrfd.py)

def _nms(boxes: np.ndarray, scores: np.ndarray) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order, keep = scores.argsort()[::-1], []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        order = order[1:][inter / (areas[i] + areas[order[1:]] - inter) <= NMS_IOU]
    return keep


def detect(det, img: Image.Image) -> list[Face]:
    scale = DET_SIZE / max(img.size)
    w, h = round(img.width * scale), round(img.height * scale)
    canvas = np.zeros((DET_SIZE, DET_SIZE, 3), dtype=np.float32)
    canvas[:h, :w] = np.asarray(img.resize((w, h), Image.BILINEAR), dtype=np.float32)
    blob = ((canvas - 127.5) / 128.0).transpose(2, 0, 1)[None]
    outs = det.run(None, {det.get_inputs()[0].name: blob})
    fmc = len(outs) // 3
    strides = [8, 16, 32, 64, 128][:fmc]
    all_scores, all_boxes, all_kps = [], [], []
    for i, stride in enumerate(strides):
        scores = outs[i].reshape(-1)
        dist = outs[i + fmc].reshape(-1, 4) * stride
        kps = outs[i + 2 * fmc].reshape(-1, 10) * stride
        fh, fw = DET_SIZE // stride, DET_SIZE // stride
        anchors = (np.stack(np.mgrid[:fh, :fw][::-1], axis=-1).astype(np.float32) * stride).reshape(-1, 2)
        anchors = np.repeat(anchors, len(scores) // len(anchors), axis=0)
        pos = scores >= DET_THRESHOLD
        c = anchors[pos]
        d = dist[pos]
        all_boxes.append(np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], 1))
        k = kps[pos].reshape(-1, 5, 2) + c[:, None, :]
        all_kps.append(k)
        all_scores.append(scores[pos])
    scores = np.concatenate(all_scores)
    if not len(scores):
        return []
    boxes, kps = np.concatenate(all_boxes) / scale, np.concatenate(all_kps) / scale
    faces = []
    for i in _nms(boxes, scores):
        x1, y1, x2, y2 = boxes[i]
        x1, y1 = max(0.0, float(x1)), max(0.0, float(y1))
        x2, y2 = min(float(img.width), float(x2)), min(float(img.height), float(y2))
        faces.append(Face((x1, y1, x2 - x1, y2 - y1), float(scores[i]), kps[i]))
    return faces


# ---- alignment + embedding

def _similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Umeyama: the 2x3 similarity transform mapping src points onto dst (least squares)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    cov = d.T @ s / len(src)
    u, sig, vt = np.linalg.svd(cov)
    e = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        e[1, 1] = -1
    r = u @ e @ vt
    scale = np.trace(np.diag(sig) @ e) / s.var(0).sum()
    t = mu_d - scale * r @ mu_s
    return np.hstack([scale * r, t[:, None]])


def align(img: Image.Image, landmarks: np.ndarray) -> Image.Image:
    m = np.vstack([_similarity(landmarks.astype(np.float64), ARCFACE_112), [0, 0, 1]])
    inv = np.linalg.inv(m)  # PIL maps OUTPUT pixels back to input pixels
    return img.transform((112, 112), Image.AFFINE, data=tuple(inv[:2].reshape(-1)), resample=Image.BILINEAR)


def blur_score(crop: Image.Image) -> float:
    """Variance of the Laplacian on the aligned grey crop (higher = sharper)."""
    g = np.asarray(crop.convert("L"), dtype=np.float64)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def embed(rec, crops: list[Image.Image]) -> np.ndarray:
    x = np.stack([np.asarray(c, dtype=np.float32) for c in crops])
    x = ((x - 127.5) / 127.5).transpose(0, 3, 1, 2)
    out = rec.run(None, {rec.get_inputs()[0].name: x})[0]
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def analyse(det, rec, img: Image.Image) -> list[Face]:
    """Detect every face, align, score blur and embed (one batch)."""
    faces = detect(det, img)
    for f in faces:
        f.crop = align(img, f.landmarks)
        f.blur = blur_score(f.crop)
    if faces:
        for f, e in zip(faces, embed(rec, [f.crop for f in faces]), strict=True):
            f.embedding = e
    return faces


# ---- matching + CLI (used by the exported bundle)

def sessions(model_dir: Path):
    import onnxruntime as ort

    det = next(iter(sorted(model_dir.glob("det_*.onnx"))))
    rec = next(iter(sorted(model_dir.glob("w600k_*.onnx"))))
    mk = lambda p: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    return mk(det), mk(rec)


def match(gallery: dict, embedding: np.ndarray) -> tuple[str | None, float]:
    """Best person by cosine similarity to their template; None ("Other") below the gallery threshold."""
    best, score = None, -1.0
    for p in gallery["people"]:
        s = float(np.asarray(p["template"], dtype=np.float64) @ embedding)
        if s > score:
            best, score = p["name"], s
    return (best if score >= gallery["threshold"] else None), score


def _describe(det, rec, gallery, img: Image.Image) -> list[str]:
    out = []
    for f in analyse(det, rec, img):
        name, score = match(gallery, f.embedding)
        x, y, w, h = (round(v) for v in f.box)
        out.append(f"{name or 'Other'}\t{score:.3f}\tbox=({x},{y},{w},{h})")
    return out


def main(argv: list[str]) -> int:
    here = Path(__file__).resolve().parent
    gallery = json.loads((here / "gallery.json").read_text())
    det, rec = sessions(here)
    if argv[:1] == ["--webcam"]:
        import cv2  # optional dependency

        cam = cv2.VideoCapture(0)
        try:
            while True:
                ok, frame = cam.read()
                if not ok:
                    break
                img = Image.fromarray(frame[:, :, ::-1])
                print(" | ".join(_describe(det, rec, gallery, img)) or "(no face)", flush=True)
        except KeyboardInterrupt:
            pass
        return 0
    if not argv:
        print(__doc__)
        return 2
    for path in argv:
        lines = _describe(det, rec, gallery, load_image(Path(path).read_bytes()))
        print(f"{path}: " + ("; ".join(lines) if lines else "no face"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
