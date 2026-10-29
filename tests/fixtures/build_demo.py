#!/usr/bin/env python3
"""Learnable demo data for a meaningful accuracy check (synthetic, so there are no licensing questions).

    python3 tests/fixtures/build_demo.py
Writes tests/fixtures/demo/:
  shapes/{circle,square,triangle}/*.png  60 per class, 128 px: one shape of random colour, size and position
                                         on a random background with per-pixel noise
  churn.csv                              400 rows with a real (noisy logistic) signal in tenure/contract/charges
  reviews.csv                            240 product reviews (positive/negative, ~25% Hinglish), learnable from wording
  detection/images/*.png + coco.json     80 images, 1-3 non-overlapping circles/squares each, exact COCO boxes (xywh px)
Standard library only, deterministic (seed 7).
"""
import csv
import math
import random
import shutil
import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "demo"
SIZE, PER_CLASS = 128, 60
rng = random.Random(7)


def png(w, h, rgb):
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        raw += rgb[y * w * 3:(y + 1) * w * 3]
    chunk = lambda k, d: struct.pack(">I", len(d)) + k + d + struct.pack(">I", zlib.crc32(k + d) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b"")


def colour_far_from(bg):
    while True:
        c = [rng.randrange(256) for _ in range(3)]
        if sum(abs(a - b) for a, b in zip(c, bg)) > 220:  # keep the shape visible against its background
            return c


def inside(kind, x, y, cx, cy, r, rot):
    dx, dy = x - cx, y - cy
    if kind == "circle":
        return dx * dx + dy * dy <= r * r
    # rotate the point into the shape's frame
    c, s = math.cos(-rot), math.sin(-rot)
    px, py = dx * c - dy * s, dx * s + dy * c
    if kind == "square":
        return abs(px) <= r * 0.8 and abs(py) <= r * 0.8
    # equilateral triangle with circumradius r, pointing up in its own frame
    v = [(0, -r), (r * math.sin(math.pi / 3 * 2), r / 2), (-r * math.sin(math.pi / 3 * 2), r / 2)]
    sign = lambda p, a, b: (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])
    d1, d2, d3 = sign((px, py), v[0], v[1]), sign((px, py), v[1], v[2]), sign((px, py), v[2], v[0])
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def shape_image(kind):
    bg = [rng.randrange(256) for _ in range(3)]
    fg = colour_far_from(bg)
    r = rng.uniform(18, 40)
    cx, cy = rng.uniform(r + 4, SIZE - r - 4), rng.uniform(r + 4, SIZE - r - 4)
    rot = rng.uniform(0, 2 * math.pi)
    noise = rng.uniform(8, 30)
    buf = bytearray(SIZE * SIZE * 3)
    i = 0
    for y in range(SIZE):
        for x in range(SIZE):
            base = fg if inside(kind, x + 0.5, y + 0.5, cx, cy, r, rot) else bg
            for ch in range(3):
                buf[i] = max(0, min(255, int(base[ch] + rng.gauss(0, noise))))
                i += 1
    return png(SIZE, SIZE, buf)


def churn_rows(n=400):
    rows = []
    for i in range(n):
        tenure = rng.randint(1, 72)
        contract = rng.choices(["monthly", "yearly", "two_year"], [0.55, 0.25, 0.20])[0]
        charges = round(rng.uniform(20, 120), 2)
        tickets = rng.choices(range(6), [40, 25, 15, 10, 6, 4])[0]
        region = rng.choice(["north", "south", "east", "west"])  # pure noise feature
        z = 1.2 - 0.06 * tenure + {"monthly": 1.4, "yearly": -0.4, "two_year": -1.6}[contract] + 0.018 * (charges - 70) + 0.45 * tickets
        p = 1 / (1 + math.exp(-z))
        rows.append({"customer_id": f"C{i:05d}", "tenure": tenure, "contract": contract, "monthly_charges": charges,
                     "support_tickets": tickets, "region": region, "churn": int(rng.random() < p)})
    return rows


POS = ["great", "excellent", "loved", "fantastic", "works perfectly", "very happy", "highly recommend", "superb", "fast delivery", "worth the money"]
NEG = ["terrible", "broken", "waste of money", "very disappointed", "stopped working", "awful", "refund please", "poor quality", "late delivery", "never buying again"]
POS_HI = ["bahut accha", "mast product", "paisa vasool", "ekdum badhiya", "dil khush ho gaya"]
NEG_HI = ["bilkul bekaar", "paise barbaad", "ghatiya quality", "kaam nahi karta", "bahut bura"]
ITEMS = ["phone", "charger", "headphones", "shoes", "mixer", "backpack", "watch", "kettle", "lamp", "keyboard"]


def review(label):
    hinglish = rng.random() < 0.25
    words = (POS_HI if hinglish else POS) if label == "positive" else (NEG_HI if hinglish else NEG)
    other = (NEG_HI if hinglish else NEG) if label == "positive" else (POS_HI if hinglish else POS)
    item = rng.choice(ITEMS)
    parts = [f"The {item}" if not hinglish else f"Ye {item}", rng.choice(words)]
    if rng.random() < 0.6:
        parts.append(rng.choice(words))
    if rng.random() < 0.15:  # label noise in the wording: one opposite phrase
        parts.append("but " + rng.choice(other))
    return ", ".join(parts) + rng.choice([".", "!", "!!", ""])


def detection_image():
    """1-3 non-overlapping circles/squares on a noisy background; returns (png, [(class, x, y, w, h)])."""
    bg = [rng.randrange(256) for _ in range(3)]
    noise = rng.uniform(6, 20)
    objs = []
    for _ in range(rng.randint(1, 3)):
        for _try in range(30):
            r = rng.uniform(12, 24)
            cx, cy = rng.uniform(r + 2, SIZE - r - 2), rng.uniform(r + 2, SIZE - r - 2)
            if all((cx - o[1]) ** 2 + (cy - o[2]) ** 2 > (r + o[3] + 4) ** 2 for o in objs):
                objs.append((rng.choice(["circle", "square"]), cx, cy, r, colour_far_from(bg)))
                break
    buf = bytearray(SIZE * SIZE * 3)
    i = 0
    for y in range(SIZE):
        for x in range(SIZE):
            base = bg
            for kind, cx, cy, r, fg in objs:
                if inside(kind, x + 0.5, y + 0.5, cx, cy, r, 0.0):
                    base = fg
                    break
            for ch in range(3):
                buf[i] = max(0, min(255, int(base[ch] + rng.gauss(0, noise))))
                i += 1
    boxes = []
    for kind, cx, cy, r, _ in objs:
        h = r if kind == "circle" else r * 0.8  # square half-side is 0.8 r (see inside())
        boxes.append((kind, round(cx - h, 1), round(cy - h, 1), round(2 * h, 1), round(2 * h, 1)))
    return png(SIZE, SIZE, buf), boxes


def main():
    shutil.rmtree(ROOT, ignore_errors=True)
    for kind in ("circle", "square", "triangle"):
        d = ROOT / "shapes" / kind
        d.mkdir(parents=True)
        for i in range(PER_CLASS):
            (d / f"{kind}_{i:03d}.png").write_bytes(shape_image(kind))
    rows = churn_rows()
    with (ROOT / "churn.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (ROOT / "reviews.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label"])
        w.writeheader()
        for i in range(240):
            lab = "positive" if i % 2 == 0 else "negative"
            w.writerow({"text": review(lab), "label": lab})
    det = ROOT / "detection" / "images"
    det.mkdir(parents=True)
    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "circle"}, {"id": 2, "name": "square"}]}
    for i in range(80):
        data, boxes = detection_image()
        (det / f"det_{i:03d}.png").write_bytes(data)
        coco["images"].append({"id": i, "file_name": f"det_{i:03d}.png", "width": SIZE, "height": SIZE})
        for kind, x, y, w, h in boxes:
            coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": i, "category_id": 1 if kind == "circle" else 2, "bbox": [x, y, w, h], "area": round(w * h, 1), "iscrowd": 0})
    (ROOT / "detection" / "coco.json").write_text(__import__("json").dumps(coco, indent=1))
    counts = {c["name"]: sum(a["category_id"] == c["id"] for a in coco["annotations"]) for c in coco["categories"]}
    print(f"detection demo: 80 images, boxes {counts}")
    print(f"demo data written to {ROOT}: {3 * PER_CLASS} shape images, {len(rows)} churn rows (churn rate {sum(r['churn'] for r in rows) / len(rows):.2f})")


if __name__ == "__main__":
    main()
