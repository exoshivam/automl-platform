#!/usr/bin/env python3
"""Builds the seeded-defect fixtures for the first-slice acceptance checklist (tests/acceptance/CHECKLIST.md).

Standard library only, deterministic (seed 42). Run from anywhere:
    python3 tests/fixtures/build_fixtures.py          # small fixtures + bombs
    python3 tests/fixtures/build_fixtures.py --no-bombs

Writes everything under tests/fixtures/ next to this file, plus EXPECTED.json: the defect counts each
fixture is built to trigger, so the checklist assertions are mechanical.
"""
import csv
import io
import json
import random
import shutil
import struct
import sys
import uuid
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rng = random.Random(42)
EXPECTED = {}


# ---------- PNG (stdlib) ----------

def _chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def png_bytes(w, h, gray=False, seed=None):
    """Blocky random image: a distinct perceptual hash per seed (8x8 blocks survive pHash downscaling)."""
    ch = 1 if gray else 3
    rows = _blocky(w, h, ch, random.Random(seed))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0 if gray else 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(bytes(rows), 9)) + _chunk(b"IEND", b"")


def _blocky(w, h, ch, r, block=8):
    grid = [[bytes(r.randrange(256) for _ in range(ch)) for _ in range((w + block - 1) // block)] for _ in range((h + block - 1) // block)]
    out = bytearray()
    for y in range(h):
        out.append(0)
        row = grid[y // block]
        for x in range(w):
            out += row[x // block]
    return out


def tweak_one_pixel(data):
    """Near-duplicate: re-encode with one byte of pixel data changed (different SHA-256, same pHash)."""
    ihdr_end = 8 + 8 + 13 + 4
    idat_len = struct.unpack(">I", data[ihdr_end:ihdr_end + 4])[0]
    raw = bytearray(zlib.decompress(data[ihdr_end + 8:ihdr_end + 8 + idat_len]))
    raw[1] = (raw[1] + 3) % 256
    return data[:ihdr_end] + _chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + _chunk(b"IEND", b"")


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def images(folder, n, w=64, h=64, start=0, gray=False):
    for i in range(n):
        write(folder / f"img_{start + i:03d}.png", png_bytes(w, h, gray=gray, seed=f"{folder.name}-{start + i}-{w}x{h}"))


# ---------- image classification sets ----------

def build_images():
    base = ROOT / "images"
    shutil.rmtree(base, ignore_errors=True)

    images(base / "clean/cat", 32)
    images(base / "clean/dog", 32)
    EXPECTED["images/clean"] = {"classes": {"cat": 32, "dog": 32}, "errors": [], "warnings": [], "note": "control set: no VAL-1..8/13 flags"}

    images(base / "corrupt/cat", 12)
    images(base / "corrupt/dog", 12)
    write(base / "corrupt/cat/zero_bytes.png", b"")
    good = png_bytes(64, 64, seed="truncate-me")
    write(base / "corrupt/cat/truncated.png", good[: len(good) // 2])
    write(base / "corrupt/dog/garbage.jpg", bytes(rng.randrange(256) for _ in range(2048)))
    write(base / "corrupt/dog/text_named_png.png", b"this is not an image\n")
    EXPECTED["images/corrupt"] = {"VAL-1": 4, "files": ["cat/zero_bytes.png", "cat/truncated.png", "dog/garbage.jpg", "dog/text_named_png.png"], "valid_after_autofix": 24}

    images(base / "too_few/cat", 5)
    images(base / "too_few/dog", 5)
    EXPECTED["images/too_few"] = {"VAL-2": ["cat", "dog"], "severity": "error"}

    images(base / "low/cat", 15)
    images(base / "low/dog", 15)
    EXPECTED["images/low"] = {"VAL-3": ["cat", "dog"], "severity": "warning"}

    images(base / "imbalance/big", 110)
    images(base / "imbalance/small", 10)
    EXPECTED["images/imbalance"] = {"VAL-4": {"ratio": 11.0, "largest": "big", "smallest": "small"}, "VAL-3": ["small"], "severity": "warning"}

    # Exact dupes, near-dupes and label conflicts.
    images(base / "dupes/cat", 32)
    images(base / "dupes/dog", 32)
    for i in range(5):  # 5 exact copies inside cat
        src = (base / f"dupes/cat/img_{i:03d}.png").read_bytes()
        write(base / f"dupes/cat/copy_{i:03d}.png", src)
    for i in range(3):  # 3 near-duplicates inside dog
        src = (base / f"dupes/dog/img_{i:03d}.png").read_bytes()
        write(base / f"dupes/dog/near_{i:03d}.png", tweak_one_pixel(src))
    for i in range(2):  # 2 identical files labelled both cat and dog
        data = png_bytes(64, 64, seed=f"conflict-{i}")
        write(base / f"dupes/cat/conflict_{i}.png", data)
        write(base / f"dupes/dog/conflict_{i}.png", data)
    EXPECTED["images/dupes"] = {
        "VAL-5": {"exact_duplicate_groups_min": 7, "detail": "5 cat copy_* groups + 2 conflict_* pairs (cross-class)"},
        "VAL-6": {"near_duplicate_pairs": 3, "files": ["dog/near_000..002.png vs dog/img_000..002.png"]},
        "VAL-8": {"conflicts": 2, "files": ["conflict_0.png", "conflict_1.png"]},
    }

    # Leakage across splits: 15 exact pairs per class. With random 70/15/15 per-sample assignment, P(no pair
    # straddles a split) = 0.535^30 ~ 7e-9, so VAL-7 must fire unless dupes are grouped before splitting.
    for cls in ("cat", "dog"):
        images(base / f"split_leak/{cls}", 30)
        for i in range(15):
            src = (base / f"split_leak/{cls}/img_{i:03d}.png").read_bytes()
            write(base / f"split_leak/{cls}/dup_{i:03d}.png", src)
    EXPECTED["images/split_leak"] = {"VAL-5": {"exact_duplicate_groups": 30}, "VAL-7": {"severity": "error", "expected_after_autofix": 0}}

    # Same leak, pre-split by folder, in case the API accepts user-assigned splits.
    for split, n in (("train", 20), ("test", 10)):
        for cls in ("cat", "dog"):
            images(base / f"split_leak_presplit/{split}/{cls}", n, start=0 if split == "train" else 100)
    for cls in ("cat", "dog"):
        for i in range(5):
            src = (base / f"split_leak_presplit/train/{cls}/img_{i:03d}.png").read_bytes()
            write(base / f"split_leak_presplit/test/{cls}/leak_{i:03d}.png", src)
    EXPECTED["images/split_leak_presplit"] = {"VAL-7": {"train_test_shared": 10, "severity": "error"}}

    # VAL-13 quality: tiny, extreme aspect ratio, grayscale mixed into RGB.
    images(base / "quality/cat", 12)
    images(base / "quality/dog", 12)
    for i in range(3):
        write(base / f"quality/cat/tiny_{i}.png", png_bytes(16, 16, seed=f"tiny-{i}"))
    for i in range(2):
        write(base / f"quality/dog/wide_{i}.png", png_bytes(400, 20, seed=f"wide-{i}"))
    for i in range(4):
        write(base / f"quality/dog/gray_{i}.png", png_bytes(64, 64, gray=True, seed=f"gray-{i}"))
    EXPECTED["images/quality"] = {"VAL-13": {"tiny_lt_32px": 3, "extreme_aspect": 2, "grayscale_in_rgb_set": 4}, "severity": "info/warning"}


# ---------- tabular ----------

def build_tabular():
    base = ROOT / "tabular"
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True)

    def clean_rows(n, seed):
        r = random.Random(seed)
        rows = []
        for i in range(n):
            tenure = r.randint(1, 72)
            charges = round(r.uniform(20, 120), 2)
            contract = r.choice(["monthly", "yearly", "two_year"])
            p = 0.6 if contract == "monthly" and tenure < 12 else 0.15
            rows.append({"tenure": tenure, "monthly_charges": charges, "contract": contract, "region": r.choice(["north", "south", "east", "west"]), "churn": int(r.random() < p)})
        return rows

    rows = clean_rows(300, "clean")
    _csv(base / "clean.csv", rows)
    EXPECTED["tabular/clean.csv"] = {"rows": 300, "target": "churn", "errors": [], "note": "control set"}

    leaky = []
    for i, row in enumerate(clean_rows(300, "leaky")):
        r = random.Random(i)
        leaky.append({
            "customer_id": str(uuid.UUID(int=r.getrandbits(128))),  # VAL-10
            "row_no": i + 1,  # VAL-10 (sequential integer ID)
            **row,
            "churn_outcome": row["churn"],  # VAL-11 by name AND single-feature AUC = 1.0
            "cancelled_date": f"2026-0{r.randint(1, 9)}-1{r.randint(0, 9)}" if row["churn"] else "",  # VAL-11 by name (+ missing pattern = target)
            "mostly_missing": "" if r.random() < 0.65 else round(r.uniform(0, 1), 3),  # VAL-9 (> 50% missing)
            "country": "IN",  # VAL-12 constant
            "session_hash": f"s{r.getrandbits(40):010x}",  # VAL-12 high-cardinality categorical (and ID-like)
        })
    _csv(base / "leaky.csv", leaky)
    missing = sum(1 for x in leaky if x["mostly_missing"] == "")
    EXPECTED["tabular/leaky.csv"] = {
        "target": "churn",
        "VAL-9": {"column": "mostly_missing", "missing_fraction": round(missing / 300, 3)},
        "VAL-10": ["customer_id", "row_no"],
        "VAL-11": {"columns": ["churn_outcome", "cancelled_date"], "severity": "error until acknowledged"},
        "VAL-12": {"constant": ["country"], "high_cardinality": ["session_hash"]},
    }

    _csv(base / "too_few.csv", clean_rows(40, "few"))
    EXPECTED["tabular/too_few.csv"] = {"rows": 40, "VAL-2": "error (< 50 rows)"}

    dup = clean_rows(100, "dup")
    dup += [dict(dup[i]) for i in range(10)]
    _csv(base / "duplicate_rows.csv", dup)
    EXPECTED["tabular/duplicate_rows.csv"] = {"rows": 110, "VAL-5": {"exact_duplicate_rows": 10}}

    # Group-split fixture for DAT-6: 40 patients x 5 rows; no patient may straddle splits.
    grouped = []
    for p in range(40):
        for k in range(5):
            grouped.append({"patient_id": f"P{p:03d}", "visit": k, "marker": round(rng.uniform(0, 10), 2), "label": p % 2})
    _csv(base / "grouped.csv", grouped)
    EXPECTED["tabular/grouped.csv"] = {"rows": 200, "group_column": "patient_id", "groups": 40, "DAT-6": "each patient_id in exactly one split"}

    _xlsx(base / "clean.xlsx", rows[:60])
    EXPECTED["tabular/clean.xlsx"] = {"rows": 60, "sheet": "Sheet1", "DAT-2": "same columns as clean.csv"}
    try:  # optional: run with an interpreter that has pyarrow (e.g. backend/.venv/bin/python) to get the Parquet fixture
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(rows[:80]), base / "clean.parquet")
        EXPECTED["tabular/clean.parquet"] = {"rows": 80, "DAT-2": "same columns and types as clean.csv"}
    except ImportError:
        print("pyarrow not available: skipped tabular/clean.parquet")


def _csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _xlsx(path, rows):
    """Minimal valid .xlsx (inline strings) without openpyxl."""
    from xml.sax.saxutils import escape

    def col(i):
        s = ""
        i += 1
        while i:
            i, rem = divmod(i - 1, 26)
            s = chr(65 + rem) + s
        return s

    def cell(ref, v):
        if isinstance(v, (int, float)):
            return f'<c r="{ref}"><v>{v}</v></c>'
        return f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(v))}</t></is></c>'

    header = list(rows[0])
    table = [header] + [[r[h] for h in header] for r in rows]
    sheet_rows = "".join(f'<row r="{y + 1}">' + "".join(cell(f"{col(x)}{y + 1}", v) for x, v in enumerate(row)) + "</row>" for y, row in enumerate(table))
    files = {
        "[Content_Types].xml": '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>',
        "_rels/.rels": '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>',
        "xl/workbook.xml": '<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>',
        "xl/_rels/workbook.xml.rels": '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>',
        "xl/worksheets/sheet1.xml": f'<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>{sheet_rows}</sheetData></worksheet>',
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, body in files.items():
            z.writestr(name, body)


# ---------- text ----------

def build_text():
    base = ROOT / "text"
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True)
    en_pos = ["Great service, delivery was quick", "Loved the product quality", "Support resolved my issue fast", "Very happy with the purchase"]
    en_neg = ["Order arrived broken", "Refund still not processed", "Worst support experience", "App keeps crashing on login"]
    rows = []
    for i in range(12):
        rows.append({"text": en_pos[i % 4] + f" #{i}", "label": "positive"})
        rows.append({"text": en_neg[i % 4] + f" #{i}", "label": "negative"})
    hinglish = ["delivery bahut late thi, bilkul bekaar", "product accha hai, paisa vasool", "customer care ne help nahi ki", "mast quality, dobara order karunga"]
    devanagari = ["सेवा बहुत अच्छी थी", "पैसे वापस नहीं मिले", "उत्पाद की गुणवत्ता खराब है", "बहुत बढ़िया अनुभव"]
    tamil = ["சேவை மிகவும் நன்றாக இருந்தது", "பணம் திரும்ப வரவில்லை"]
    for i, t in enumerate(hinglish):
        rows.append({"text": t, "label": "negative" if i % 2 == 0 else "positive"})
    for i, t in enumerate(devanagari):
        rows.append({"text": t, "label": "positive" if i in (0, 3) else "negative"})
    for i, t in enumerate(tamil):
        rows.append({"text": t, "label": "positive" if i == 0 else "negative"})
    empties = ["", "   ", "ok", "k", "👍"]
    for i, t in enumerate(empties):
        rows.append({"text": t, "label": "positive" if i % 2 else "negative"})
    _csv(base / "mixed.csv", rows)
    EXPECTED["text/mixed.csv"] = {
        "rows": len(rows),
        "VAL-14": {"empty_or_whitespace": 2, "very_short": 3, "scripts": {"Latin": 24 + 4 + 2, "Devanagari": 4, "Tamil": 2, "emoji_only": 1}, "hinglish_romanised": 4},
        "note": f"non-English or code-mixed = 10/{len(rows)} ({100 * 10 / len(rows):.1f}%) > 10%, so expect the Indic/multilingual backbone suggestion",
    }


# ---------- detection ----------

def build_detection():
    base = ROOT / "detection"
    shutil.rmtree(base, ignore_errors=True)
    W = H = 128
    n_img = 20
    for i in range(n_img):
        write(base / f"images/det_{i:03d}.png", png_bytes(W, H, seed=f"det-{i}"))
    cats = [{"id": 1, "name": "helmet"}, {"id": 2, "name": "vest"}, {"id": 3, "name": "glove"}]
    anns = []
    aid = 1

    def add(img, cat, x, y, w, h):
        nonlocal aid
        anns.append({"id": aid, "image_id": img, "category_id": cat, "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
        aid += 1

    boxed = list(range(n_img - 2))  # last 2 images: no boxes and not flagged "no objects"
    # helmet: 3 boxes on each of 18 images = 54 (>= 50, no count flag)
    for img in boxed:
        for k in range(3):
            add(img, 1, 4 + k * 40, 10, 30, 30)
    # vest: 20 boxes (15-49 -> warning)
    for k in range(20):
        add(boxed[k % len(boxed)], 2, 10, 60, 40, 40)
    # glove: 10 boxes (< 15 -> error)
    for k in range(10):
        add(boxed[k % len(boxed)], 3, 70, 80, 20, 20)
    # defects (counted inside the class totals above? no: these are extra helmet boxes)
    add(0, 1, 100, 100, 60, 60)   # out of bounds (x+w = 160 > 128)
    add(1, 1, 50, 50, 0, 20)      # zero area
    add(2, 1, 20, 100, 5, 5)      # tiny (< 8 px)
    coco = {
        "images": [{"id": i, "file_name": f"det_{i:03d}.png", "width": W, "height": H} for i in range(n_img)],
        "annotations": anns,
        "categories": cats,
    }
    write(base / "coco_defects.json", json.dumps(coco, indent=1).encode())
    EXPECTED["detection/coco_defects.json"] = {
        "images": n_img,
        "boxes": {"helmet": 54 + 3, "vest": 20, "glove": 10},
        "VAL-16": {
            "out_of_bounds": 1, "zero_area": 1, "tiny_lt_8px": 1,
            "images_without_boxes_unflagged": 2,
            "class_count_warning": ["vest"], "class_count_error": ["glove"],
        },
    }

    # Clean small import set for DAT-8 in all three formats (first 4 images, helmet + vest only).
    clean = [a for a in anns if a["image_id"] < 4 and a["category_id"] in (1, 2) and a["bbox"][2] >= 8 and a["bbox"][3] >= 8 and a["bbox"][0] + a["bbox"][2] <= W]
    write(base / "coco_clean.json", json.dumps({"images": coco["images"][:4], "annotations": clean, "categories": cats[:2]}, indent=1).encode())
    for i in range(4):
        lines, objs = [], []
        for a in (a for a in clean if a["image_id"] == i):
            x, y, w, h = a["bbox"]
            lines.append(f"{a['category_id'] - 1} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
            name = cats[a["category_id"] - 1]["name"]
            objs.append(f"<object><name>{name}</name><bndbox><xmin>{x}</xmin><ymin>{y}</ymin><xmax>{x + w}</xmax><ymax>{y + h}</ymax></bndbox></object>")
        write(base / f"yolo/labels/det_{i:03d}.txt", ("\n".join(lines) + "\n").encode())
        write(base / f"voc/det_{i:03d}.xml", f"<annotation><filename>det_{i:03d}.png</filename><size><width>{W}</width><height>{H}</height><depth>3</depth></size>{''.join(objs)}</annotation>".encode())
    write(base / "yolo/classes.txt", b"helmet\nvest\n")
    per_class = {"helmet": sum(1 for a in clean if a["category_id"] == 1), "vest": sum(1 for a in clean if a["category_id"] == 2)}
    EXPECTED["detection/clean_import"] = {"images": 4, "boxes": per_class, "formats": ["coco_clean.json", "yolo/", "voc/"], "DAT-8": "same box counts from all three formats"}


# ---------- hostile archives (NFR-SEC6) ----------

def build_zips(bombs):
    base = ROOT / "zips"
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True)

    with zipfile.ZipFile(base / "valid_folders.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for cls in ("cat", "dog"):
            for i in range(12):
                z.writestr(f"{cls}/img_{i:03d}.png", png_bytes(64, 64, seed=f"zip-{cls}-{i}"))
    EXPECTED["zips/valid_folders.zip"] = {"DAT-1": {"labels": {"cat": 12, "dog": 12}}}

    with zipfile.ZipFile(base / "traversal.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("cat/ok.png", png_bytes(64, 64, seed="trav-ok"))
        z.writestr("../../escape.txt", b"path traversal marker AUTOML-TRAVERSAL-1\n")
        z.writestr("cat/../../../outside.png", png_bytes(64, 64, seed="trav-2"))
        z.writestr("/abs/absolute.png", png_bytes(64, 64, seed="trav-3"))
        z.writestr("..\\..\\windows_escape.txt", b"backslash traversal marker AUTOML-TRAVERSAL-2\n")
    EXPECTED["zips/traversal.zip"] = {"NFR-SEC6": "reject the archive, or drop the 4 hostile entries; nothing written outside the upload prefix", "hostile_entries": ["../../escape.txt", "cat/../../../outside.png", "/abs/absolute.png", "..\\..\\windows_escape.txt"]}

    info = zipfile.ZipInfo("cat/link.png")
    info.create_system = 3
    info.external_attr = (0o120777 << 16)  # symlink
    with zipfile.ZipFile(base / "symlink.zip", "w") as z:
        z.writestr("cat/ok.png", png_bytes(64, 64, seed="sym-ok"))
        z.writestr(info, "/etc/passwd")
    EXPECTED["zips/symlink.zip"] = {"NFR-SEC6": "symlink entry not followed or materialised"}

    if not bombs:
        return
    # 1 GiB of zeros -> ~1 MB deflated (ratio ~1000:1).
    with zipfile.ZipFile(base / "bomb_1gib.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        with z.open("cat/big.png", "w", force_zip64=True) as f:
            chunk = b"\0" * (1 << 20)
            for _ in range(1024):
                f.write(chunk)
    EXPECTED["zips/bomb_1gib.zip"] = {"declared_uncompressed_bytes": 1 << 30, "NFR-SEC6": "rejected by ratio/size limit before full extraction; worker memory and disk stay bounded"}

    # Decompression-bomb PNG: 20000 x 20000 grayscale (400 MP) of zeros, a few hundred KB on disk.
    w = h = 20000
    comp = zlib.compressobj(9)
    row = b"\0" * (w + 1)
    idat = bytearray()
    for _ in range(h):
        idat += comp.compress(row)
    idat += comp.flush()
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    write(ROOT / "images/decompression_bomb/cat/huge.png", b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", bytes(idat)) + _chunk(b"IEND", b""))
    EXPECTED["images/decompression_bomb/cat/huge.png"] = {"pixels": w * h, "NFR-SEC6": "image decompression-bomb limit: rejected or flagged VAL-1, never fully decoded"}


def main():
    bombs = "--no-bombs" not in sys.argv
    build_images()
    build_tabular()
    build_text()
    build_detection()
    build_zips(bombs)
    (ROOT / "EXPECTED.json").write_text(json.dumps(EXPECTED, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"fixtures written to {ROOT}")


if __name__ == "__main__":
    main()
