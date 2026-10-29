#!/usr/bin/env python3
"""Self-check: the built fixtures really contain the defects EXPECTED.json promises.
Run after build_fixtures.py:  python3 tests/fixtures/check_fixtures.py"""
import csv
import hashlib
import json
import struct
import zipfile
import zlib
from pathlib import Path

R = Path(__file__).resolve().parent
E = json.loads((R / "EXPECTED.json").read_text())


def png_ok(p):
    b = p.read_bytes()
    if not b.startswith(b"\x89PNG"):
        return False
    try:
        i, idat = 8, b""
        while i < len(b):
            n = struct.unpack(">I", b[i:i + 4])[0]
            if b[i + 4:i + 8] == b"IDAT":
                idat += b[i + 8:i + 8 + n]
            i += 12 + n
        zlib.decompress(idat)
        return True
    except Exception:
        return False


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


bad = [p.relative_to(R / "images/corrupt").as_posix() for p in sorted((R / "images/corrupt").rglob("*")) if p.is_file() and not png_ok(p)]
assert sorted(bad) == sorted(E["images/corrupt"]["files"]), bad
assert all(png_ok(p) for p in (R / "images/clean").rglob("*.png"))
assert len({sha(p) for p in (R / "images/clean").rglob("*.png")}) == 64

groups = {}
for p in (R / "images/dupes").rglob("*.png"):
    groups.setdefault(sha(p), []).append(p.relative_to(R / "images/dupes").as_posix())
dups = [v for v in groups.values() if len(v) > 1]
assert len(dups) == 7, dups
assert sum(len({x.split("/")[0] for x in v}) > 1 for v in dups) == 2
assert all(sha(R / f"images/dupes/dog/near_{i:03d}.png") != sha(R / f"images/dupes/dog/img_{i:03d}.png") for i in range(3))

rows = list(csv.DictReader(open(R / "tabular/leaky.csv", encoding="utf-8")))
assert all(r["churn"] == r["churn_outcome"] for r in rows)
assert all((r["cancelled_date"] != "") == (r["churn"] == "1") for r in rows)
assert len({r["customer_id"] for r in rows}) == 300 and {r["country"] for r in rows} == {"IN"}
assert sum(r["mostly_missing"] == "" for r in rows) / 300 > 0.5

names = zipfile.ZipFile(R / "zips/traversal.zip").namelist()
assert any(n.startswith("../") for n in names) and any(n.startswith("/") for n in names), names
link = zipfile.ZipFile(R / "zips/symlink.zip").getinfo("cat/link.png")
assert (link.external_attr >> 16) & 0o170000 == 0o120000
bomb = R / "zips/bomb_1gib.zip"
if bomb.exists():
    b = zipfile.ZipFile(bomb).infolist()[0]
    assert b.file_size == 1 << 30 and b.compress_size < 5_000_000, (b.file_size, b.compress_size)
assert "xl/worksheets/sheet1.xml" in zipfile.ZipFile(R / "tabular/clean.xlsx").namelist()

c = json.loads((R / "detection/coco_defects.json").read_text())
A = c["annotations"]
oob = [a for a in A if a["bbox"][0] + a["bbox"][2] > 128 or a["bbox"][1] + a["bbox"][3] > 128]
zero = [a for a in A if a["bbox"][2] * a["bbox"][3] == 0]
tiny = [a for a in A if 0 < min(a["bbox"][2], a["bbox"][3]) < 8]
unboxed = set(range(20)) - {a["image_id"] for a in A}
assert (len(oob), len(zero), len(tiny), len(unboxed)) == (1, 1, 1, 2)
per_class = {}
for a in A:
    per_class[a["category_id"]] = per_class.get(a["category_id"], 0) + 1
assert per_class == {1: 57, 2: 20, 3: 10}, per_class
print("fixture self-check OK")
