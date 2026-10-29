"""Group 5: uploads (real presigned PUTs to MinIO), ingestion, SEC6 guards, imports, splits, tenant isolation."""

import hashlib
import io
import json
import random
import stat
import zipfile

import httpx
import pytest
from PIL import Image

from app.settings import get_settings
from tests.test_auth import problem

API = "/api/v1"


def png(color=(200, 10, 10), size=(48, 48), mode="RGB") -> bytes:
    """Seeded noise so different colours give different perceptual hashes (solid images all hash alike)."""
    buf = io.BytesIO()
    if mode == "RGB":
        rnd = random.Random(str(color))
        img = Image.frombytes("RGB", size, bytes(rnd.randrange(256) for _ in range(size[0] * size[1] * 3)))
    else:
        img = Image.new(mode, size, 128)
    img.save(buf, "PNG")
    return buf.getvalue()


def project(c, u, template="image_classification", labels=("cat", "dog"), **kw):
    body = {"name": "P", "template": template, **({"labels": list(labels)} if labels else {}), **kw}
    r = c.post(f"{API}/orgs/{u.org_id}/projects", json=body, headers=u.h)
    assert r.status_code == 201, r.text
    return r.json()


def upload(c, u, p, files, source="folder", label_from_folder=True, sha=True, complete=True, **extra):
    r = c.post(f"{API}/projects/{p['id']}/datasets/{p['dataset_id']}/uploads", headers=u.h,
               json={"source": source, "label_from_folder": label_from_folder, "file_count": len(files),
                     "total_bytes": sum(len(b) for _, b in files), **extra})
    assert r.status_code == 201, r.text
    up = r.json()
    done = []
    for rel, data in files:
        key = up["key_prefix"] + rel
        s = c.post(f"{API}/uploads/{up['upload_id']}/sign", headers=u.h,
                   json={"method": "PUT", "key": key, "size": len(data)})
        assert s.status_code == 200, s.text
        assert httpx.put(s.json()["url"], content=data).status_code == 200
        done.append({"key": key, "name": rel.split("/")[-1], "relative_path": rel, "size": len(data),
                     "sha256": hashlib.sha256(data).hexdigest() if sha else None})
    if not complete:
        return up, done
    r = c.post(f"{API}/uploads/{up['upload_id']}/complete", headers=u.h, json={"files": done})
    return up, r


def version(c, u, vid):
    return c.get(f"{API}/dataset-versions/{vid}", headers=u.h).json()


def folder_files(per_class=12):
    return [(f"{cls}/{i:03d}.png", png((i * 7 % 255, 30 if cls == "cat" else 200, i * 3 % 255)))
            for cls in ("cat", "dog") for i in range(per_class)]


def test_folder_upload_labels_thumbnails_checksums(client, alice):
    p = project(client, alice)
    _, r = upload(client, alice, p, folder_files())
    assert r.status_code == 202, r.text
    v = version(client, alice, r.json()["dataset_version_id"])
    assert v["ingest_status"] == "ready" and v["sample_count"] == 24
    assert v["label_stats"] == {"cat": {"images": 12}, "dog": {"images": 12}}
    page = client.get(f"{API}/dataset-versions/{v['id']}/samples?limit=200", headers=alice.h).json()
    s = page["items"][0]
    assert s["thumb_url"] and "X-Amz-Expires=900" in s["url"] and s["width"] == 48
    assert httpx.get(s["url"]).status_code == 200 and httpx.get(s["thumb_url"]).status_code == 200
    src = dict(folder_files())
    for item in page["items"]:
        assert item["sha256"] == hashlib.sha256(src[item["name"]]).hexdigest()
    assert len(client.get(f"{API}/dataset-versions/{v['id']}/samples?label=cat", headers=alice.h).json()["items"]) == 12
    assert client.get(f"{API}/projects/{p['id']}", headers=alice.h).json()["sample_count"] == 24


def test_uploads_append_to_one_draft_and_dedup_objects(client, alice):
    p = project(client, alice)
    _, r1 = upload(client, alice, p, folder_files(3))
    _, r2 = upload(client, alice, p, folder_files(3))
    assert r1.json()["dataset_version_id"] == r2.json()["dataset_version_id"]  # same open draft, no version spam
    v = version(client, alice, r2.json()["dataset_version_id"])
    assert v["sample_count"] == 12 and v["sealed_at"] is None
    keys = {s["url"].split("?")[0] for s in client.get(f"{API}/dataset-versions/{v['id']}/samples?limit=200",
                                                       headers=alice.h).json()["items"]}
    assert len(keys) == 6  # identical bytes stored once
    listed = client.get(f"{API}/datasets/{p['dataset_id']}/versions", headers=alice.h).json()["items"]
    assert [x["version_no"] for x in listed] == [1] and listed[0]["is_draft"]
    # after sealing, the next upload starts a NEW draft that carries the sealed samples
    client.post(f"{API}/dataset-versions/{v['id']}/seal", headers=alice.h)
    _, r3 = upload(client, alice, p, folder_files(1))
    v2 = version(client, alice, r3.json()["dataset_version_id"])
    assert v2["parent_version_id"] == v["id"] and v2["sample_count"] == 14 and v2["sealed_at"] is None


def test_stale_version_fix_and_labels_rejected(client, alice):
    """The Human-vs-Phone loss: acting on an older sealed version must not replace newer data."""
    p = project(client, alice)
    _, r = upload(client, alice, p, folder_files(5) + [("cat/dup.png", folder_files(5)[0][1])])
    old = r.json()["dataset_version_id"]
    client.post(f"{API}/dataset-versions/{old}/seal", headers=alice.h)  # VAL-5 offers dedupe_exact
    sid = client.get(f"{API}/dataset-versions/{old}/samples?limit=1", headers=alice.h).json()["items"][0]["id"]
    _, r2 = upload(client, alice, p, folder_files(2))  # newer data lands in a new draft
    new = r2.json()["dataset_version_id"]
    assert new != old
    for path, payload in ((f"/dataset-versions/{old}/validation/fixes", {"fix_ids": ["dedupe_exact"]}),
                          (f"/dataset-versions/{old}/labels:batch",
                           {"upserts": [{"sample_id": sid, "labels": [{"class_name": "dog"}]}]})):
        body = problem(client.post(f"{API}{path}", headers=alice.h, json=payload), 409, "stale_version")
        assert body["current_version_id"] == new
    assert version(client, alice, new)["sample_count"] == 15
    assert client.get(f"{API}/projects/{p['id']}", headers=alice.h).json()["current_version_id"] == new


def _zip(entries) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data, *attr in entries:
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            if attr:
                info.external_attr = attr[0] << 16
            zf.writestr(info, data)
    return buf.getvalue()


def test_zip_with_hostile_entries(client, alice):
    p = project(client, alice)
    z = _zip([("cat/ok.png", png()), ("dog/ok2.png", png((1, 2, 3))), ("../escape.txt", b"AUTOML-TRAVERSAL-1"),
              ("/abs/escape.txt", b"AUTOML-TRAVERSAL-2"), ("cat/../../x.png", png()),
              ("cat/link.png", b"/etc/passwd", stat.S_IFLNK | 0o777), ("__MACOSX/._ok.png", b"junk")])
    _, r = upload(client, alice, p, [("data.zip", z)], source="zip")
    v = version(client, alice, r.json()["dataset_version_id"])
    assert v["ingest_status"] == "ready" and v["sample_count"] == 2
    codes = sorted(e["code"] for e in v["ingest_errors"])
    assert codes == ["path_traversal", "path_traversal", "path_traversal", "symlink"]
    names = {s["name"] for s in client.get(f"{API}/dataset-versions/{v['id']}/samples", headers=alice.h).json()["items"]}
    assert names == {"cat/ok.png", "dog/ok2.png"}


def test_zip_bomb_rejected(client, alice):
    p = project(client, alice)
    bomb = _zip([("cat/zeros.bin", b"\0" * (40 * 1024 * 1024))])  # ~40 KB compressed, ratio ~1000
    up, r = upload(client, alice, p, [("bomb.zip", bomb)], source="zip")
    assert r.status_code == 202
    v = version(client, alice, r.json()["dataset_version_id"])
    assert v["ingest_status"] == "failed" and v["ingest_errors"][0]["code"] == "zip_bomb"
    assert client.get(f"{API}/uploads/{up['upload_id']}", headers=alice.h).json()["status"] == "failed"
    assert client.get(f"{API}/datasets/{p['dataset_id']}/versions", headers=alice.h).json()["items"] == []
    _, ok = upload(client, alice, p, folder_files(1))  # next upload still works
    assert version(client, alice, ok.json()["dataset_version_id"])["ingest_status"] == "ready"


def test_decompression_bomb_flagged_without_decode(client, alice, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_image_pixels", 1_000_000)
    p = project(client, alice)
    _, r = upload(client, alice, p, [("cat/huge.png", png(size=(2000, 2000), mode="L")), ("cat/ok.png", png())])
    v = version(client, alice, r.json()["dataset_version_id"])
    items = {s["name"]: s for s in client.get(f"{API}/dataset-versions/{v['id']}/samples", headers=alice.h)
             .json()["items"]}
    assert items["cat/huge.png"]["width"] == 2000 and items["cat/huge.png"]["thumb_url"] is None
    assert items["cat/ok.png"]["thumb_url"]


def test_corrupt_and_misnamed_files_kept_for_validation(client, alice):
    p = project(client, alice)
    good = png()
    _, r = upload(client, alice, p, [("cat/zero.png", b""), ("cat/trunc.png", good[:60]), ("dog/garbage.jpg", b"xx" * 50),
                                     ("dog/text_named_png.png", b"hello, not an image"), ("dog/fine.png", good)])
    v = version(client, alice, r.json()["dataset_version_id"])
    assert v["sample_count"] == 5  # VAL-1 (group 7) flags them; ingestion keeps them
    from sqlalchemy import select

    from app import models as m
    from app.db import session_factory

    with session_factory()() as db:
        meta = {s.original_name: s.meta for s in db.scalars(select(m.Sample).where(
            m.Sample.dataset_version_id == v["id"]))}
    assert meta["cat/zero.png"]["corrupt"] in ("zero_bytes", "unreadable")
    assert meta["cat/trunc.png"]["corrupt"] == "truncated_or_invalid"
    assert meta["dog/text_named_png.png"]["corrupt"] == "unreadable"  # by content, not by the .png name
    assert "corrupt" not in meta["dog/fine.png"]
    api = {x["name"]: x for x in client.get(f"{API}/dataset-versions/{v['id']}/samples", headers=alice.h).json()["items"]}
    assert api["cat/trunc.png"]["corrupt"] == "truncated_or_invalid" and api["cat/trunc.png"]["flags"] == ["VAL-1"]
    assert api["dog/fine.png"]["corrupt"] is None and api["dog/fine.png"]["flags"] == []
    flagged = client.get(f"{API}/dataset-versions/{v['id']}/samples?flag=VAL-1", headers=alice.h).json()["items"]
    assert len(flagged) == 4


def test_checksum_mismatch_fails_ingest(client, alice):
    p = project(client, alice)
    up, done = upload(client, alice, p, folder_files(1), complete=False)
    done[0]["sha256"] = "0" * 64
    r = client.post(f"{API}/uploads/{up['upload_id']}/complete", headers=alice.h, json={"files": done})
    assert r.status_code == 202
    v = version(client, alice, r.json()["dataset_version_id"])
    assert v["ingest_status"] == "failed" and v["ingest_errors"][0]["code"] == "checksum_mismatch"
    assert client.get(f"{API}/datasets/{p['dataset_id']}/versions", headers=alice.h).json()["items"] == []


def test_complete_with_missing_file_409_no_version(client, alice):
    p = project(client, alice)
    up, done = upload(client, alice, p, folder_files(1), complete=False)
    done.append({"key": up["key_prefix"] + "cat/never.png", "name": "never.png", "size": 10})
    body = problem(client.post(f"{API}/uploads/{up['upload_id']}/complete", headers=alice.h, json={"files": done}), 409,
                   "upload_incomplete")
    assert body["files"][0]["found"] is None
    assert client.get(f"{API}/datasets/{p['dataset_id']}/versions", headers=alice.h).json()["items"] == []


def test_sign_rules_and_limits(client, alice):
    p = project(client, alice)
    s = get_settings()
    path = f"{API}/projects/{p['id']}/datasets/{p['dataset_id']}/uploads"
    problem(client.post(path, headers=alice.h, json={"file_count": 50_001, "total_bytes": 1}), 413, "too_many_files")
    up = client.post(path, headers=alice.h, json={"file_count": 50_000, "total_bytes": 1}).json()
    sign = f"{API}/uploads/{up['upload_id']}/sign"
    key = up["key_prefix"] + "a/b.png"
    problem(client.post(sign, headers=alice.h, json={"method": "PUT", "key": key, "size": s.max_file_bytes + 1}), 413,
            "file_too_large")
    assert client.post(sign, headers=alice.h, json={"method": "PUT", "key": key, "size": s.max_file_bytes}).status_code == 200
    for bad in ("org/other/uploads/x/a.png", up["key_prefix"] + "../x.png", up["key_prefix"] + "/abs.png"):
        problem(client.post(sign, headers=alice.h, json={"method": "PUT", "key": bad, "size": 1}), 422, "invalid_key")
    url = client.post(sign, headers=alice.h, json={"method": "POST", "key": key, "size": 10**9}).json()["url"]
    assert "uploads" in url and "X-Amz-Expires=900" in url
    problem(client.post(sign, headers=alice.h, json={"method": "PUT", "key": key, "upload_id": "x"}), 422)


def test_multipart_upload_resume_and_complete(client, alice):
    """64 MB-ish file via multipart; list parts after 'crashing' (DAT-3b)."""
    p = project(client, alice)
    up = client.post(f"{API}/projects/{p['id']}/datasets/{p['dataset_id']}/uploads", headers=alice.h,
                     json={"source": "files", "file_count": 1, "total_bytes": 11 * 1024 * 1024}).json()
    sign = f"{API}/uploads/{up['upload_id']}/sign"
    key = up["key_prefix"] + "cat/big.png"
    data = png() + b"\0" * (11 * 1024 * 1024)
    create = httpx.post(client.post(sign, headers=alice.h, json={"method": "POST", "key": key, "size": len(data)})
                        .json()["url"])
    s3_id = create.text.split("<UploadId>")[1].split("</UploadId>")[0]
    part = 5 * 1024 * 1024
    chunks = [data[i:i + part] for i in range(0, len(data), part)]
    etags = []
    for n, chunk in enumerate(chunks[:2], 1):
        u = client.post(sign, headers=alice.h, json={"method": "PUT", "key": key, "upload_id": s3_id, "part_number": n})
        etags.append(httpx.put(u.json()["url"], content=chunk).headers["etag"])
    listed = httpx.get(client.post(sign, headers=alice.h, json={"method": "GET", "key": key, "upload_id": s3_id})
                       .json()["url"]).text
    assert listed.count("<PartNumber>") == 2  # resume knows parts 1-2 are done
    u = client.post(sign, headers=alice.h, json={"method": "PUT", "key": key, "upload_id": s3_id, "part_number": 3})
    etags.append(httpx.put(u.json()["url"], content=chunks[2]).headers["etag"])
    xml = "<CompleteMultipartUpload>" + "".join(
        f"<Part><PartNumber>{i}</PartNumber><ETag>{e}</ETag></Part>" for i, e in enumerate(etags, 1)) + \
        "</CompleteMultipartUpload>"
    done = httpx.post(client.post(sign, headers=alice.h, json={"method": "POST", "key": key, "upload_id": s3_id})
                      .json()["url"], content=xml)
    assert done.status_code == 200, done.text
    r = client.post(f"{API}/uploads/{up['upload_id']}/complete", headers=alice.h, json={"files": [
        {"key": key, "name": "big.png", "relative_path": "cat/big.png", "size": len(data),
         "sha256": hashlib.sha256(data).hexdigest()}]})
    v = version(client, alice, r.json()["dataset_version_id"])
    assert v["ingest_status"] == "ready" and v["sample_count"] == 1
    problem(client.post(sign, headers=alice.h, json={"method": "PUT", "key": key, "size": 1}), 409, "upload_closed")


CSV = "customer_id,tenure,monthly_charges,contract,region,churn,patient_id\n" + "\n".join(
    f"c{i},{i % 60},{20 + i % 50}.5,{'monthly' if i % 3 else 'yearly'},{'north' if i % 2 else 'south'},"
    f"{1 if i % 4 == 0 else 0},p{i // 5}" for i in range(200)) + "\n"


def _import(c, u, p, name, data, fmt, mapping=None, extra_files=()):
    up, r = upload(c, u, p, [(name, data), *extra_files], source="annotations", label_from_folder=False)
    assert r.status_code == 200 and r.json()["status"] == "uploaded", r.text
    body = {"upload_id": up["upload_id"], "format": fmt, "key": up["key_prefix"] + name}
    if mapping is not None:
        body["mapping"] = mapping
    return c.post(f"{API}/projects/{p['id']}/datasets/{p['dataset_id']}/imports", headers=u.h, json=body)


def test_tabular_csv_import_preview_mapping_and_grouped_split(client, alice):
    p = project(client, alice, template="tabular", labels=None)
    prev = _import(client, alice, p, "clean.csv", CSV.encode(), "csv")
    assert prev.status_code == 200, prev.text
    cols = {c["name"]: c for c in prev.json()["columns"]}
    assert prev.json()["row_count"] == 200
    assert cols["tenure"]["inferred_type"] == "numeric" and cols["contract"]["inferred_type"] == "categorical"
    assert cols["churn"]["inferred_type"] == "boolean"
    problem(_import(client, alice, p, "clean.csv", CSV.encode(), "csv", {"columns": [{"name": "nope", "role": "target"}]}),
            422, "invalid_mapping")
    problem(_import(client, alice, p, "clean.csv", CSV.encode(), "csv", {"columns": [
        {"name": "churn", "role": "target"}, {"name": "tenure", "role": "target"}]}), 422, "invalid_mapping")
    mapping = {"columns": [{"name": "churn", "role": "target"}, {"name": "customer_id", "role": "id"},
                           {"name": "patient_id", "role": "group"}, {"name": "tenure", "type": "categorical"}]}
    r = _import(client, alice, p, "clean.csv", CSV.encode(), "csv", mapping)
    assert r.status_code == 202, r.text
    v = version(client, alice, r.json()["dataset_version_id"])
    assert v["sample_count"] == 200 and v["ingest_status"] == "ready"
    types = {c["name"]: c["type"] for c in v["column_mapping"]["columns"]}
    assert types["tenure"] == "categorical"  # override kept (DAT-2b)
    assert client.post(f"{API}/dataset-versions/{v['id']}/seal", headers=alice.h).status_code == 202
    sealed = version(client, alice, v["id"])
    assert sealed["sealed_at"] and sealed["split_strategy"] == "grouped"
    rows, cursor = [], None
    while True:
        page = client.get(f"{API}/dataset-versions/{v['id']}/rows?limit=500" + (f"&cursor={cursor}" if cursor else ""),
                          headers=alice.h).json()
        rows += page["items"]
        if not (cursor := page["next_cursor"]):
            break
    by_group: dict = {}
    for row in rows:
        by_group.setdefault(row["values"]["patient_id"], set()).add(row["split"])
    assert len(by_group) == 40 and all(len(s) == 1 for s in by_group.values())  # DAT-6d


def test_xlsx_parquet_and_text_import(client, alice):
    import pandas as pd

    df = pd.read_csv(io.StringIO(CSV)).head(60)
    xbuf, pbuf = io.BytesIO(), io.BytesIO()
    df.to_excel(xbuf, index=False)
    df.to_parquet(pbuf, index=False)
    p = project(client, alice, template="tabular", labels=None)
    for name, data, fmt in (("t.xlsx", xbuf.getvalue(), "xlsx"), ("t.parquet", pbuf.getvalue(), "parquet")):
        prev = _import(client, alice, p, name, data, fmt).json()
        assert prev["row_count"] == 60 and {c["name"] for c in prev["columns"]} == set(df.columns)
    t = project(client, alice, template="text_classification", labels=("pos", "neg"))
    text_csv = "text,label\n" + "\n".join(f"\"review number {i} is {'great' if i % 2 else 'bad'}\",{'pos' if i % 2 else 'neg'}"
                                         for i in range(20)) + "\n"
    r = _import(client, alice, t, "reviews.csv", text_csv.encode(), "csv",
                {"columns": [{"name": "label", "role": "target"}, {"name": "text", "role": "text"}]})
    assert r.status_code == 202, r.text
    v = version(client, alice, r.json()["dataset_version_id"])
    assert v["label_stats"] == {"neg": {"images": 10}, "pos": {"images": 10}}
    s = client.get(f"{API}/dataset-versions/{v['id']}/samples?limit=1", headers=alice.h).json()["items"][0]
    assert s["text"].startswith("review number")


def _det_images():
    return [(f"images/det_{i:03d}.png", png((i * 40, 80, 120), size=(100, 80))) for i in range(4)]


COCO = {"images": [{"id": i + 1, "file_name": f"det_{i:03d}.png", "width": 100, "height": 80} for i in range(4)],
        "categories": [{"id": 1, "name": "helmet"}, {"id": 2, "name": "vest"}],
        "annotations": [{"id": k, "image_id": 1 + k % 4, "category_id": 1 if k < 12 else 2,
                         "bbox": [10 + k, 5, 20, 30]} for k in range(18)]}


def test_coco_and_yolo_detection_imports(client, alice):
    p = project(client, alice, template="object_detection", labels=("helmet", "vest"))
    r = _import(client, alice, p, "coco.json", json.dumps(COCO).encode(), "coco", extra_files=_det_images())
    assert r.status_code == 202, r.text
    v = version(client, alice, r.json()["dataset_version_id"])
    assert v["label_stats"] == {"helmet": {"images": 4, "boxes": 12}, "vest": {"images": 4, "boxes": 6}}
    s = next(x for x in client.get(f"{API}/dataset-versions/{v['id']}/samples", headers=alice.h).json()["items"]
         if x["name"] == "images/det_000.png")
    assert {"x": 10.0, "y": 5.0, "w": 20.0, "h": 30.0} in [lb["bbox"] for lb in s["labels"]]

    yolo = {f"labels/det_{i:03d}.txt": "".join(
        f"{0 if k < 12 else 1} {(10 + k + 10) / 100:.6f} {(5 + 15) / 80:.6f} {20 / 100:.6f} {30 / 80:.6f}\n"
        for k in range(18) if 1 + k % 4 == i + 1).encode() for i in range(4)}
    p2 = project(client, alice, template="object_detection", labels=("helmet", "vest"))
    r = _import(client, alice, p2, "classes.txt", b"helmet\nvest\n", "yolo",
                extra_files=_det_images() + list(yolo.items()))
    v2 = version(client, alice, r.json()["dataset_version_id"])
    assert v2["label_stats"] == v["label_stats"]

    bad = dict(COCO, annotations=COCO["annotations"] + [{"id": 99, "image_id": 1, "category_id": 7, "bbox": [0, 0, 1, 1]}])
    body = problem(_import(client, alice, p, "coco.json", json.dumps(bad).encode(), "coco", extra_files=_det_images()),
                   422, "import_invalid")
    assert "unknown_category" in json.dumps(body["errors"])


def test_seal_stratified_split_deterministic_and_idempotent(client, alice):
    p = project(client, alice)
    _, r = upload(client, alice, p, folder_files(32))
    vid = r.json()["dataset_version_id"]
    assert client.post(f"{API}/dataset-versions/{vid}/seal", headers=alice.h).status_code == 202
    v = version(client, alice, vid)
    for cls in ("cat", "dog"):
        c = v["split_summary"]["per_class"][cls]
        assert (c["train"], c["val"], c["test"]) == (22, 5, 5)
    assert v["content_hash"] and v["validation_status"] == "passed"
    assert client.post(f"{API}/dataset-versions/{vid}/seal", headers=alice.h).status_code == 200  # idempotent
    assert version(client, alice, vid)["content_hash"] == v["content_hash"]
    # same content + same seed in another project -> identical assignment
    p2 = project(client, alice)
    _, r2 = upload(client, alice, p2, folder_files(32))
    client.post(f"{API}/dataset-versions/{r2.json()['dataset_version_id']}/seal", headers=alice.h)

    def splits_of(vid_):
        items = client.get(f"{API}/dataset-versions/{vid_}/samples?limit=200", headers=alice.h).json()["items"]
        return {s["name"]: s["split"] for s in items}

    assert splits_of(vid) == splits_of(r2.json()["dataset_version_id"])
    assert client.get(f"{API}/projects/{p['id']}", headers=alice.h).json()["status"] == "Ready to train"


def test_cross_tenant_uploads_and_datasets(client, alice, bob):
    p = project(client, alice)
    up, r = upload(client, alice, p, folder_files(1))
    vid = r.json()["dataset_version_id"]
    for method, path, body in (
            ("post", f"/projects/{p['id']}/datasets/{p['dataset_id']}/uploads", {"file_count": 1, "total_bytes": 1}),
            ("post", f"/uploads/{up['upload_id']}/sign", {"method": "PUT", "key": up["key_prefix"] + "x.png", "size": 1}),
            ("post", f"/uploads/{up['upload_id']}/complete", {"files": [{"key": "k", "name": "n", "size": 1}]}),
            ("get", f"/uploads/{up['upload_id']}", None),
            ("get", f"/datasets/{p['dataset_id']}/versions", None),
            ("get", f"/dataset-versions/{vid}", None),
            ("get", f"/dataset-versions/{vid}/samples", None),
            ("post", f"/dataset-versions/{vid}/seal", None)):
        problem(client.request(method.upper(), f"{API}{path}", json=body, headers=bob.h), 404)
    # B imports into its own dataset citing A's upload id -> 404
    pb = project(client, bob)
    problem(client.post(f"{API}/projects/{pb['id']}/datasets/{pb['dataset_id']}/imports", headers=bob.h,
                        json={"upload_id": up["upload_id"], "format": "csv", "key": "k"}), 404)
    # B's project id + A's dataset id
    problem(client.post(f"{API}/projects/{pb['id']}/datasets/{p['dataset_id']}/uploads", headers=bob.h,
                        json={"file_count": 1, "total_bytes": 1}), 404)


@pytest.fixture(autouse=True)
def _fast_parts(monkeypatch):
    monkeypatch.setattr(get_settings(), "part_size_bytes", 5 * 1024 * 1024)


def test_capture_target_hint(client, alice, bob):
    p = project(client, alice, template="object_detection", labels=("charger", "cable"))
    url = f"{API}/projects/{p['id']}/datasets/{p['dataset_id']}/uploads"
    base = {"source": "camera", "file_count": 1, "total_bytes": 10}
    problem(client.post(url, headers=alice.h, json={**base, "capture_target": "phone"}), 422, "unknown_class")
    cls = project(client, alice)
    problem(client.post(f"{API}/projects/{cls['id']}/datasets/{cls['dataset_id']}/uploads", headers=alice.h,
                        json={**base, "capture_target": "cat"}), 422, "capture_target_detection_only")
    _, r = upload(client, alice, p, [("a.png", png((1, 2, 3))), ("b.png", png((4, 5, 6)))], source="camera",
                  label_from_folder=False, capture_target="charger")
    vid = r.json()["dataset_version_id"]
    upload(client, alice, p, [("c.png", png((7, 8, 9)))], source="files", label_from_folder=False)
    rows = client.get(f"{API}/dataset-versions/{vid}/samples?capture_target=charger", headers=alice.h).json()["items"]
    assert len(rows) == 2 and all(s["capture_target"] == "charger" and s["labels"] == [] for s in rows)  # hint only
    none = client.get(f"{API}/dataset-versions/{vid}/samples?capture_target=__none", headers=alice.h).json()["items"]
    assert [s["capture_target"] for s in none] == [None]
    assert version(client, alice, vid)["capture_counts"] == {"charger": 2, "__none": 1}
    problem(client.get(f"{API}/dataset-versions/{vid}/samples?capture_target=charger", headers=bob.h), 404)
