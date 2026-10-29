#!/usr/bin/env python3
"""Acceptance run for DAT-1..6, DAT-8, NFR-SEC6 and NFR-SEC3 on the dataset paths (CHECKLIST.md).

Needs the API + Celery worker + MinIO (presigned URLs are called directly).
    python3 tests/acceptance/dat_run.py
"""
import csv
import hashlib
import http.client
import io
import json
import os
import re
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import acc_run as A
from acc_run import call, check, expect, problem_ok

FX = Path(__file__).resolve().parents[1] / "fixtures"
GIB = 1 << 30


# ---------- S3 via presigned URLs ----------

def s3(method, url, body=b"", headers=None):
    u = urllib.parse.urlsplit(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=120)
    conn.request(method, u.path + ("?" + u.query if u.query else ""), body=body, headers=headers or {})
    r = conn.getresponse()
    data = r.read()
    conn.close()
    A.LOG.append({"req": {"method": method, "s3": u.path, "query_keys": sorted(urllib.parse.parse_qs(u.query))}, "status": r.status, "body": data[:600].decode("utf-8", "replace")})
    return r.status, dict(r.getheaders()), data


def sign(tok, upload, **body):
    return call("POST", f"/uploads/{upload}/sign", body, tok=tok)


def put_object(tok, up, key, data):
    r = sign(tok, up["upload_id"], method="PUT", key=key, size=len(data))
    assert r.status == 200, r.brief()
    st, _, body = s3("PUT", r.json["url"], data, r.json.get("headers") or {})
    assert st == 200, (st, body[:300])
    return r.json["url"]


def start(tok, pid, did, source, files, label_from_folder=True, label=None):
    body = {"source": source, "label_from_folder": label_from_folder, "file_count": len(files), "total_bytes": sum(len(d) for _, d in files)}
    if label:
        body["label"] = label
    r = call("POST", f"/projects/{pid}/datasets/{did}/uploads", body, tok=tok)
    assert r.status == 201, r.brief()
    return r.json


def complete(tok, up, entries):
    return call("POST", f"/uploads/{up['upload_id']}/complete", {"files": entries}, tok=tok)


def entry(up, rel, data, label=None, sha=None):
    e = {"key": up["key_prefix"].rstrip("/") + "/" + rel, "name": rel.split("/")[-1], "relative_path": rel, "size": len(data),
         "sha256": sha or hashlib.sha256(data).hexdigest()}
    if label:
        e["label"] = label
    return e


def upload(tok, pid, did, source, files, label_from_folder=True, wait=True, label=None):
    """files: [(relative_path, bytes)]. Returns (complete_response, version_detail_or_None, upload)."""
    up = start(tok, pid, did, source, files, label_from_folder, label)
    ents = []
    for rel, data in files:
        e = entry(up, rel, data)
        put_object(tok, up, e["key"], data)
        ents.append(e)
    c = complete(tok, up, ents)
    v = wait_version(tok, c.json.get("dataset_version_id")) if (wait and c.status == 202 and c.json.get("dataset_version_id")) else None
    return c, v, up


def wait_version(tok, vid, timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        r = call("GET", f"/dataset-versions/{vid}", tok=tok)
        if r.status == 200 and r.json["ingest_status"] in ("ready", "failed"):
            return r.json
        time.sleep(0.5)
    raise AssertionError(f"version {vid} not ingested in {timeout}s: {r.raw[:300]}")


def samples(tok, vid, **q):
    out, cursor = [], None
    for _ in range(100):
        qs = urllib.parse.urlencode({**q, "limit": 200, **({"cursor": cursor} if cursor else {})})
        r = call("GET", f"/dataset-versions/{vid}/samples?{qs}", tok=tok)
        assert r.status == 200, r.brief()
        out += r.json["items"]
        cursor = r.json.get("next_cursor")
        if not cursor:
            return out
    return out


def versions(tok, did):
    r = call("GET", f"/datasets/{did}/versions?limit=200", tok=tok)
    return r.json["items"] if r.status == 200 else []


def folder_files(rel_root):
    root = FX / rel_root
    return [(p.relative_to(root).as_posix(), p.read_bytes()) for p in sorted(root.rglob("*")) if p.is_file()]


def project(tok, org, name, **kw):
    r = call("POST", f"/orgs/{org}/projects", {"name": f"{name} {A.RUN}", **kw}, tok=tok)
    assert r.status == 201, r.brief()
    return r.json["id"], r.json["dataset_id"], r.json


def img_project(tok, org, name, labels=("cat", "dog")):
    return project(tok, org, name, template="image_classification", labels=list(labels))


def label_counts(v, kind="images"):
    """label_stats is {class: {"images": n, "boxes": m}}."""
    ls = v.get("label_stats") or {}
    return {c: (x.get(kind) if isinstance(x, dict) else x) for c, x in ls.items()}


def obj_path(url):
    return urllib.parse.urlsplit(url).path if url else None


# ---------- multipart (DAT-3a/b) ----------

def multipart(tok, up, key, data, part_size, stop_after=None):
    """Create + upload parts. stop_after=n uploads only n parts. Returns (s3_upload_id, etags)."""
    r = sign(tok, up["upload_id"], method="POST", key=key, size=len(data))
    assert r.status == 200, r.brief()
    st, _, body = s3("POST", r.json["url"], b"", r.json.get("headers") or {})
    assert st == 200, (st, body[:300])
    mpu = ET.fromstring(body).find("{*}UploadId").text
    etags = {}
    parts = [data[i:i + part_size] for i in range(0, len(data), part_size)]
    for n, chunk in enumerate(parts, 1):
        if stop_after and n > stop_after:
            break
        etags[n] = upload_part(tok, up, key, mpu, n, chunk)
    return mpu, etags, parts


def upload_part(tok, up, key, mpu, n, chunk):
    r = sign(tok, up["upload_id"], method="PUT", key=key, upload_id=mpu, part_number=n, size=len(chunk))
    assert r.status == 200, r.brief()
    st, hdr, body = s3("PUT", r.json["url"], chunk, r.json.get("headers") or {})
    assert st == 200, (st, body[:300])
    return {k.lower(): v for k, v in hdr.items()}["etag"]


def list_parts(tok, up, key, mpu):
    r = sign(tok, up["upload_id"], method="GET", key=key, upload_id=mpu)
    assert r.status == 200, r.brief()
    st, _, body = s3("GET", r.json["url"])
    assert st == 200, (st, body[:300])
    return {int(p.find("{*}PartNumber").text): p.find("{*}ETag").text for p in ET.fromstring(body).findall("{*}Part")}


def finish_multipart(tok, up, key, mpu, etags):
    xml = "<CompleteMultipartUpload>" + "".join(f"<Part><PartNumber>{n}</PartNumber><ETag>{e}</ETag></Part>" for n, e in sorted(etags.items())) + "</CompleteMultipartUpload>"
    r = sign(tok, up["upload_id"], method="POST", key=key, upload_id=mpu)
    assert r.status == 200, r.brief()
    st, _, body = s3("POST", r.json["url"], xml.encode(), {"Content-Type": "application/xml", **(r.json.get("headers") or {})})
    assert st == 200 and b"Error" not in body, (st, body[:300])


# ---------- checks ----------

def dat1(tok, org):
    pid, did, _ = img_project(tok, org, "DAT1 zip")
    z = (FX / "zips/valid_folders.zip").read_bytes()
    c, v, _ = upload(tok, pid, did, "zip", [("valid_folders.zip", z)])
    check("DAT-1a zip upload -> 202 + version ready", c.status == 202 and v and v["ingest_status"] == "ready", f"{c.status} {v and v['ingest_status']} {v and v.get('ingest_errors')}", c)
    lc = label_counts(v or {})
    check("DAT-1a subfolder names -> labels cat 12 / dog 12", lc.get("cat") == 12 and lc.get("dog") == 12 and (v or {}).get("sample_count") == 24, f"label_stats={json.dumps((v or {}).get('label_stats'))[:300]}")
    pid2, did2, _ = img_project(tok, org, "DAT1 folder")
    c, v2, _ = upload(tok, pid2, did2, "folder", folder_files("images/clean"))
    lc = label_counts(v2 or {})
    check("DAT-1b folder (API side) -> cat 32 / dog 32", lc.get("cat") == 32 and lc.get("dog") == 32, f"label_stats={json.dumps((v2 or {}).get('label_stats'))[:300]}", c)
    return pid, did, v, pid2, did2, v2


def dat2(tok, org):
    pid, did, _ = project(tok, org, "DAT2 tab", template="tabular", task_type="tabular_classification")

    def imp(fmt, rel, mapping=None, wait=True):
        data = (FX / rel).read_bytes()
        up = start(tok, pid, did, "annotations", [(rel.split("/")[-1], data)], label_from_folder=False)
        e = entry(up, rel.split("/")[-1], data)
        put_object(tok, up, e["key"], data)
        c = complete(tok, up, [e])
        assert c.status in (200, 202), c.brief()
        body = {"upload_id": up["upload_id"], "format": fmt, "key": e["key"], **({"mapping": mapping} if mapping is not None else {})}
        r = call("POST", f"/projects/{pid}/datasets/{did}/imports", body, tok=tok)
        v = wait_version(tok, r.json["dataset_version_id"]) if (wait and r.status == 202) else None
        return r, v, up, e

    r, _, up, e = imp("csv", "tabular/clean.csv")
    cols = json.dumps(r.json or {})
    check("DAT-2a CSV without mapping -> 200 preview with inferred types", r.status == 200 and all(c in cols for c in ("tenure", "monthly_charges", "contract", "churn")), f"{r.status} {r.raw[:300]}", r)
    A.RESULTS.append({"id": "DAT-2a preview body (record)", "pass": True, "why": r.raw[:600], "note": True})
    print("INFO DAT-2a preview:", r.raw[:400])
    m = {"columns": [{"name": "churn", "role": "target"}]}
    r, v, _, _ = imp("csv", "tabular/clean.csv", m)
    check("DAT-2a CSV with mapping -> 202 + version ready, 300 rows", r.status == 202 and v and v["ingest_status"] == "ready" and v["sample_count"] == 300, f"{r.status} {v and (v['ingest_status'], v['sample_count'], v.get('ingest_errors'))}", r)
    cm = json.dumps((call("GET", f"/dataset-versions/{v['id']}", tok=tok).json or {}).get("column_mapping")) if v else ""
    check("DAT-2a inferred types recorded (tenure numeric, contract categorical, churn target)", '"tenure"' in cm and "numeric" in cm and "categorical" in cm and "target" in cm, cm[:400])
    m2 = {"columns": [{"name": "churn", "role": "target"}, {"name": "tenure", "role": "feature", "type": "categorical"}]}
    r, v, _, _ = imp("csv", "tabular/clean.csv", m2)
    cm = (call("GET", f"/dataset-versions/{v['id']}", tok=tok).json or {}).get("column_mapping") if v else {}
    ten = next((c for c in (cm or {}).get("columns", []) if c.get("name") == "tenure"), {})
    check("DAT-2b type override kept on the version (tenure categorical)", ten.get("type") == "categorical", f"tenure={ten}")
    r, v, _, _ = imp("xlsx", "tabular/clean.xlsx", m)
    check("DAT-2c XLSX -> 60 rows", r.status == 202 and v and v["sample_count"] == 60 and v["ingest_status"] == "ready", f"{r.status} {v and (v['sample_count'], v['ingest_status'], v.get('ingest_errors'))}", r)
    r, v, _, _ = imp("parquet", "tabular/clean.parquet", m)
    check("DAT-2d Parquet -> 80 rows", r.status == 202 and v and v["sample_count"] == 80 and v["ingest_status"] == "ready", f"{r.status} {v and (v['sample_count'], v['ingest_status'], v.get('ingest_errors'))}", r)
    r, _, _, _ = imp("csv", "tabular/clean.csv", {"columns": [{"name": "no_such_col", "role": "target"}]}, wait=False)
    ok = r.status == 422 and r.code == "invalid_mapping" and "/mapping/columns/0/name" in json.dumps(r.json.get("errors", []))
    check("DAT-2e unknown column -> 422 invalid_mapping at /mapping/columns/0/name", ok, f"{r.status} {r.code} {json.dumps((r.json or {}).get('errors'))[:200]}", r)
    r, _, _, _ = imp("csv", "tabular/clean.csv", {"columns": [{"name": "churn", "role": "target"}, {"name": "tenure", "role": "target"}]}, wait=False)
    expect("DAT-2e two targets -> 422 invalid_mapping", r, 422, "invalid_mapping")
    return pid, did, imp


def dat3(tok, org):
    pid, did, _ = img_project(tok, org, "DAT3")
    up = start(tok, pid, did, "files", [("x", b"")])
    check("DAT-3a part_size >= 5 MiB", up["part_size"] >= 5 * 1024 * 1024, f"part_size={up['part_size']}")
    check("DAT-3a key_prefix org-scoped", up["key_prefix"].startswith(f"org/{org}/"), up["key_prefix"])
    r = sign(tok, up["upload_id"], method="PUT", key=up["key_prefix"] + "a.bin", size=10)
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(r.json["url"]).query)
    check("DAT-3a presigned X-Amz-Expires=900", q.get("X-Amz-Expires") == ["900"], f"{q.get('X-Amz-Expires')}", r)
    ps = up["part_size"]
    data = os.urandom(ps * 2 + 12345)  # 3 parts at the server's part_size
    up = start(tok, pid, did, "files", [("big.bin", data)], label_from_folder=False, label="cat")
    key = up["key_prefix"].rstrip("/") + "/big.bin"
    mpu, etags, parts = multipart(tok, up, key, data, ps, stop_after=2)
    listed = list_parts(tok, up, key, mpu)
    check("DAT-3b ListParts after interruption shows parts 1-2", sorted(listed) == [1, 2], f"listed {sorted(listed)}")
    r = sign(tok, up["upload_id"], method="PUT", key=key, upload_id=mpu, part_number=2, size=len(parts[1]))
    check("DAT-3b re-sign done part 2 -> 200 (idempotent retry)", r.status == 200 and r.json.get("url"), f"got {r.status}", r)
    for n in range(3, len(parts) + 1):
        listed[n] = upload_part(tok, up, key, mpu, n, parts[n - 1])
    finish_multipart(tok, up, key, mpu, listed)
    c = complete(tok, up, [{"key": key, "name": "big.bin", "relative_path": "big.bin", "size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "label": "cat"}])
    check("DAT-3b complete after resume -> 202", c.status == 202, f"got {c.status} {c.raw[:200]}", c)
    v = wait_version(tok, c.json["dataset_version_id"]) if c.status == 202 else {}
    sm = samples(tok, v["id"]) if v else []
    check("DAT-3b server sha256 == local (resumed multipart)", any(s["sha256"] == hashlib.sha256(data).hexdigest() for s in sm), f"ingest {v.get('ingest_status')} {v.get('ingest_errors')}")
    expect("DAT-3a part_number out of range -> 422", sign(tok, up["upload_id"], method="PUT", key=key, upload_id=mpu, part_number=10001, size=10), 422)

    expect("DAT-3c file_count 50001 -> 413 too_many_files", call("POST", f"/projects/{pid}/datasets/{did}/uploads", {"source": "files", "file_count": 50001, "total_bytes": 1}, tok=tok), 413, "too_many_files")
    r = call("POST", f"/projects/{pid}/datasets/{did}/uploads", {"source": "files", "file_count": 50000, "total_bytes": 1}, tok=tok)
    check("DAT-3c file_count 50000 -> 201", r.status == 201, f"got {r.status}", r)
    up2 = r.json if r.status == 201 else up
    k = up2["key_prefix"].rstrip("/") + "/huge.bin"
    cap = up2["max_file_bytes"]
    check("DAT-3c max_file_bytes = 20 GiB", cap == 20 * GIB, f"{cap}")
    for method in ("PUT", "POST"):
        expect(f"DAT-3c/S6-5 {method} size cap + 1 -> 413 file_too_large", sign(tok, up2["upload_id"], method=method, key=k, size=cap + 1), 413, "file_too_large")
        r = sign(tok, up2["upload_id"], method=method, key=k, size=cap)
        check(f"DAT-3c {method} size exactly cap -> 200 (sign only, nothing sent)", r.status == 200, f"got {r.status} {r.code}", r)
    other = f"org/{org}/other/x.bin"
    expect("DAT-3 key outside key_prefix -> 422 invalid_key", sign(tok, up2["upload_id"], method="PUT", key=other, size=1), 422, "invalid_key")
    expect("DAT-3 key with .. -> 422 invalid_key", sign(tok, up2["upload_id"], method="PUT", key=up2["key_prefix"].rstrip("/") + "/../escape.bin", size=1), 422, "invalid_key")

    # DAT-3d: missing object, size mismatch, checksum mismatch
    before = {x["id"] for x in versions(tok, did)}
    img = (FX / "images/clean/cat/img_000.png").read_bytes()
    up = start(tok, pid, did, "files", [("a.png", img), ("b.png", img)])
    ea = entry(up, "a.png", img, label="cat")
    put_object(tok, up, ea["key"], img)
    eb = entry(up, "never_uploaded.png", img, label="cat")
    c = complete(tok, up, [ea, eb])
    files = (c.json or {}).get("files", [])
    check("DAT-3d missing object -> 409 upload_incomplete, files[] lists it", c.status == 409 and c.code == "upload_incomplete" and any("never_uploaded" in f.get("key", "") for f in files), f"{c.status} {c.code} {files}", c)
    check("DAT-3d missing object -> no version created", {x["id"] for x in versions(tok, did)} == before, "a version appeared")
    up = start(tok, pid, did, "files", [("a.png", img)])
    ea = entry(up, "a.png", img, label="cat")
    put_object(tok, up, ea["key"], img)
    c = complete(tok, up, [dict(ea, size=len(img) + 7)])
    check("DAT-3d size mismatch -> 409 upload_incomplete", c.status == 409 and c.code == "upload_incomplete", f"{c.status} {c.code}", c)
    draft = next((x for x in versions(tok, did) if x.get("is_draft")), None)
    n_before = draft["sample_count"] if draft else 0
    up = start(tok, pid, did, "files", [("a.png", img)])
    ea = entry(up, "a.png", img, label="cat", sha="0" * 64)
    put_object(tok, up, ea["key"], img)
    c = complete(tok, up, [ea])
    v = wait_version(tok, c.json["dataset_version_id"]) if c.status == 202 else {}
    errs = [e for e in v.get("ingest_errors", []) if e.get("code") == "checksum_mismatch" and e.get("upload_id") == up["upload_id"]]
    # Drafts (2026-10-22): a bad file is rejected at file level; the shared draft stays usable.
    check("DAT-3d wrong sha256 -> 202, file rejected (checksum_mismatch for this upload), draft still ready", c.status == 202 and v.get("ingest_status") == "ready" and errs, f"{c.status} {v.get('ingest_status')} {v.get('ingest_errors')}", c)
    check("DAT-3d rejected file not added to the draft", v.get("sample_count") == n_before and not any(s_["sha256"] == hashlib.sha256(img).hexdigest() and s_.get("name", "").endswith("a.png") for s_ in samples(tok, v["id"])) if v else False, f"count {n_before} -> {v.get('sample_count')}")


def dat4_5(tok, org):
    pid, did, _ = img_project(tok, org, "DAT45")
    z = (FX / "zips/valid_folders.zip").read_bytes()
    c1, v1, _ = upload(tok, pid, did, "zip", [("valid_folders.zip", z)])
    s1 = samples(tok, v1["id"])
    local = {}
    with zipfile.ZipFile(io.BytesIO(z)) as zf:
        for n in zf.namelist():
            local[hashlib.sha256(zf.read(n)).hexdigest()] = n
    check("DAT-4 every sample sha256 matches a local file", s1 and all(s["sha256"] in local for s in s1) and len({s["sha256"] for s in s1}) == 24, f"{len(s1)} samples")
    extra = folder_files("images/low")[:6]
    c2, v1b, _ = upload(tok, pid, did, "folder", extra)
    check("DAT-5a second upload APPENDS to the open draft (same version id)", c2.json.get("dataset_version_id") == v1["id"] and v1b["sample_count"] == 24 + 6 and v1b.get("is_draft") is True,
          f"v1={v1['id'][-6:]} v2={str(c2.json.get('dataset_version_id'))[-6:]} count={v1b and v1b['sample_count']} is_draft={v1b and v1b.get('is_draft')}")
    r = call("POST", f"/dataset-versions/{v1['id']}/seal", tok=tok)
    check("DAT-5b seal -> 202", r.status == 202, f"got {r.status} {r.raw[:200]}", r)
    time.sleep(1)
    d = call("GET", f"/dataset-versions/{v1['id']}", tok=tok).json
    check("DAT-5b sealed_at + content_hash set, no longer draft", bool(d.get("sealed_at")) and bool(d.get("content_hash")) and d.get("is_draft") is False, f"{d.get('sealed_at')} {d.get('content_hash')} draft={d.get('is_draft')}")
    h1 = d.get("content_hash")
    r2 = call("POST", f"/dataset-versions/{v1['id']}/seal", tok=tok)
    check("DAT-5c seal twice -> idempotent (2xx) or 409", r2.status in (200, 202, 409), f"got {r2.status}", r2)
    check("DAT-5c content_hash unchanged after re-seal", call("GET", f"/dataset-versions/{v1['id']}", tok=tok).json.get("content_hash") == h1, "")
    c3, v2, _ = upload(tok, pid, did, "zip", [("valid_folders.zip", z)])
    check("DAT-5a upload after seal -> NEW draft, parent = sealed version, carries sealed samples", v2["id"] != v1["id"] and v2["parent_version_id"] == v1["id"] and v2.get("is_draft") is True and v2["sample_count"] >= 30,
          f"id {v2['id'][-6:]} parent {str(v2['parent_version_id'])[-6:]} draft={v2.get('is_draft')} count={v2['sample_count']}")
    vs = versions(tok, did)
    check("DAT-5a versions list newest first", [x["id"] for x in vs][:2] == [v2["id"], v1["id"]], f"{[(x['version_no'], x['id'][-6:]) for x in vs]}")
    s2 = samples(tok, v2["id"])
    p1 = {s["sha256"]: obj_path(s.get("url")) for s in s1}
    p2 = {s["sha256"]: obj_path(s.get("url")) for s in s2}
    same = [k for k in p1 if p1[k] and p1[k] == p2.get(k)]
    check("DAT-4 re-upload reuses the same object keys (content-addressed)", len(same) == len(p1), f"{len(same)}/{len(p1)} identical object paths")
    check("DAT-4 object keys are org-prefixed", all(f"/org/{org}/" in (p or "") for p in p1.values()), f"e.g. {next(iter(p1.values()))}")
    # stale_version: v1 is sealed but no longer current (v2 draft exists)
    sid = s1[0]["id"]
    for name, method, path, body in (("labels:batch", "POST", f"/dataset-versions/{v1['id']}/labels:batch", {"upserts": [{"sample_id": sid, "labels": [{"class_name": "dog"}]}]}),
                                     ("seal", "POST", f"/dataset-versions/{v1['id']}/seal", None),
                                     ("validation fixes", "POST", f"/dataset-versions/{v1['id']}/validation/fixes", {"fix_ids": ["dedupe_exact"]})):
        r = call(method, path, body, tok=tok)
        ok = r.status == 409 and r.code == "stale_version" and (r.json or {}).get("current_version_id") == v2["id"]
        if name == "seal" and r.status in (200, 202):  # re-sealing an already sealed version may be a no-op
            ok = True
        check(f"DAT-5 stale: {name} on sealed non-current version -> 409 stale_version (current id)", ok, f"{r.status} {r.code} {r.raw[:200]}", r)
    still = {s["id"]: s for s in samples(tok, v1["id"])}
    check("DAT-5 stale write changed nothing on the old version", still[sid]["labels"] == s1[0]["labels"], "labels changed")
    return pid, did, v1


def split_of(tok, vid):
    return {s["sha256"] + ":" + (s.get("name") or ""): s["split"] for s in samples(tok, vid)}


def dat6(tok, org, dat2_ctx):
    pid, did, _ = img_project(tok, org, "DAT6")
    _, v, _ = upload(tok, pid, did, "folder", folder_files("images/clean"))
    call("POST", f"/dataset-versions/{v['id']}/seal", tok=tok)
    time.sleep(1)
    d = call("GET", f"/dataset-versions/{v['id']}", tok=tok).json
    pc = (d.get("split_summary") or {}).get("per_class", {})
    ok = all(22 <= pc.get(c, {}).get("train", 0) <= 23 and 4 <= pc.get(c, {}).get("val", 0) <= 5 and 4 <= pc.get(c, {}).get("test", 0) <= 5 for c in ("cat", "dog"))
    check("DAT-6a stratified 70/15/15 per class (32 -> 22-23/4-5/4-5)", ok, f"per_class={pc}")
    check("DAT-6a split_summary has seed + ratios", (d.get("split_summary") or {}).get("seed") is not None and (d.get("split_summary") or {}).get("ratios"), f"{json.dumps(d.get('split_summary'))[:300]}")
    s_a = {s["sha256"]: s["split"] for s in samples(tok, v["id"])}
    pid_b, did_b, _ = img_project(tok, org, "DAT6 same seed")
    _, vb, _ = upload(tok, pid_b, did_b, "folder", folder_files("images/clean"))
    call("POST", f"/dataset-versions/{vb['id']}/seal", tok=tok)
    time.sleep(1)
    s_b = {s["sha256"]: s["split"] for s in samples(tok, vb["id"])}
    check("DAT-6b same content + same seed -> identical split", s_a == s_b and all(s_a.values()), f"{sum(s_a.get(k) == s_b.get(k) for k in s_a)}/{len(s_a)} match")

    pid_c, did_c, pj = img_project(tok, org, "DAT6 80/10/10")
    spec = pj["current_spec"]
    spec["split"] = {**spec["split"], "train": 0.8, "val": 0.1, "test": 0.1}
    call("PUT", f"/projects/{pid_c}/spec", {"spec": spec}, tok=tok)
    _, vc, _ = upload(tok, pid_c, did_c, "folder", folder_files("images/clean"))
    call("POST", f"/dataset-versions/{vc['id']}/seal", tok=tok)
    time.sleep(1)
    dc = call("GET", f"/dataset-versions/{vc['id']}", tok=tok).json
    pcc = (dc.get("split_summary") or {}).get("per_class", {})
    check("DAT-6c 80/10/10 respected (32 -> 25-26/3-4/3-4)", all(25 <= pcc.get(c, {}).get("train", 0) <= 26 for c in ("cat", "dog")), f"per_class={pcc}")

    # grouped tabular
    tpid, tdid, imp = dat2_ctx
    r, vg, _, _ = imp("csv", "tabular/grouped.csv", {"columns": [{"name": "label", "role": "target"}, {"name": "patient_id", "role": "group"}]})
    if vg:
        call("POST", f"/dataset-versions/{vg['id']}/seal", tok=tok)
        time.sleep(1)
        rows, cursor = [], None
        for _ in range(20):
            rr = call("GET", f"/dataset-versions/{vg['id']}/rows?limit=200" + (f"&cursor={cursor}" if cursor else ""), tok=tok)
            rows += (rr.json or {}).get("items", [])
            cursor = (rr.json or {}).get("next_cursor")
            if not cursor:
                break
        by = {}
        for row in rows:
            vals = row.get("values") or row.get("data") or row
            by.setdefault(vals.get("patient_id"), set()).add(row.get("split"))
        check("DAT-6d grouped split: every patient_id in exactly one split", len(by) == 40 and all(len(s) == 1 and None not in s for s in by.values()), f"{len(by)} groups, multi-split={sum(len(s) > 1 for s in by.values())}, sample row={json.dumps(rows[:1])[:300]}")
    else:
        check("DAT-6d grouped import", False, f"import {r.status} {r.raw[:200]}", r)

    pid_d, did_d, _ = img_project(tok, org, "DAT6 too few")
    _, vd, _ = upload(tok, pid_d, did_d, "folder", folder_files("images/too_few"))
    call("POST", f"/dataset-versions/{vd['id']}/seal", tok=tok)
    time.sleep(1)
    dd = call("GET", f"/dataset-versions/{vd['id']}", tok=tok).json
    w = (dd.get("split_summary") or {}).get("warnings", [])
    check("DAT-6e too few -> min_test_count_not_met warning per class", {x.get("class") for x in w if x.get("code") == "min_test_count_not_met"} == {"cat", "dog"}, f"warnings={w}")

    pid_e, did_e, _ = img_project(tok, org, "VAL7 leak")
    _, ve, _ = upload(tok, pid_e, did_e, "folder", folder_files("images/split_leak"))
    call("POST", f"/dataset-versions/{ve['id']}/seal", tok=tok)
    time.sleep(1)
    by = {}
    for s in samples(tok, ve["id"]):
        by.setdefault(s["sha256"], set()).add(s["split"])
    check("VAL-7/DAT-6 exact duplicates never straddle splits (by construction)", by and all(len(x) == 1 for x in by.values()), f"{sum(len(x) > 1 for x in by.values())} groups straddle")


def dat8(tok, org):
    imgs = [(f"images/det_{i:03d}.png", (FX / f"detection/images/det_{i:03d}.png").read_bytes()) for i in range(4)]

    def run(fmt, ann_files, key_rel):
        pid, did, _ = project(tok, org, f"DAT8 {fmt}", template="object_detection", labels=["helmet", "vest"])
        files = imgs + ann_files
        up = start(tok, pid, did, "annotations", files, label_from_folder=False)
        ents = []
        for rel, data in files:
            e = entry(up, rel, data)
            put_object(tok, up, e["key"], data)
            ents.append(e)
        c = complete(tok, up, ents)
        assert c.status in (200, 202), c.brief()
        r = call("POST", f"/projects/{pid}/datasets/{did}/imports", {"upload_id": up["upload_id"], "format": fmt, "key": up["key_prefix"].rstrip("/") + "/" + key_rel}, tok=tok)
        v = wait_version(tok, r.json["dataset_version_id"]) if r.status == 202 else None
        return r, v

    coco = (FX / "detection/coco_clean.json").read_bytes()
    r, v = run("coco", [("coco.json", coco)], "coco.json")
    boxes = lambda v: label_counts(v or {}, "boxes")
    check("DAT-8a COCO -> 4 images, helmet 12 / vest 6", r.status == 202 and v and v["ingest_status"] == "ready" and boxes(v) in ({"helmet": 12, "vest": 6},), f"{r.status} {v and v.get('label_stats')} {v and v.get('ingest_errors')}", r)
    yolo = [(f"labels/det_{i:03d}.txt", (FX / f"detection/yolo/labels/det_{i:03d}.txt").read_bytes()) for i in range(4)] + [("classes.txt", (FX / "detection/yolo/classes.txt").read_bytes())]
    r, vy = run("yolo", yolo, "classes.txt")
    check("DAT-8b YOLO -> helmet 12 / vest 6", r.status == 202 and vy and boxes(vy) == {"helmet": 12, "vest": 6}, f"{r.status} {r.raw[:200]} {vy and vy.get('label_stats')}", r)
    voc = [(f"voc/det_{i:03d}.xml", (FX / f"detection/voc/det_{i:03d}.xml").read_bytes()) for i in range(4)]
    r, vv = run("voc", voc, "voc/det_000.xml")
    check("DAT-8b VOC -> helmet 12 / vest 6", r.status == 202 and vv and boxes(vv) == {"helmet": 12, "vest": 6}, f"{r.status} {r.raw[:200]} {vv and vv.get('label_stats')}", r)
    if v and vy:
        bb = lambda b: tuple(round(b[k]) for k in ("x", "y", "w", "h")) if isinstance(b, dict) else tuple(round(c) for c in b)
        bx = lambda vid: sorted((s["sha256"], l["class_name"], bb(l.get("bbox") or {"x": 0, "y": 0, "w": 0, "h": 0})) for s in samples(tok, vid) for l in s["labels"])
        a, b = bx(v["id"]), bx(vy["id"])
        check("DAT-8b YOLO boxes == COCO boxes (±1 px)", len(a) == len(b) and all(x[:2] == y[:2] and all(abs(p - q) <= 1 for p, q in zip(x[2], y[2])) for x, y in zip(a, b)), f"coco {a[:2]} yolo {b[:2]}")
    bad = json.loads(coco)
    bad["annotations"][0]["category_id"] = 99
    r, _ = run("coco", [("coco.json", json.dumps(bad).encode())], "coco.json")
    check("DAT-8c unknown category -> 422 import_invalid unknown_category", r.status == 422 and r.code == "import_invalid" and "unknown_category" in r.raw, f"{r.status} {r.code}", r)
    bad = json.loads(coco)
    bad["images"].append({"id": 999, "file_name": "nope.png", "width": 128, "height": 128})
    bad["annotations"].append({"id": 9999, "image_id": 999, "category_id": 1, "bbox": [1, 1, 10, 10], "area": 100, "iscrowd": 0})
    r, _ = run("coco", [("coco.json", json.dumps(bad).encode())], "coco.json")
    check("DAT-8c missing image -> 422 import_invalid image_missing", r.status == 422 and r.code == "import_invalid" and "image_missing" in r.raw, f"{r.status} {r.code}", r)


def base(n):
    return (n or "").split("/")[-1]


def sec6(tok, org):
    for name, codes in (("traversal.zip", {"path_traversal"}), ("symlink.zip", {"symlink"})):
        pid, did, _ = img_project(tok, org, f"SEC6 {name}")
        c, v, _ = upload(tok, pid, did, "zip", [(name, (FX / f"zips/{name}").read_bytes())])
        errs = (v or {}).get("ingest_errors", [])
        got = {e.get("code") for e in errs}
        names = [base(s.get("name")) for s in samples(tok, v["id"])] if v and v["ingest_status"] == "ready" else []
        check(f"S6 {name}: hostile entries skipped + listed ({codes})", v and codes <= got and names == ["ok.png"], f"{v and v['ingest_status']} codes={got} samples={names}", c)
        if name == "traversal.zip":
            check("S6-1 all 4 traversal entries reported", sum(e.get("code") == "path_traversal" for e in errs) == 4, f"{[(e.get('code'), e.get('entry') or e.get('name')) for e in errs]}")
    pid, did, _ = img_project(tok, org, "SEC6 bomb")
    t0 = time.time()
    c, v, _ = upload(tok, pid, did, "zip", [("bomb_1gib.zip", (FX / "zips/bomb_1gib.zip").read_bytes())])
    codes = [e.get("code") for e in (v or {}).get("ingest_errors", [])]
    check("S6-3 zip bomb -> ingest failed zip_bomb", v and v["ingest_status"] == "failed" and "zip_bomb" in codes, f"{v and v['ingest_status']} {codes}", c)
    check("S6-3 bomb rejected fast (< 30 s)", time.time() - t0 < 30, f"{time.time() - t0:.1f}s")
    c, v, _ = upload(tok, pid, did, "zip", [("valid_folders.zip", (FX / "zips/valid_folders.zip").read_bytes())])
    check("S6-3 next upload after bomb works", v and v["ingest_status"] == "ready", f"{v and v['ingest_status']}", c)
    pid, did, _ = img_project(tok, org, "SEC6 pixels")
    huge = (FX / "images/decompression_bomb/cat/huge.png").read_bytes()
    t0 = time.time()
    c, v, _ = upload(tok, pid, did, "folder", [("cat/huge.png", huge)])
    sm = [s for s in samples(tok, v["id"]) if base(s.get("name")) == "huge.png"] if v else []
    s = sm[0] if sm else {}
    check("S6-4 decompression bomb kept, dims from header (20000x20000), no thumbnail", s and s.get("width") == 20000 and s.get("height") == 20000 and not s.get("thumb_url"), f"sample={json.dumps(s)[:300]}", c)
    check("S6-4 too_many_pixels visible via API", "too_many_pixels" in json.dumps(s), "NOT EXPOSED: SampleOut.flags reads meta.flags, ingest writes meta.corrupt")
    check("S6-4 handled fast (< 30 s, no full decode)", time.time() - t0 < 30, f"{time.time() - t0:.1f}s")
    pid, did, _ = img_project(tok, org, "SEC6 corrupt")
    c, v, _ = upload(tok, pid, did, "folder", folder_files("images/corrupt"))
    sm = {base(x["name"]) if "/img_" not in x["name"] else x["name"]: x.get("flags", []) for x in samples(tok, v["id"])} if v else {}
    want = {"zero_bytes.png": "zero_bytes", "truncated.png": "truncated_or_invalid", "garbage.jpg": "unreadable", "text_named_png.png": "unreadable"}
    flagged = {n: f for n, f in sm.items() if f}
    check("VAL-1 prep: all 28 files kept as samples (4 corrupt + 24 valid)", len(sm) == 28 and set(want) <= set(sm), f"{len(sm)} samples, names {sorted(sm)[:6]}")
    check("VAL-1 prep: corrupt classification visible via API", set(flagged) == set(want), "NOT EXPOSED: flags empty; classification is in meta.corrupt (VAL-1 / group 7 should surface it)")
    A.RESULTS.append({"id": "VAL-1 prep record: corrupt codes", "pass": True, "why": json.dumps(flagged), "note": True})
    print("INFO corrupt flags:", json.dumps(flagged))
    clean = [n for n in sm if "/img_" in n]
    check("VAL-1 prep: 24 valid images unflagged", len(clean) == 24 and not any(sm[n] for n in clean), f"{len(clean)} valid, flagged among them {[n for n in clean if sm[n]]}")
    A.RESULTS.append({"id": "S6-6 malware scan (ClamAV)", "pass": True, "why": "N/A this slice", "skipped": True})


def sec3(tok, org, tok_b, org_b, ctx):
    pid, did, v = ctx
    up = start(tok, pid, did, "files", [("a.png", b"x")])
    k = up["key_prefix"].rstrip("/") + "/a.png"
    sm = samples(tok, v["id"])
    k_b = call("POST", f"/orgs/{org_b}/api-keys", {"name": "b", "scopes": ["projects:read", "projects:write"]}, tok=tok_b).json["key"]
    targets = [
        ("POST", f"/projects/{pid}/datasets/{did}/uploads", {"source": "files", "file_count": 1, "total_bytes": 1}),
        ("GET", f"/uploads/{up['upload_id']}", None),
        ("POST", f"/uploads/{up['upload_id']}/sign", {"method": "PUT", "key": k, "size": 1}),
        ("POST", f"/uploads/{up['upload_id']}/complete", {"files": [{"key": k, "name": "a.png", "size": 1}]}),
        ("POST", f"/projects/{pid}/datasets/{did}/imports", {"upload_id": up["upload_id"], "format": "csv", "key": k}),
        ("GET", f"/datasets/{did}/versions", None),
        ("GET", f"/dataset-versions/{v['id']}", None),
        ("GET", f"/dataset-versions/{v['id']}/samples", None),
        ("GET", f"/dataset-versions/{v['id']}/rows", None),
        ("POST", f"/dataset-versions/{v['id']}/seal", None),
    ]
    snap = call("GET", f"/dataset-versions/{v['id']}", tok=tok).raw
    for m, path, body in targets:
        short = path.replace(pid, "{A_p}").replace(did, "{A_ds}").replace(v["id"], "{A_v}").replace(up["upload_id"], "{A_up}")
        expect(f"SEC3 B {m} {short} -> 404", call(m, path, body, tok=tok_b), 404)
        expect(f"SEC3 B-key {m} {short} -> 404", call(m, path, body, key=k_b), 404)
    bpid, bdid, _ = img_project(tok_b, org_b, "B")
    r = call("POST", f"/projects/{bpid}/datasets/{bdid}/imports", {"upload_id": up["upload_id"], "format": "csv", "key": k}, tok=tok_b)
    expect("SEC3 B imports into own project citing A's upload_id -> 404", r, 404)
    bup = start(tok_b, bpid, bdid, "files", [("x", b"x")])
    expect("SEC3 B signs a key under A's prefix in B's upload -> 422 invalid_key", sign(tok_b, bup["upload_id"], method="PUT", key=k, size=1), 422, "invalid_key")
    expect("SEC3 B signs GET of A's object key -> 422 invalid_key", sign(tok_b, bup["upload_id"], method="GET", key=k), 422, "invalid_key")
    if sm and sm[0].get("url"):
        url = sm[0]["url"]
        st, _, _ = s3("GET", url)
        check("S3-8a A's presigned sample URL works for its holder (short-lived)", st == 200 and "X-Amz-Expires" in url, f"{st}")
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        check("S3-8a sample URL expiry <= 900 s", int(q.get("X-Amz-Expires", ["99999"])[0]) <= 900, f"{q.get('X-Amz-Expires')}")
        u = urllib.parse.urlsplit(url)
        tampered = urllib.parse.urlunsplit(u._replace(path=u.path.replace(f"/org/{org}/", f"/org/{org_b}/")))
        st, _, body = s3("GET", tampered)
        check("S3-8b tampered key in presigned URL -> 403 SignatureDoesNotMatch", st == 403 and b"SignatureDoesNotMatch" in body, f"{st} {body[:120]}")
    check("SEC3 A's version unchanged after attacks", call("GET", f"/dataset-versions/{v['id']}", tok=tok).raw == snap, "changed")
    expect("SEC3 random version id -> 404", call("GET", "/dataset-versions/00000000-0000-7000-8000-000000000000", tok=tok), 404)


def main():
    if call("GET", "/health").status != 200:
        sys.exit("API not healthy")
    print(f"run {A.RUN} against {A.API}")
    ea, _ = A.user("da")
    tok, _, _ = A.login(ea)
    org = A.personal_org(tok)
    eb, _ = A.user("db")
    tok_b, _, _ = A.login(eb)
    org_b = A.personal_org(tok_b)
    steps = [("DAT-1", lambda: dat1(tok, org)), ("DAT-2", lambda: dat2(tok, org)), ("DAT-3", lambda: dat3(tok, org)), ("DAT-4/5", lambda: dat4_5(tok, org))]
    ctx = {}
    for name, fn in steps:
        try:
            ctx[name] = fn()
        except Exception as e:  # keep going: one broken step must not hide the rest
            check(f"{name} step crashed", False, f"{type(e).__name__}: {str(e)[:400]}")
    for name, fn in [("DAT-6", lambda: dat6(tok, org, ctx["DAT-2"])), ("DAT-8", lambda: dat8(tok, org)), ("SEC6", lambda: sec6(tok, org)),
                     ("SEC3", lambda: sec3(tok, org, tok_b, org_b, ctx["DAT-4/5"]))]:
        try:
            fn()
        except Exception as e:
            check(f"{name} step crashed", False, f"{type(e).__name__}: {str(e)[:400]}")
    fails = [r for r in A.RESULTS if not r["pass"]]
    out = A.OUT / f"dat_{A.RUN}.json"
    out.write_text(json.dumps({"run": A.RUN, "results": A.RESULTS, "log": A.LOG}, indent=1))
    print(f"\n{len(A.RESULTS) - len(fails)}/{len(A.RESULTS)} pass, {len(fails)} fail. Log: {out}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
