#!/usr/bin/env python3
"""Acceptance run for VAL-1..14/16, LBL-1/2/4a, PRJ-7c and NFR-SEC3 S3-6 (CHECKLIST.md), driven by fixtures/EXPECTED.json.

    python3 tests/acceptance/val_run.py
"""
import json
import sys
import time

import acc_run as A
import dat_run as D
from acc_run import call, check, expect

FX = D.FX
EXP = json.loads((FX / "EXPECTED.json").read_text())


def report(tok, vid, timeout=180):
    end = time.time() + timeout
    while time.time() < end:
        r = call("GET", f"/dataset-versions/{vid}/validation", tok=tok)
        assert r.status == 200, r.brief()
        if r.json["status"] not in ("pending", "running"):
            return r.json
        time.sleep(0.7)
    raise AssertionError(f"validation of {vid} still running after {timeout}s")


def sealed(tok, v):
    r = call("POST", f"/dataset-versions/{v['id']}/seal", tok=tok)
    assert r.status == 202, r.brief()
    return report(tok, v["id"])


def items(rep, code):
    return [i for i in rep["items"] if i["code"] == code]


def kinds(rep, code):
    return {(i.get("details") or {}).get("kind"): i for i in items(rep, code)}


def blob(x):
    return json.dumps(x, ensure_ascii=False)


def image_set(tok, org, rel, labels=("cat", "dog"), name=None):
    pid, did, pj = D.img_project(tok, org, name or rel, labels)
    _, v, _ = D.upload(tok, pid, did, "folder", D.folder_files(rel))
    assert v and v["ingest_status"] == "ready", v
    return pid, did, v


def table_set(tok, org, rel, mapping, template="tabular", task_type="tabular_classification", labels=None):
    kw = {"template": template}
    if task_type:
        kw["task_type"] = task_type
    if labels:
        kw["labels"] = labels
    pid, did, _ = D.project(tok, org, rel, **kw)
    data = (FX / rel).read_bytes()
    up = D.start(tok, pid, did, "annotations", [(rel.split("/")[-1], data)], label_from_folder=False)
    e = D.entry(up, rel.split("/")[-1], data)
    D.put_object(tok, up, e["key"], data)
    assert D.complete(tok, up, [e]).status in (200, 202)
    fmt = rel.rsplit(".", 1)[-1]
    r = call("POST", f"/projects/{pid}/datasets/{did}/imports", {"upload_id": up["upload_id"], "format": fmt, "key": e["key"], "mapping": mapping}, tok=tok)
    assert r.status == 202, r.brief()
    return pid, did, D.wait_version(tok, r.json["dataset_version_id"])


def detection_set(tok, org, ann_rel="detection/coco_defects.json", n=20, labels=("helmet", "vest", "glove")):
    pid, did, _ = D.project(tok, org, f"det {ann_rel}", template="object_detection", labels=list(labels))
    files = [(f"images/det_{i:03d}.png", (FX / f"detection/images/det_{i:03d}.png").read_bytes()) for i in range(n)] + [("coco.json", (FX / ann_rel).read_bytes())]
    up = D.start(tok, pid, did, "annotations", files, label_from_folder=False)
    ents = []
    for rel, data in files:
        e = D.entry(up, rel, data)
        D.put_object(tok, up, e["key"], data)
        ents.append(e)
    assert D.complete(tok, up, ents).status in (200, 202)
    r = call("POST", f"/projects/{pid}/datasets/{did}/imports", {"upload_id": up["upload_id"], "format": "coco", "key": up["key_prefix"] + "coco.json"}, tok=tok)
    assert r.status == 202, r.brief()
    return pid, did, D.wait_version(tok, r.json["dataset_version_id"])


def fix(tok, vid, rep, code):
    it = items(rep, code)
    fid = next((i["fix"].get("auto_fix_id") for i in it if i.get("fix") and i["fix"].get("auto_fix_id")), None)
    assert fid, f"no fix offered for {code}: {blob(it)[:300]}"
    r = call("POST", f"/dataset-versions/{vid}/validation/fixes", {"fix_ids": [fid]}, tok=tok)
    assert r.status == 202, r.brief()
    nv = D.wait_version(tok, r.json["dataset_version_id"])
    return nv, report(tok, nv["id"])


def run_val(tok, org):
    # VAL-1
    pid, did, v = image_set(tok, org, "images/corrupt")
    sm = {D.base(s["name"]) if "/img_" not in s["name"] else s["name"]: s for s in D.samples(tok, v["id"])}
    want = {"zero_bytes.png": "zero_bytes", "truncated.png": "truncated_or_invalid", "garbage.jpg": "unreadable", "text_named_png.png": "unreadable"}
    got = {n: sm[n].get("corrupt") for n in want if n in sm}
    check("VAL-1 SampleOut.corrupt set from ingest (by content)", all(got.get(n) for n in want), f"corrupt={got}")
    A.RESULTS.append({"id": "VAL-1 record corrupt codes", "pass": True, "why": blob(got), "note": True})
    print("INFO VAL-1 corrupt codes:", blob(got))
    pre = D.samples(tok, v["id"], flag="VAL-1")
    check("VAL-1 ?flag=VAL-1 before seal -> 4", len(pre) == 4, f"{len(pre)}")
    rep = sealed(tok, v)
    it = items(rep, "VAL-1")
    check("VAL-1 error count 4, examples + why + fix", it and it[0]["severity"] == "error" and sum(i["count"] for i in it) == 4 and it[0].get("why") and it[0].get("examples") and it[0].get("fix"), blob(it)[:300])
    check("VAL-1 report status failed", rep["status"] == "failed", rep["status"])
    expect("VAL-1 acknowledge -> 422 not_acknowledgeable", call("POST", f"/dataset-versions/{v['id']}/validation/acknowledge", {"codes": ["VAL-1"]}, tok=tok), 422, "not_acknowledgeable")
    expect("VAL fix not offered -> 422 fix_not_offered", call("POST", f"/dataset-versions/{v['id']}/validation/fixes", {"fix_ids": ["dedupe_exact"]}, tok=tok), 422, "fix_not_offered")
    nv, nrep = fix(tok, v["id"], rep, "VAL-1")
    check("VAL-1 auto-fix -> new version origin fix, 24 samples, no VAL-1", nv["origin"] == "fix" and nv["sample_count"] == 24 and not items(nrep, "VAL-1") and nv["parent_version_id"] == v["id"], f"{nv['origin']} {nv['sample_count']} {[i['code'] for i in nrep['items']]}")
    st = call("GET", f"/projects/{pid}", tok=tok).json["status"]
    A.RESULTS.append({"id": "PRJ-7c record status after fix", "pass": True, "why": st, "note": True})

    # VAL-2/3/4
    rep = sealed(tok, image_set(tok, org, "images/too_few")[2])
    it = items(rep, "VAL-2")
    check("VAL-2 too few (5/class) -> error naming cat and dog", it and all(i["severity"] == "error" for i in it) and "cat" in blob(it) and "dog" in blob(it), blob(it)[:300])
    rep = sealed(tok, image_set(tok, org, "images/low")[2])
    it = items(rep, "VAL-3")
    check("VAL-3 low (15/class) -> warning cat + dog with estimated accuracy band", it and all(i["severity"] == "warning" for i in it) and "cat" in blob(it) and "dog" in blob(it) and "estimated" in blob(it), blob(it)[:300])
    check("VAL-3 not an error (no VAL-2)", not items(rep, "VAL-2"), "")
    rep = sealed(tok, image_set(tok, org, "images/imbalance", labels=("big", "small"))[2])
    it = items(rep, "VAL-4")
    check("VAL-4 imbalance -> warning ratio 11.0 big/small", it and it[0]["severity"] == "warning" and "11" in blob(it) and "big" in blob(it) and "small" in blob(it), blob(it)[:300])
    check("VAL-4 also VAL-3 for small", "small" in blob(items(rep, "VAL-3")), blob(items(rep, "VAL-3"))[:200])

    # VAL-5/6/8 + dedupe fix
    _, _, v = image_set(tok, org, "images/dupes")
    rep = sealed(tok, v)
    v5, v6, v8 = items(rep, "VAL-5"), items(rep, "VAL-6"), items(rep, "VAL-8")
    check("VAL-5 exact duplicates >= 7 groups (warning)", v5 and v5[0]["severity"] == "warning" and sum(i["count"] for i in v5) >= 7, blob(v5)[:300])
    check("VAL-6 near-duplicates = 3 pairs (warning)", v6 and sum(i["count"] for i in v6) == 3 and "near_" in blob(v6), blob(v6)[:300])
    check("VAL-8 label conflicts = 2 (warning)", v8 and sum(i["count"] for i in v8) == 2 and "conflict_" in blob(v8), blob(v8)[:300])
    nv, nrep = fix(tok, v["id"], rep, "VAL-5")
    check("VAL-5 dedupe fix -> new version without VAL-5", not items(nrep, "VAL-5") and nv["sample_count"] < v["sample_count"], f"{v['sample_count']} -> {nv['sample_count']} {[i['code'] for i in nrep['items']]}")
    flagged = D.samples(tok, v["id"], flag="VAL-6")
    check("VAL flags on samples after seal (?flag=VAL-6)", len(flagged) >= 3, f"{len(flagged)}")

    # VAL-7 by construction
    rep = sealed(tok, image_set(tok, org, "images/split_leak")[2])
    check("VAL-7 split_leak: no VAL-7 (exact dups grouped before splitting, by design)", not items(rep, "VAL-7"), blob(items(rep, "VAL-7"))[:200])
    check("VAL-7 split_leak: VAL-5 still reports the 30 duplicate groups", sum(i["count"] for i in items(rep, "VAL-5")) == 30, blob(items(rep, "VAL-5"))[:200])

    # VAL-13
    rep = sealed(tok, image_set(tok, org, "images/quality")[2])
    k = kinds(rep, "VAL-13")
    cnt = {kk: vv["count"] for kk, vv in k.items()}
    exp = EXP["images/quality"]["VAL-13"]
    check("VAL-13 tiny 3 / extreme aspect 2 / grayscale 4", sorted(cnt.values()) == sorted(exp.values()) and len(cnt) == 3, f"{cnt}")

    # control
    rep = sealed(tok, image_set(tok, org, "images/clean")[2])
    check("VAL-ctl images/clean -> 0 errors, 0 warnings", rep["summary"]["errors"] == 0 and rep["summary"]["warnings"] == 0, f"{rep['summary']} {[i['code'] for i in rep['items']]}")

    # tabular
    pid, _, v = table_set(tok, org, "tabular/leaky.csv", {"columns": [{"name": "churn", "role": "target"}]})
    rep = sealed(tok, v)
    v9, v10, v11, v12 = items(rep, "VAL-9"), items(rep, "VAL-10"), items(rep, "VAL-11"), items(rep, "VAL-12")
    check("VAL-9 mostly_missing 58.3% (warning)", "mostly_missing" in blob(v9) and ("0.583" in blob(v9) or "58.3" in blob(v9)), blob(v9)[:300])
    check("VAL-10 customer_id + row_no ID-like (warning)", "customer_id" in blob(v10) and "row_no" in blob(v10), blob(v10)[:300])
    names11 = {e for i in v11 for e in ("churn_outcome", "cancelled_date") if e in blob(i)}
    check("VAL-11 exactly churn_outcome + cancelled_date, error, acknowledgeable", names11 == {"churn_outcome", "cancelled_date"} and all(i["severity"] == "error" and i["acknowledgeable"] for i in v11)
          and not any(c in blob(v11) for c in ("tenure", "monthly_charges", "contract")), blob(v11)[:400])
    check("VAL-12 country constant", "country" in blob(v12), blob(v12)[:300])
    check("VAL-12/10 session_hash reported (high-card or ID-like)", "session_hash" in blob(v12) + blob(v10), "")
    check("VAL-11 blocks: status failed before ack", rep["status"] == "failed", rep["status"])
    st = call("GET", f"/projects/{pid}", tok=tok).json["status"]
    check("PRJ-7c errors keep project out of Ready to train", st != "Ready to train", st)
    r = call("POST", f"/dataset-versions/{v['id']}/validation/acknowledge", {"codes": ["VAL-11"]}, tok=tok)
    rep2 = report(tok, v["id"])
    check("VAL-11 acknowledge -> no longer failed", r.status == 200 and rep2["status"] != "failed" and all(i["acknowledged"] for i in items(rep2, "VAL-11")), f"{r.status} {rep2['status']}")
    info = [i["code"] for i in rep2["items"] if i["severity"] == "info"]
    if info:
        r = call("POST", f"/dataset-versions/{v['id']}/validation/acknowledge", {"codes": [info[0]]}, tok=tok)
        check(f"VAL ack of info-only {info[0]} -> 200 no-op", r.status == 200, f"{r.status}", r)
    nv, nrep = fix(tok, v["id"], rep2, "VAL-10")
    cm = blob(call("GET", f"/dataset-versions/{nv['id']}", tok=tok).json.get("column_mapping"))
    check("VAL-10 exclude_id_columns -> new version, customer_id role id", '"customer_id"' in cm and '"id"' in cm, cm[:300])

    rep = sealed(tok, table_set(tok, org, "tabular/too_few.csv", {"columns": [{"name": "churn", "role": "target"}]})[2])
    check("VAL-2 too_few.csv (40 rows) -> error", any(i["severity"] == "error" for i in items(rep, "VAL-2")), blob(items(rep, "VAL-2"))[:200])
    rep = sealed(tok, table_set(tok, org, "tabular/duplicate_rows.csv", {"columns": [{"name": "churn", "role": "target"}]})[2])
    check("VAL-5 duplicate_rows.csv -> 10", sum(i["count"] for i in items(rep, "VAL-5")) == 10, blob(items(rep, "VAL-5"))[:200])
    pid, _, v = table_set(tok, org, "tabular/clean.csv", {"columns": [{"name": "churn", "role": "target"}]})
    rep = sealed(tok, v)
    check("VAL-ctl clean.csv -> 0 errors", rep["summary"]["errors"] == 0, f"{rep['summary']} {[i['code'] for i in rep['items']]}")
    st = call("GET", f"/projects/{pid}", tok=tok).json["status"]
    check("PRJ-7c clean + validated -> Ready to train (after acknowledging any warnings)", st == "Ready to train" or rep["status"] == "warnings", f"status {st}, report {rep['status']}")
    if rep["status"] == "warnings":
        call("POST", f"/dataset-versions/{v['id']}/validation/acknowledge", {"codes": sorted({i['code'] for i in rep['items'] if i['severity'] == 'warning'})}, tok=tok)
        st = call("GET", f"/projects/{pid}", tok=tok).json["status"]
        check("PRJ-7c after acknowledging warnings -> Ready to train", st == "Ready to train", st)

    # text
    rep = sealed(tok, table_set(tok, org, "text/mixed.csv", {"columns": [{"name": "text", "role": "text"}, {"name": "label", "role": "target"}]}, template="text_classification", task_type=None, labels=["positive", "negative"])[2])
    det = {}
    for i in items(rep, "VAL-14"):
        det.update(i.get("details") or {})
    exp = EXP["text/mixed.csv"]["VAL-14"]
    check("VAL-14 empty 2 / very short 3", det.get("empty_or_whitespace") == exp["empty_or_whitespace"] and det.get("very_short") == exp["very_short"], blob(det))
    check("VAL-14 script mix == EXPECTED (Latin/Devanagari/Tamil/emoji) + Hinglish 4", det.get("scripts") == exp["scripts"] and det.get("hinglish_romanised") == exp["hinglish_romanised"], blob(det))
    check("VAL-14 non-English 25.6% -> multilingual backbone suggested", abs(det.get("non_english_share", 0) - 0.256) < 0.002 and det.get("suggest_multilingual") is True, blob(det))
    check("VAL-14 is info", all(i["severity"] == "info" for i in items(rep, "VAL-14")), "")

    # detection
    pid, did, v = detection_set(tok, org)
    rep = sealed(tok, v)
    k = kinds(rep, "VAL-16")
    cnt = {kk: vv["count"] for kk, vv in k.items()}
    b = blob(items(rep, "VAL-16"))
    vals = sorted(cnt.values())
    check("VAL-16 oob 1 / zero-area 1 / tiny 1 / unflagged images 2", vals.count(1) >= 3 and 2 in vals, f"{cnt}")
    A.RESULTS.append({"id": "VAL-16 record kinds", "pass": True, "why": blob(cnt), "note": True})
    print("INFO VAL-16 kinds:", blob(cnt))
    check("VAL-16 vest warning, glove error", any(i["severity"] == "warning" and "vest" in blob(i) for i in items(rep, "VAL-16")) and any(i["severity"] == "error" and "glove" in blob(i) for i in items(rep, "VAL-16")), b[:400])
    return pid, did, v, rep


def run_val18(tok, org):
    import importlib.util
    spec = importlib.util.spec_from_file_location("bf", FX / "build_fixtures.py")
    bf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bf)
    # burst-like: 3 base frames per class, each with 9 one-pixel variants -> near_duplicate_share ~90%
    files = []
    for cls in ("cat", "dog"):
        for b in range(3):
            base = bf.png_bytes(64, 64, seed=f"burst-{cls}-{b}")
            files.append((f"{cls}/b{b}_0.png", base))
            frame = base
            for k in range(1, 10):
                frame = bf.tweak_one_pixel(frame)
                files.append((f"{cls}/b{b}_{k}.png", frame))
    pid, did, _ = D.img_project(tok, org, "VAL18 burst")
    _, v, _ = D.upload(tok, pid, did, "folder", files)
    rep = sealed(tok, v)
    it = items(rep, "VAL-18")
    dv = rep.get("diversity") or {}
    check("VAL-18 burst near-duplicates -> warning, acknowledgeable, diversity.low", it and it[0]["severity"] == "warning" and it[0]["acknowledgeable"] and dv.get("low") is True and dv.get("near_duplicate_share", 0) > 0.2,
          f"items={blob(it)[:200]} diversity={blob(dv)}")
    check("VAL-18 report.diversity has all fields", {"samples", "effective_samples", "near_duplicate_share", "sessions_per_class", "min_sessions_per_class", "camera_share", "low"} <= set(dv), blob(dv))
    r = call("POST", f"/dataset-versions/{v['id']}/validation/acknowledge", {"codes": ["VAL-18"]}, tok=tok)
    fresh = report(tok, v["id"])
    check("VAL-18 acknowledge persists on a fresh GET (status stays failed because of VAL-7)", r.status == 200 and all(i["acknowledged"] for i in items(fresh, "VAL-18")) and fresh["status"] == "failed",
          f"{r.status} fresh status={fresh['status']} acked={[i['acknowledged'] for i in items(fresh, 'VAL-18')]}", r)
    check("VAL-7 on near-duplicate bursts (only exact dups are grouped) -> error", any(i["severity"] == "error" for i in items(fresh, "VAL-7")), blob(items(fresh, "VAL-7"))[:200])
    nv, nrep = fix(tok, v["id"], fresh, "VAL-7")
    check("VAL-7 resplit_groups fix -> new version with no VAL-7", not items(nrep, "VAL-7") and nv["origin"] == "fix", f"{[i['code'] for i in nrep['items']]}")
    # camera: 20 distinct frames per class, one capture session each -> session rule
    pid, did, _ = D.img_project(tok, org, "VAL18 camera")
    vid = None
    for cls in ("cat", "dog"):
        cam = [(f"{cls}_{i:03d}.png", bf.png_bytes(64, 64, seed=f"cam-{cls}-{i}")) for i in range(20)]
        c, vv, _ = D.upload(tok, pid, did, "camera", cam, label_from_folder=False, label=cls)
        vid = vv["id"]
    rep = sealed(tok, {"id": vid})
    dv = rep.get("diversity") or {}
    check("VAL-18 camera: 1 session per class (n>=20) -> warning", bool(items(rep, "VAL-18")) and dv.get("min_sessions_per_class") == 1 and dv.get("camera_share") == 1.0, f"diversity={blob(dv)}")
    # control: one plain folder batch of distinct images must not trip it
    rep = sealed(tok, image_set(tok, org, "images/clean", name="VAL18 control")[2])
    check("VAL-18 plain folder upload (distinct images) -> no VAL-18", not items(rep, "VAL-18"), blob(rep.get("diversity")))


def run_lbl(tok, org, det_ctx):
    pid, did, v = image_set(tok, org, "images/clean", name="LBL")
    sm = D.samples(tok, v["id"])
    ids = [s["id"] for s in sm[:20]]
    r = call("POST", f"/dataset-versions/{v['id']}/labels:batch", {"upserts": [{"sample_id": i, "labels": [{"class_name": "cat"}]} for i in ids]}, tok=tok)
    check("LBL-1a labels:batch 20 on unsealed version -> updated 20, not forked", r.status == 200 and r.json["updated"] == 20 and r.json["forked"] is False, r.raw[:200], r)
    after = {s["id"]: s for s in D.samples(tok, v["id"])}
    check("LBL-1a labels stored as user, class cat", all(after[i]["labels"] and after[i]["labels"][0]["class_name"] == "cat" and after[i]["labels"][0]["source"] == "user" for i in ids), blob(after[ids[0]]["labels"]))
    call("POST", f"/dataset-versions/{v['id']}/seal", tok=tok)
    report(tok, v["id"])
    before = call("GET", f"/dataset-versions/{v['id']}", tok=tok).json
    r = call("POST", f"/dataset-versions/{v['id']}/labels:batch", {"upserts": [{"sample_id": ids[0], "labels": [{"class_name": "dog"}]}]}, tok=tok)
    check("DAT-5b labels on sealed version -> forked new version + sample_id_map", r.status == 200 and r.json["forked"] is True and r.json["dataset_version_id"] != v["id"] and r.json.get("sample_id_map"), r.raw[:300], r)
    still = {s["id"]: s for s in D.samples(tok, v["id"])}
    check("DAT-5b sealed version unchanged (still cat, same content_hash)", still[ids[0]]["labels"][0]["class_name"] == "cat" and call("GET", f"/dataset-versions/{v['id']}", tok=tok).json["content_hash"] == before["content_hash"], "")
    fv = r.json["dataset_version_id"]
    # label set ops
    ls_id = call("GET", f"/projects/{pid}", tok=tok).json["label_set_id"]
    ls0 = call("GET", f"/label-sets/{ls_id}", tok=tok).json
    r = call("POST", f"/label-sets/{ls_id}/classes", {"add": ["kitten"]}, tok=tok)
    check("LBL-1b add class kitten -> label_set version +1", r.status == 200 and r.json["label_set"]["version"] == ls0["version"] + 1 and "kitten" in r.json["label_set"]["classes"], r.raw[:200], r)
    fsm = D.samples(tok, fv)
    kit = [s["id"] for s in fsm if s["labels"] and s["labels"][0]["class_name"] == "cat"][:3]
    call("POST", f"/dataset-versions/{fv}/labels:batch", {"upserts": [{"sample_id": i, "labels": [{"class_name": "kitten"}]} for i in kit]}, tok=tok)
    ls_id = call("GET", f"/projects/{pid}", tok=tok).json["label_set_id"]  # changes return a new id; always use the current one
    r = call("POST", f"/label-sets/{ls_id}/merge", {"from": ["kitten"], "into": "cat"}, tok=tok)
    spec_labels = call("GET", f"/projects/{pid}", tok=tok).json["current_spec"]["labels"]
    check("LBL-1b merge kitten -> cat: relabeled 3, kitten gone from set + spec", r.status == 200 and r.json["relabeled"] == 3 and "kitten" not in r.json["label_set"]["classes"] and "kitten" not in spec_labels, f"{r.status} {r.raw[:300]} spec={spec_labels}", r)
    ls_id = call("GET", f"/projects/{pid}", tok=tok).json["label_set_id"]
    r = call("POST", f"/label-sets/{ls_id}/rename", {"from": "dog", "to": "puppy"}, tok=tok)
    spec_labels = call("GET", f"/projects/{pid}", tok=tok).json["current_spec"]["labels"]
    check("LBL-1b rename dog -> puppy (set + spec)", r.status == 200 and "puppy" in r.json["label_set"]["classes"] and "dog" not in r.json["label_set"]["classes"] and "puppy" in spec_labels, f"{r.status} {r.raw[:300]}", r)
    ls_id = call("GET", f"/projects/{pid}", tok=tok).json["label_set_id"]
    expect("LBL-1b merge into a class not in the set -> 422 unknown_class (use rename)", call("POST", f"/label-sets/{call('GET', f'/projects/{pid}', tok=tok).json['label_set_id']}/merge", {"from": ["cat"], "into": "nope"}, tok=tok), 422, "unknown_class")
    # Stale label-set id: a change sent with a superseded id must not undo newer changes.
    cur = call("GET", f"/projects/{pid}", tok=tok).json
    stale = ls0["id"]
    r = call("POST", f"/label-sets/{stale}/classes", {"add": ["bird"]}, tok=tok)
    now = call("GET", f"/projects/{pid}", tok=tok).json["current_spec"]["labels"]
    lost = [c for c in cur["current_spec"]["labels"] if c not in now]
    check("LBL-1b stale label-set id -> 409 stale_label_set with current id/classes, nothing lost", r.status == 409 and r.code == "stale_label_set" and (r.json or {}).get("current_label_set_id") == cur["label_set_id"]
          and (r.json or {}).get("current_classes") == cur["current_spec"]["labels"] and not lost, f"{r.status} {r.code} lost {lost}", r)
    ls_id = cur["label_set_id"]
    expect("LBL-1b merge FROM a class not in the set -> 422 unknown_class", call("POST", f"/label-sets/{ls_id}/merge", {"from": ["ghost"], "into": "puppy"}, tok=tok), 422, "unknown_class")
    expect("LBL-1b rename to an existing name -> 409 class_exists", call("POST", f"/label-sets/{ls_id}/rename", {"from": "puppy", "to": "cat"}, tok=tok), 409, "class_exists")
    expect("LBL-1b rename unknown source -> 422 unknown_class", call("POST", f"/label-sets/{ls_id}/rename", {"from": "ghost", "to": "x"}, tok=tok), 422, "unknown_class")

    # detection
    dpid, ddid, dv, drep = det_ctx
    dsm = D.samples(tok, dv["id"])
    empty = [s["id"] for s in dsm if not s["labels"]]
    r = call("POST", f"/dataset-versions/{dv['id']}/labels:batch", {"upserts": [{"sample_id": i, "labels": [], "no_objects": True} for i in empty]}, tok=tok)
    nv = r.json["dataset_version_id"]
    rep = sealed(tok, {"id": nv}) if not call("GET", f"/dataset-versions/{nv}", tok=tok).json.get("sealed_at") else report(tok, nv)
    unflagged = [i for i in items(rep, "VAL-16") if "no_objects" in blob(i) or "without" in blob(i).lower() or (i.get("details") or {}).get("kind") in ("images_without_boxes", "no_boxes", "unlabeled")]
    check("LBL-2b no_objects on det_018/019 -> unflagged-image VAL-16 gone", len(empty) == 2 and not unflagged, f"empty={len(empty)} remaining={blob(unflagged)[:200]}")
    tgt = next(s for s in D.samples(tok, nv) if s["labels"])
    r = call("POST", f"/dataset-versions/{nv}/labels:batch", {"upserts": [{"sample_id": tgt["id"], "labels": [{"class_name": "helmet", "bbox": {"x": 10, "y": 12, "w": 30, "h": 20}}]}]}, tok=tok)
    nv2 = r.json["dataset_version_id"]
    m = r.json.get("sample_id_map") or {}
    got = next(s for s in D.samples(tok, nv2) if s["id"] == m.get(tgt["id"], tgt["id"]))
    check("LBL-2a bbox round-trip exact", got["labels"] and {k: got["labels"][0]["bbox"][k] for k in "xywh"} == {"x": 10, "y": 12, "w": 30, "h": 20}, blob(got["labels"]))
    ex = call("GET", f"/dataset-versions/{dv['id']}/labels/export?format=coco", tok=tok)
    cj = ex.json or {}
    check("LBL-2c COCO export: same box count as imported", ex.status == 200 and len(cj.get("annotations", [])) == len(json.loads((FX / "detection/coco_defects.json").read_text())["annotations"]), f"{ex.status} {len(cj.get('annotations', []))}")
    ey = call("GET", f"/dataset-versions/{dv['id']}/labels/export?format=yolo", tok=tok)
    check("LBL-2c YOLO export is a zip", ey.status == 200 and "zip" in ";".join(ey.header("content-type")), f"{ey.status} {ey.header('content-type')}")
    expect("LBL-2c unknown export format -> 422", call("GET", f"/dataset-versions/{dv['id']}/labels/export?format=voc2", tok=tok), 422)
    # LBL-4a
    before = {s["id"]: len(s["labels"]) for s in D.samples(tok, nv2)}
    ids3 = list(before)[:3]
    r = call("POST", f"/dataset-versions/{nv2}/box-suggestions", {"sample_ids": ids3, "classes": ["helmet", "vest"]}, tok=tok)
    sug = (r.json or {}).get("suggestions", {})
    check("LBL-4a suggestions returned with confidence (stub detector recorded)", r.status == 200 and all(sug.get(i) for i in ids3) and all("confidence" in b_ for i in ids3 for b_ in sug[i]), f"{r.status} model={(r.json or {}).get('model')}", r)
    after = {s["id"]: len(s["labels"]) for s in D.samples(tok, nv2)}
    check("LBL-4a suggestions not stored until accepted", after == before, "labels changed")
    accept = [{"sample_id": ids3[0], "labels": [dict(b_, source="ai_prelabel") for b_ in sug[ids3[0]]]}, {"sample_id": ids3[1], "labels": [dict(sug[ids3[1]][0], source="ai_prelabel")]}]
    r = call("POST", f"/dataset-versions/{nv2}/labels:batch", {"upserts": accept}, tok=tok)
    nv3 = r.json["dataset_version_id"]
    m = r.json.get("sample_id_map") or {}
    s3 = {s["id"]: s for s in D.samples(tok, nv3)}
    a0, a1 = s3[m.get(ids3[0], ids3[0])], s3[m.get(ids3[1], ids3[1])]
    check("LBL-4a accept all / accept one stored with source ai_prelabel + confidence", len(a0["labels"]) == len(sug[ids3[0]]) and len(a1["labels"]) == 1 and all(l["source"] == "ai_prelabel" and l["confidence"] is not None for l in a0["labels"] + a1["labels"]), blob(a1["labels"]))
    A.RESULTS.append({"id": "LBL-4a record: detector", "pass": True, "why": "stub-detector-v0 (no weights): licence-cleared real model still to come", "note": True})
    return v, sm, dv


def run_sec3(tok, org, tok_b, org_b, ctx):
    v, sm, dv = ctx
    # Writes must target the dataset's CURRENT version (older sealed ones answer 409 stale_version), so use the newest.
    ds = call("GET", f"/dataset-versions/{v['id']}", tok=tok).json["dataset_id"]
    v = call("GET", f"/datasets/{ds}/versions?limit=1", tok=tok).json["items"][0]
    sm = D.samples(tok, v["id"])
    bpid, bdid, bv = image_set(tok_b, org_b, "images/too_few", name="B")
    bsm = D.samples(tok_b, bv["id"])
    expect("S3-6 A's version, B's sample_id in labels:batch -> 404", call("POST", f"/dataset-versions/{v['id']}/labels:batch", {"upserts": [{"sample_id": bsm[0]["id"], "labels": [{"class_name": "cat"}]}]}, tok=tok), 404)
    expect("S3-6 B's version, A's sample_id -> 404", call("POST", f"/dataset-versions/{bv['id']}/labels:batch", {"upserts": [{"sample_id": sm[0]["id"], "labels": [{"class_name": "cat"}]}]}, tok=tok_b), 404)
    other = D.samples(tok, dv["id"])[0]["id"]
    expect("S3-6 sample_id from another version of the same org -> 404", call("POST", f"/dataset-versions/{v['id']}/labels:batch", {"upserts": [{"sample_id": other, "labels": [{"class_name": "cat"}]}]}, tok=tok), 404)
    k_b = call("POST", f"/orgs/{org_b}/api-keys", {"name": "b", "scopes": ["projects:read", "projects:write"]}, tok=tok_b).json["key"]
    targets = [("GET", f"/dataset-versions/{v['id']}/validation", None), ("POST", f"/dataset-versions/{v['id']}/validation/acknowledge", {"codes": ["VAL-3"]}),
               ("POST", f"/dataset-versions/{v['id']}/validation/fixes", {"fix_ids": ["exclude_corrupt"]}),
               ("POST", f"/dataset-versions/{v['id']}/labels:batch", {"upserts": [{"sample_id": sm[0]["id"], "labels": [{"class_name": "cat"}]}]}),
               ("GET", f"/dataset-versions/{v['id']}/labels/export?format=coco", None), ("POST", f"/dataset-versions/{dv['id']}/box-suggestions", {"sample_ids": []})]
    for m, path, body in targets:
        short = path.replace(v["id"], "{A_v}").replace(dv["id"], "{A_dv}")
        expect(f"SEC3 B {m} {short} -> 404", call(m, path, body, tok=tok_b), 404)
        expect(f"SEC3 B-key {m} {short} -> 404", call(m, path, body, key=k_b), 404)


def run_prj7c(tok, org):
    pid, did, _ = D.img_project(tok, org, "PRJ7c")
    seen = [call("GET", f"/projects/{pid}", tok=tok).json["status"]]
    _, v, _ = D.upload(tok, pid, did, "folder", D.folder_files("images/clean"))
    seen.append(call("GET", f"/projects/{pid}", tok=tok).json["status"])
    call("POST", f"/dataset-versions/{v['id']}/seal", tok=tok)
    seen.append(call("GET", f"/projects/{pid}", tok=tok).json["status"])
    report(tok, v["id"])
    seen.append(call("GET", f"/projects/{pid}", tok=tok).json["status"])
    dedup = [s for i, s in enumerate(seen) if i == 0 or s != seen[i - 1]]
    check("PRJ-7c Draft -> Data needed -> (Validating) -> Ready to train", dedup[0] == "Draft" and dedup[-1] == "Ready to train" and all(x in ("Draft", "Data needed", "Validating", "Ready to train") for x in dedup), f"observed {seen}")


def main():
    if call("GET", "/health").status != 200:
        sys.exit("API not healthy")
    print(f"run {A.RUN} against {A.API}")
    tok, _, _ = A.login(A.user("va")[0])
    org = A.personal_org(tok)
    tok_b, _, _ = A.login(A.user("vb")[0])
    org_b = A.personal_org(tok_b)
    det = lbl = None
    for name, fn in (("VAL", lambda: run_val(tok, org)), ("PRJ-7c", lambda: run_prj7c(tok, org)), ("VAL-18", lambda: run_val18(tok, org))):
        try:
            out = fn()
            if name == "VAL":
                det = out
        except Exception as e:
            check(f"{name} step crashed", False, f"{type(e).__name__}: {str(e)[:500]}")
    if det:
        try:
            lbl = run_lbl(tok, org, det)
        except Exception as e:
            check("LBL step crashed", False, f"{type(e).__name__}: {str(e)[:500]}")
    if lbl:
        try:
            run_sec3(tok, org, tok_b, org_b, lbl)
        except Exception as e:
            check("SEC3 step crashed", False, f"{type(e).__name__}: {str(e)[:500]}")
    fails = [r for r in A.RESULTS if not r["pass"]]
    out = A.OUT / f"val_{A.RUN}.json"
    out.write_text(json.dumps({"run": A.RUN, "results": A.RESULTS, "log": A.LOG}, indent=1))
    print(f"\n{len(A.RESULTS) - len(fails)}/{len(A.RESULTS)} pass, {len(fails)} fail. Log: {out}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
