#!/usr/bin/env python3
"""Local end-to-end run through the API: project -> upload -> label -> validate -> train -> evaluate -> playground -> export.

One image-classification flow (fixtures/images/clean, 32 cat + 32 dog) and one detection flow
(fixtures/detection, COCO, helmet/vest). A stage whose endpoint isn't in openapi.json yet is reported
NOT BUILT and the flow stops there; nothing is faked.

    python3 tests/acceptance/e2e_run.py              # both flows
    python3 tests/acceptance/e2e_run.py --only image # or --only detection
Env: AUTOML_API (default http://127.0.0.1:8010/api/v1), E2E_TRAIN_TIMEOUT seconds (default 1800).
"""
import json
import os
import sys
import time
import urllib.parse
from pathlib import Path

import acc_run as A
import dat_run as D
from acc_run import call

FX = D.FX
TRAIN_TIMEOUT = int(os.environ.get("E2E_TRAIN_TIMEOUT", "1800"))
PATHS = set()
CAL = {}


class NotBuilt(Exception):
    pass


def need(path):
    if path not in PATHS:
        raise NotBuilt(path)


def stage(flow, name, fn):
    t0 = time.time()
    try:
        out = fn()
        A.check(f"E2E[{flow}] {name}", True)
        print(f"     {time.time() - t0:5.1f}s")
        return out
    except NotBuilt as e:
        A.RESULTS.append({"id": f"E2E[{flow}] {name}", "pass": True, "skipped": True, "why": f"NOT BUILT: {e}"})
        print(f"SKIP E2E[{flow}] {name} -- NOT BUILT: {e}")
        raise
    except AssertionError as e:
        A.check(f"E2E[{flow}] {name}", False, str(e)[:600])
        raise


def wait(path, tok, done, timeout, what):
    end = time.time() + timeout
    last = None
    while time.time() < end:
        r = call("GET", path, tok=tok)
        last = r
        if r.status == 200 and done(r.json):
            return r.json
        assert r.status == 200, r.brief()
        time.sleep(2)
    raise AssertionError(f"{what} not done in {timeout}s; last: {last.raw[:400] if last else None}")


def flow_image(tok, org):
    f = "image"
    rel = os.environ.get("E2E_IMAGES", "images/clean")  # e.g. demo/shapes (learnable) or images/clean (random noise)
    classes = sorted(p.name for p in (FX / rel).iterdir() if p.is_dir())
    files = D.folder_files(rel)
    pid, did, _ = stage(f, f"create project (image classification, {'/'.join(classes)})", lambda: D.img_project(tok, org, f"E2E image {rel}", classes))

    def up():
        _, v, _ = D.upload(tok, pid, did, "folder", files)
        assert v and v["ingest_status"] == "ready", v
        lc = D.label_counts(v)
        want = {c: sum(1 for r, _ in files if r.startswith(c + "/")) for c in classes}
        assert lc == want, (lc, want)
        return v
    v = stage(f, f"upload folder ({len(files)} images from {rel}, labels from folders)", up)
    v = stage(f, "label (already labelled by folder; labels:batch spot-edit)", lambda: label_spot_check(tok, v))
    v = stage(f, "seal + validate", lambda: seal_and_validate(tok, v))
    probe_rel, probe = files[0]
    return train_eval_export(tok, f, pid, v, sample_bytes=probe, expect_label=probe_rel.split("/")[0], classes=classes)


def _dur(job):
    from datetime import datetime
    f = lambda x: datetime.fromisoformat(x.replace("Z", "+00:00"))
    return round((f(job["finished_at"]) - f(job["started_at"])).total_seconds(), 1) if job.get("started_at") and job.get("finished_at") else None


def flow_tabular(tok, org):
    f = "tabular"
    rel = os.environ.get("E2E_TABLE", "demo/churn.csv")
    pid, did, _ = stage(f, "create project (tabular classification)", lambda: D.project(tok, org, f"E2E tabular {rel}", template="tabular", task_type="tabular_classification"))
    data = (FX / rel).read_bytes()
    rows = list(__import__("csv").DictReader(__import__("io").StringIO(data.decode())))

    def up():
        upl = D.start(tok, pid, did, "annotations", [("data.csv", data)], label_from_folder=False)
        e = D.entry(upl, "data.csv", data)
        D.put_object(tok, upl, e["key"], data)
        assert D.complete(tok, upl, [e]).status in (200, 202)
        mapping = {"columns": [{"name": "churn", "role": "target", "type": "categorical"}, {"name": "customer_id", "role": "id"}]}
        r = call("POST", f"/projects/{pid}/datasets/{did}/imports", {"upload_id": upl["upload_id"], "format": "csv", "key": e["key"], "mapping": mapping}, tok=tok)
        assert r.status == 202, r.brief()
        v = D.wait_version(tok, r.json["dataset_version_id"])
        assert v["ingest_status"] == "ready" and v["sample_count"] == len(rows), v
        return v
    v = stage(f, f"upload + import {rel} ({len(rows)} rows, target churn)", up)
    v = stage(f, "seal + validate", lambda: seal_and_validate(tok, v))
    probe = {k: val for k, val in rows[0].items() if k not in ("churn", "customer_id")}
    return train_eval_export(tok, f, pid, v, sample_bytes=probe, expect_label=None, classes=["0", "1"])


def flow_text(tok, org):
    f = "text"
    rel = os.environ.get("E2E_TEXT", "demo/reviews.csv")
    pid, did, _ = stage(f, "create project (text classification, positive/negative)", lambda: D.project(tok, org, f"E2E text {rel}", template="text_classification", labels=["positive", "negative"]))
    data = (FX / rel).read_bytes()
    rows = list(__import__("csv").DictReader(__import__("io").StringIO(data.decode("utf-8"))))

    def up():
        upl = D.start(tok, pid, did, "annotations", [("reviews.csv", data)], label_from_folder=False)
        e = D.entry(upl, "reviews.csv", data)
        D.put_object(tok, upl, e["key"], data)
        assert D.complete(tok, upl, [e]).status in (200, 202)
        mapping = {"columns": [{"name": "text", "role": "text"}, {"name": "label", "role": "target"}]}
        r = call("POST", f"/projects/{pid}/datasets/{did}/imports", {"upload_id": upl["upload_id"], "format": "csv", "key": e["key"], "mapping": mapping}, tok=tok)
        assert r.status == 202, r.brief()
        v = D.wait_version(tok, r.json["dataset_version_id"])
        assert v["ingest_status"] == "ready" and v["sample_count"] == len(rows), v
        return v
    v = stage(f, f"upload + import {rel} ({len(rows)} reviews)", up)
    v = stage(f, "seal + validate", lambda: seal_and_validate(tok, v))
    return train_eval_export(tok, f, pid, v, sample_bytes={"__text__": "Ye charger bilkul bekaar, paise barbaad"}, expect_label="negative", classes=["positive", "negative"])


def _iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax2, bx2) - max(a[0], b[0])), max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter) if inter else 0.0


def detection_quality(tok, mv_id, n=10):
    """Playground boxes on n demo images vs ground truth: count, IoU>=0.5 match, <=20 boxes, confidence floor."""
    root = os.environ.get("E2E_DET", "demo/detection")
    coco = json.loads((FX / root / "coco.json").read_text())
    names = {c["id"]: c["name"] for c in coco["categories"]}
    exact = matched = total_gt = 0
    worst = []
    vid = call("GET", f"/model-versions/{mv_id}", tok=tok).json["dataset_version_id"]
    test_names = {s_["name"].split("/")[-1] for s_ in D.samples(tok, vid, split="test")}
    pick = [im for im in coco["images"] if im["file_name"] in test_names][:n]
    assert pick, "no test-split images found"
    n = len(pick)
    for im in pick:
        gt = [(names[a["category_id"]], a["bbox"]) for a in coco["annotations"] if a["image_id"] == im["id"]]
        r = multipart(f"/model-versions/{mv_id}/predict", tok, "file", im["file_name"], (FX / root / "images" / im["file_name"]).read_bytes())
        assert r.status == 200, r.brief()
        boxes = [(b["class_name"], [b["bbox"][k] for k in "xywh"], b["confidence"]) for b in r.json["boxes"]]
        assert len(boxes) <= 20 and all(c >= 0.5 for _, _, c in boxes), f"{im['file_name']}: {len(boxes)} boxes, min conf {min((c for *_, c in boxes), default=None)}"
        exact += len(boxes) == len(gt)
        total_gt += len(gt)
        for cls, g in gt:
            matched += any(_iou(g, bb) >= 0.5 and c == cls for c, bb, _ in boxes)
        if len(boxes) != len(gt):
            worst.append((im["file_name"], len(gt), len(boxes)))
    print(f"     detection quality on {n} TEST-split images: exact box count {exact}/{n}, GT boxes matched (IoU>=0.5, right class) {matched}/{total_gt}; mismatches {worst[:5]}")
    A.RESULTS.append({"id": "E2E[detection] record playground quality", "pass": True, "note": True, "why": f"exact {exact}/{n} matched {matched}/{total_gt} mismatches {worst}"})
    assert matched >= 0.9 * total_gt and exact >= 0.8 * n, f"exact {exact}/{n}, matched {matched}/{total_gt}"


def flow_detection(tok, org):
    f = "detection"
    root = os.environ.get("E2E_DET", "demo/detection")  # learnable synthetic circles/squares with exact COCO boxes
    coco = json.loads((FX / root / "coco.json").read_text())
    classes = [c["name"] for c in coco["categories"]]
    pid, did, _ = stage(f, f"create project (object detection, {'/'.join(classes)})", lambda: D.project(tok, org, f"E2E detection {root}", template="object_detection", labels=classes))

    def up():
        imgs = [(f"images/{im['file_name']}", (FX / root / "images" / im["file_name"]).read_bytes()) for im in coco["images"]]
        files = imgs + [("coco.json", json.dumps(coco).encode())]
        upl = D.start(tok, pid, did, "annotations", files, label_from_folder=False)
        ents = []
        for rel, data in files:
            e = D.entry(upl, rel, data)
            D.put_object(tok, upl, e["key"], data)
            ents.append(e)
        c = D.complete(tok, upl, ents)
        assert c.status in (200, 202), c.brief()
        r = call("POST", f"/projects/{pid}/datasets/{did}/imports", {"upload_id": upl["upload_id"], "format": "coco", "key": upl["key_prefix"] + "coco.json"}, tok=tok)
        assert r.status == 202, r.brief()
        v = D.wait_version(tok, r.json["dataset_version_id"])
        assert v["ingest_status"] == "ready", v
        want = {c: sum(1 for a in coco["annotations"] if coco["categories"][a["category_id"] - 1]["name"] == c) for c in classes}
        assert D.label_counts(v, "boxes") == want, (D.label_counts(v, "boxes"), want)
        return v
    v = stage(f, f"upload {len(coco['images'])} images + COCO import ({len(coco['annotations'])} boxes)", up)
    v = stage(f, "label (stub box suggestions before any model)", lambda: label_detection(tok, v))
    v = stage(f, "seal + validate", lambda: seal_and_validate(tok, v))
    probe = (FX / root / "images" / coco["images"][0]["file_name"]).read_bytes()
    out = train_eval_export(tok, f, pid, v, sample_bytes=probe, expect_label=None, classes=classes)

    def own_suggestions():
        sm = D.samples(tok, v["id"])[:2]
        r = call("POST", f"/dataset-versions/{v['id']}/box-suggestions", {"sample_ids": [x["id"] for x in sm], "classes": classes}, tok=tok)
        assert r.status == 200 and str(r.json.get("model", "")).startswith("own:"), f"model={r.json.get('model') if r.json else r.raw[:200]}"
        print(f"     box-suggestions model={r.json['model']} boxes={sum(len(x) for x in r.json['suggestions'].values())}")
    stage(f, "box suggestions now from the project's own detector", own_suggestions)
    return out


def label_spot_check(tok, v):
    need("/api/v1/dataset-versions/{version_id}/labels:batch")
    sm = D.samples(tok, v["id"])
    first = sm[0]
    r = call("POST", f"/dataset-versions/{v['id']}/labels:batch", {"upserts": [{"sample_id": first["id"], "labels": [{"class_name": first["labels"][0]["class_name"]}]}]}, tok=tok)
    assert r.status in (200, 202), r.brief()
    return D.wait_version(tok, r.json["dataset_version_id"]) if r.json and r.json.get("dataset_version_id") else v


def label_detection(tok, v):
    need("/api/v1/dataset-versions/{version_id}/labels:batch")
    sm = D.samples(tok, v["id"])
    empty = [s["id"] for s in sm if not s["labels"]]
    nv = v
    if empty:  # flag images that genuinely have no objects (the demo set has none)
        r = call("POST", f"/dataset-versions/{v['id']}/labels:batch", {"upserts": [{"sample_id": i, "labels": [], "no_objects": True} for i in empty]}, tok=tok)
        assert r.status in (200, 202), r.brief()
        nv = D.wait_version(tok, r.json["dataset_version_id"]) if r.json and r.json.get("dataset_version_id") else v
    if "/api/v1/dataset-versions/{version_id}/box-suggestions" in PATHS:
        cls = [c for c in {l["class_name"] for s_ in sm for l in s_["labels"]}] or ["object"]
        r = call("POST", f"/dataset-versions/{nv['id']}/box-suggestions", {"sample_ids": [sm[0]["id"]], "classes": cls}, tok=tok)
        assert r.status in (200, 202), r.brief()
        print(f"     box-suggestions before training: model={(r.json or {}).get('model')}")
    return nv


def seal_and_validate(tok, v):
    r = call("POST", f"/dataset-versions/{v['id']}/seal", tok=tok)
    assert r.status == 202, r.brief()
    need("/api/v1/dataset-versions/{version_id}/validation")
    rep = wait(f"/dataset-versions/{v['id']}/validation", tok, lambda j: j.get("status") not in ("pending", "running"), 300, "validation")
    errors = [i for i in rep.get("items", rep.get("issues", [])) if i.get("severity") == "error"]
    assert not errors, f"validation errors on a clean E2E set: {json.dumps(errors)[:400]}"
    warn = [i.get("code") for i in rep.get("items", rep.get("issues", [])) if i.get("severity") == "warning"]
    if warn:
        r = call("POST", f"/dataset-versions/{v['id']}/validation/acknowledge", {"codes": warn}, tok=tok)
        assert r.status in (200, 202, 204), r.brief()
    return call("GET", f"/dataset-versions/{v['id']}", tok=tok).json


def multipart(path, tok, field, filename, data, mime="image/png"):
    """POST a single file as multipart/form-data (the playground's upload form)."""
    import http.client
    import uuid as _uuid
    b = "----e2e" + _uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{filename}\"\r\nContent-Type: {mime}\r\n\r\n").encode() + data + f"\r\n--{b}--\r\n".encode()
    conn = http.client.HTTPConnection(A.U.hostname, A.U.port, timeout=60)
    conn.request("POST", A.U.path + path, body=body, headers={"Authorization": f"Bearer {tok}", "Content-Type": f"multipart/form-data; boundary={b}"})
    r = conn.getresponse()
    raw = r.read().decode("utf-8", "replace")
    conn.close()
    out = A.R(r.status, r.getheaders(), raw, {"method": "POST", "path": A.U.path + path, "multipart": filename})
    A.LOG.append({"req": out.req, "status": out.status, "body": raw[:2000]})
    return out


def train_eval_export(tok, f, pid, v, sample_bytes, expect_label, classes=None):
    def train():
        need("/api/v1/projects/{project_id}/training-jobs")
        r = call("POST", f"/projects/{pid}/training-jobs", {"dataset_version_id": v["id"], "budget_mode": "quick"}, tok=tok)
        assert r.status == 202, r.brief()
        job = r.json["job"]
        print(f"     job {job['id']} estimate {json.dumps(r.json.get('estimate'))[:200]}")
        done = wait(f"/training-jobs/{job['id']}", tok, lambda j: j.get("status") in ("succeeded", "failed", "cancelled"), TRAIN_TIMEOUT, "training")
        assert done["status"] == "succeeded" and done.get("model_version_id"), f"{done['status']} {done.get('error_code')} {done.get('error_message')}"
        rm = done.get("repro_manifest") or {}
        need_keys = {"dataset_version_id", "dataset_content_hash", "spec_version", "split_seed", "budget", "framework", "architecture", "base_model", "library_versions", "device"}
        assert need_keys <= set(rm), f"repro_manifest missing {sorted(need_keys - set(rm))}"
        prog = done.get("progress") or {}
        print(f"     trained in {_dur(done)}s budget={rm.get('budget')} device={rm.get('device')} epochs={prog.get('epoch')}/{prog.get('epochs_total')} trials={prog.get('trials_total')} best_params={json.dumps(rm.get('best_params'))[:160]}")
        A.RESULTS.append({"id": f"E2E[{f}] record training", "pass": True, "note": True, "why": json.dumps({"seconds": _dur(done), "budget": rm.get("budget"), "progress": prog, "estimate": r.json.get("estimate")})[:1500]})
        return done
    job = stage(f, "train (quick budget)", train)
    mv_id = job["model_version_id"]

    def evaluate():
        need("/api/v1/model-versions/{mv_id}/evaluation")
        mvd = call("GET", f"/model-versions/{mv_id}", tok=tok)
        bm = (mvd.json or {}).get("base_model") or {}
        assert mvd.status == 200 and mvd.json.get("framework") and mvd.json.get("architecture") and bm.get("registry_id") and bm.get("licence"), f"model version metadata: {mvd.raw[:400]}"
        print(f"     model {mvd.json['framework']}/{mvd.json['architecture']} licence={bm.get('licence')} commercial_use={bm.get('commercial_use')}")
        if f in ("image", "tabular"):
            cal = ((mvd.json.get("model_json") or {}).get("output") or {}).get("calibration") or {}
            need_c = {"temperature", "fitted_temperature", "applied", "val_ece_before", "val_ece_after", "fitted_on"}
            assert need_c <= set(cal), f"calibration fields missing: {sorted(need_c - set(cal))} in {json.dumps(cal)}"
            if cal["applied"]:
                assert cal["val_ece_after"] < cal["val_ece_before"], f"applied but val ECE not lower: {cal}"
            else:
                assert cal["temperature"] == 1, f"not applied but T={cal['temperature']}"
            CAL[f] = cal
            print(f"     calibration applied={cal['applied']} T={cal['temperature']} (fitted {cal['fitted_temperature']}) val ECE {cal['val_ece_before']} -> {cal['val_ece_after']}")
        ev = wait(f"/model-versions/{mv_id}/evaluation", tok, lambda j: j.get("status") in ("ready", "failed"), 600, "evaluation")
        assert ev["status"] == "ready", f"evaluation {ev['status']}: {ev.get('error')}"
        rep = ev["report"]
        assert rep.get("metrics") and rep.get("primary_metric") and rep.get("n_test"), json.dumps(rep)[:300]
        if f == "image":
            assert rep.get("per_class") and rep.get("confusion", {}).get("labels"), "per-class / confusion missing"
        if f == "detection":
            assert "map_50" in json.dumps(rep["metrics"]), f"no map_50 in {json.dumps(rep['metrics'])[:200]}"
        mc = call("GET", f"/model-versions/{mv_id}/model-card", tok=tok)
        assert mc.status == 200 and "markdown" in ";".join(mc.header("content-type")) and len(mc.raw) > 200, mc.brief()
        if f in CAL and "ece" in rep["metrics"]:
            print(f"     eval ECE {rep['metrics']['ece']} (val ECE after calibration {CAL[f]['val_ece_after']})")
        print(f"     {rep['primary_metric']}={rep['metrics'].get(rep['primary_metric'])} metrics={json.dumps(rep['metrics'])[:200]} n_test={rep['n_test']} baseline={json.dumps(rep.get('baseline'))[:160]}")
        print(f"     per_class={json.dumps(rep.get('per_class'))[:300]}")
        print(f"     confusion={json.dumps((rep.get('confusion') or {}).get('counts'))} labels={(rep.get('confusion') or {}).get('labels')}")
        A.RESULTS.append({"id": f"E2E[{f}] record metrics", "pass": True, "note": True, "why": json.dumps({"metrics": rep["metrics"], "baseline": rep.get("baseline"), "n_test": rep["n_test"]})})
        return ev
    stage(f, "evaluate (report + per-class + model card)", evaluate)

    def playground():
        need("/api/v1/model-versions/{mv_id}/predict")
        if isinstance(sample_bytes, dict) and "__text__" in sample_bytes:
            r = call("POST", f"/model-versions/{mv_id}/predict", {"text": sample_bytes["__text__"]}, tok=tok)
        elif isinstance(sample_bytes, dict):
            r = call("POST", f"/model-versions/{mv_id}/predict", {"row": sample_bytes}, tok=tok)
        else:
            r = multipart(f"/model-versions/{mv_id}/predict", tok, "file", "probe.png", sample_bytes)
        assert r.status == 200, r.brief()
        j = r.json or {}
        if f in ("image", "tabular", "text"):
            assert str(j.get("label")) in [str(c) for c in (classes or [])] and 0 <= j.get("confidence", -1) <= 1 and j.get("probs"), r.brief()
        else:
            assert isinstance(j.get("boxes"), list), r.brief()
            if f == "detection":
                detection_quality(tok, mv_id)
        assert "latency_ms" in j, r.brief()
        if j.get("probs"):
            assert abs(sum(j["probs"].values()) - 1) < 1e-4, f"probs sum {sum(j['probs'].values())}"
        if f == "image" and CAL.get("image", {}).get("applied"):
            ood = (FX / "images/clean/cat/img_000.png").read_bytes()  # random-noise image: no shape in it
            o = multipart(f"/model-versions/{mv_id}/predict", tok, "file", "noise.png", ood).json
            A.RESULTS.append({"id": "E2E[image] record OOD confidence (calibrated)", "pass": True, "note": True, "why": json.dumps({k: o.get(k) for k in ("label", "confidence")})})
            print(f"     OOD noise image -> {o.get('label')} conf {o.get('confidence')} (was 0.99994 before calibration)")
            assert o["confidence"] < 0.99, f"OOD confidence still {o['confidence']}"
        if f == "image":
            assert j["confidence"] < 1.0, f"confidence exactly {j['confidence']} (label smoothing expected)"
        print(f"     predict -> {json.dumps({k: j.get(k) for k in ('label', 'confidence', 'latency_ms')})} boxes={len(j.get('boxes') or [])}")
        sn = call("GET", f"/model-versions/{mv_id}/snippets", tok=tok)
        assert sn.status == 200 and sn.json.get("python") and sn.json.get("curl"), sn.brief()
        return j
    stage(f, "playground prediction + snippets", playground)

    def export():
        need("/api/v1/model-versions/{mv_id}/exports")
        r = call("POST", f"/model-versions/{mv_id}/exports", {"format": "onnx"}, tok=tok)
        if f == "detection":
            assert r.status == 422 and r.code == "phase_2" and A.problem_ok(r), f"detection export should be a clean 422 phase_2: {r.brief()}"
            print(f"     detection ONNX export -> 422 phase_2 ({(r.json or {}).get('detail')})")
            return r.json
        assert r.status == 202, r.brief()
        ex = wait(f"/exports/{r.json['id']}", tok, lambda j: j.get("status") in ("ready", "failed"), 600, "export")
        assert ex["status"] == "ready" and ex.get("download_url"), f"{ex['status']} {ex.get('error')}"
        st, _, body = D.s3("GET", ex["download_url"])
        assert st == 200, st
        import io
        import zipfile as zf
        names = set(zf.ZipFile(io.BytesIO(body)).namelist())
        want = {"model.onnx", "model.json", "labels.txt", "README.md", "MODEL_CARD.md"}
        assert want <= names, f"zip has {sorted(names)}"
        print(f"     export {len(body)} bytes, verified={ex.get('verified')} report={json.dumps(ex.get('verify_report'))[:160]} delta={ex.get('accuracy_delta')}")
        again = call("POST", f"/model-versions/{mv_id}/exports", {"format": "onnx"}, tok=tok)
        assert again.status == 202 and again.json["id"] == r.json["id"], f"export not idempotent: {again.raw[:200]}"
        return ex
    stage(f, "export ONNX + download zip" if f != "detection" else "export: detection ONNX is Phase 2 (clean 422)", export)


def main():
    if call("GET", "/health").status != 200:
        sys.exit("API not healthy")
    PATHS.update(call("GET", "/openapi.json").json["paths"])
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    print(f"run {A.RUN} against {A.API} ({len(PATHS)} paths)")
    e, _ = A.user("e2e")
    tok, _, _ = A.login(e)
    org = A.personal_org(tok)
    for name, fn in (("image", flow_image), ("tabular", flow_tabular), ("text", flow_text), ("detection", flow_detection)):
        if only and only != name:
            continue
        try:
            fn(tok, org)
        except NotBuilt:
            pass
        except Exception as ex:  # a stage already recorded its failure; keep the other flow running
            if not any(r["id"].startswith(f"E2E[{name}]") and not r["pass"] for r in A.RESULTS):
                A.check(f"E2E[{name}] crashed", False, f"{type(ex).__name__}: {str(ex)[:400]}")
    fails = [r for r in A.RESULTS if not r["pass"]]
    skipped = [r for r in A.RESULTS if r.get("skipped")]
    out = A.OUT / f"e2e_{A.RUN}.json"
    out.write_text(json.dumps({"run": A.RUN, "results": A.RESULTS, "log": A.LOG}, indent=1))
    print(f"\n{len(A.RESULTS) - len(fails) - len(skipped)} stages pass, {len(fails)} fail, {len(skipped)} not built. Log: {out}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
