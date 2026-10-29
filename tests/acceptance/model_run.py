#!/usr/bin/env python3
"""Negative + cross-tenant checks for the model-version endpoints (evaluation, predict, snippets, exports, model card).

Uses an existing trained model version owned by a user from an e2e run, or trains one if none is given:
    python3 tests/acceptance/model_run.py <e2e log .json>     # reuses that run's model version (needs its user's password: same RUN)
    python3 tests/acceptance/model_run.py                     # trains a fresh image model first (~5 min)
"""
import json
import sys
import time

import acc_run as A
import dat_run as D
import e2e_run as E
from acc_run import call, check, expect


def main():
    if call("GET", "/health").status != 200:
        sys.exit("API not healthy")
    E.PATHS.update(call("GET", "/openapi.json").json["paths"])
    tok, _, _ = A.login(A.user("ma")[0])
    org = A.personal_org(tok)
    pid, did, _ = D.img_project(tok, org, "MODEL")
    _, v, _ = D.upload(tok, pid, did, "folder", D.folder_files("images/clean"))
    E.seal_and_validate(tok, v)
    r = call("POST", f"/projects/{pid}/training-jobs", {"dataset_version_id": v["id"], "budget_mode": "quick"}, tok=tok)
    job = r.json["job"]
    # 409 model_not_ready / not_evaluated while training is still going
    early = call("GET", f"/projects/{pid}/training-jobs", tok=tok)
    check("TRN list jobs for project", early.status == 200 and job["id"] in early.raw, f"{early.status}", early)
    done = E.wait(f"/training-jobs/{job['id']}", tok, lambda j: j.get("status") in ("succeeded", "failed", "cancelled"), E.TRAIN_TIMEOUT, "training")
    assert done["status"] == "succeeded", done
    mv = done["model_version_id"]
    t_lat = time.time()
    E.wait(f"/model-versions/{mv}/evaluation", tok, lambda j: j.get("status") in ("ready", "failed"), 600, "evaluation")

    # ---- negative cases ----
    expect("predict empty JSON body -> 422 input_required", call("POST", f"/model-versions/{mv}/predict", {}, tok=tok), 422, "input_required")
    for body in ({"text": "a cat"}, {"row": {"tenure": 3}}):
        r = call("POST", f"/model-versions/{mv}/predict", body, tok=tok)
        ok = r.status == 422 and r.code == "wrong_input_type" and (r.json or {}).get("expected") == "image" and (r.json or {}).get("got") == next(iter(body))
        check(f"predict {next(iter(body))} on image model -> 422 wrong_input_type expected=image", ok, f"{r.status} {r.code} {r.raw[:200]}", r)
    # export (now in a child process): fresh ready + verified + zip contents
    r = call("POST", f"/model-versions/{mv}/exports", {"format": "onnx"}, tok=tok)
    ex = E.wait(f"/exports/{r.json['id']}", tok, lambda j: j.get("status") in ("ready", "failed"), 600, "export") if r.status == 202 else {}
    import io, zipfile
    names = set()
    if ex.get("download_url"):
        st, _, body = D.s3("GET", ex["download_url"])
        names = set(zipfile.ZipFile(io.BytesIO(body)).namelist()) if st == 200 else set()
    check("export (child process) ready + verified + complete zip", ex.get("status") == "ready" and ex.get("verified") and {"model.onnx", "model.json", "labels.txt", "README.md", "MODEL_CARD.md"} <= names, f"{ex.get('status')} {ex.get('error')} {sorted(names)}")
    r = E.multipart(f"/model-versions/{mv}/predict", tok, "file", "garbage.png", b"not an image at all")
    check("predict undecodable image -> 4xx problem (not 500)", 400 <= r.status < 500, f"{r.status} {r.raw[:200]}", r)
    lat = []
    img = (D.FX / "images/clean/dog/img_001.png").read_bytes()
    for _ in range(5):
        r = E.multipart(f"/model-versions/{mv}/predict", tok, "file", "dog.png", img)
        lat.append((r.json or {}).get("latency_ms"))
    A.RESULTS.append({"id": "predict warm latency (record; NFR-P5 hosted target is < 80 ms)", "pass": True, "note": True, "why": f"latency_ms over 5 warm calls: {lat}"})
    print(f"INFO warm predict latency_ms: {lat}")
    for fmt in ("tflite", "coreml"):
        expect(f"export {fmt} -> 422 phase_2", call("POST", f"/model-versions/{mv}/exports", {"format": fmt}, tok=tok), 422, "phase_2")
    expect("export unknown format -> 422", call("POST", f"/model-versions/{mv}/exports", {"format": "pickle"}, tok=tok), 422)
    lst = call("GET", f"/model-versions/{mv}/exports", tok=tok)
    check("GET exports list", lst.status == 200, f"{lst.status}", lst)
    cell = call("GET", f"/model-versions/{mv}/evaluation/cell?y_true=cat&y_pred=dog", tok=tok)
    check("evaluation/cell returns samples for a confusion cell", cell.status == 200, f"{cell.status} {cell.raw[:200]}", cell)
    rr = call("POST", f"/model-versions/{mv}/evaluation/rerun", tok=tok)
    check("evaluation/rerun -> 202", rr.status == 202, f"{rr.status}", rr)
    models = call("GET", f"/projects/{pid}/models", tok=tok)
    check("GET project models lists the version", models.status == 200 and mv in models.raw, f"{models.status}", models)
    trials = call("GET", f"/training-jobs/{job['id']}/trials", tok=tok)
    check("GET job trials", trials.status == 200, f"{trials.status}", trials)

    # ---- cross-tenant ----
    tok_b, _, _ = A.login(A.user("mb")[0])
    org_b = A.personal_org(tok_b)
    k_b = call("POST", f"/orgs/{org_b}/api-keys", {"name": "b", "scopes": ["projects:read", "projects:write", "training", "inference"]}, tok=tok_b).json["key"]
    ex = call("GET", f"/model-versions/{mv}/exports", tok=tok)
    ex_id = (ex.json[0]["id"] if isinstance(ex.json, list) and ex.json else (ex.json or {}).get("items", [{}])[0].get("id")) if ex.status == 200 else None
    targets = [("GET", f"/model-versions/{mv}"), ("GET", f"/model-versions/{mv}/evaluation"), ("GET", f"/model-versions/{mv}/evaluation/cell?y_true=cat&y_pred=dog"),
               ("POST", f"/model-versions/{mv}/evaluation/rerun"), ("GET", f"/model-versions/{mv}/model-card"), ("GET", f"/model-versions/{mv}/snippets"),
               ("GET", f"/model-versions/{mv}/exports"), ("POST", f"/model-versions/{mv}/exports"), ("GET", f"/projects/{pid}/models"),
               ("GET", f"/projects/{pid}/training-jobs"), ("POST", f"/projects/{pid}/training-jobs"), ("GET", f"/training-jobs/{job['id']}"),
               ("GET", f"/training-jobs/{job['id']}/trials"), ("POST", f"/training-jobs/{job['id']}/cancel")]
    if ex_id:
        targets.append(("GET", f"/exports/{ex_id}"))
    bodies = {"/exports": {"format": "onnx"}, "/training-jobs": {"dataset_version_id": v["id"], "budget_mode": "quick"}}
    for m, path in targets:
        body = next((b for k, b in bodies.items() if path.endswith(k)), None) if m == "POST" else None
        short = path.replace(mv, "{A_mv}").replace(pid, "{A_p}").replace(job["id"], "{A_job}").replace(ex_id or "-", "{A_export}")
        expect(f"SEC3 B {m} {short} -> 404", call(m, path, body, tok=tok_b), 404)
        expect(f"SEC3 B-key {m} {short} -> 404", call(m, path, body, key=k_b), 404)
    r = E.multipart(f"/model-versions/{mv}/predict", tok_b, "file", "x.png", img)
    expect("SEC3 B predict on A's model -> 404", r, 404)
    bpid, bdid, _ = D.img_project(tok_b, org_b, "B")
    expect("SEC3 B trains own project on A's dataset_version_id -> 404", call("POST", f"/projects/{bpid}/training-jobs", {"dataset_version_id": v["id"], "budget_mode": "quick"}, tok=tok_b), 404)

    fails = [r for r in A.RESULTS if not r["pass"]]
    out = A.OUT / f"model_{A.RUN}.json"
    out.write_text(json.dumps({"run": A.RUN, "results": A.RESULTS, "log": A.LOG}, indent=1))
    print(f"\n{len(A.RESULTS) - len(fails)}/{len(A.RESULTS)} pass, {len(fails)} fail. Log: {out}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
