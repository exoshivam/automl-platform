#!/usr/bin/env python3
"""Permanent project purge + ProjectOut best metric. Only purges projects this script creates.

Object removal is checked through presigned URLs captured before the purge (sample objects + export zip);
MLflow through its REST API (read-only); DB row counts are printed as SQL for the service runner to run.
    python3 tests/acceptance/purge_run.py
"""
import json
import os
import sys
import time
import urllib.parse

import acc_run as A
import dat_run as D
import e2e_run as E
from acc_run import call, check, expect

MLFLOW = os.environ.get("MLFLOW_URL", "http://127.0.0.1:5005")


def mlflow_experiment(name):
    import http.client
    u = urllib.parse.urlsplit(MLFLOW)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
    c.request("GET", "/api/2.0/mlflow/experiments/get-by-name?" + urllib.parse.urlencode({"experiment_name": name}))
    r = c.getresponse()
    body = r.read().decode()
    c.close()
    return r.status, (json.loads(body) if body.startswith("{") else {})


def url_status(url):
    return D.s3("GET", url)[0]


def build_project(tok, org, name, train=True):
    pid, did, pj = D.img_project(tok, org, name)
    _, v, _ = D.upload(tok, pid, did, "folder", D.folder_files("images/clean"))
    E.seal_and_validate(tok, v)
    out = {"pid": pid, "did": did, "name": pj["name"], "vid": v["id"], "urls": [s["url"] for s in D.samples(tok, v["id"])[:6]]}
    if train:
        r = call("POST", f"/projects/{pid}/training-jobs", {"dataset_version_id": v["id"], "budget_mode": "quick"}, tok=tok)
        job = E.wait(f"/training-jobs/{r.json['job']['id']}", tok, lambda j: j.get("status") in ("succeeded", "failed", "cancelled"), E.TRAIN_TIMEOUT, "training")
        assert job["status"] == "succeeded", job
        out["job"], out["mv"] = job["id"], job["model_version_id"]
        E.wait(f"/model-versions/{out['mv']}/evaluation", tok, lambda j: j.get("status") in ("ready", "failed"), 600, "evaluation")
        ex = call("POST", f"/model-versions/{out['mv']}/exports", {"format": "onnx"}, tok=tok)
        exd = E.wait(f"/exports/{ex.json['id']}", tok, lambda j: j.get("status") in ("ready", "failed"), 600, "export")
        out["export_url"] = exd.get("download_url")
    return out


def main():
    if call("GET", "/health").status != 200:
        sys.exit("API not healthy")
    E.PATHS.update(call("GET", "/openapi.json").json["paths"])
    print(f"run {A.RUN} against {A.API}")
    tok, _, _ = A.login(A.user("pu")[0])
    team = call("POST", "/orgs", {"name": f"Purge Team {A.RUN}"}, tok=tok).json["id"]
    roles = {}
    for role in ("admin", "member", "viewer"):
        e, _ = A.user(f"pu{role}")
        assert call("POST", f"/orgs/{team}/members", {"email": e, "role": role}, tok=tok).status == 201
        roles[role] = A.login(e)[0]
    tok_b, _, _ = A.login(A.user("pub")[0])

    P = build_project(tok, team, "PURGE target")
    S = build_project(tok, team, "PURGE sibling", train=False)

    # ---- best metric on ProjectOut ----
    pj = call("GET", f"/projects/{P['pid']}", tok=tok).json
    ev = call("GET", f"/model-versions/{P['mv']}/evaluation", tok=tok).json["report"]
    check("BEST ProjectOut best_metric = latest ready model's primary metric", pj.get("latest_model_version_id") == P["mv"] and pj.get("best_metric_name") == ev["primary_metric"]
          and abs((pj.get("best_metric") or -1) - ev["metrics"][ev["primary_metric"]]) < 1e-9 and pj.get("best_metric_split") in ("test", "val"),
          json.dumps({k: pj.get(k) for k in ("best_metric", "best_metric_name", "best_metric_split", "latest_model_version_id")}))
    sib = call("GET", f"/projects/{S['pid']}", tok=tok).json
    check("BEST project without a model -> best_metric null", sib.get("best_metric") is None and sib.get("latest_model_version_id") is None, json.dumps({k: sib.get(k) for k in ("best_metric", "latest_model_version_id")}))

    # ---- pre-purge state ----
    st, exp = mlflow_experiment(f"project-{P['pid']}")
    check("PURGE pre: MLflow experiment project-<id> exists", st == 200 and exp.get("experiment", {}).get("lifecycle_stage") == "active", f"{st} {json.dumps(exp)[:200]}")
    check("PURGE pre: sample objects + export zip readable", all(url_status(u) == 200 for u in P["urls"] + [P["export_url"]]), "")

    # ---- guards ----
    body = {"confirm_name": P["name"]}
    for role in ("member", "viewer"):
        expect(f"PURGE [{role}] -> 403", call("POST", f"/projects/{P['pid']}:purge", body, tok=roles[role]), 403)
    expect("PURGE other tenant -> 404", call("POST", f"/projects/{P['pid']}:purge", body, tok=tok_b), 404)
    expect("PURGE wrong confirm_name -> 422 confirmation_mismatch", call("POST", f"/projects/{P['pid']}:purge", {"confirm_name": P["name"] + "x"}, tok=tok), 422, "confirmation_mismatch")
    expect("PURGE case-variant confirm_name -> 422", call("POST", f"/projects/{P['pid']}:purge", {"confirm_name": P["name"].lower()}, tok=tok), 422, "confirmation_mismatch")

    # ---- purge (admin) ----
    t0 = time.time()
    r = call("POST", f"/projects/{P['pid']}:purge", body, tok=roles["admin"])
    check("PURGE [admin] -> 202 Deleting + task_id", r.status == 202 and r.json.get("status") == "Deleting" and r.json.get("task_id") and r.json.get("project_id") == P["pid"], r.raw[:300], r)
    lst = call("GET", f"/orgs/{team}/projects?limit=100", tok=tok)
    check("PURGE project gone from list immediately; sibling still listed", P["pid"] not in lst.raw and S["pid"] in lst.raw, "")
    g = call("GET", f"/projects/{P['pid']}", tok=tok)
    check("PURGE GET during purge -> Deleting (or already 404)", (g.status == 200 and g.json.get("status") == "Deleting") or g.status == 404, f"{g.status} {g.raw[:150]}", g)
    w = call("PATCH", f"/projects/{P['pid']}", {"name": "renamed"}, tok=tok)
    check("PURGE write during purge -> 409 project_deleting (or 404 if already gone)", (w.status == 409 and w.code == "project_deleting") or w.status == 404, f"{w.status} {w.code}", w)
    end = time.time() + 120
    while time.time() < end and call("GET", f"/projects/{P['pid']}", tok=tok).status != 404:
        time.sleep(1)
    check("PURGE project -> 404 when done", call("GET", f"/projects/{P['pid']}", tok=tok).status == 404, f"after {time.time() - t0:.0f}s")
    print(f"INFO purge completed in {time.time() - t0:.1f}s")
    for path in (f"/model-versions/{P['mv']}", f"/datasets/{P['did']}/versions", f"/dataset-versions/{P['vid']}", f"/training-jobs/{P['job']}"):
        expect(f"PURGE child {path.split('/')[1]} -> 404", call("GET", path, tok=tok), 404)
    check("PURGE MinIO: sample objects gone (404)", all(url_status(u) == 404 for u in P["urls"]), str([url_status(u) for u in P["urls"]]))
    check("PURGE MinIO: export zip gone (404)", url_status(P["export_url"]) == 404, str(url_status(P["export_url"])))
    st, exp = mlflow_experiment(f"project-{P['pid']}")
    check("PURGE MLflow experiment deleted", st == 404 or exp.get("experiment", {}).get("lifecycle_stage") == "deleted", f"{st} {json.dumps(exp)[:200]}")
    audit = call("GET", f"/orgs/{team}/audit-logs?limit=200", tok=tok).raw
    check("PURGE audit has project.purge_requested + project.purged", "project.purge_requested" in audit and "project.purged" in audit and P["pid"] in audit, "")
    sg = call("GET", f"/projects/{S['pid']}", tok=tok)
    check("PURGE sibling project untouched (200, samples readable)", sg.status == 200 and all(url_status(u) == 200 for u in S["urls"]), f"{sg.status}")

    # ---- purge with a running job: cancelled, not 409 ----
    R = build_project(tok, team, "PURGE running", train=False)
    r = call("POST", f"/projects/{R['pid']}/training-jobs", {"dataset_version_id": R["vid"], "budget_mode": "quick"}, tok=tok)
    job = r.json["job"]["id"]
    time.sleep(2)
    r = call("POST", f"/projects/{R['pid']}:purge", {"confirm_name": R["name"]}, tok=tok)
    check("PURGE with a queued/running job -> 202 (job cancelled, not 409)", r.status == 202, f"{r.status} {r.raw[:200]}", r)
    end = time.time() + 180
    while time.time() < end and call("GET", f"/projects/{R['pid']}", tok=tok).status != 404:
        time.sleep(1)
    check("PURGE running-job project -> 404 when done", call("GET", f"/projects/{R['pid']}", tok=tok).status == 404, "")

    print("\nSQL for the service runner (expect 0 everywhere):")
    for pid in (P["pid"], R["pid"]):
        print(f"  -- project {pid}")
        print(f"  SELECT 'projects',count(*) FROM projects WHERE id='{pid}' UNION ALL SELECT 'datasets',count(*) FROM datasets WHERE project_id='{pid}' UNION ALL SELECT 'training_jobs',count(*) FROM training_jobs WHERE project_id='{pid}' UNION ALL SELECT 'models',count(*) FROM models WHERE project_id='{pid}' UNION ALL SELECT 'task_spec_versions',count(*) FROM task_spec_versions WHERE project_id='{pid}' UNION ALL SELECT 'label_sets',count(*) FROM label_sets WHERE project_id='{pid}';")
    print(f"  SELECT 'dataset_versions',count(*) FROM dataset_versions WHERE dataset_id='{P['did']}' UNION ALL SELECT 'samples',count(*) FROM samples s JOIN dataset_versions v ON v.id=s.dataset_version_id WHERE v.dataset_id='{P['did']}' UNION ALL SELECT 'model_versions',count(*) FROM model_versions WHERE id='{P['mv']}' UNION ALL SELECT 'exports',count(*) FROM exports WHERE model_version_id='{P['mv']}' UNION ALL SELECT 'trials',count(*) FROM trials WHERE training_job_id='{P['job']}';")
    print(f"  MinIO prefixes to list (expect empty): org/{team}/datasets/{P['did']}/  org/{team}/models/  (model of {P['mv']})")
    fails = [x for x in A.RESULTS if not x["pass"]]
    out = A.OUT / f"purge_{A.RUN}.json"
    out.write_text(json.dumps({"run": A.RUN, "ids": {"team": team, "P": {k: v for k, v in P.items() if k != "urls"}, "S": S["pid"], "R": R["pid"]}, "results": A.RESULTS, "log": A.LOG}, indent=1))
    print(f"\n{len(A.RESULTS) - len(fails)}/{len(A.RESULTS)} pass, {len(fails)} fail. Log: {out}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
