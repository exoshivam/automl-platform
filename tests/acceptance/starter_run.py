#!/usr/bin/env python3
"""Starter dataset library (PRD DAT-13): every catalog entry through import -> seal + validate -> train (quick) -> evaluate,
plus the contract checks (task_mismatch, import_running, already_imported, user uploads into the same draft, cross-tenant).

    python3 tests/acceptance/starter_run.py                 # every catalog entry
    python3 tests/acceptance/starter_run.py beans tf_flowers  # only these ids
    STARTER_NO_TRAIN=1 python3 tests/acceptance/starter_run.py   # import + validate only (no train-worker use)
"""
import json
import os
import sys
import time

import acc_run as A
import dat_run as D
import e2e_run as E
import val_run as V
from acc_run import call, check, expect

TEMPLATE = {"image_classification": "image_classification", "object_detection": "object_detection", "text_classification": "text_classification",
            "tabular_classification": "tabular", "tabular_regression": "tabular"}
WRONG = {"image_classification": ("text_classification", None), "object_detection": ("image_classification", None), "text_classification": ("image_classification", None),
         "tabular_classification": ("image_classification", None), "tabular_regression": ("image_classification", None)}
ROWS = []


def project_for(tok, org, st, suffix=""):
    t = st["task_type"]
    kw = {"template": TEMPLATE[t]}
    if t.startswith("tabular"):
        kw["task_type"] = t
    else:
        kw["labels"] = st["classes"]
    return D.project(tok, org, f"starter {st['id']}{suffix}", **kw)


def start_import(tok, pid, sid):
    return call("POST", f"/projects/{pid}/starter-imports", {"starter_id": sid}, tok=tok)


def run_entry(tok, org, st):
    row = {"id": st["id"], "task_type": st["task_type"], "catalog_samples": st["sample_count"], "licence": st["licence"]}
    t0 = time.time()
    try:
        pid, did, _ = project_for(tok, org, st)
        r = start_import(tok, pid, st["id"])
        assert r.status == 202, r.brief()
        v = D.wait_version(tok, r.json["dataset_version_id"], timeout=900)
        row["import_s"] = round(time.time() - t0, 1)
        assert v["ingest_status"] == "ready", f"ingest {v['ingest_status']} {v.get('ingest_errors')}"
        row["samples"] = v["sample_count"]
        check(f"STARTER {st['id']} import -> sample_count == catalog ({st['sample_count']})", v["sample_count"] == st["sample_count"], f"got {v['sample_count']}")
        if st["classes"] and not st["task_type"].startswith("tabular"):
            kind = "boxes" if st["task_type"] == "object_detection" else "images"
            got = sorted(c for c, n in D.label_counts(v, kind).items() if n)
            check(f"STARTER {st['id']} classes == catalog", got == sorted(st["classes"]), f"got {got}")
        r = call("POST", f"/dataset-versions/{v['id']}/seal", tok=tok)
        assert r.status == 202, r.brief()
        rep = V.report(tok, v["id"], timeout=600)
        row["validation"] = {"status": rep["status"], "codes": sorted({f"{i['code']}:{i['severity'][0]}" for i in rep["items"]})}
        errors = [i for i in rep["items"] if i["severity"] == "error" and not i.get("acknowledgeable")]
        check(f"STARTER {st['id']} validation has no blocking errors", not errors, json.dumps([(i["code"], i["title"], i["count"]) for i in errors]))
        ack = sorted({i["code"] for i in rep["items"] if i.get("acknowledgeable") and not i.get("acknowledged")})
        if ack:
            call("POST", f"/dataset-versions/{v['id']}/validation/acknowledge", {"codes": ack}, tok=tok)
        status = call("GET", f"/projects/{pid}", tok=tok).json["status"]
        check(f"STARTER {st['id']} project Ready to train after validation", status == "Ready to train", status)
        if os.environ.get("STARTER_NO_TRAIN"):
            row["train"] = "skipped (STARTER_NO_TRAIN)"
            return row
        r = call("POST", f"/projects/{pid}/training-jobs", {"dataset_version_id": v["id"], "budget_mode": "quick"}, tok=tok)
        assert r.status == 202, r.brief()
        row["estimate_s"] = r.json.get("estimate", {}).get("eta_seconds")
        job = E.wait(f"/training-jobs/{r.json['job']['id']}", tok, lambda j: j.get("status") in ("succeeded", "failed", "cancelled"), E.TRAIN_TIMEOUT, "training")
        row["train_s"] = E._dur(job)
        row["job"] = job["id"]
        check(f"STARTER {st['id']} training succeeded", job["status"] == "succeeded", f"{job['status']} {job.get('error_code')} {str(job.get('error_message'))[:300]}")
        if job["status"] != "succeeded":
            row["error"] = f"{job.get('error_code')}: {str(job.get('error_message'))[:200]}"
            return row
        ev = E.wait(f"/model-versions/{job['model_version_id']}/evaluation", tok, lambda j: j.get("status") in ("ready", "failed"), 900, "evaluation")
        check(f"STARTER {st['id']} evaluation ready", ev["status"] == "ready", f"{ev['status']} {ev.get('error')}")
        rep = ev.get("report") or {}
        m = rep.get("metrics", {})
        row.update({"primary": rep.get("primary_metric"), "value": m.get(rep.get("primary_metric")), "n_test": rep.get("n_test"),
                    "baseline": (rep.get("baseline") or {}).get("metrics"),
                    "metrics": {k: round(v_, 4) for k, v_ in m.items() if isinstance(v_, (int, float))}})
        if st["task_type"] in ("image_classification", "text_classification", "tabular_classification"):
            base = (rep.get("baseline") or {}).get("metrics", {}).get(rep.get("primary_metric"))
            check(f"STARTER {st['id']} beats the baseline", base is None or (row["value"] or 0) > base, f"{row['value']} vs baseline {base}")
    except AssertionError as e:
        check(f"STARTER {st['id']} flow", False, str(e)[:500])
        row["error"] = str(e)[:300]
    except Exception as e:
        check(f"STARTER {st['id']} crashed", False, f"{type(e).__name__}: {str(e)[:400]}")
        row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        row["total_s"] = round(time.time() - t0, 1)
        ROWS.append(row)
    return row


def contract(tok, org, tok_b, catalog):
    st = next((s for s in catalog if s["task_type"] == "image_classification"), catalog[0])
    # previews
    check("STARTER catalog entries have source + licence + previews", all(s.get("source") and s.get("licence") for s in catalog), "")
    if st.get("preview_urls"):
        check("STARTER preview URLs load", all(D.s3("GET", u)[0] == 200 for u in st["preview_urls"][:3]), "")
    flt = call("GET", f"/starter-datasets?task_type={st['task_type']}", tok=tok).json
    check("STARTER ?task_type filter", flt and all(s["task_type"] == st["task_type"] for s in flt), f"{[s['task_type'] for s in flt]}")
    # wrong task type
    wt, _ = WRONG[st["task_type"]]
    wpid, _, _ = D.project(tok, org, f"starter wrong {st['id']}", template=TEMPLATE[wt], labels=["a", "b"])
    expect(f"STARTER {st['id']} into a {wt} project -> 422 task_mismatch", start_import(tok, wpid, st["id"]), 422, "task_mismatch")
    expect("STARTER unknown starter_id -> 404", start_import(tok, wpid, "no_such_starter"), 404)
    # double import + user upload into the same draft
    pid, did, _ = project_for(tok, org, st, " (contract)")
    r1 = start_import(tok, pid, st["id"])
    r2 = start_import(tok, pid, st["id"])
    check("STARTER second import while the first runs -> 409 import_running (or already_imported)", r1.status == 202 and r2.status == 409 and r2.code in ("import_running", "already_imported"), f"{r1.status} / {r2.status} {r2.code}", r2)
    v = D.wait_version(tok, r1.json["dataset_version_id"], timeout=900)
    expect("STARTER import again after it finished -> 409 already_imported", start_import(tok, pid, st["id"]), 409, "already_imported")
    extra = D.folder_files("images/clean")[:3]
    extra = [(f"{st['classes'][0]}/{rel.split('/')[-1]}", data) for rel, data in extra]
    c, v2, _ = D.upload(tok, pid, did, "folder", extra)
    check("STARTER user upload on top lands in the SAME draft (+3 samples)", c.json.get("dataset_version_id") == v["id"] and v2["sample_count"] == v["sample_count"] + 3, f"{v['id'][-6:]} vs {str(c.json.get('dataset_version_id'))[-6:]}, {v['sample_count']} -> {v2 and v2['sample_count']}")
    # cross-tenant
    expect("STARTER B imports into A's project -> 404", start_import(tok_b, pid, st["id"]), 404)
    k_b = call("POST", f"/orgs/{A.personal_org(tok_b)}/api-keys", {"name": "b", "scopes": ["projects:write"]}, tok=tok_b).json["key"]
    expect("STARTER B's key imports into A's project -> 404", call("POST", f"/projects/{pid}/starter-imports", {"starter_id": st["id"]}, key=k_b), 404)
    check("STARTER catalog readable by any signed-in user", call("GET", "/starter-datasets", tok=tok_b).status == 200, "")
    expect("STARTER catalog without auth -> 401", call("GET", "/starter-datasets"), 401)


def main():
    if call("GET", "/health").status != 200:
        sys.exit("API not healthy")
    E.PATHS.update(call("GET", "/openapi.json").json["paths"])
    tok, _, _ = A.login(A.user("st")[0])
    org = A.personal_org(tok)
    tok_b, _, _ = A.login(A.user("stb")[0])
    catalog = call("GET", "/starter-datasets", tok=tok).json
    ids = [a for a in sys.argv[1:] if not a.startswith("-")]
    todo = [s for s in catalog if not ids or s["id"] in ids]
    print(f"run {A.RUN}: catalog {[s['id'] for s in catalog]}; running {[s['id'] for s in todo]}")
    try:
        contract(tok, org, tok_b, catalog)
    except Exception as e:
        check("STARTER contract crashed", False, f"{type(e).__name__}: {str(e)[:400]}")
    for st in todo:
        print(f"--- {st['id']} ({st['task_type']}, {st['sample_count']} samples, {st['size_bytes'] / 1e6:.0f} MB, {st['licence']})")
        row = run_entry(tok, org, st)
        print("    " + json.dumps({k: row.get(k) for k in ("samples", "import_s", "validation", "train_s", "primary", "value", "n_test", "baseline", "error")}))
    fails = [x for x in A.RESULTS if not x["pass"]]
    out = A.OUT / f"starter_{A.RUN}.json"
    out.write_text(json.dumps({"run": A.RUN, "rows": ROWS, "results": A.RESULTS, "log": A.LOG}, indent=1))
    print("\nSUMMARY")
    for r in ROWS:
        print(f"  {r['id']:<22} {r['task_type']:<24} samples={r.get('samples')} import={r.get('import_s')}s train={r.get('train_s')}s {r.get('primary')}={r.get('value')} {('ERROR ' + r['error']) if r.get('error') else ''}")
    print(f"\n{len(A.RESULTS) - len(fails)}/{len(A.RESULTS)} pass, {len(fails)} fail. Log: {out}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
