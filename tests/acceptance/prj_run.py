#!/usr/bin/env python3
"""Acceptance run for PRJ-1..7 + project-path role checks and NFR-SEC3 (CHECKLIST.md).

    python3 tests/acceptance/prj_run.py                 # server stub mode valid (default)
    python3 tests/acceptance/prj_run.py --mode invalid  # server restarted with LLM_STUB_MODE=invalid
    python3 tests/acceptance/prj_run.py --mode down     # server restarted with LLM_STUB_MODE=down
The stub mode is server env; --mode says which one to assert.
"""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import acc_run as A
from acc_run import call, check, expect, problem_ok

MODE = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "valid"
FX = Path(__file__).resolve().parents[1] / "fixtures"
DESC = "Detect whether workers wear helmets and vests in CCTV photos; must run offline on Android"


def llm_logs(tok):
    r = call("GET", "/_dev/llm-log", tok=tok)
    return r.json if isinstance(r.json, list) else (r.json or {}).get("items", [])


def projects(tok, org, **q):
    qs = "&".join(f"{k}={v}" for k, v in q.items())
    r = call("GET", f"/orgs/{org}/projects" + (f"?{qs}" if qs else ""), tok=tok)
    return r, (r.json or {}).get("items", [])


def all_pages(tok, org, limit=1, **q):
    seen, cursor = [], None
    for _ in range(50):
        r, items = projects(tok, org, limit=limit, **({"cursor": cursor} if cursor else {}), **q)
        seen += [p["id"] for p in items]
        cursor = (r.json or {}).get("next_cursor")
        if not cursor:
            break
    return seen


def create(tok, org, name, **body):
    return call("POST", f"/orgs/{org}/projects", {"name": name, **body}, tok=tok)


def run_valid(tok, org, tok_b, org_b, roles):
    # ---------- PRJ-1 ----------
    r = call("GET", "/templates", tok=tok)
    tm = {t["id"]: t for t in r.json or []}
    enabled = sorted(t for t, v in tm.items() if v["enabled"])
    check("PRJ-1a enabled templates = image cls, detection, text, tabular", enabled == sorted(["image_classification", "object_detection", "text_classification", "tabular"]), f"enabled {enabled}", r)
    check("PRJ-1a audio + later shown as disabled", tm.get("audio_classification", {}).get("enabled") is False and len(tm) > len(enabled), f"templates {sorted(tm)}", r)
    need = {"spec_version", "task_type", "labels", "primary_metric", "constraints", "budget_mode", "split"}
    check("PRJ-1a enabled templates have full default spec", all(need <= set(tm[t]["default_spec"] or {}) for t in enabled), "missing keys", r)
    check("PRJ-1a tabular default has target/feature/id/group columns", {"target_column", "feature_columns", "id_columns", "group_column"} <= set(tm["tabular"]["default_spec"]), "", r)

    labels = ["healthy", "early_blight", "late_blight"]
    p = create(tok, org, "Plants", template="image_classification", labels=labels)
    pj = p.json or {}
    sp = pj.get("current_spec", {})
    ok = p.status == 201 and pj.get("task_type") == "image_classification" and sp.get("labels") == labels and pj.get("spec_version") == 1 \
        and sp.get("split", {}).get("train") == 0.7 and sp.get("split", {}).get("val") == 0.15 and sp.get("split", {}).get("test") == 0.15 and sp["split"].get("seed") is not None
    check("PRJ-1b create from template prefilled", ok, f"{p.status} {json.dumps(pj)[:300]}", p)
    check("PRJ-1b status Draft or Data needed", pj.get("status") in ("Draft", "Data needed"), f"status {pj.get('status')}", p)
    check("PRJ-1b dataset + label set created", bool(pj.get("dataset_id")) and bool(pj.get("label_set_id")), "", p)
    pid = pj.get("id")
    v = call("GET", f"/projects/{pid}/spec/versions", tok=tok)
    vs = v.json if isinstance(v.json, list) else (v.json or {}).get("items", [])
    check("PRJ-1b spec v1 source template", len(vs) == 1 and vs[0]["version"] == 1 and vs[0]["source"] == "template", f"versions {[(x.get('version'), x.get('source')) for x in vs]}", v)

    expect("PRJ-1c disabled template (audio) -> 422", create(tok, org, "Audio", template="audio_classification", labels=["a", "b"]), 422)
    expect("PRJ-1c unknown template -> 422", create(tok, org, "X", template="no_such_template", labels=["a", "b"]), 422)
    for name, lab in [("1 label", ["only"]), ("501 labels", [f"c{i}" for i in range(501)]), ("duplicate", ["a", "a"]), ("empty string", ["a", ""]),
                      ("case-duplicate", ["Cat", "cat"])]:
        r = create(tok, org, f"L-{name}", template="image_classification", labels=lab)
        ok = r.status == 422 and problem_ok(r)
        check(f"PRJ-1d labels {name} -> 422", ok, f"got {r.status}", r)
    r = create(tok, org, "L-500", template="image_classification", labels=[f"c{i}" for i in range(500)])
    check("PRJ-1d 500 labels accepted (boundary)", r.status == 201, f"got {r.status}", r)
    r = create(tok, org, "Tab-reg", template="tabular", task_type="tabular_regression")
    check("PRJ-1b tabular template with task_type regression", r.status == 201 and (r.json or {}).get("task_type") == "tabular_regression", f"got {r.status} {(r.json or {}).get('task_type')}", r)
    expect("PRJ-1c tabular template with image task_type -> 422", create(tok, org, "Tab-bad", template="tabular", task_type="image_classification"), 422)
    expect("PRJ-1 create with no template/spec -> 422", create(tok, org, "Nothing"), 422)

    # ---------- PRJ-2 (valid stub) ----------
    before = len(all_pages(tok, org, limit=100))
    d = call("POST", f"/orgs/{org}/projects/from-description", {"description": DESC}, tok=tok)
    dj = d.json or {}
    ds = dj.get("draft_spec") or {}
    ok = d.status == 200 and dj.get("status") == "ok" and dj.get("category") == "object_detection" and ds.get("task_type") == "object_detection" \
        and {"helmet", "vest"} <= set(ds.get("labels", [])) and "android" in ds.get("constraints", {}).get("target_platforms", [])
    check("PRJ-2a from-description -> detection draft (helmet, vest, android)", ok, f"{d.status} {dj.get('status')} {dj.get('category')}", d)
    check("PRJ-2a draft has full spec keys", need <= set(ds), f"keys {sorted(ds)}", d)
    after = len(all_pages(tok, org, limit=100))
    check("PRJ-2a draft creates no project", after == before, f"{before} -> {after}")
    r = create(tok, org, "From LLM", spec=ds, llm_log_id=dj.get("llm_log_id"))
    check("PRJ-2a create from draft spec -> 201", r.status == 201, f"got {r.status}", r)
    if r.status == 201:
        vv = call("GET", f"/projects/{r.json['id']}/spec/versions/1", tok=tok)
        check("PRJ-2a v1 source llm", (vv.json or {}).get("source") == "llm", f"source {(vv.json or {}).get('source')}", vv)
    bad = dict(ds, primary_metric="macro_f1", task_type="object_detection")
    expect("PRJ-2a/NFR-SEC9 tampered draft (invalid metric) rejected", create(tok, org, "Tampered", spec=dict(bad, labels=["x"])), 422)

    # ---------- PRJ-3 ----------
    s1 = sp
    s2 = json.loads(json.dumps(s1))
    s2["constraints"]["max_model_size_mb"] = 10
    r = call("PUT", f"/projects/{pid}/spec", {"spec": s2}, tok=tok)
    check("PRJ-3a valid edit -> 200, spec_version 2", r.status == 200 and (r.json or {}).get("spec_version") == 2, f"got {r.status} v={(r.json or {}).get('spec_version')}", r)
    v2 = call("GET", f"/projects/{pid}/spec/versions/2", tok=tok)
    check("PRJ-3a v2 source user_edit", (v2.json or {}).get("source") == "user_edit" and v2.json["spec"]["constraints"]["max_model_size_mb"] == 10, "", v2)
    v1 = call("GET", f"/projects/{pid}/spec/versions/1", tok=tok)
    check("PRJ-3a v1 unchanged", (v1.json or {}).get("spec") == s1, "v1 changed", v1)

    def mut(**kw):
        x = json.loads(json.dumps(s2))
        for k, val in kw.items():
            if val is _DEL:
                x.pop(k, None)
            else:
                x[k] = val
        return x

    cases = [
        ("primary_metric map_50 on image cls", mut(primary_metric="map_50"), "/primary_metric"),
        ("unknown task_type", mut(task_type="generate_images"), "/task_type"),
        ("split sums to 1.1", mut(split={**s2["split"], "train": 0.8}), "/split"),
        ("missing labels", mut(labels=_DEL), "/labels"),
        ("labels not a list", mut(labels="cat"), "/labels"),
        ("1 label", mut(labels=["cat"]), "/labels"),
        ("max_model_size_mb negative", mut(constraints={**s2["constraints"], "max_model_size_mb": -5}), "/constraints"),
        ("unknown budget_mode", mut(budget_mode="infinite"), "/budget_mode"),
    ]
    for name, spec, ptr in cases:
        r = call("PUT", f"/projects/{pid}/spec", {"spec": spec}, tok=tok)
        locs = json.dumps((r.json or {}).get("errors", []))
        ok = r.status == 422 and problem_ok(r) and ptr in locs
        check(f"PRJ-3b {name} -> 422 pointer {ptr}", ok, f"got {r.status} errors={locs[:200]}", r)
    g = call("GET", f"/projects/{pid}", tok=tok)
    check("PRJ-3b spec_version unchanged after rejects", (g.json or {}).get("spec_version") == 2, f"v={(g.json or {}).get('spec_version')}", g)
    s3 = mut(task_type="tabular_regression")
    r = call("PUT", f"/projects/{pid}/spec", {"spec": s3}, tok=tok)
    check("PRJ-3b task_type change across categories rejected (record)", r.status == 422, f"got {r.status} (accepted a category switch?)", r)

    v = call("GET", f"/projects/{pid}/spec/versions", tok=tok)
    vs = v.json if isinstance(v.json, list) else (v.json or {}).get("items", [])
    check("PRJ-3c history lists every version", sorted(x["version"] for x in vs) == [1, 2], f"{[x['version'] for x in vs]}", v)
    expect("PRJ-3c missing version -> 404", call("GET", f"/projects/{pid}/spec/versions/99", tok=tok), 404)

    # ---------- PRJ-4 ----------
    rows = list(csv.DictReader(open(FX / "tabular/leaky.csv", encoding="utf-8")))
    cols = list(rows[0])
    d = call("POST", f"/orgs/{org}/projects/from-description", {"description": "Predict which customers will churn", "include_sample_rows": True, "columns": cols, "sample_rows": rows}, tok=tok)
    log = next((l for l in llm_logs(tok) if l["id"] == (d.json or {}).get("llm_log_id")), None)
    pl = (log or {}).get("request", {}).get("payload", {})
    sent = pl.get("sample_rows") or pl.get("rows") or []
    check("PRJ-4a opt-in: payload has description + columns", d.status == 200 and log and pl.get("description") and pl.get("columns") == cols, f"{d.status} payload keys {sorted(pl)}", d)
    check("PRJ-4a opt-in: at most 5 rows sent (300 supplied)", 0 < len(sent) <= 5, f"rows sent {len(sent)}", d)
    raw = json.dumps(log or {})
    leaked = [r["customer_id"] for r in rows[5:] if r["customer_id"] in raw]
    check("PRJ-4a opt-in: no row beyond 5 anywhere in the logged request", not leaked, f"{len(leaked)} later rows found in log")
    d = call("POST", f"/orgs/{org}/projects/from-description", {"description": "Predict which customers will churn", "include_sample_rows": False, "columns": cols, "sample_rows": rows}, tok=tok)
    log = next((l for l in llm_logs(tok) if l["id"] == (d.json or {}).get("llm_log_id")), None)
    raw = json.dumps((log or {}).get("request", {}))
    check("PRJ-4b no opt-in: zero sample rows sent", d.status == 200 and log and not any(r["customer_id"] in raw for r in rows), "rows present without opt-in", d)
    pl = (log or {}).get("request", {}).get("payload", {})
    A.RESULTS.append({"id": "PRJ-4b record: columns sent without opt-in?", "pass": True, "why": f"payload keys {sorted(pl)}", "note": True})
    print(f"INFO PRJ-4b payload keys without opt-in: {sorted(pl)}")
    d = call("POST", f"/orgs/{org}/projects/from-description", {"description": "x", "include_sample_rows": True, "columns": cols, "sample_rows": [{"customer_id": "1"}] * 6}, tok=tok)
    check("PRJ-4a 6 rows with opt-in: accepted but capped, or 422", d.status in (200, 422), f"got {d.status}", d)
    logs = llm_logs(tok)
    ok = logs and all(l.get("expires_at") and _days(l["created_at"], l["expires_at"]) == 30 for l in logs)
    check("PRJ-4c every llm_log expires after 30 days", ok, f"{[(l.get('created_at'), l.get('expires_at')) for l in logs[:2]]}")
    # Dev hooks are global by design but dev/test-only and loopback-only. A request that a
    # trusted local proxy marks as coming from a foreign IP must not get through.
    for hook in ("/_dev/llm-log", "/_dev/mail"):
        r = call("GET", hook, headers={"X-Forwarded-For": "10.1.2.3"})
        check(f"PRJ-4/SEC dev hook {hook} refused for non-loopback client", r.status in (400, 403, 404), f"got {r.status}", r)

    # ---------- PRJ-5 ----------
    for text in ("generate images of cats", "predict stock prices with certainty"):
        d = call("POST", f"/orgs/{org}/projects/from-description", {"description": text}, tok=tok)
        dj = d.json or {}
        ok = d.status in (200, 422) and dj.get("status") != "ok" and dj.get("message") and not dj.get("draft_spec")
        check(f"PRJ-5 '{text}' flagged unsupported with explanation", ok, f"{d.status} status={dj.get('status')} draft={'yes' if dj.get('draft_spec') else 'no'}", d)

    # ---------- PRJ-6 ----------
    s6 = json.loads(json.dumps(s2))
    s6["constraints"].update({"target_platforms": ["android"], "max_model_size_mb": 1, "max_latency_ms": 1})
    s6["budget_mode"] = "quick"
    r = call("PUT", f"/projects/{pid}/spec", {"spec": s6}, tok=tok)
    check("PRJ-6 constraints stored (filtering is TRN: N/A this slice)", r.status == 200 and r.json["current_spec"]["constraints"]["max_model_size_mb"] == 1, f"got {r.status}", r)

    # ---------- PRJ-7 ----------
    ids = [create(tok, org, f"List {n} {A.RUN}", template="text_classification", labels=["pos", "neg"]).json["id"] for n in ("alpha", "beta", "gamma")]
    d = call("DELETE", f"/projects/{ids[2]}", tok=tok)
    check("PRJ-7a archive (DELETE) -> 204", d.status == 204, f"got {d.status}", d)
    _, default = projects(tok, org, limit=100)
    _, arch = projects(tok, org, archived="true", limit=100)
    check("PRJ-7a archived hidden from default list", ids[2] not in [p["id"] for p in default] and ids[0] in [p["id"] for p in default], "")
    check("PRJ-7a archived listed with ?archived=true", ids[2] in [p["id"] for p in arch], "")
    g = call("GET", f"/projects/{ids[2]}", tok=tok)
    check("PRJ-7a archived row kept (archived_at set)", g.status == 200 and (g.json or {}).get("archived_at"), f"got {g.status}", g)
    u = call("PATCH", f"/projects/{ids[2]}", {"archived": False}, tok=tok)
    _, default = projects(tok, org, limit=100)
    check("PRJ-7a unarchive via PATCH archived:false", u.status == 200 and ids[2] in [p["id"] for p in default], f"got {u.status}", u)
    r = call("PATCH", f"/projects/{ids[0]}", {"name": f"List alpha renamed {A.RUN}"}, tok=tok)
    check("PRJ-7 PATCH name", r.status == 200 and "renamed" in (r.json or {}).get("name", ""), f"got {r.status}", r)
    _, hits = projects(tok, org, search=f"renamed%20{A.RUN}", limit=100)
    check("PRJ-7b search", [p["id"] for p in hits] == [ids[0]], f"hits {[p['name'] for p in hits]}")
    _, drafts = projects(tok, org, status="Draft", limit=100)
    check("PRJ-7b status=Draft filter", drafts and all(p["status"] == "Draft" for p in drafts), f"{len(drafts)} items")
    expect("PRJ-7b unknown status -> 422", projects(tok, org, status="Bogus")[0], 422)
    expect("PRJ-7b limit=0 -> 422", projects(tok, org, limit=0)[0], 422)
    every = all_pages(tok, org, limit=1)
    full = [p["id"] for p in projects(tok, org, limit=100)[1]]
    check("PRJ-7b cursor pagination: each project exactly once", sorted(every) == sorted(full) and len(every) == len(set(every)), f"paged {len(every)} vs {len(full)}")
    A.RESULTS.append({"id": "PRJ-7c status flow Draft -> Data needed -> Validating -> Ready to train", "pass": True, "why": "NOT RUN: needs uploads/validation (groups 5-7)", "skipped": True})

    # ---------- role matrix on project paths ----------
    for role in ("admin", "member", "viewer"):
        rtok = roles[role][1]
        tgt = roles["owner_org"]
        r = create(rtok, tgt, f"by {role}", template="image_classification", labels=["a", "b"])
        if role == "viewer":
            expect(f"ACC-4/PRJ [viewer] POST project -> 403", r, 403)
        else:
            check(f"ACC-4/PRJ [{role}] POST project -> 201", r.status == 201, f"got {r.status}", r)
    tp = create(tok, roles["owner_org"], "role target", template="image_classification", labels=["a", "b"]).json["id"]
    for role in ("admin", "member", "viewer"):
        rtok = roles[role][1]
        g = call("GET", f"/projects/{tp}", tok=rtok)
        check(f"ACC-4/PRJ [{role}] GET project -> 200", g.status == 200, f"got {g.status}", g)
        sp_ = g.json["current_spec"]
        calls = [("PATCH", f"/projects/{tp}", {"name": f"{role} edit"}), ("PUT", f"/projects/{tp}/spec", {"spec": sp_}),
                 ("POST", f"/orgs/{roles['owner_org']}/projects/from-description", {"description": DESC})]
        for m, path, body in calls:
            r = call(m, path, body, tok=rtok)
            if role == "viewer":
                expect(f"ACC-4/PRJ [viewer] {m} {path.split('/')[-1]} -> 403", r, 403)
            else:
                check(f"ACC-4/PRJ [{role}] {m} {path.split('/')[-1]} -> 2xx", 200 <= r.status < 300, f"got {r.status}", r)
    expect("ACC-4/PRJ [viewer] DELETE project -> 403", call("DELETE", f"/projects/{tp}", tok=roles["viewer"][1]), 403)
    k_ro = call("POST", f"/orgs/{roles['owner_org']}/api-keys", {"name": "prj-ro", "scopes": ["projects:read"]}, tok=tok).json["key"]
    k_rw = call("POST", f"/orgs/{roles['owner_org']}/api-keys", {"name": "prj-rw", "scopes": ["projects:write"]}, tok=tok).json["key"]
    check("ACC-6/PRJ read key GET project -> 200", call("GET", f"/projects/{tp}", key=k_ro).status == 200, "")
    check("ACC-6/PRJ read key list -> 200", call("GET", f"/orgs/{roles['owner_org']}/projects", key=k_ro).status == 200, "")
    expect("ACC-6/PRJ read key POST project -> 403", call("POST", f"/orgs/{roles['owner_org']}/projects", {"name": "k", "template": "image_classification", "labels": ["a", "b"]}, key=k_ro), 403)
    r = call("POST", f"/orgs/{roles['owner_org']}/projects", {"name": "k", "template": "image_classification", "labels": ["a", "b"]}, key=k_rw)
    check("ACC-6/PRJ write key POST project -> 201", r.status == 201, f"got {r.status}", r)

    # ---------- NFR-SEC3 project paths ----------
    k_b = call("POST", f"/orgs/{org_b}/api-keys", {"name": "b", "scopes": ["projects:read", "projects:write"]}, tok=tok_b).json["key"]
    snap = call("GET", f"/projects/{pid}", tok=tok).raw
    targets = [("GET", f"/projects/{pid}", None), ("PATCH", f"/projects/{pid}", {"name": "pwned"}), ("PATCH", f"/projects/{pid}", {"archived": True}),
               ("DELETE", f"/projects/{pid}", None), ("PUT", f"/projects/{pid}/spec", {"spec": s2}), ("GET", f"/projects/{pid}/spec/versions", None),
               ("GET", f"/projects/{pid}/spec/versions/1", None), ("GET", f"/orgs/{org}/projects", None),
               ("POST", f"/orgs/{org}/projects", {"name": "pwn", "template": "image_classification", "labels": ["a", "b"]}),
               ("POST", f"/orgs/{org}/projects/from-description", {"description": DESC})]
    for m, path, body in targets:
        short = path.replace(pid, "{A_project}").replace(org, "{ORG_A}")
        expect(f"SEC3 B {m} {short} -> 404", call(m, path, body, tok=tok_b), 404)
        expect(f"SEC3 B-key {m} {short} -> 404", call(m, path, body, key=k_b), 404)
    a_log = next((l["id"] for l in llm_logs(tok)), None)
    r = create(tok_b, org_b, "steal draft", spec=ds, llm_log_id=a_log)
    check("SEC3 B creates project in own org citing A's llm_log_id -> rejected", 400 <= r.status < 500, f"got {r.status} (cross-tenant id in body accepted)", r)
    _, b_items = projects(tok_b, org_b, limit=100)
    check("SEC3 B's list has none of A's projects", not any(p["organization_id"] == org for p in b_items), "")
    check("SEC3 A's project unchanged after attacks", call("GET", f"/projects/{pid}", tok=tok).raw == snap, "A's project changed")
    expect("SEC3 random project id -> 404", call("GET", "/projects/00000000-0000-7000-8000-000000000000", tok=tok), 404)


_DEL = object()


def _days(a, b):
    f = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
    return round((f(b) - f(a)).total_seconds() / 86400)


def run_invalid(tok, org):
    before = {l["id"] for l in llm_logs(tok)}
    d = call("POST", f"/orgs/{org}/projects/from-description", {"description": DESC}, tok=tok)
    new = [l for l in llm_logs(tok) if l["id"] not in before and l.get("organization_id") == org]
    expect("PRJ-2b invalid output -> 422 llm_invalid_output", d, 422, "llm_invalid_output")
    check("PRJ-2b fallback category", (d.json or {}).get("fallback") == "category", f"body {d.raw[:300]}", d)
    check("PRJ-2b exactly 3 LLM attempts logged", len(new) == 3 and sorted(l.get("attempt") for l in new) == [1, 2, 3], f"{len(new)} new logs, attempts {[l.get('attempt') for l in new]}")
    check("PRJ-2b no project created", not projects(tok, org, limit=100)[1], "projects exist")


def run_down(tok, org):
    d = call("POST", f"/orgs/{org}/projects/from-description", {"description": DESC}, tok=tok)
    expect("PRJ-2c LLM down -> 503 llm_unavailable", d, 503, "llm_unavailable")
    check("PRJ-2c templates still served (NFR-R5)", call("GET", "/templates", tok=tok).status == 200, "")
    r = create(tok, org, "fallback", template="image_classification", labels=["a", "b"])
    check("PRJ-2c template create still works", r.status == 201, f"got {r.status}", r)


def main():
    h = call("GET", "/health")
    if h.status != 200:
        sys.exit(f"API not healthy: {h.status}")
    print(f"run {A.RUN} mode {MODE} against {A.API}")
    a_email, _ = A.user("pa")
    tok, _, _ = A.login(a_email)
    org = A.personal_org(tok)
    if MODE == "invalid":
        run_invalid(tok, org)
    elif MODE == "down":
        run_down(tok, org)
    else:
        b_email, _ = A.user("pb")
        tok_b, _, _ = A.login(b_email)
        org_b = A.personal_org(tok_b)
        team = call("POST", "/orgs", {"name": f"Prj Team {A.RUN}"}, tok=tok).json["id"]
        roles = {"owner_org": team}
        for role in ("admin", "member", "viewer"):
            e, _ = A.user(f"p{role}")
            assert call("POST", f"/orgs/{team}/members", {"email": e, "role": role}, tok=tok).status == 201
            roles[role] = (e, A.login(e)[0], None)
        run_valid(tok, org, tok_b, org_b, roles)
        call("DELETE", f"/orgs/{team}", tok=tok)
    fails = [r for r in A.RESULTS if not r["pass"]]
    out = A.OUT / f"prj_{MODE}_{A.RUN}.json"
    out.write_text(json.dumps({"run": A.RUN, "mode": MODE, "results": A.RESULTS, "log": A.LOG}, indent=1))
    print(f"\n{len(A.RESULTS) - len(fails)}/{len(A.RESULTS)} pass, {len(fails)} fail. Log: {out}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
