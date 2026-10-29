#!/usr/bin/env python3
"""Storage end to end: version delete + GC against MinIO, prune dry-run vs apply, guards, roles, cross-tenant.

Object presence is checked through presigned sample URLs captured before the delete (GET -> 200 vs 404 NoSuchKey),
so no MinIO credentials are needed. Trains one small image model (images/clean) for the version_has_models guard.
    python3 tests/acceptance/storage_run.py
"""
import json
import sys
import time

import acc_run as A
import dat_run as D
import e2e_run as E
import val_run as V
from acc_run import call, check, expect


def storage(tok, pid):
    r = call("GET", f"/projects/{pid}/storage", tok=tok)
    assert r.status == 200, r.brief()
    return r.json


def exists(url):
    st, _, body = D.s3("GET", url)
    return st == 200, st, body[:120]


def run(tok, org, tok_b, org_b, viewer_tok, team):
    # ---- delete a version: unique objects go, shared objects stay ----
    pid, did, v1 = V.image_set(tok, org, "images/corrupt", name="STORAGE")
    s1 = {s["sha256"]: s for s in D.samples(tok, v1["id"])}
    rep = V.sealed(tok, v1)
    v2, _ = V.fix(tok, v1["id"], rep, "VAL-1")
    s2 = {s["sha256"]: s for s in D.samples(tok, v2["id"])}
    unique = [s1[h] for h in s1 if h not in s2]
    shared = [s1[h] for h in s1 if h in s2]
    check("STO setup: v1 28 samples, fix v2 24, 4 unique to v1", len(s1) == 28 and len(s2) == 24 and len(unique) == 4, f"{len(s1)} {len(s2)} {len(unique)}")
    before = storage(tok, pid)
    print("INFO storage shape:", json.dumps(before)[:600])
    A.RESULTS.append({"id": "STO record storage before", "pass": True, "note": True, "why": json.dumps(before)[:1500]})
    row1 = next((x for x in before["versions"] if x.get("id") == v1["id"] or x.get("version_id") == v1["id"]), {})
    reclaim = row1.get("reclaimable_bytes") or row1.get("unique_bytes")
    want_reclaim = sum(s["bytes"] or 0 for s in unique)
    check("STO v1 reclaimable_bytes == bytes of its 4 unique objects", reclaim == want_reclaim, f"reported {reclaim}, unique objects {want_reclaim}, row={json.dumps(row1)[:300]}")
    check("STO before delete: all v1 objects present in MinIO", all(exists(s["url"])[0] for s in s1.values()), "")
    expect("STO guard: DELETE current version -> 409 version_in_use", call("DELETE", f"/dataset-versions/{v2['id']}", tok=tok), 409, "version_in_use")
    d = call("DELETE", f"/dataset-versions/{v1['id']}", tok=tok)
    check("STO DELETE old version -> 204", d.status == 204, f"{d.status} {d.raw[:200]}", d)
    expect("STO deleted version -> 404", call("GET", f"/dataset-versions/{v1['id']}", tok=tok), 404)
    end = time.time() + 60
    while time.time() < end and any(exists(s["url"])[0] for s in unique):
        time.sleep(1)
    gone = [exists(s["url"]) for s in unique]
    check("STO GC removed the 4 unique objects from MinIO (404 NoSuchKey)", all(not g[0] and g[1] == 404 for g in gone), f"{[(g[1], g[2][:40]) for g in gone]}")
    kept = [exists(s["url"])[0] for s in shared]
    check("STO shared objects (24, still used by v2) remain in MinIO", all(kept), f"{kept.count(False)} shared objects missing")
    after = storage(tok, pid)
    tb, ta = before["totals"].get("total_bytes"), after["totals"].get("total_bytes")
    db, da = before["totals"].get("data_bytes"), after["totals"].get("data_bytes")
    check("STO data_bytes went down by exactly reclaimable_bytes", db - da == want_reclaim, f"data before {db} after {da} diff {db - da} want {want_reclaim}")
    check("STO total_bytes went down by reclaimable + the deleted version's manifest", tb - ta >= want_reclaim and after["totals"]["manifests_bytes"] < before["totals"]["manifests_bytes"], f"total {tb} -> {ta}")
    check("STO samples of v2 still readable after GC", all(exists(s["url"])[0] for s in D.samples(tok, v2["id"])), "")

    # ---- prune: dry-run matches apply ----
    ppid, pdid, _ = D.img_project(tok, org, "PRUNE")
    files = D.folder_files("images/clean")
    sealed_ids = []
    for chunk in (files[:20], files[20:40], files[40:]):
        _, v, _ = D.upload(tok, ppid, pdid, "folder", chunk)
        V.sealed(tok, v)
        sealed_ids.append(v["id"])
    vs = D.versions(tok, pdid)
    check("STO prune setup: 3 sealed versions", len(vs) == 3 and all(not x.get("is_draft") for x in vs), f"{[(x['version_no'], x.get('is_draft')) for x in vs]}")
    dry = call("POST", f"/datasets/{pdid}/versions:prune", {"keep_latest": 1, "dry_run": True}, tok=tok)
    ids = lambda xs: sorted(x["id"] if isinstance(x, dict) else x for x in xs)
    # keep_latest = N latest versions besides the current one (current is always kept): with 3 sealed, keep_latest 1 keeps v3 (current) + v2.
    check("STO prune dry_run keep_latest 1 -> deletes v1, keeps current v3 + v2", dry.status == 200 and dry.json["dry_run"] is True and ids(dry.json["deleted"]) == sorted(sealed_ids[:1]) and ids(dry.json["kept"]) == sorted(sealed_ids[1:]),
          f"{dry.status} {dry.raw[:300]}", dry)
    check("STO prune dry_run changed nothing", len(D.versions(tok, pdid)) == 3, "")
    app = call("POST", f"/datasets/{pdid}/versions:prune", {"keep_latest": 1, "dry_run": False}, tok=tok)
    check("STO prune apply matches dry run (same deleted set + reclaimable_bytes)", app.status == 200 and ids(app.json["deleted"]) == ids(dry.json["deleted"]) and app.json["reclaimable_bytes"] == dry.json["reclaimable_bytes"],
          f"dry {dry.json.get('deleted')} {dry.json.get('reclaimable_bytes')} / apply {(app.json or {}).get('deleted')} {(app.json or {}).get('reclaimable_bytes')}", app)
    check("STO after prune: v3 + v2 remain, newest samples readable", sorted(x["id"] for x in D.versions(tok, pdid)) == sorted(sealed_ids[1:]) and all(exists(s["url"])[0] for s in D.samples(tok, sealed_ids[2])), "")
    z = call("POST", f"/datasets/{pdid}/versions:prune", {"keep_latest": 0, "dry_run": True}, tok=tok)
    check("STO prune keep_latest 0 -> only the current version kept", z.status == 200 and ids(z.json["kept"]) == [sealed_ids[2]], f"{z.status} {z.raw[:200]}", z)

    # ---- version_has_models + model delete ----
    mpid, mdid, _ = D.img_project(tok, org, "STO model")
    _, mv_ds, _ = D.upload(tok, mpid, mdid, "folder", files)
    E.seal_and_validate(tok, mv_ds)
    r = call("POST", f"/projects/{mpid}/training-jobs", {"dataset_version_id": mv_ds["id"], "budget_mode": "quick"}, tok=tok)
    job = E.wait(f"/training-jobs/{r.json['job']['id']}", tok, lambda j: j.get("status") in ("succeeded", "failed", "cancelled"), E.TRAIN_TIMEOUT, "training")
    mv = job.get("model_version_id")
    check("STO model trained for guard test", job["status"] == "succeeded" and mv, f"{job['status']} {job.get('error_code')}")
    _, newer, _ = D.upload(tok, mpid, mdid, "folder", D.folder_files("images/low")[:4])  # makes the trained version non-current
    expect("STO guard: DELETE version with models -> 409 version_has_models", call("DELETE", f"/dataset-versions/{mv_ds['id']}", tok=tok), 409, "version_has_models")
    # roles + cross-tenant before deleting
    return pid, did, v2, pdid, mpid, mdid, mv_ds, mv, newer


def roles_and_tenancy(tok, tok_b, team_ctx, ctx):
    pid, did, v2, pdid, mpid, mdid, mv_ds, mv, newer = ctx
    k_b = call("POST", f"/orgs/{A.personal_org(tok_b)}/api-keys", {"name": "b", "scopes": ["projects:read", "projects:write", "training"]}, tok=tok_b).json["key"]
    targets = [("DELETE", f"/dataset-versions/{mv_ds['id']}", None), ("DELETE", f"/model-versions/{mv}", None),
               ("POST", f"/datasets/{pdid}/versions:prune", {"keep_latest": 1, "dry_run": True}), ("GET", f"/projects/{pid}/storage", None)]
    for m, path, body in targets:
        expect(f"SEC3 B {m} {path.split('/')[1]}/{path.split('/')[-1][:14]} -> 404", call(m, path, body, tok=tok_b), 404)
        expect(f"SEC3 B-key {m} {path.split('/')[1]}/{path.split('/')[-1][:14]} -> 404", call(m, path, body, key=k_b), 404)
    # viewer in a team org: build a tiny project there
    team, owner_tok, viewer_tok = team_ctx
    tpid, tdid, _ = D.img_project(owner_tok, team, "STO team")
    _, tv1, _ = D.upload(owner_tok, tpid, tdid, "folder", D.folder_files("images/too_few"))
    V.sealed(owner_tok, tv1)
    D.upload(owner_tok, tpid, tdid, "folder", D.folder_files("images/low")[:2])  # new draft -> tv1 non-current
    expect("STO viewer DELETE dataset version -> 403", call("DELETE", f"/dataset-versions/{tv1['id']}", tok=viewer_tok), 403)
    expect("STO viewer prune (even dry_run) -> 403", call("POST", f"/datasets/{tdid}/versions:prune", {"keep_latest": 1, "dry_run": True}, tok=viewer_tok), 403)
    check("STO viewer GET storage -> 200", call("GET", f"/projects/{tpid}/storage", tok=viewer_tok).status == 200, "")
    # finally: delete the model, then the version is deletable
    d = call("DELETE", f"/model-versions/{mv}", tok=tok)
    check("STO DELETE model version -> 204", d.status == 204, f"{d.status} {d.raw[:200]}", d)
    expect("STO deleted model version -> 404", call("GET", f"/model-versions/{mv}", tok=tok), 404)
    d = call("DELETE", f"/dataset-versions/{mv_ds['id']}", tok=tok)
    check("STO version deletable once its model is gone -> 204", d.status == 204, f"{d.status} {d.raw[:200]}", d)


def main():
    if call("GET", "/health").status != 200:
        sys.exit("API not healthy")
    E.PATHS.update(call("GET", "/openapi.json").json["paths"])
    print(f"run {A.RUN} against {A.API}")
    tok, _, _ = A.login(A.user("sa")[0])
    org = A.personal_org(tok)
    tok_b, _, _ = A.login(A.user("sb")[0])
    team = call("POST", "/orgs", {"name": f"Sto Team {A.RUN}"}, tok=tok).json["id"]
    ve, _ = A.user("sv")
    assert call("POST", f"/orgs/{team}/members", {"email": ve, "role": "viewer"}, tok=tok).status == 201
    viewer_tok = A.login(ve)[0]
    try:
        ctx = run(tok, org, tok_b, None, viewer_tok, team)
        roles_and_tenancy(tok, tok_b, (team, tok, viewer_tok), ctx)
    except Exception as e:
        check("storage run crashed", False, f"{type(e).__name__}: {str(e)[:500]}")
    fails = [r for r in A.RESULTS if not r["pass"]]
    out = A.OUT / f"storage_{A.RUN}.json"
    out.write_text(json.dumps({"run": A.RUN, "results": A.RESULTS, "log": A.LOG}, indent=1))
    print(f"\n{len(A.RESULTS) - len(fails)}/{len(A.RESULTS)} pass, {len(fails)} fail. Log: {out}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
