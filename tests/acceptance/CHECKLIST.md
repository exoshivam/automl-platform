# Acceptance checklist: first slice

Covers ACC-1/2/3/4/6, PRJ-1..7, DAT-1..6/8/9, LBL-1/2/4a, VAL-1..14/16, NFR-SEC3 and NFR-SEC6 from [PRD.md](../../PRD.md) v0.2 (§6, §7.4, §11).
**Scope (user decision, 2026-10-22):** the target is a **local end-to-end test** (project → upload/camera → label → validate → train → evaluate → playground → export download). Hosted deploy, billing and Google sign-in are **deferred to Phase 2**: ACC-1e/1f below are marked *deferred*. Functional correctness of the flow comes before exhaustive prod-hardening checks.

**Status:**
- Run: ACC (`acc_run.py` 154/154), PRJ (`prj_run.py` 108/108 + invalid/down modes), DAT/SEC6/SEC3 (`dat_run.py` 96/98: corrupt classification isn't exposed in SampleOut yet).
- Not built yet: LBL, VAL, training.
- End to end: `e2e_run.py`.

## Conventions

```sh
API=http://127.0.0.1:8010/api/v1          # app API (§11); live spec at $API/openapi.json, docs at $API/docs
A=/tmp/acc-a.jar  B=/tmp/acc-b.jar        # cookie jars: user A (org A), user B (org B)
FX=tests/fixtures                         # build first: python3 tests/fixtures/build_fixtures.py && python3 tests/fixtures/check_fixtures.py
```

- Requests are written as `METHOD path {body}` against `$API`. Bearer token = the access token returned by `/auth/login`, sent as `Authorization: Bearer $TOK_A`.
- **Every error** must be RFC 9457 Problem Details: `Content-Type: application/problem+json` with `type`, `title`, `status` (and `detail`). This is checked on every negative case below and not repeated.
- Field names follow `$API/openapi.json` (committed copy: `backend/openapi.json`). Groups 1-3 (auth, orgs, members, api-keys, usage, audit-logs) are pinned. **⚠** marks names still waiting for groups 4-7.
- Problem Details always carry `type, title, status, detail, instance, code`, plus `errors[]` (loc/msg/type) on 422. Expected `code` values are given below where fixed.
- Refresh and logout need the header `X-CSRF: 1`.
- Automated run of ACC + SEC3 org paths: `python3 tests/acceptance/acc_run.py` (stdlib; `AUTOML_API` overrides the base URL).
- "Expected" is what passes. Anything else fails, and the report records the exact request, status and body.
- Expected defect counts come from [`tests/fixtures/EXPECTED.json`](../fixtures/EXPECTED.json) (generated alongside the fixtures).
- UI-only steps (folder drag-and-drop, camera, labeling canvas) need Chrome, and the user hasn't approved Chrome yet. They're marked **[UI]** and are run only after approval. Their API side is still checked.

## 0. Setup (seed data used by later items)

| Step | Request | Expected |
|---|---|---|
| 0.1 | `POST /auth/register {email:"acc-a@test.local", password:"Str0ng!pass-A", name:"Acc A"}`, then the same for `acc-b@test.local` (`.local`/`.test` domains are accepted in dev only) | 201 `{user:{id,email,name,email_verified:false}, verification_required:true}`. **No tokens** |
| 0.2 | `GET /_dev/mail` (newest first `[{to,subject,body,…}]`), take the `Token: <token>` line for each user, then `POST /auth/verify-email {token}`. Dev hooks exist only with `APP_ENV=dev\|test` | 200. Login works only after this |
| 0.3 | `POST /auth/login` for each; keep `TOK_A`, `TOK_B`, and the cookie jars | 200 + access token + refresh cookie |
| 0.4 | `GET /me` (A, B); keep `ORG_A`, `ORG_B` = the personal org ids | Each has exactly 1 membership, role `owner` |
| 0.5 | Register and verify `acc-admin`, `acc-member` and `acc-viewer`, then as A: `POST /orgs/{ORG_A}/members {email, role}` for each (existing verified users only; owner/admin only) | `GET /orgs/{ORG_A}/members` lists 4 members with the right roles. The same call for an unverified or unknown email gives 4xx |

---

## ACC: accounts, workspaces, API keys

| ID | Step / request | Expected |
|---|---|---|
| ACC-1a | `POST /auth/register` with an existing email in another case (`ACC-A@test.local`, `acc-a@TEST.local`) | 409 each (citext unique). Also: register `Mixed@TEST.local` and the stored email has a lowercased domain (`Mixed@test.local`) |
| ACC-1b | `POST /auth/register` with a password under 10 characters / missing email / malformed email / a password over 256 characters | 422 each, with `errors[]` naming the field |
| ACC-1c | `POST /auth/login` before email verification | 403 code `email_not_verified`. No tokens, no cookie |
| ACC-1d | `POST /auth/login` with a wrong password, 6 times for the same email from the same IP | Attempts 1-5: 401 code `invalid_credentials` with an identical body, also identical to an unknown email apart from `instance` (no user enumeration). Attempt 6: 429 code `too_many_attempts` with `Retry-After`. The limit is 5 failures per 15 min per email + IP (`LOGIN_RATE_LIMIT`). A different email from the same IP is not blocked |
| ACC-1e *(deferred: Phase 2)* | `GET /auth/google/start` | 302 to `accounts.google.com/...` with `response_type=code`, `scope` containing `openid email`, plus `state` and `nonce` (or PKCE `code_challenge`). **Don't follow it**: this checks only the redirect parameters |
| ACC-1f *(deferred: Phase 2)* | `GET /auth/google/callback?code=x&state=forged` | 400/401. No session cookie set |
| ACC-2a | Decode `TOK_A` (JWT) | `exp - iat == 900` (15 min). Has `sub` and no password or secret claims |
| ACC-2b | Inspect `Set-Cookie` from login | `refresh_token=…; HttpOnly; SameSite=lax; Path=/api/v1/auth; Max-Age=2592000`. `Secure` is **absent in dev** over http (expected; `COOKIE_SECURE=true` in prod, so re-check this before any prod release) |
| ACC-2c | `POST /auth/refresh` with cookie R1 and `X-CSRF: 1`; then again **without** the header | 200, a new access token, new cookie R2 ≠ R1. Without the header: 403 code `csrf_header_missing` |
| ACC-2d | `POST /auth/refresh` with **R1 again** (reuse) | 401 code `refresh_reuse`. **Then** `POST /auth/refresh` with R2 gets 401 code `session_expired`, because reuse revokes the whole family |
| ACC-2e | `POST /auth/logout` (with `X-CSRF: 1`), then `POST /auth/refresh` with the last cookie | 204, then 401 |
| ACC-2f | Call `GET /me` with an access token that has a tampered signature, then with an expired one (server restarted with `ACCESS_TOKEN_TTL_SECONDS=5`, wait 6 s). Also refresh with `REFRESH_TOKEN_TTL_SECONDS=5` after 6 s | 401 for all three |
| ACC-3a | `GET /me` right after 0.3 | Personal org exists automatically, role `owner` |
| ACC-3b | `POST /orgs {name:"Acc A Team"}` | 201. Creator is `owner`. `slug` is unique: creating the same name again gives a different slug or 409 |
| ACC-3c | `DELETE /orgs/{ORG_A}` (the personal org) as owner | 409 code `personal_workspace` |
| ACC-4 | **Role matrix** on ORG_A . Each call is made as owner, admin, member, viewer, a non-member (B), and API keys:<br>• **All roles** get 200 on every GET/list: `/orgs/{ORG_A}`, projects, datasets, versions, samples, validation, members<br>• **owner/admin/member** get 2xx, **viewer** 403: create/edit project, `PUT .../spec`, uploads, imports, `labels:batch`, seal, validation fixes, box suggestions<br>• **owner/admin** get 2xx, **member/viewer** 403: `PATCH /orgs/{ORG_A}`, `POST/DELETE .../members`, list/create/revoke `api-keys`, `GET .../audit-logs`<br>• **owner only**: `DELETE /orgs/{ORG_A}`, and granting the owner role. Admin trying to grant, change or remove `owner` gets 403<br>• **non-member** on any ORG_A path, or any resource of ORG_A by id: **404**<br>• **API keys** (`Authorization: Bearer ak_…`, `X-API-Key` also accepted): `projects:read` allows only GETs; `projects:write` allows project/dataset/label mutations; no scope allows member, key or org management (403) | Exactly that matrix. Every 403/404 is Problem Details. Members: unknown or unverified email gives 404 `user_not_found`; an existing member gives 409; demoting or removing the last owner gives 409 `last_owner` (`PATCH/DELETE /orgs/{id}/members/{user_id}`) |
| ACC-6a | `POST /orgs/{ORG_A}/api-keys {name:"ci", scopes:["projects:read"], expires_at:"<now+1d>"}` | 201 `ApiKeyCreated`: `key` matches `^ak_[0-9a-f]{8}_.+$` and is shown **once** (keep `KEY_RO`), plus `prefix`, `scopes`, `expires_at` |
| ACC-6b | `GET /orgs/{ORG_A}/api-keys` | Lists name, prefix, scopes, expires_at and last_used_at. **No** full key and no hash in any field |
| ACC-6c | Ask the service runner to run `SELECT key_hash, prefix FROM api_keys WHERE name='ci'` | `key_hash` = `sha256(key)` hex (64 chars) and ≠ the key. The full key appears nowhere in the DB |
| ACC-6d | `GET /orgs/{ORG_A}/projects` with `Authorization: Bearer $KEY_RO`, then again with `X-API-Key: $KEY_RO`, then list the keys again | 200, then `last_used_at` is set to roughly now |
| ACC-6e | `POST /orgs/{ORG_A}/projects` with `KEY_RO` | 403 (scope `projects:write` missing) |
| ACC-6f | Create a key with `expires_at` 3 s out, wait 4 s, use it. Also an unknown key `ak_00000000_nope` | 401 each |
| ACC-6g | `DELETE /orgs/{ORG_A}/api-keys/{id}`, then use the key | 204, then 401 |
| ACC-6h | `POST .../api-keys {scopes:["admin:everything"]}` | 422 (unknown scope) |
| ACC-6i | With a key that has every scope: `POST /orgs/{ORG_A}/members`, `POST /orgs/{ORG_A}/api-keys`, `PATCH /orgs/{ORG_A}`, `GET /me` | 403 each. The same key on another org's path gives 404 |

## PRJ: projects and task specs

| ID | Step / request | Expected |
|---|---|---|
| PRJ-1a | `GET /templates` | Enabled: image classification, object detection, text/sentiment, tabular. Audio and later categories are listed as "coming soon" / disabled. Each has a default spec that validates against Appendix A |
| PRJ-1b | `POST /orgs/{ORG_A}/projects {name:"Plants", template:"image_classification", labels:["healthy","early_blight","late_blight"]}` (⚠ field names) | 201. `current_spec` has `task_type:"image_classification"`, those 3 labels, the default `split` 0.7/0.15/0.15 with seed, `spec_version` 1, status `Draft` or `Data needed` |
| PRJ-1c | Same, with `template:"audio_classification"` | 422 (coming soon) |
| PRJ-1d | `labels` with 1 entry, with 501 entries, with a duplicate entry `["a","a"]`, and with an empty-string entry | 422 each (2-500, Appendix A). A duplicate or empty label is ⚠ assumed rejected |
| PRJ-2a | `POST /orgs/{ORG_A}/projects/from-description {description:"Detect whether workers wear helmets and vests in CCTV photos; must run offline on Android"}` | 200/201 draft spec: `task_type:"object_detection"`, labels like helmet/vest, `constraints.target_platforms` containing `android`. The spec validates against the schema |
| PRJ-2b | Server with `LLM_PROVIDER=stub LLM_STUB_MODE=invalid`; same request as PRJ-2a | `GET /_dev/llm-log` shows exactly **3** calls (1 + 2 retries) for this request. Response: 422 Problem Details with code `llm_invalid_output` and `fallback:"category"`. No project is created |
| PRJ-2c | `LLM_STUB_MODE=down` | 503 Problem Details, code `llm_unavailable`. `GET /templates` and template-based create (PRJ-1b) still work (NFR-R5). No 500 |
| PRJ-3a | `PUT /projects/{id}/spec` with a valid edit (change `max_model_size_mb` to 10) | 200. `spec_version` +1, and a `task_spec_versions` row with `source:"user_edit"`. The old version is unchanged |
| PRJ-3b | `PUT .../spec` with `primary_metric:"mAP"` on image classification / an unknown `task_type` / `split` that sums to 1.1 / a missing `labels` | 422 with a JSON pointer to the bad field. `spec_version` unchanged |
| PRJ-3c | `GET /projects/{id}/spec/versions`, then `GET /projects/{id}/spec/versions/1` | Every version is present with its `source` (template/llm/user_edit). Version 1 still holds the original spec after PRJ-3a |
| PRJ-4a | Run from-description **with** sample rows opted in on `tabular/leaky.csv` (⚠ opt-in field name from openapi), then read the payload at `GET /_dev/llm-log` | Payload has the description + column names + **≤ 5 rows**, and no other rows. 300-row CSV → 5 rows max |
| PRJ-4b | Same, **without** the opt-in | No sample rows in the payload at all, only the description (+ column names ⚠: check the PRD intent with the PRD owner) |
| PRJ-4c | `SELECT created_at, expires_at FROM llm_logs` (via the service runner) | `expires_at = created_at + 30 days`. The vendor call sets no-training/zero-retention flags where the provider supports them (check the real provider's request config; the stub can't show it) |
| PRJ-5 | from-description with `"generate images of cats"`, then `"predict stock prices with certainty"` | Each is flagged unsupported/ambiguous, explains what *is* supported, and creates no project |
| PRJ-6 | Create a spec with `constraints {target_platforms:["android"], max_model_size_mb:1, max_latency_ms:1, budget_mode:"quick"}` | Accepted and stored. Where candidate filtering exists, the response or GET shows "no backbone qualifies" (Appendix A rule). Otherwise mark it **N/A this slice** (model selection is TRN) |
| PRJ-7a | Create 3 projects, archive one: `DELETE /projects/{id}` | The archived project is hidden from the default list and appears with `?archived=true`. The row isn't hard-deleted (`archived_at` set) |
| PRJ-7b | List with `?search=Plan` and `?status=Draft`; paginate with `?limit=1&cursor=` | Filtered correctly. Cursor pagination returns each project exactly once |
| PRJ-7c | Status flow on one project: create, upload, seal with errors, fix, seal clean | `Draft` → `Data needed` → `Validating` → `Ready to train`. A validation Error keeps it out of `Ready to train` |

## DAT: data ingestion

| ID | Step / request | Expected |
|---|---|---|
| DAT-1a | Upload `zips/valid_folders.zip` to an image classification project: `POST /projects/{p}/datasets/{d}/uploads {files:[{name,size,sha256}]}`, PUT the parts, then `POST /uploads/{u}/complete` | A new dataset version with labels `cat` 12 and `dog` 12 taken from subfolder names |
| DAT-1b **[UI]** | Drag the folder `images/clean/` (webkitdirectory) | Same result: cat 32 and dog 32 |
| DAT-2a | Import `tabular/clean.csv`: `POST /projects/{p}/datasets/{d}/imports {format:"csv", mapping:{target:"churn", ignore:[]}}` | 300 rows. Inferred types: tenure int, monthly_charges float, contract and region categorical, churn binary target |
| DAT-2b | Override `tenure` to categorical in the mapping | The override is kept on the version and shown in the version details |
| DAT-2c | Import `tabular/clean.xlsx` | 60 rows, same columns and types as the first 60 rows of clean.csv |
| DAT-2d | Parquet import | **Fixture missing**: no pyarrow here. a backend developer provides one (⚠) |
| DAT-2e | Mapping names a column that doesn't exist, or 2 targets | 422 |
| DAT-3a | Start a multipart upload of a 64 MB random file (`head -c 67108864 /dev/urandom`) | The response has `upload_id` + presigned part URLs, with part size ≥ 5 MB except the last. URLs expire in ≤ 15 min (⚠ check `X-Amz-Expires`) |
| DAT-3b | PUT parts 1-3, kill the client, then `GET /uploads/{u}`; then `POST /uploads/{u}/parts {part_numbers:[4,…]}` | `GET` lists parts 1-3 in `parts_done` (part_number, etag, size). `POST .../parts` returns fresh URLs for the requested parts. Asking for done part 2 again gives 200 + a URL, and re-PUTting it replaces it (still one part 2 at complete). A part number out of range gives 422. Finish, complete, and the server SHA-256 equals the local SHA-256 |
| DAT-3c | Declare size `20 GB + 1` byte; declare 50 001 files in one batch | 413/422 each, before any URL is issued. Exactly 20 GB and 50 000 files are accepted (declare only, never send) |
| DAT-3d | `POST /uploads/{u}/complete` with a part missing, or with a wrong checksum | 409/422, and no dataset version is created |
| DAT-4 | After DAT-1a, list samples. Then upload the same zip again | `samples.sha256` equals `shasum -a 256` of each file. The re-upload creates a new version, but the objects are deduplicated (same key or reference count, not stored twice ⚠: confirm the dedup level) |
| DAT-5a | `GET /datasets/{d}/versions` after 2 uploads | v1 and v2, with v2 `parent_version_id` = v1. Each has `content_hash` and `sample_count` |
| DAT-5b | `POST /dataset-versions/{v1}/seal`, then `POST /dataset-versions/{v1}/labels:batch` | The labels go to a **new** version, and v1's manifest and `content_hash` stay unchanged |
| DAT-5c | Seal the same version twice | Idempotent (200) or 409. Not a second sealed copy |
| DAT-6a | Seal `images/clean` (32/32) with the default split | Per class, train/val/test are 70/15/15 with rounding off by at most 1 (22-23 / 4-5 / 4-5). Every class has ≥ the minimum test count |
| DAT-6b | Re-seal the same content with the same `seed` | An identical split assignment |
| DAT-6c | Spec split 0.8/0.1/0.1 | The ratios are respected |
| DAT-6d | Import `tabular/grouped.csv` with `group_by:"patient_id"` | Every `patient_id` has all 5 rows in **one** split: 40 groups → ~28/6/6 |
| DAT-6e | Seal `images/too_few` (5 per class) | The minimum test count can't be met, so the split reports it (together with VAL-2) rather than producing a test split with 0 of a class |
| DAT-8a | Import `detection/coco_clean.json` + `detection/images/det_000..003.png` | 4 images, helmet 12 and vest 6 boxes, matching coordinates |
| DAT-8b | Import the same images with `detection/yolo/` (labels + classes.txt), then with `detection/voc/` | The same counts (helmet 12, vest 6) and the same boxes within ±1 px |
| DAT-8c | COCO whose `category_id` isn't in `categories`, or whose image file is missing | 422 or a VAL entry naming the bad record. Not silently dropped |
| DAT-9 **[UI]** | Mobile + desktop camera capture:<br>(1) rear camera by default on a phone, and the switch works<br>(2) burst count 10, interval 300 ms, gives 10 images in that class<br>(3) a detection project sends captures unlabelled to the box queue<br>(4) the per-class counter matches the server count<br>(5) offline mid-burst, reconnect, and the queue resumes with no duplicates (check sha256) | All 5 hold. Without Chrome approval, only the API side is checked: captured blobs use the same upload API (DAT-3) |

## LBL: labeling

| ID | Step / request | Expected |
|---|---|---|
| LBL-1a | `POST /dataset-versions/{v}/labels:batch {labels:[{sample_id, class_name:"cat"} × 20]}` | 200. 20 labels with `source:"user"` and `labeled_by` = the user. The seal produces a new version |
| LBL-1b | Create a class `kitten` and label 3 samples with it, then `POST /label-sets/{id}/merge {from:["kitten"], into:"cat"}`; also `POST /label-sets/{id}/rename {from:"dog", to:"puppy"}` | Each call creates a new `label_sets` version. All kitten labels become cat, dog becomes puppy, and the old label-set version is unchanged. Merging into a missing class gives 422 |
| LBL-1c **[UI]** | Keyboard shortcuts 1..9 assign classes; shift-click multi-select, then label all | Works; the server count matches |
| LBL-2a | `labels:batch` with bbox `{x,y,w,h}` in pixels on `det_000.png` | Stored as given, and the round trip through GET matches |
| LBL-2b | `labels:batch` setting sample-level `no_objects:true` on det_018 and det_019 (⚠ exact name from openapi) | Those 2 images are no longer counted in VAL-16 "images without boxes" |
| LBL-2c | `GET /dataset-versions/{v}/labels/export?format=coco`, then `?format=yolo` | Re-importing each export (DAT-8) gives identical boxes. `format=voc` or unknown gives 422 |
| LBL-2d **[UI]** | Draw, resize, move and delete a box; copy it to the next image; zoom and pan; autosave (reload and the boxes are still there); touch on a tablet viewport | All work |
| LBL-4a | `POST /dataset-versions/{v}/box-suggestions {sample_ids:[…], classes:["helmet","vest"]}` | Suggestions have `source:"ai_prelabel"` + `confidence` and are **not** stored as labels until accepted. Accept all for one image, one for another, none for a third, via `labels:batch`, then confirm the stored labels match exactly. In this slice the detector is a stub behind an interface: record that the licence-cleared real model is still to come |

## VAL: data validation

Seal each fixture as its own dataset version, then `GET /dataset-versions/{v}/validation`. Each report item must carry **count, examples, reason and suggested fix** (§6.5). Errors block `POST /projects/{p}/training-jobs` (expect 409/422 there). Warnings need an acknowledgement before training.

| ID | Fixture | Expected |
|---|---|---|
| VAL-1 | `images/corrupt` | **Error**, count **4**: `cat/zero_bytes.png`, `cat/truncated.png`, `dog/garbage.jpg`, `dog/text_named_png.png`. The auto-fix `POST .../validation/fixes {fix:"exclude_corrupt"}` (⚠) creates a new version with 24 samples and no VAL-1 |
| VAL-2a | `images/too_few` (5 + 5) | **Error** for both `cat` and `dog` (< 10 per class) |
| VAL-2b | `tabular/too_few.csv` (40 rows) | **Error** (< 50 rows) |
| VAL-3 | `images/low` (15 + 15) | **Warning** for both classes (< 30), with the estimated accuracy band shown |
| VAL-4 | `images/imbalance` (110 / 10) | **Warning**, ratio **11:1**, largest `big`, smallest `small`. Also VAL-3 for `small` |
| VAL-5a | `images/dupes` | **Warning**: at least **7** exact-duplicate groups (5 `cat/copy_*` + 2 `conflict_*`). The auto-fix dedups, leaving one per group |
| VAL-5b | `tabular/duplicate_rows.csv` | **Warning**: **10** duplicate rows |
| VAL-6 | `images/dupes` | **Warning**: **3** near-duplicate pairs (`dog/near_00k.png` ≈ `dog/img_00k.png`). Their SHA-256 differ, so this is not VAL-5 |
| VAL-7a | `images/split_leak` (15 exact pairs per class) | **Error**: leakage across splits. The auto-fix re-split keeps duplicate groups together, so the new version has VAL-7 = 0 and each pair sits in one split. If the platform groups dupes *before* splitting and VAL-7 never fires, record that and check that no pair straddles splits |
| VAL-7b | `images/split_leak_presplit` (only if user-assigned splits are supported) | **Error**: **10** train/test shared files |
| VAL-8 | `images/dupes` | **Warning**: **2** conflicts (`conflict_0.png`, `conflict_1.png` labelled both cat and dog) |
| VAL-9 | `tabular/leaky.csv` | **Warning**: `mostly_missing` 58.3% missing (> 50%), with an impute/drop choice. Other columns report their missing counts too |
| VAL-10 | `tabular/leaky.csv` | **Warning**: `customer_id` (UUID) and `row_no` (sequential). Excluded from features by default (auto-fix on) |
| VAL-11 | `tabular/leaky.csv` | **Error until acknowledged**: `churn_outcome` (AUC 1.0 and the `*_outcome` name) and `cancelled_date` (name, plus non-missing exactly when churn=1). Training is blocked until each is acknowledged or excluded |
| VAL-12 | `tabular/leaky.csv` | `country` constant: **Info/Warning**. `session_hash` high-cardinality: **Warning** (it may also be reported under VAL-10; either is acceptable) |
| VAL-13 | `images/quality` | Info/Warning: **3** tiny (16 px), **2** extreme aspect (400×20), **4** grayscale in an RGB set |
| VAL-14 | `text/mixed.csv` | **Info**: 2 empty/whitespace and 3 very short texts. The script mix lists Latin, Devanagari and Tamil, plus Hinglish. Non-English or code-mixed is 25.6% (> 10%), so the report suggests an Indic/multilingual backbone |
| VAL-16 | `detection/coco_defects.json` + `detection/images` | 1 box out of bounds, 1 zero-area, 1 tiny (< 8 px), 2 images without boxes and not flagged. `vest` 20 boxes gives a **Warning** (< 50); `glove` 10 boxes gives an **Error** (< 15). After LBL-2b flags det_018 and det_019, the unflagged-image count drops to 0 |
| VAL-ctl | `images/clean`, `tabular/clean.csv` | No Errors. A clean report proves the checks don't fire on good data |
| VAL-run | Every seal | Validation runs automatically (no extra call). The project status is `Validating` during the run, and the report `GET` works after it |

## NFR-SEC3: tenant isolation

User B (ORG_B) attacks ORG_A's resources, using ids collected while setting up as A.

| # | Request as B (`TOK_B`, or B's API key) | Expected |
|---|---|---|
| S3-1 | `GET /orgs/{ORG_A}`, `/orgs/{ORG_A}/projects`, `/orgs/{ORG_A}/members`, `/orgs/{ORG_A}/api-keys` | 404 (preferred: no existence oracle) or 403. **Never 200**, and no ORG_A data in the body |
| S3-2 | `GET`/`PATCH`/`DELETE /projects/{A_project}`; `PUT /projects/{A_project}/spec` | 404/403. A's project is unchanged afterwards (A re-GETs it) |
| S3-3 | `GET /datasets/{A_ds}/versions`, `/dataset-versions/{A_v}`, `.../samples`, `.../validation` | 404/403 |
| S3-4 | `POST /projects/{A_project}/datasets/{A_ds}/uploads`; `POST /uploads/{A_upload}/complete` | 404/403. No object is written under A's prefix (MinIO listing via the service runner) |
| S3-5 | `POST /dataset-versions/{A_v}/labels:batch`, `.../seal`, `.../validation/fixes` | 404/403. A's versions are unchanged |
| S3-6 | B creates a project in ORG_B, then `POST /dataset-versions/{A_v}/labels:batch` using B's own `sample_id`s, and the reverse (A's `sample_id` inside B's version) | Rejected: cross-tenant ids inside bodies are checked too, not just ids in paths |
| S3-7 | B's API key with all scopes on ORG_A paths | 404/403 |
| S3-8 | Take a presigned thumbnail URL from A's sample list. (a) B fetches it as-is: works only until expiry (expiry ≤ 15 min ⚠). (b) Edit the object key to another org's prefix and keep the signature | (a) Short-lived. (b) 403 `SignatureDoesNotMatch` |
| S3-9 | MinIO listing | Every object key starts with the org id or org prefix (§7.4 SEC3). Nothing sits outside an org prefix |
| S3-10 | `GET /me` as B, and every list endpoint as B | Zero ORG_A rows anywhere |
| S3-11 | Spot-check the SQL query log for one request per resource type | Every tenant-table query has an `organization_id =` predicate |

## NFR-SEC6: hostile files

| # | Fixture / step | Expected |
|---|---|---|
| S6-1 | Upload `zips/traversal.zip` | The archive is rejected (422), **or** only `cat/ok.png` is ingested and the 4 hostile entries are rejected and listed. Nothing is written outside the upload's own prefix in MinIO, and nothing on the worker filesystem outside its temp dir: `find / -name escape.txt -newer <marker>` inside the worker container finds nothing, and the marker strings `AUTOML-TRAVERSAL-1/2` appear nowhere |
| S6-2 | Upload `zips/symlink.zip` | The symlink entry isn't followed or materialised. `/etc/passwd` content is never ingested as a sample |
| S6-3 | Upload `zips/bomb_1gib.zip` (1.0 MB compressed, 1 GiB declared) | Rejected by a ratio or uncompressed-size limit **before** full extraction, with a clear 413/422 or ingestion error. The worker's memory peak stays under a few hundred MB and its disk isn't filled (`docker stats` + `df` via the service runner); the worker stays healthy and the next upload works |
| S6-4 | Upload `images/decompression_bomb/cat/huge.png` (20000×20000, 389 KB on disk) | Rejected or VAL-1 "too many pixels" **without** a full decode. The worker isn't OOM-killed and the thumbnail job doesn't hang |
| S6-5 | A single file over the size limit (declared) | 413 before any presigned URL is issued |
| S6-6 | A malware scan with the EICAR test string, **generated at test time** (`printf 'X5O!P%%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*' > /tmp/eicar.txt`) inside a zip | Flagged/quarantined by ClamAV and not ingested. If scanning isn't in this slice, record **N/A this slice**. It isn't stored as a fixture so local antivirus doesn't quarantine the repo |
| S6-7 | A file whose extension doesn't match its content (`corrupt/dog/text_named_png.png`) | Classified by content, not by name (VAL-1 here) |

---

## Fixtures

Build with `python3 tests/fixtures/build_fixtures.py`, then run `python3 tests/fixtures/check_fixtures.py`, which must print `fixture self-check OK`. It's standard library only, deterministic (seed 42), and takes about 3 s.

| Path | Seeded for |
|---|---|
| `images/clean/` | control (32 + 32) |
| `images/corrupt/` | VAL-1 (4 bad files) |
| `images/too_few/`, `images/low/`, `images/imbalance/` | VAL-2 / VAL-3 / VAL-4 |
| `images/dupes/` | VAL-5, VAL-6, VAL-8 |
| `images/split_leak/`, `images/split_leak_presplit/` | VAL-7 |
| `images/quality/` | VAL-13 |
| `images/decompression_bomb/` | SEC6 (git-ignored, regenerated) |
| `tabular/clean.csv`, `clean.xlsx`, `grouped.csv` | DAT-2, DAT-6 grouped split, control |
| `tabular/leaky.csv` | VAL-9, VAL-10, VAL-11, VAL-12, PRJ-4 |
| `tabular/too_few.csv`, `duplicate_rows.csv` | VAL-2, VAL-5 |
| `text/mixed.csv` | VAL-14 |
| `detection/` (images, COCO defects + clean, YOLO, VOC) | DAT-8, LBL-2, VAL-16 |
| `zips/valid_folders.zip`, `traversal.zip`, `symlink.zip`, `bomb_1gib.zip` | DAT-1, SEC6 (the bomb is git-ignored) |
| `EXPECTED.json` | Expected counts per fixture, used as the source for the numbers above |

## Open points

Resolved on 2026-10-22: the role matrix, test hooks, extra endpoints, list params and member add are folded in above.

1. Field names still marked ⚠ get pinned from `GET http://localhost:8010/openapi.json` once the API is up: the project-create body, import mapping, auto-fix body, sample-row opt-in, `no_objects`.
2. Parquet fixture (DAT-2d): pyarrow is being added to the backend deps. I'll generate `tabular/clean.parquet` with the backend venv once uv/Python 3.12 is installed.
3. LBL-4a uses a stub detector this slice. The real licence-cleared model needs its own check later.
