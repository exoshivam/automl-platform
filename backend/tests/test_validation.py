"""Group 7: validation against the seeded-defect fixtures (tests/fixtures, read-only; EXPECTED.json counts)."""

import json
import pathlib

import pytest

from tests.test_auth import problem
from tests.test_uploads import _import, project, upload, version

API = "/api/v1"
FX = pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures"
EXPECTED = json.loads((FX / "EXPECTED.json").read_text()) if (FX / "EXPECTED.json").exists() else {}
pytestmark = pytest.mark.skipif(not EXPECTED, reason="fixtures not built (python3 tests/fixtures/build_fixtures.py)")


def folder(path: str):
    root = FX / path
    return [(str(p.relative_to(root)), p.read_bytes()) for p in sorted(root.rglob("*")) if p.is_file()]


def sealed_report(c, u, p, files=None, version_id=None):
    if version_id is None:
        _, r = upload(c, u, p, files)
        version_id = r.json()["dataset_version_id"]
    assert c.post(f"{API}/dataset-versions/{version_id}/seal", headers=u.h).status_code == 202
    rep = c.get(f"{API}/dataset-versions/{version_id}/validation", headers=u.h).json()
    return version_id, rep


def items(rep, code):
    return [i for i in rep["items"] if i["code"] == code]


def test_clean_images_no_errors(client, alice):
    _, rep = sealed_report(client, alice, project(client, alice), folder("images/clean"))
    assert rep["summary"]["errors"] == 0
    assert not [i for i in rep["items"] if i["code"] in ("VAL-1", "VAL-2", "VAL-3", "VAL-4", "VAL-5", "VAL-6", "VAL-7",
                                                        "VAL-8")], rep["items"]
    assert rep["status"] in ("passed", "warnings")


def test_val1_corrupt_and_autofix(client, alice):
    p = project(client, alice)
    vid, rep = sealed_report(client, alice, p, folder("images/corrupt"))
    (v1,) = items(rep, "VAL-1")
    assert v1["severity"] == "error" and v1["count"] == EXPECTED["images/corrupt"]["VAL-1"]
    assert set(v1["examples"]) == set(EXPECTED["images/corrupt"]["files"])
    assert rep["status"] == "failed"
    assert client.get(f"{API}/projects/{p['id']}", headers=alice.h).json()["status"] == "Data needed"
    out = client.post(f"{API}/dataset-versions/{vid}/validation/fixes", headers=alice.h,
                      json={"fix_ids": ["exclude_corrupt"]})
    assert out.status_code == 202
    nv = version(client, alice, out.json()["dataset_version_id"])
    assert nv["sample_count"] == EXPECTED["images/corrupt"]["valid_after_autofix"] and nv["parent_version_id"] == vid
    assert nv["sealed_at"] and nv["origin"] == "fix"
    rep2 = client.get(f"{API}/dataset-versions/{nv['id']}/validation", headers=alice.h).json()
    assert not items(rep2, "VAL-1")
    problem(client.post(f"{API}/dataset-versions/{vid}/validation/fixes", headers=alice.h,
                        json={"fix_ids": ["dedupe_rows"]}), 409, "stale_version")  # vid is no longer current
    problem(client.post(f"{API}/dataset-versions/{nv['id']}/validation/fixes", headers=alice.h,
                        json={"fix_ids": ["dedupe_rows"]}), 422, "fix_not_offered")


def test_val2_val3_val4(client, alice):
    _, rep = sealed_report(client, alice, project(client, alice), folder("images/too_few"))
    (v2,) = items(rep, "VAL-2")
    assert v2["severity"] == "error" and set(v2["examples"]) == set(EXPECTED["images/too_few"]["VAL-2"])
    _, rep = sealed_report(client, alice, project(client, alice), folder("images/low"))
    (v3,) = items(rep, "VAL-3")
    assert v3["severity"] == "warning" and set(v3["examples"]) == {"cat", "dog"} and v3["details"]["estimated_accuracy"]
    assert rep["status"] == "warnings"
    _, rep = sealed_report(client, alice, project(client, alice, labels=("big", "small")), folder("images/imbalance"))
    (v4,) = items(rep, "VAL-4")
    exp = EXPECTED["images/imbalance"]["VAL-4"]
    assert v4["details"]["ratio"] == exp["ratio"] and v4["details"]["largest"] == exp["largest"]
    assert v4["details"]["smallest"] == exp["smallest"]
    assert "small" in items(rep, "VAL-3")[0]["examples"]


def test_val5_val6_val8_dupes(client, alice):
    vid, rep = sealed_report(client, alice, project(client, alice), folder("images/dupes"))
    exp = EXPECTED["images/dupes"]
    (v5,) = items(rep, "VAL-5")
    assert v5["count"] >= exp["VAL-5"]["exact_duplicate_groups_min"]
    (v6,) = items(rep, "VAL-6")
    assert v6["count"] == exp["VAL-6"]["near_duplicate_pairs"], v6["examples"]
    (v8,) = items(rep, "VAL-8")
    assert v8["count"] == exp["VAL-8"]["conflicts"]
    out = client.post(f"{API}/dataset-versions/{vid}/validation/fixes", headers=alice.h,
                      json={"fix_ids": ["dedupe_exact"]}).json()
    assert not items(client.get(f"{API}/dataset-versions/{out['dataset_version_id']}/validation",
                                headers=alice.h).json(), "VAL-5")
    flagged = client.get(f"{API}/dataset-versions/{vid}/samples?flag=VAL-6&limit=200", headers=alice.h).json()["items"]
    assert len(flagged) == 2 * exp["VAL-6"]["near_duplicate_pairs"]


def test_val7_split_leak_never_straddles(client, alice):
    vid, rep = sealed_report(client, alice, project(client, alice), folder("images/split_leak"))
    assert not items(rep, "VAL-7")  # exact dupes are grouped before splitting (recorded, see PRD VAL-7 note)
    by_sha: dict = {}
    for s in client.get(f"{API}/dataset-versions/{vid}/samples?limit=200", headers=alice.h).json()["items"]:
        by_sha.setdefault(s["sha256"], set()).add(s["split"])
    assert all(len(sp) == 1 for sp in by_sha.values())


def test_val13_image_quality(client, alice):
    _, rep = sealed_report(client, alice, project(client, alice), folder("images/quality"))
    kinds = {i["details"]["kind"]: i["count"] for i in items(rep, "VAL-13")}
    exp = EXPECTED["images/quality"]["VAL-13"]
    assert kinds["tiny"] == exp["tiny_lt_32px"] and kinds["extreme_aspect"] == exp["extreme_aspect"]
    assert kinds["grayscale_in_rgb"] == exp["grayscale_in_rgb_set"]


def _tabular(client, alice, name, mapping_cols):
    p = project(client, alice, template="tabular", labels=None)
    r = _import(client, alice, p, name, (FX / "tabular" / name).read_bytes(), "csv", {"columns": mapping_cols})
    assert r.status_code == 202, r.text
    return sealed_report(client, alice, p, version_id=r.json()["dataset_version_id"])


def test_tabular_leaky(client, alice):
    vid, rep = _tabular(client, alice, "leaky.csv", [{"name": "churn", "role": "target"}])
    exp = EXPECTED["tabular/leaky.csv"]
    (v9,) = items(rep, "VAL-9")
    assert abs(v9["details"]["missing_fraction"][exp["VAL-9"]["column"]] - exp["VAL-9"]["missing_fraction"]) < 0.01
    (v10,) = items(rep, "VAL-10")
    assert set(exp["VAL-10"]) <= set(v10["details"]["columns"])
    (v11,) = items(rep, "VAL-11")
    assert set(v11["details"]["columns"]) == set(exp["VAL-11"]["columns"]) and v11["acknowledgeable"]
    v12 = {i["details"]["kind"]: i["details"]["columns"] for i in items(rep, "VAL-12")}
    assert exp["VAL-12"]["constant"][0] in v12["constant"]
    assert exp["VAL-12"]["high_cardinality"][0] in v12.get("high_cardinality", []) + v10["details"]["columns"]
    assert rep["status"] == "failed"
    rep = client.post(f"{API}/dataset-versions/{vid}/validation/acknowledge", headers=alice.h,
                      json={"codes": ["VAL-11"]}).json()
    assert rep["status"] == "warnings" and items(rep, "VAL-11")[0]["acknowledged"]
    rep = client.post(f"{API}/dataset-versions/{vid}/validation/acknowledge", headers=alice.h,
                      json={"codes": ["VAL-9", "VAL-10", "VAL-12"]}).json()
    assert rep["status"] == "passed"
    problem(client.post(f"{API}/dataset-versions/{vid}/validation/acknowledge", headers=alice.h,
                        json={"codes": ["VAL-1"]}), 422, "not_acknowledgeable")
    out = client.post(f"{API}/dataset-versions/{vid}/validation/fixes", headers=alice.h,
                      json={"fix_ids": ["exclude_id_columns"]}).json()
    nv = version(client, alice, out["dataset_version_id"])
    roles = {c["name"]: c["role"] for c in nv["column_mapping"]["columns"]}
    assert all(roles[c] == "id" for c in exp["VAL-10"])


def test_tabular_too_few_and_duplicates_and_clean(client, alice):
    _, rep = _tabular(client, alice, "too_few.csv", [{"name": "churn", "role": "target"}])
    assert items(rep, "VAL-2")[0]["severity"] == "error"
    _, rep = _tabular(client, alice, "duplicate_rows.csv", [{"name": "churn", "role": "target"}])
    assert items(rep, "VAL-5")[0]["count"] == EXPECTED["tabular/duplicate_rows.csv"]["VAL-5"]["exact_duplicate_rows"]
    _, rep = _tabular(client, alice, "clean.csv", [{"name": "churn", "role": "target"}])
    assert rep["summary"]["errors"] == 0, rep["items"]


def test_text_mixed_val14(client, alice):
    p = project(client, alice, template="text_classification", labels=("a", "b"))
    head = (FX / "text" / "mixed.csv").read_text().splitlines()[0].split(",")
    cols = [{"name": "label", "role": "target"} if "label" in h else {"name": h, "role": "text"} for h in head[:2]]
    r = _import(client, alice, p, "mixed.csv", (FX / "text" / "mixed.csv").read_bytes(), "csv", {"columns": cols})
    assert r.status_code == 202, r.text
    _, rep = sealed_report(client, alice, p, version_id=r.json()["dataset_version_id"])
    exp = EXPECTED["text/mixed.csv"]["VAL-14"]
    v14 = {i["title"]: i for i in items(rep, "VAL-14")}
    short = v14["Empty or very short texts"]["details"]
    assert short["empty_or_whitespace"] == exp["empty_or_whitespace"] and short["very_short"] == exp["very_short"]
    mix = v14["Language / script mix"]["details"]
    assert mix["suggest_multilingual"] and mix["scripts"]["Devanagari"] == exp["scripts"]["Devanagari"]
    assert mix["scripts"]["Tamil"] == exp["scripts"]["Tamil"] and mix["hinglish_romanised"] == exp["hinglish_romanised"]


def test_detection_val16(client, alice):
    p = project(client, alice, template="object_detection", labels=("helmet", "vest", "glove"))
    imgs = [(f"images/{q.name}", q.read_bytes()) for q in sorted((FX / "detection" / "images").iterdir())]
    r = _import(client, alice, p, "coco_defects.json", (FX / "detection" / "coco_defects.json").read_bytes(), "coco",
                extra_files=imgs)
    assert r.status_code == 202, r.text
    _vid, rep = sealed_report(client, alice, p, version_id=r.json()["dataset_version_id"])
    exp = EXPECTED["detection/coco_defects.json"]["VAL-16"]
    kinds = {}
    for i in items(rep, "VAL-16"):
        kinds.setdefault(i["details"]["kind"], []).append(i)
    assert kinds["out_of_bounds"][0]["count"] == exp["out_of_bounds"]
    assert kinds["zero_area"][0]["count"] == exp["zero_area"]
    assert kinds["tiny"][0]["count"] == exp["tiny_lt_8px"]
    assert kinds["unlabeled_images"][0]["count"] == exp["images_without_boxes_unflagged"]
    by_cls = {i["details"]["class_name"]: i["severity"] for i in kinds["class_count"]}
    assert by_cls["vest"] == "warning" and by_cls["glove"] == "error"


def test_ack_persists_even_when_status_unchanged(client, alice):
    """VAL-18 ack was lost: the overall status stayed 'failed' (another error), so no UPDATE was written."""
    from tests.test_uploads import png

    p = project(client, alice)
    files = [(f"{c}/f{i}.png", png((i // 10, 1 if c == "cat" else 2, 3))) for c in ("cat", "dog") for i in range(25)]
    files.append(("cat/broken.png", png()[:60]))  # VAL-1 error keeps the status 'failed'
    vid, rep = sealed_report(client, alice, p, files)
    assert rep["status"] == "failed"
    ackable = sorted({i["code"] for i in rep["items"] if i["acknowledgeable"]})
    assert ackable
    client.post(f"{API}/dataset-versions/{vid}/validation/acknowledge", headers=alice.h, json={"codes": ackable})
    again = client.get(f"{API}/dataset-versions/{vid}/validation", headers=alice.h).json()
    assert all(i["acknowledged"] for i in again["items"] if i["code"] in ackable), again["items"]
    assert again["status"] == "failed"  # VAL-1 still blocks
