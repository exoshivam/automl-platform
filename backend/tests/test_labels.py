"""Group 6: labels:batch (draft + fork of sealed), label sets merge/rename, export round trip, box suggestions."""

import io
import json
import zipfile

from tests.test_auth import problem
from tests.test_uploads import COCO, _det_images, _import, folder_files, project, upload, version

API = "/api/v1"


def samples(c, u, vid, **q):
    qs = "&".join(f"{k}={v}" for k, v in {"limit": 200, **q}.items())
    return c.get(f"{API}/dataset-versions/{vid}/samples?{qs}", headers=u.h).json()["items"]


def test_classification_labels_on_draft_then_sealed_fork(client, alice):
    p = project(client, alice, labels=("cat", "dog"))
    _, r = upload(client, alice, p, folder_files(5), label_from_folder=False)  # unlabeled
    v1 = r.json()["dataset_version_id"]
    items = samples(client, alice, v1)
    assert all(not s["labels"] for s in items) and len(samples(client, alice, v1, unlabeled="true")) == 10
    body = {"upserts": [{"sample_id": s["id"], "labels": [{"class_name": "cat"}]} for s in items[:6]]}
    out = client.post(f"{API}/dataset-versions/{v1}/labels:batch", headers=alice.h, json=body).json()
    assert out == {"updated": 6, "dataset_version_id": v1, "forked": False, "sample_id_map": {}}
    labeled = [s for s in samples(client, alice, v1) if s["labels"]]
    assert len(labeled) == 6 and all(lb["source"] == "user" for s in labeled for lb in s["labels"])
    # seal v1, then label again: goes to a NEW version, v1 unchanged (DAT-5b)
    client.post(f"{API}/dataset-versions/{v1}/seal", headers=alice.h)
    before = version(client, alice, v1)
    out = client.post(f"{API}/dataset-versions/{v1}/labels:batch", headers=alice.h,
                      json={"upserts": [{"sample_id": items[9]["id"], "labels": [{"class_name": "dog"}]}]}).json()
    assert out["forked"] and out["dataset_version_id"] != v1 and items[9]["id"] in out["sample_id_map"]
    after = version(client, alice, v1)
    assert after["content_hash"] == before["content_hash"] and after["label_stats"] == before["label_stats"]
    v2 = version(client, alice, out["dataset_version_id"])
    assert v2["parent_version_id"] == v1 and v2["origin"] == "labels" and v2["label_stats"]["dog"]["images"] == 1
    assert client.get(f"{API}/projects/{p['id']}", headers=alice.h).json()["current_version_id"] == v2["id"]


def test_labels_batch_rejects_foreign_and_cross_version_samples(client, alice, bob):
    pa = project(client, alice)
    _, ra = upload(client, alice, pa, folder_files(1))
    pb = project(client, bob)
    _, rb = upload(client, bob, pb, folder_files(1))
    va, vb = ra.json()["dataset_version_id"], rb.json()["dataset_version_id"]
    a_sample = samples(client, alice, va)[0]["id"]
    b_sample = samples(client, bob, vb)[0]["id"]
    # B's sample inside B's own version request is fine; A's sample inside B's version is rejected (S3-6)
    problem(client.post(f"{API}/dataset-versions/{vb}/labels:batch", headers=bob.h,
                        json={"upserts": [{"sample_id": a_sample, "labels": [{"class_name": "x"}]}]}), 404)
    problem(client.post(f"{API}/dataset-versions/{va}/labels:batch", headers=bob.h,
                        json={"upserts": [{"sample_id": b_sample, "labels": [{"class_name": "x"}]}]}), 404)
    # same org, sample from another version -> rejected too
    _, ra2 = upload(client, alice, project(client, alice), folder_files(1))
    other = samples(client, alice, ra2.json()["dataset_version_id"])[0]["id"]
    problem(client.post(f"{API}/dataset-versions/{va}/labels:batch", headers=alice.h,
                        json={"upserts": [{"sample_id": other, "labels": [{"class_name": "x"}]}]}), 404)
    problem(client.post(f"{API}/dataset-versions/{va}/labels:batch", headers=alice.h,
                        json={"upserts": [{"sample_id": a_sample, "labels": [{"class_name": "  "}]}]}), 422)


def test_merge_and_rename_classes(client, alice):
    p = project(client, alice, labels=("cat", "dog", "kitten"))
    files = [(f"{c}/{i}.png", f) for c in ("cat", "kitten") for i, (_, f) in enumerate(folder_files(2)[:2])]
    upload(client, alice, p, files)
    ls_id = client.get(f"{API}/projects/{p['id']}", headers=alice.h).json()["label_set_id"]
    out = client.post(f"{API}/label-sets/{ls_id}/merge", headers=alice.h, json={"from": ["kitten"], "into": "cat"}).json()
    assert out["label_set"]["classes"] == ["cat", "dog"] and out["label_set"]["version"] == 2 and out["relabeled"] == 2
    v = version(client, alice, out["dataset_version_id"])
    assert v["label_stats"] == {"cat": {"images": 4}}
    spec = client.get(f"{API}/projects/{p['id']}", headers=alice.h).json()["current_spec"]
    assert spec["labels"] == ["cat", "dog"]
    out = client.post(f"{API}/label-sets/{out['label_set']['id']}/rename", headers=alice.h,
                      json={"from": "dog", "to": "puppy"}).json()
    assert out["label_set"]["classes"] == ["cat", "puppy"] and out["label_set"]["version"] == 3
    problem(client.post(f"{API}/label-sets/{ls_id}/rename", headers=alice.h, json={"from": "nope", "to": "x"}), 409,
            "stale_label_set")
    problem(client.post(f"{API}/label-sets/{out['label_set']['id']}/rename", headers=alice.h,
                        json={"from": "nope", "to": "x"}), 422, "unknown_class")


def test_detection_boxes_no_objects_export_roundtrip_and_suggestions(client, alice):
    p = project(client, alice, template="object_detection", labels=("helmet", "vest"))
    r = _import(client, alice, p, "coco.json", json.dumps(COCO).encode(), "coco", extra_files=_det_images())
    vid = r.json()["dataset_version_id"]
    items = samples(client, alice, vid)
    s0 = items[0]
    box = {"x": 1.5, "y": 2.0, "w": 10.0, "h": 12.0}
    client.post(f"{API}/dataset-versions/{vid}/labels:batch", headers=alice.h, json={"upserts": [
        {"sample_id": s0["id"], "labels": [{"class_name": "helmet", "bbox": box}]},
        {"sample_id": items[1]["id"], "labels": [], "no_objects": True}]})
    got = {s["id"]: s for s in samples(client, alice, vid)}
    assert [lb["bbox"] for lb in got[s0["id"]]["labels"]] == [box]
    assert got[items[1]["id"]]["no_objects"] is True and not got[items[1]["id"]]["labels"]
    assert items[1]["id"] not in {s["id"] for s in samples(client, alice, vid, unlabeled="true")}

    coco = client.get(f"{API}/dataset-versions/{vid}/labels/export?format=coco", headers=alice.h).json()
    assert len(coco["images"]) == 4 and {c["name"] for c in coco["categories"]} == {"helmet", "vest"}
    exported = {(im["file_name"], tuple(a["bbox"])) for im in coco["images"] for a in coco["annotations"]
                if a["image_id"] == im["id"]}
    p2 = project(client, alice, template="object_detection", labels=("helmet", "vest"))
    imgs = [(f"images/{n}", d) for n, d in ((f.split("/")[-1], d) for f, d in _det_images())]
    coco_named = dict(coco, images=[dict(im, file_name=im["file_name"].split("/")[-1]) for im in coco["images"]])
    r2 = _import(client, alice, p2, "coco.json", json.dumps(coco_named).encode(), "coco", extra_files=imgs)
    re_items = samples(client, alice, r2.json()["dataset_version_id"])
    reimported = {(s["name"], (lb["bbox"]["x"], lb["bbox"]["y"], lb["bbox"]["w"], lb["bbox"]["h"]))
                  for s in re_items for lb in s["labels"]}
    assert reimported == exported  # LBL-2c round trip

    yz = client.get(f"{API}/dataset-versions/{vid}/labels/export?format=yolo", headers=alice.h)
    assert yz.headers["content-type"] == "application/zip"
    names = zipfile.ZipFile(io.BytesIO(yz.content)).namelist()
    assert "classes.txt" in names and len([n for n in names if n.startswith("labels/")]) == 4

    sug = client.post(f"{API}/dataset-versions/{vid}/box-suggestions", headers=alice.h,
                      json={"sample_ids": [s0["id"]]}).json()
    boxes = sug["suggestions"][s0["id"]]
    assert sug["model"] and {b["class_name"] for b in boxes} == {"helmet", "vest"}
    assert all(0 <= b["confidence"] <= 1 for b in boxes)
    before = len(got[s0["id"]]["labels"])
    assert len({s["id"]: s for s in samples(client, alice, vid)}[s0["id"]]["labels"]) == before  # not stored
    accepted = [{"class_name": b["class_name"], "bbox": b["bbox"], "source": "ai_prelabel", "confidence": b["confidence"]}
                for b in boxes]
    client.post(f"{API}/dataset-versions/{vid}/labels:batch", headers=alice.h,
                json={"upserts": [{"sample_id": s0["id"], "labels": accepted}]})
    stored = {s["id"]: s for s in samples(client, alice, vid)}[s0["id"]]["labels"]
    assert {lb["source"] for lb in stored} == {"ai_prelabel"} and all(lb["confidence"] for lb in stored)
    problem(client.post(f"{API}/dataset-versions/{vid}/box-suggestions", headers=alice.h, json={"sample_ids": []}), 422)


def test_new_folder_classes_reported_and_added(client, alice):
    p = project(client, alice, labels=("cat", "dog"))
    files = folder_files(1) + [("bird/0.png", folder_files(1)[0][1])]
    _, r = upload(client, alice, p, files)
    v = version(client, alice, r.json()["dataset_version_id"])
    assert v["unknown_classes"] == ["bird"]
    ls_id = client.get(f"{API}/projects/{p['id']}", headers=alice.h).json()["label_set_id"]
    out = client.post(f"{API}/label-sets/{ls_id}/classes", headers=alice.h, json={"add": ["bird", "CAT"]}).json()
    assert out["label_set"]["classes"] == ["cat", "dog", "bird"] and out["dataset_version_id"] is None
    assert version(client, alice, v["id"])["unknown_classes"] == []
    assert client.get(f"{API}/projects/{p['id']}", headers=alice.h).json()["current_spec"]["labels"] == ["cat", "dog",
                                                                                                        "bird"]


def test_stale_label_set_id_conflicts_and_merge_into_unknown_422(client, alice):
    p = project(client, alice, labels=("cat", "dog"))
    ls0 = client.get(f"{API}/projects/{p['id']}", headers=alice.h).json()["label_set_id"]
    r = client.post(f"{API}/label-sets/{ls0}/rename", headers=alice.h, json={"from": "dog", "to": "puppy"}).json()
    body = problem(client.post(f"{API}/label-sets/{ls0}/classes", headers=alice.h, json={"add": ["bird"]}), 409,
                   "stale_label_set")
    assert body["current_label_set_id"] == r["label_set"]["id"] and body["current_classes"] == ["cat", "puppy"]
    cur = r["label_set"]["id"]
    assert client.post(f"{API}/label-sets/{cur}/classes", headers=alice.h, json={"add": ["bird"]}).json()["label_set"][
        "classes"] == ["cat", "puppy", "bird"]
    latest = client.get(f"{API}/projects/{p['id']}", headers=alice.h).json()["label_set_id"]
    problem(client.post(f"{API}/label-sets/{latest}/merge", headers=alice.h, json={"from": ["cat"], "into": "nope"}),
            422, "unknown_class")
    problem(client.post(f"{API}/label-sets/{latest}/merge", headers=alice.h, json={"from": ["ghost"], "into": "cat"}),
            422, "unknown_class")
