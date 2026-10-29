from tests.test_uploads import folder_files, project, upload

API = "/api/v1"


def status(c, u, p):
    return c.get(f"{API}/projects/{p['id']}", headers=u.h).json()


def test_status_flow_and_sample_count(client, alice):
    p = project(client, alice)
    assert status(client, alice, p)["status"] == "Draft" and status(client, alice, p)["sample_count"] == 0
    _, r = upload(client, alice, p, folder_files(5))  # 5 per class -> VAL-2 error
    s = status(client, alice, p)
    assert s["status"] == "Data needed" and s["sample_count"] == 10
    vid = r.json()["dataset_version_id"]
    client.post(f"{API}/dataset-versions/{vid}/seal", headers=alice.h)
    assert status(client, alice, p)["status"] == "Data needed"  # validation failed (too few)
    p2 = project(client, alice)
    _, r = upload(client, alice, p2, folder_files(32))
    client.post(f"{API}/dataset-versions/{r.json()['dataset_version_id']}/seal", headers=alice.h)
    assert status(client, alice, p2)["status"] == "Ready to train"
    listed = client.get(f"{API}/orgs/{alice.org_id}/projects?status=Ready%20to%20train", headers=alice.h).json()
    assert [x["id"] for x in listed["items"]] == [p2["id"]]
    # labelling a sealed version forks a draft -> back to Data needed until sealed again
    sid = client.get(f"{API}/dataset-versions/{r.json()['dataset_version_id']}/samples?limit=1",
                     headers=alice.h).json()["items"][0]["id"]
    client.post(f"{API}/dataset-versions/{r.json()['dataset_version_id']}/labels:batch", headers=alice.h,
                json={"upserts": [{"sample_id": sid, "labels": [{"class_name": "dog"}]}]})
    assert status(client, alice, p2)["status"] == "Data needed"
