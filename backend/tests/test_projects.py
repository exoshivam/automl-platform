from sqlalchemy import select

from app import models as m
from app.db import session_factory
from app.settings import get_settings
from tests.conftest import make_user
from tests.test_auth import problem

API = "/api/v1"
LABELS = ["healthy", "early_blight", "late_blight"]


def create(c, u, org_id=None, **body):
    body = {"name": "Plants", "template": "image_classification", "labels": LABELS, **body}
    return c.post(f"{API}/orgs/{org_id or u.org_id}/projects", json=body, headers=u.h)


def test_templates(client):
    t = {x["id"]: x for x in client.get(f"{API}/templates").json()}
    assert {k for k, v in t.items() if v["enabled"]} == {"image_classification", "object_detection",
                                                         "text_classification", "tabular", "face_recognition",
                                                         "speaker_recognition"}
    assert t["audio_classification"]["enabled"] is False and t["audio_classification"]["default_spec"] is None


def test_create_project_from_template(client, alice):
    r = create(client, alice)
    assert r.status_code == 201, r.text
    p = r.json()
    spec = p["current_spec"]
    assert spec["task_type"] == "image_classification" and spec["labels"] == LABELS
    assert spec["split"] == {"train": 0.7, "val": 0.15, "test": 0.15, "stratify": True, "group_by": None, "seed": 42}
    assert p["spec_version"] == 1 and p["status"] == "Draft" and p["dataset_id"] and p["label_set_id"]
    tab = create(client, alice, template="tabular", task_type="tabular_regression", labels=None).json()
    assert tab["current_spec"]["primary_metric"] == "rmse" and tab["category"] == "tabular"


def test_create_project_rejections(client, alice):
    problem(create(client, alice, template="audio_classification"), 422, "template_coming_soon")
    problem(create(client, alice, template="nope"), 422, "unknown_template")
    for labels in (["only"], [f"c{i}" for i in range(501)], ["a", "a"], ["a", "A"], ["a", ""]):
        problem(create(client, alice, labels=labels), 422, "invalid_spec")


def test_spec_versions_and_validation(client, alice):
    p = create(client, alice).json()
    spec = p["current_spec"]
    spec["constraints"]["max_model_size_mb"] = 10
    r = client.put(f"{API}/projects/{p['id']}/spec", json={"spec": spec}, headers=alice.h)
    assert r.status_code == 200 and r.json()["spec_version"] == 2
    bad_cases = [
        ({**spec, "primary_metric": "map_50"}, "/primary_metric"),
        ({**spec, "task_type": "dance_recognition"}, "/task_type"),
        ({**spec, "split": {**spec["split"], "train": 0.8}}, "/split"),
        ({k: v for k, v in spec.items() if k != "labels"}, "/labels"),
        ({**spec, "unknown_field": 1}, "/"),
    ]
    for bad, loc in bad_cases:
        body = problem(client.put(f"{API}/projects/{p['id']}/spec", json={"spec": bad}, headers=alice.h), 422,
                       "invalid_spec")
        assert any(e["loc"] == loc for e in body["errors"]), body["errors"]
    versions = client.get(f"{API}/projects/{p['id']}/spec/versions", headers=alice.h).json()
    assert [v["version"] for v in versions] == [1, 2] and [v["source"] for v in versions] == ["template", "user_edit"]
    assert versions[0]["spec"]["constraints"]["max_model_size_mb"] is None  # v1 unchanged
    assert client.get(f"{API}/projects/{p['id']}", headers=alice.h).json()["spec_version"] == 2
    # labels change -> new label set version
    spec["labels"] = LABELS + ["mosaic"]
    client.put(f"{API}/projects/{p['id']}/spec", json={"spec": spec}, headers=alice.h)
    with session_factory()() as db:
        assert db.scalar(select(m.LabelSet.version).where(m.LabelSet.project_id == p["id"])
                         .order_by(m.LabelSet.version.desc())) == 2


def test_list_search_status_archive_pagination(client, alice):
    ids = [create(client, alice, name=n).json()["id"] for n in ("Plants", "Plan B", "Cats")]
    assert client.delete(f"{API}/projects/{ids[2]}", headers=alice.h).status_code == 204
    live = client.get(f"{API}/orgs/{alice.org_id}/projects", headers=alice.h).json()["items"]
    assert {p["id"] for p in live} == set(ids[:2])
    arch = client.get(f"{API}/orgs/{alice.org_id}/projects?archived=true", headers=alice.h).json()["items"]
    assert [p["id"] for p in arch] == [ids[2]]
    found = client.get(f"{API}/orgs/{alice.org_id}/projects?search=plan", headers=alice.h).json()["items"]
    assert {p["name"] for p in found} == {"Plants", "Plan B"}
    assert client.get(f"{API}/orgs/{alice.org_id}/projects?status=Draft", headers=alice.h).json()["items"]
    seen, cursor = [], None
    while True:
        page = client.get(f"{API}/orgs/{alice.org_id}/projects?limit=1" + (f"&cursor={cursor}" if cursor else ""),
                          headers=alice.h).json()
        seen += [p["id"] for p in page["items"]]
        if not (cursor := page["next_cursor"]):
            break
    assert sorted(seen) == sorted(ids[:2])
    restored = client.patch(f"{API}/projects/{ids[2]}", json={"archived": False}, headers=alice.h).json()
    assert restored["archived_at"] is None


def test_role_matrix_projects(client, alice):
    roles = {r: make_user(client, f"{r}@test.local") for r in ("admin", "member", "viewer")}
    for r, u in roles.items():
        client.post(f"{API}/orgs/{alice.org_id}/members", json={"email": u.email, "role": r}, headers=alice.h)
    p = create(client, alice).json()
    for r, u in roles.items():
        assert client.get(f"{API}/orgs/{alice.org_id}/projects", headers=u.h).status_code == 200
        want = 403 if r == "viewer" else 201
        assert create(client, u, org_id=alice.org_id).status_code == want, r
        assert client.patch(f"{API}/projects/{p['id']}", json={"name": "x"}, headers=u.h).status_code == \
            (403 if r == "viewer" else 200)
    ro = client.post(f"{API}/orgs/{alice.org_id}/api-keys", json={"name": "ro", "scopes": ["projects:read"]},
                     headers=alice.h).json()["key"]
    kh = {"Authorization": f"Bearer {ro}"}
    assert client.get(f"{API}/orgs/{alice.org_id}/projects", headers=kh).status_code == 200
    problem(create(client, type("K", (), {"org_id": alice.org_id, "h": kh})()), 403)


def test_cross_tenant_projects(client, alice, bob):
    p = create(client, alice).json()
    for method, path, body in (("get", f"/projects/{p['id']}", None), ("patch", f"/projects/{p['id']}", {"name": "x"}),
                               ("delete", f"/projects/{p['id']}", None),
                               ("put", f"/projects/{p['id']}/spec", {"spec": p["current_spec"]}),
                               ("get", f"/projects/{p['id']}/spec/versions", None),
                               ("get", f"/orgs/{alice.org_id}/projects", None),
                               ("post", f"/orgs/{alice.org_id}/projects", {"name": "x", "template": "tabular"})):
        r = client.request(method.upper(), f"{API}{path}", json=body, headers=bob.h)
        problem(r, 404)
    after = client.get(f"{API}/projects/{p['id']}", headers=alice.h).json()
    assert after["name"] == "Plants" and after["spec_version"] == 1 and after["archived_at"] is None


def test_from_description_stub(client, alice, monkeypatch):
    s = get_settings()
    url = f"{API}/orgs/{alice.org_id}/projects/from-description"
    r = client.post(url, json={"description": "Detect whether workers wear helmets and vests in CCTV photos; must "
                                              "run offline on Android"}, headers=alice.h)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["status"] == "ok" and d["category"] == "object_detection"
    assert d["draft_spec"]["task_type"] == "object_detection" and {"helmet", "vest"} <= set(d["draft_spec"]["labels"])
    assert "android" in d["draft_spec"]["constraints"]["target_platforms"]
    # creating from the reviewed draft records source=llm
    p = client.post(f"{API}/orgs/{alice.org_id}/projects", headers=alice.h,
                    json={"name": "Helmets", "spec": d["draft_spec"], "llm_log_id": d["llm_log_id"]}).json()
    assert client.get(f"{API}/projects/{p['id']}/spec/versions", headers=alice.h).json()[0]["source"] == "llm"

    for desc in ("generate images of cats for my blog", "predict stock prices with certainty for tomorrow"):
        u = client.post(url, json={"description": desc}, headers=alice.h).json()
        assert u["status"] == "unsupported" and u["draft_spec"] is None and u["message"]

    rows = [{"a": i, "b": "x"} for i in range(300)]
    client.post(url, headers=alice.h, json={"description": "predict churn from this csv", "include_sample_rows": True,
                                            "columns": ["a", "b"], "sample_rows": rows})
    client.post(url, headers=alice.h, json={"description": "predict churn from this csv", "columns": ["a", "b"],
                                            "sample_rows": rows})
    logs = client.get(f"{API}/_dev/llm-log", headers=alice.h).json()
    opted, not_opted = logs[1]["request"]["payload"], logs[0]["request"]["payload"]
    assert len(opted["sample_rows"]) == 5 and opted["columns"] == ["a", "b"]
    assert set(not_opted) == {"description"}

    monkeypatch.setattr(s, "llm_stub_mode", "invalid")
    before = len(client.get(f"{API}/_dev/llm-log").json())
    body = problem(client.post(url, json={"description": "classify my photos please"}, headers=alice.h), 422,
                   "llm_invalid_output")
    assert body["fallback"] == "category"
    assert len(client.get(f"{API}/_dev/llm-log").json()) - before == 3  # 1 + 2 retries
    monkeypatch.setattr(s, "llm_stub_mode", "down")
    assert problem(client.post(url, json={"description": "classify my photos please"}, headers=alice.h), 503,
                   "llm_unavailable")["fallback"] == "category"
    monkeypatch.setattr(s, "llm_provider", None)
    problem(client.post(url, json={"description": "classify my photos please"}, headers=alice.h), 503,
            "llm_unavailable")
    with session_factory()() as db:
        assert all(r.expires_at > r.created_at for r in db.scalars(select(m.LlmLog)))
    assert client.get(f"{API}/orgs/{alice.org_id}/projects", headers=alice.h).json()["items"].__len__() == 1


def test_llm_log_id_from_another_org_rejected(client, alice, bob):
    d = client.post(f"{API}/orgs/{alice.org_id}/projects/from-description",
                    json={"description": "classify my plant photos"}, headers=alice.h).json()
    r = client.post(f"{API}/orgs/{bob.org_id}/projects", headers=bob.h,
                    json={"name": "x", "spec": d["draft_spec"], "llm_log_id": d["llm_log_id"]})
    problem(r, 404)
