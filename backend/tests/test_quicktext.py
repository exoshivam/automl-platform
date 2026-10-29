"""Quick text labels (examples, no training) + ready-made sentiment; train-for-accuracy switches the predictor."""

import time

import pytest

from app.ml import idle, quicktext
from tests.test_auth import problem

API = "/api/v1"
EX = {"refund": ["mujhe refund chahiye", "paise wapas karo", "I want my money back", "refund kab milega",
                 "cancel karke refund do"],
      "delivery late": ["order abhi tak nahi aaya", "delivery late hai", "my parcel is delayed", "kab tak deliver hoga",
                        "courier nahi aaya"]}


def quick(c, u, labels=("refund", "delivery late"), **kw):
    r = c.post(f"{API}/orgs/{u.org_id}/projects", headers=u.h,
               json={"name": "Quick", "template": "text_classification", "mode": "quick", "labels": list(labels), **kw})
    assert r.status_code == 201, r.text
    return r.json()


def add(c, u, p, examples=EX):
    for label, xs in examples.items():
        for t in xs:
            r = c.post(f"{API}/projects/{p['id']}/examples", headers=u.h, json={"text": t, "label": label})
            assert r.status_code == 201, r.text


def pred(c, u, p, text):
    r = c.post(f"{API}/projects/{p['id']}/predict", headers=u.h, json={"text": text})
    assert r.status_code == 200, r.text
    return r.json()


def test_quick_mode_examples_predict_other_threshold(client, alice, bob):
    p = quick(client, alice, ready_made=["sentiment"])
    assert p["status"] == "Ready"  # label names alone already work (zero-shot)
    problem(client.post(f"{API}/projects/{p['id']}/examples", headers=alice.h, json={"text": "x", "label": "nope"}),
            422, "unknown_class")
    add(client, alice, p)
    problem(client.post(f"{API}/projects/{p['id']}/examples", headers=alice.h,
                        json={"text": "delivery late hai", "label": "delivery late"}), 409, "duplicate_example")
    assert len(client.get(f"{API}/projects/{p['id']}/examples?label=refund", headers=alice.h).json()) == 5
    r = pred(client, alice, p, "refund please, paisa wapas chahiye")
    assert r["label"] == "refund" and r["source"] == "examples" and not r["other"] and len(r["top3"]) == 2
    assert pred(client, alice, p, "parcel abhi tak deliver nahi hua")["label"] == "delivery late"
    off = pred(client, alice, p, "aaj mausam bahut accha hai")  # unrelated Hinglish chit-chat
    assert off["label"] is None and off["other"]
    assert r["sentiment"]["label"] in ("positive", "neutral", "negative")
    s = client.patch(f"{API}/projects/{p['id']}/text-settings", headers=alice.h, json={"threshold": 0.95}).json()
    assert s["threshold"] == 0.95 and s["mode"] == "quick" and s["embedding_model"]["licence"] == "Apache-2.0"
    assert pred(client, alice, p, "refund please, paisa wapas chahiye")["other"]  # nothing clears 0.95
    rep = client.get(f"{API}/projects/{p['id']}/text-report", headers=alice.h).json()
    assert rep["ready"] and rep["loo_accuracy"] >= 0.8 and rep["examples"] == 10 and 0.2 <= rep["suggested_threshold"] <= 0.9
    assert {x["label"] for x in rep["per_label"]} == {"refund", "delivery late"}
    eid = client.get(f"{API}/projects/{p['id']}/examples?label=refund", headers=alice.h).json()[0]["id"]
    assert client.delete(f"{API}/projects/{p['id']}/examples/{eid}", headers=alice.h).status_code == 204
    assert len(client.get(f"{API}/projects/{p['id']}/examples?label=refund", headers=alice.h).json()) == 4
    for call in (lambda u: client.post(f"{API}/projects/{p['id']}/predict", headers=u.h, json={"text": "x"}),
                 lambda u: client.get(f"{API}/projects/{p['id']}/examples", headers=u.h),
                 lambda u: client.get(f"{API}/projects/{p['id']}/text-report", headers=u.h)):
        problem(call(bob), 404)


def test_zero_shot_and_sentiment_only(client, alice):
    p = quick(client, alice)
    r = pred(client, alice, p, "I want a refund for this order")  # no examples: label names are the gallery
    assert r["top3"][0]["label"] == "refund"
    rep = client.get(f"{API}/projects/{p['id']}/text-report", headers=alice.h).json()
    assert not rep["ready"] and {w["code"] for x in rep["per_label"] for w in x["warnings"]} == {"zero_shot"}
    s = quick(client, alice, labels=(), ready_made=["sentiment"])
    assert s["status"] == "Ready"
    out = pred(client, alice, s, "यह फिल्म बहुत अच्छी थी")  # Hindi positive
    assert out["source"] == "sentiment_only" and out["label"] is None and out["sentiment"]["label"] == "positive"
    assert pred(client, alice, s, "service bahut kharab thi, never again")["sentiment"]["label"] == "negative"
    problem(client.post(f"{API}/orgs/{alice.org_id}/projects", headers=alice.h,
                        json={"name": "x", "template": "text_classification", "mode": "quick", "labels": []}),
            422, "invalid_spec")  # empty labels only with ready-made sentiment


def test_idle_unload_text_models(client, alice, monkeypatch):
    from app.settings import get_settings

    p = quick(client, alice, ready_made=["sentiment"])
    monkeypatch.setattr(get_settings(), "predict_idle_unload_s", 3600)  # the background reaper must not race the load checks
    quicktext._CACHE.clear()
    idle.sweep(time.monotonic() + 3600)
    assert not quicktext.loaded("embed") and not quicktext.loaded("sentiment")
    first = pred(client, alice, p, "refund chahiye")
    assert first["model_load_ms"] > 0 and quicktext.loaded("embed") and quicktext.loaded("sentiment")
    monkeypatch.setattr(get_settings(), "predict_idle_unload_s", 0.2)
    time.sleep(0.3)
    idle.sweep()
    assert not quicktext.loaded("embed") and not quicktext.loaded("sentiment")


@pytest.mark.parametrize("n", [12])
def test_train_for_accuracy_switches_to_the_model(client, alice, n):
    p = quick(client, alice)
    more = {k: [f"{t} {i}" for i in range(3) for t in v][:n] for k, v in EX.items()}
    add(client, alice, p, more)
    vid = client.get(f"{API}/projects/{p['id']}/examples", headers=alice.h).json()[0]["dataset_version_id"]
    client.post(f"{API}/dataset-versions/{vid}/seal", headers=alice.h)
    rep = client.get(f"{API}/dataset-versions/{vid}/validation", headers=alice.h).json()
    acks = sorted({i["code"] for i in rep["items"] if i["acknowledgeable"]})
    if acks:
        client.post(f"{API}/dataset-versions/{vid}/validation/acknowledge", headers=alice.h, json={"codes": acks})
    r = client.post(f"{API}/projects/{p['id']}/training-jobs", headers=alice.h, json={"dataset_version_id": vid})
    assert r.status_code == 202, r.text
    job = client.get(f"{API}/training-jobs/{r.json()['job']['id']}", headers=alice.h).json()
    assert job["status"] == "succeeded", job
    out = pred(client, alice, p, "mujhe refund chahiye")
    assert out["source"] == "model" and out["label"] in ("refund", "delivery late")
    assert client.get(f"{API}/projects/{p['id']}/text-settings", headers=alice.h).json()["has_trained_model"]


def test_other_decision_regression():
    """Fixed eval (tests/fixtures/quicktext_eval.py): paraphrases must not be swallowed by the background 'Other'
    set (the 'bhai mera paisa kab wapas aayega' regression), and everyday chit-chat must be rejected."""
    from tests.fixtures.quicktext_eval import EXAMPLES, OFF_TOPIC, ON_TOPIC

    on = [quicktext.classify(EXAMPLES, t, quicktext.DEFAULT_THRESHOLD)["label"] == y for t, y in ON_TOPIC]
    off = [quicktext.classify(EXAMPLES, t, quicktext.DEFAULT_THRESHOLD)["other"] for t in OFF_TOPIC]
    assert quicktext.classify(EXAMPLES, "bhai mera paisa kab wapas aayega", 0.45)["label"] == "refund"
    assert sum(on) / len(on) >= 0.85 and sum(off) / len(off) >= 0.85, (sum(on), sum(off))
