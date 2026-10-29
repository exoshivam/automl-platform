"""Speaker recognition (gallery pattern, no training): consent, clip quality, identify/Unknown, delete, tenancy.
Synthetic harmonic 'voices' cover the plumbing; real public-domain speech (tests/fixtures/speakers) when present."""

import io
import pathlib
import time
import uuid

import numpy as np
import pytest
import soundfile as sf
from sqlalchemy import func, select

from app import models as m
from app.db import session_factory
from app.ml import idle, speaker
from app.services import storage
from tests.test_auth import problem
from tests.test_uploads import project

API = "/api/v1"
FIX = pathlib.Path(__file__).parent / "fixtures" / "speakers"


def voice(f0, formants, seed, secs=4.0, sr=16000) -> bytes:
    rng = np.random.default_rng(seed)
    t = np.arange(int(secs * sr)) / sr
    f = f0 * (1 + 0.03 * np.sin(2 * np.pi * 5 * t) + 0.02 * rng.standard_normal())
    x = np.zeros_like(t)
    for h in range(1, 25):
        amp = sum(np.exp(-((h * f0 - fm) / 120) ** 2) for fm in formants) + 0.05
        x += amp / h * np.sin(2 * np.pi * h * np.cumsum(f) / sr)
    x *= (np.sin(2 * np.pi * rng.uniform(2, 4) * t) > -0.3).astype(float)  # syllable-like gating
    b = io.BytesIO()
    sf.write(b, (0.3 * x / np.abs(x).max()).astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return b.getvalue()


A = [voice(110, (700, 1200), s) for s in range(4)]
B = [voice(210, (400, 2300), s + 10) for s in range(4)]
C = voice(300, (300, 3000), 99)  # measured: best cosine to A/B 0.47 (held-out A/B clips score ~0.93)


def sp_project(c, u):
    return project(c, u, template="speaker_recognition", labels=None)


def person(c, u, p, name):
    return c.post(f"{API}/projects/{p['id']}/people", headers=u.h, json={"name": name, "consent": {"attested": True}})


def clips(c, u, pid, datas):
    return c.post(f"{API}/people/{pid}/clips", headers=u.h, files=[("files", (f"c{i}.wav", d)) for i, d in enumerate(datas)])


def ident(c, u, p, data):
    return c.post(f"{API}/projects/{p['id']}/identify", headers=u.h, files={"file": ("x.wav", data)})


def test_speaker_enrol_identify_unknown_delete(client, alice, bob):
    p = sp_project(client, alice)
    assert p["task_type"] == "speaker_recognition" and p["status"] == "Draft"
    s = client.get(f"{API}/projects/{p['id']}/speaker-settings", headers=alice.h).json()
    assert s["kind"] == "speaker" and s["embedding_model"]["id"] == "microsoft/wavlm-base-plus-sv"
    assert "CC BY-SA 3.0" in s["embedding_model"]["licence"] and s["min_enrol_seconds"] == 3.0
    problem(client.post(f"{API}/projects/{p['id']}/people", headers=alice.h, json={"name": "x"}), 422, "consent_required")
    a, b = person(client, alice, p, "Reader A").json(), person(client, alice, p, "Reader B").json()
    silence = io.BytesIO()
    sf.write(silence, np.zeros(48000, np.float32), 16000, format="WAV")
    r = clips(client, alice, a["id"], A[:3] + [silence.getvalue(), voice(110, (700, 1200), 7, secs=1.0), b"junk"]).json()
    assert len(r["accepted"]) == 3 and r["accepted"][0]["seconds"] >= 3
    assert [x["reason"] for x in r["rejected"]] == ["too_quiet", "too_short", "unsupported_format"]
    assert len(clips(client, alice, b["id"], B[:3]).json()["accepted"]) == 3
    assert client.get(f"{API}/projects/{p['id']}", headers=alice.h).json()["status"] == "Ready"
    client.patch(f"{API}/projects/{p['id']}/speaker-settings", headers=alice.h,
                 json={"phrase": "Mera naam ... hai aur main yahan kaam karta hoon"})  # default threshold 0.86
    st = client.get(f"{API}/projects/{p['id']}/speaker-settings", headers=alice.h).json()
    assert st["threshold_source"] == "auto" and 0.86 <= st["threshold"] <= 0.97 and st["score_hint"]  # FAR-first
    got_a, got_b = ident(client, alice, p, A[3]).json(), ident(client, alice, p, B[3]).json()  # held-out clips
    assert got_a["threshold"] == st["threshold"]
    assert got_a["name"] == "Reader A" and got_b["name"] == "Reader B", (got_a, got_b)
    assert len(got_a["top3"]) == 2 and got_a["seconds"] >= 1.5 and "latency_ms" in got_a
    assert ident(client, alice, p, C).json()["name"] is None  # a stranger -> Unknown
    problem(ident(client, alice, p, voice(110, (700, 1200), 8, secs=0.8)), 422, "too_short")
    hand = client.patch(f"{API}/projects/{p['id']}/speaker-settings", headers=alice.h, json={"threshold": 0.95}).json()
    assert hand["threshold"] == 0.95 and hand["threshold_source"] == "user"  # a hand-set threshold sticks
    back = client.patch(f"{API}/projects/{p['id']}/speaker-settings", headers=alice.h, json={"threshold_auto": True})
    assert back.json()["threshold_source"] == "auto"
    rep = client.get(f"{API}/projects/{p['id']}/speaker-report", headers=alice.h).json()
    assert rep["ready"] and rep["rank1_accuracy"] == 1.0 and rep["genuine"]["mean"] > rep["impostor"]["mean"]
    detail = client.get(f"{API}/people/{a['id']}", headers=alice.h).json()
    assert len(detail["photos"]) == 3 and detail["photos"][0]["audio_url"] and detail["photos"][0]["seconds"] >= 3
    problem(client.post(f"{API}/people/{a['id']}/photos", headers=alice.h, files=[("files", ("x.png", b"x"))]),
            422, "not_face_project")
    problem(client.post(f"{API}/projects/{p['id']}/face-export", headers=alice.h,
                        json={"acknowledge_biometric": True}), 409, "export_unavailable")
    for call in (lambda u: client.get(f"{API}/people/{a['id']}", headers=u.h),
                 lambda u: clips(client, u, a["id"], A[:1]), lambda u: ident(client, u, p, A[3])):
        problem(call(bob), 404)
    prefix = f"{storage.org_prefix(alice.org_id)}faces/"
    assert len([k for k in storage.existing_sizes(prefix) if a["id"] in k]) == 3
    assert client.delete(f"{API}/people/{a['id']}", headers=alice.h).status_code == 204
    assert not [k for k in storage.existing_sizes(prefix) if a["id"] in k]
    with session_factory()() as db:
        assert db.scalar(select(func.count()).select_from(m.FaceEmbedding).where(
            m.FaceEmbedding.identity_id == uuid.UUID(a["id"]))) == 0
        assert db.scalar(select(m.AuditLog).where(m.AuditLog.action == "face.identity_deleted",
                                                  m.AuditLog.target_id == a["id"]))
    assert ident(client, alice, p, A[3]).json()["name"] != "Reader A"  # instant removal, nothing retrained
    assert client.post(f"{API}/projects/{p['id']}:purge", headers=alice.h,
                       json={"confirm_name": p["name"]}).status_code == 202
    assert not storage.existing_sizes(prefix)


def test_speaker_idle_unload(client, alice, monkeypatch):
    from app.settings import get_settings

    p = sp_project(client, alice)
    monkeypatch.setattr(get_settings(), "predict_idle_unload_s", 3600)  # the background reaper must not race the load checks
    idle.sweep(time.monotonic() + 3600)
    assert not speaker.loaded()
    assert ident(client, alice, p, A[0]).json()["model_load_ms"] > 0
    monkeypatch.setattr(get_settings(), "predict_idle_unload_s", 0.2)
    time.sleep(0.3)
    idle.sweep()
    assert not speaker.loaded()


@pytest.mark.skipif(not (FIX / "a").exists(), reason="public-domain speech fixtures missing")
def test_real_speech(client, alice):
    files = {k: sorted((FIX / k).glob("*.wav")) for k in ("a", "b", "c")}
    p = sp_project(client, alice)
    a, b = person(client, alice, p, "A").json(), person(client, alice, p, "B").json()
    assert not clips(client, alice, a["id"], [f.read_bytes() for f in files["a"][:-1]]).json()["rejected"]
    assert not clips(client, alice, b["id"], [f.read_bytes() for f in files["b"][:-1]]).json()["rejected"]
    assert ident(client, alice, p, files["a"][-1].read_bytes()).json()["name"] == "A"
    assert ident(client, alice, p, files["b"][-1].read_bytes()).json()["name"] == "B"
    assert ident(client, alice, p, files["c"][0].read_bytes()).json()["name"] is None
