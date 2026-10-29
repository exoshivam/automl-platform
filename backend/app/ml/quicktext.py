"""Quick text labels (no training) + ready-made sentiment.

Quick mode is face-gallery style: texts are embedded with the multilingual MiniLM (mean pooling, L2-normalised), and a
text is scored against each label as the mean of its top-3 cosine matches among that label's examples (a label with no
examples falls back to its own name). Below the project threshold -> "Other". Sentiment is a pretrained multilingual
classifier. Both models are lazy and freed by app.ml.idle.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict

import numpy as np
import structlog

from app.ml import idle

log = structlog.get_logger()
EMBED_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED = {"id": EMBED_ID, "licence": "Apache-2.0"}
SENTIMENT_LICENCES = {"tabularisai/multilingual-sentiment-analysis": "CC BY-NC 4.0 (non-commercial)"}


# Tuned on tests/fixtures/quicktext_eval.py (20 on-topic paraphrases + 20 off-topic, EN/HI/Hinglish):
#   top-3 mean, no background ........ on-topic 0.70 / off-topic rejected 0.65
#   top-3 mean, background wins ties .. 0.60 / 1.00   (swallowed paraphrases of a single example)
#   max similarity, background must beat the best label by > 0.05 ... 0.85 / 0.90   <- shipped
DEFAULT_THRESHOLD, TOP_K, BG_MARGIN = 0.45, 1, 0.05
# Everyday off-topic text (English / Hindi / Hinglish) competing as an implicit "Other" label: romanised Hindi shares
# so many surface tokens that unrelated chit-chat clears a bare threshold (measured: a weather remark scored 0.59 for
# "delivery late"). If the background matches better than every label, the answer is Other.
BACKGROUND = [
    "how are you doing today", "good morning, have a nice day", "what time is it now", "thank you so much",
    "the weather is nice today", "it is raining outside", "I watched a movie last night", "let's go for a walk",
    "my favourite food is pizza", "who won the cricket match", "I am going to the office", "happy birthday",
    "kya haal hai bhai", "sab theek hai", "aaj bahut garmi hai", "baarish ho rahi hai", "khana kha liya kya",
    "kal milte hain", "chai peene chalein", "movie bahut lambi thi", "main ghar ja raha hoon", "shubh prabhat",
    "aap kaise hain", "आज मौसम अच्छा है", "मैं घर जा रहा हूँ", "आप कैसे हैं", "धन्यवाद", "क्रिकेट मैच कौन जीता",
    "hello", "ok", "yes", "no", "please tell me more", "what is this", "nice photo", "see you tomorrow",
]
OTHER = "__other__"
_LOCK = threading.Lock()
_STATE: dict = {"embed": None, "sentiment": None, "used": {}, "load_ms": {}}
_CACHE: OrderedDict[str, np.ndarray] = OrderedDict()  # text sha256 -> embedding (process-local)
CACHE_SIZE = 20_000


def sentiment_ids() -> list[tuple[str, str | None]]:
    """SENTIMENT_MODELS: comma list of "hf_id@revision"; the first one present in the local HF cache is used."""
    from app.settings import get_settings

    out = []
    for x in get_settings().sentiment_models.split(","):
        if x.strip():
            mid, _, rev = x.strip().partition("@")
            out.append((mid, rev or None))
    return out


def _touch(name: str) -> None:
    _STATE["used"][name] = time.monotonic()


def _load(name: str):
    """Call with _LOCK held. Returns (tokenizer, model), loading it from the local HF cache if needed."""
    if _STATE[name] is None:
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

        t0 = time.perf_counter()
        if name == "embed":
            tok = AutoTokenizer.from_pretrained(EMBED_ID, local_files_only=True)
            model = AutoModel.from_pretrained(EMBED_ID, local_files_only=True)
            _STATE["embed_id"] = EMBED_ID
        else:
            last = None
            for mid, rev in sentiment_ids():
                try:
                    tok = AutoTokenizer.from_pretrained(mid, revision=rev, local_files_only=True)
                    model = AutoModelForSequenceClassification.from_pretrained(mid, revision=rev, local_files_only=True)
                    _STATE["sentiment_id"] = mid
                    break
                except OSError as e:  # not downloaded
                    last = e
            else:
                raise FileNotFoundError(f"no sentiment model in the local cache: {last}")
        _STATE[name] = (tok, model.eval())
        _STATE["load_ms"][name] = round((time.perf_counter() - t0) * 1000)
        log.info("text_model_loaded", model=name, load_ms=_STATE["load_ms"][name])
        idle.register(f"text_{name}", _unload_idle)
    _touch(name)
    return _STATE[name]


def _unload_idle(now: float) -> int:
    from app.settings import get_settings

    limit, freed = get_settings().predict_idle_unload_s, 0
    with _LOCK:
        for name in ("embed", "sentiment"):
            if _STATE[name] is not None and now - _STATE["used"].get(name, 0) >= limit:
                _STATE[name] = None
                freed += 1
                log.info("text_model_unloaded", model=name)
    return freed


def loaded(name: str) -> bool:
    return _STATE[name] is not None


def sentiment_model_id() -> str | None:
    """Without loading: the first sentiment model present in the local cache."""
    from huggingface_hub import try_to_load_from_cache

    for mid, rev in sentiment_ids():
        if isinstance(try_to_load_from_cache(mid, "config.json", revision=rev), str):
            return mid
    return None


def _key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def embed(texts: list[str]) -> tuple[np.ndarray, int]:
    """(L2-normalised embeddings [n, 384], model_load_ms). Cached by text hash."""
    import torch

    out: list[np.ndarray | None] = [_CACHE.get(_key(t)) for t in texts]
    todo = [i for i, v in enumerate(out) if v is None]
    load_ms = 0
    if todo:
        with _LOCK:
            cold = _STATE["embed"] is None
            tok, model = _load("embed")
            load_ms = _STATE["load_ms"]["embed"] if cold else 0
            for k in range(0, len(todo), 32):
                idx = todo[k:k + 32]
                enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=256, return_tensors="pt")
                with torch.no_grad():
                    h = model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                vec = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # mean pooling
                vec = torch.nn.functional.normalize(vec, dim=1).numpy()
                for i, v in zip(idx, vec, strict=True):
                    out[i] = v
                    _CACHE[_key(texts[i])] = v
            while len(_CACHE) > CACHE_SIZE:
                _CACHE.popitem(last=False)
    else:
        _touch("embed")
    return np.stack(out), load_ms


def sentiment(text: str) -> tuple[dict, int]:
    """({label: positive|neutral|negative, score}, model_load_ms)."""
    import torch

    with _LOCK:
        cold = _STATE["sentiment"] is None
        tok, model = _load("sentiment")
        enc = tok([text], truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            p = torch.softmax(model(**enc).logits[0], dim=0).numpy()
        load_ms = _STATE["load_ms"]["sentiment"] if cold else 0
    # map BY NAME (index orders differ between models) and collapse 5-way scales to 3 classes
    coarse: dict[str, float] = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
    names = [str(model.config.id2label[i]).lower() for i in range(len(p))]
    for name, prob in zip(names, p, strict=True):
        coarse["negative" if "neg" in name else "positive" if "pos" in name else "neutral"] += float(prob)
    k = int(p.argmax())
    label = max(coarse, key=coarse.get)
    return {"label": label, "score": round(coarse[label], 4), "detail": names[k]}, load_ms


def score_matrix(q: np.ndarray, examples: dict[str, np.ndarray]) -> dict[str, float]:
    """Per label: mean of the TOP_K best cosine similarities between q and that label's example embeddings."""
    out = {}
    for label, e in examples.items():
        sims = np.sort(e @ q)[::-1][:TOP_K]
        out[label] = float(sims.mean())
    return out


def gallery(examples: dict[str, list[str]]) -> tuple[dict[str, np.ndarray], set[str], int]:
    """(embeddings per label, labels that fell back to their own name, model_load_ms)."""
    texts, owners, zero = [], [], set()
    for label, xs in examples.items():
        if not xs:
            zero.add(label)
            xs = [label]  # zero-shot: the label name itself
        for t in xs:
            texts.append(t)
            owners.append(label)
    if not texts:
        return {}, zero, 0
    vecs, load_ms = embed(texts)
    by: dict[str, list] = {}
    for label, v in zip(owners, vecs, strict=True):
        by.setdefault(label, []).append(v)
    return {k: np.stack(v) for k, v in by.items()}, zero, load_ms


def classify(examples: dict[str, list[str]], text: str, threshold: float) -> dict:
    g, _, load_a = gallery({**examples, OTHER: BACKGROUND})
    q, load_b = embed([text])
    raw = score_matrix(q[0], g)
    background = raw.pop(OTHER)
    scores = sorted(raw.items(), key=lambda x: -x[1])
    top = [{"label": k, "score": round(v, 4)} for k, v in scores[:3]]
    # a user label wins when it clears the threshold and isn't clearly beaten by everyday off-topic text
    hit = top[0] if top and top[0]["score"] >= threshold and background <= top[0]["score"] + BG_MARGIN else None
    return {"label": hit["label"] if hit else None, "score": top[0]["score"] if top else None, "top3": top,
            "other": hit is None, "background_score": round(background, 4), "model_load_ms": load_a + load_b}


def _hist(x: np.ndarray) -> dict:
    edges = np.round(np.arange(-0.2, 1.0001, 0.05), 2)
    counts, _ = np.histogram(np.clip(x, -0.2, 1.0), bins=edges)
    return {"count": len(x), "mean": round(float(x.mean()), 4) if len(x) else None,
            "hist": [{"lo": float(a), "hi": float(b), "count": int(c)} for a, b, c in zip(edges[:-1], edges[1:], counts,
                                                                                          strict=True)]}


def report(examples: dict[str, list[str]], threshold: float) -> dict:
    """Leave-one-out over the examples: each example is scored against every label with itself removed."""
    n = sum(len(v) for v in examples.values())
    base = {"examples": n, "labels": len(examples), "current_threshold": threshold}
    usable = [k for k, v in examples.items() if len(v) >= 2]
    if len(usable) < 2:
        per = [{"label": k, "examples": len(v), "loo_hits": None,
                "warnings": [{"code": "zero_shot" if not v else "few_examples",
                              "detail": "No examples yet: the label name is used" if not v else
                              f"Only {len(v)} example(s); 5-10 recommended"}] if len(v) < 5 else []}
               for k, v in examples.items()]
        return {**base, "ready": False, "loo_accuracy": None, "suggested_threshold": None, "genuine": None,
                "impostor": None, "per_label": per, "warnings": [
                    {"code": "not_enough_examples", "detail": "Add at least 2 examples to 2 labels to see the report"}]}
    g, zero, _ = gallery(examples)
    genuine, impostor, per = [], [], []
    for label, xs in examples.items():
        hits, worst, worst_s = 0, None, -1.0
        for i in range(len(xs)):
            q = g[label][i]
            own = np.delete(g[label], i, axis=0) if label not in zero else g[label]
            others = {k: v for k, v in g.items() if k != label}
            gs = float(np.sort(own @ q)[::-1][:TOP_K].mean()) if len(own) else None
            imp = score_matrix(q, others)
            impostor += list(imp.values())
            if gs is not None:
                genuine.append(gs)
                best_other = max(imp.values(), default=-1.0)
                hits += int(gs > best_other)
            for k, s in imp.items():
                if s > worst_s:
                    worst, worst_s = k, s
        warns = []
        if not xs:
            warns.append({"code": "zero_shot", "detail": "No examples yet: the label name is used"})
        elif len(xs) < 5:
            warns.append({"code": "few_examples", "detail": f"Only {len(xs)} example(s); 5-10 recommended"})
        if len(xs) >= 2 and hits < len(xs):
            warns.append({"code": "not_matched", "detail": f"{len(xs) - hits} of {len(xs)} example(s) matched another "
                                                           "label better"})
        if worst is not None and worst_s >= threshold:
            warns.append({"code": "confusable_with", "detail": f"Examples look like '{worst}' (score {worst_s:.2f})",
                          "other_label": worst})
        per.append({"label": label, "examples": len(xs), "loo_hits": hits if len(xs) >= 2 else None, "warnings": warns})
    gen, imp = np.array(genuine), np.array(impostor)
    tried = [p for p in per if p["loo_hits"] is not None]
    loo = sum(p["loo_hits"] for p in tried) / max(1, sum(p["examples"] for p in tried))
    # between the impostor and genuine bulk: the 95th impostor percentile, nudged up, never above the genuine median
    t = float(np.quantile(imp, 0.95)) + 0.02 if len(imp) else DEFAULT_THRESHOLD
    suggested = round(min(max(t, 0.2), float(np.median(gen)) if len(gen) else 0.9, 0.9), 2)
    return {**base, "ready": True, "loo_accuracy": round(loo, 4), "suggested_threshold": suggested,
            "genuine": _hist(gen), "impostor": _hist(imp), "per_label": per,
            "warnings": [] if n >= 20 else [{"code": "few_examples_total",
                                             "detail": f"{n} examples in total; the threshold estimate is rough"}]}
