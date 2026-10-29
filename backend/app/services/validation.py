"""Data validation (PRD §6.5, VAL-1..14 + VAL-16). Runs after every seal; also re-run after auto-fixes.

Report item: {code, severity: error|warning|info, title, why, count, examples, fix: {text, auto_fix_id}, acknowledgeable,
acknowledged, details}. Errors block training; warnings (and VAL-11) need an acknowledgement.
"""

import io
import re
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

import pandas as pd
from sqlalchemy import select

from app import models as m
from app.services import storage
from app.services.project_status import refresh_status

NEAR_DUP_HAMMING = 3  # 64-bit pHash: <= 3 differing bits ~ similarity > 0.95
MIN_PER_CLASS, LOW_PER_CLASS, MIN_TABULAR_ROWS, IMBALANCE = 10, 30, 50, 10.0
BOX_WARN, BOX_ERROR, TINY_BOX = 50, 15, 8
LEAKY_NAMES = re.compile(r"(_outcome|_result|_label|_target)$|^(cancel|churned|outcome|target_)|cancel(l)?ed", re.IGNORECASE)
HINGLISH = re.compile(r"\b(hai|hain|nahi|nahin|kya|kyun|bahut|acha|accha|achha|bhai|yaar|raha|rahi|rahe|mera|meri|"
                      r"tera|hum|aap|kiya|karna|karunga|karenge|dobara|bilkul|bekaar|bekar|paisa|vasool|wasool|thi|"
                      r"tha|bohot|jaldi|abhi|kabhi|sab|kuch|zyada|thoda|lekin|aur)\b", re.IGNORECASE)


def _item(code, severity, title, why, count, examples=(), fix_text="", auto_fix_id=None, ackable=None, **details):
    return {"code": code, "severity": severity, "title": title, "why": why, "count": count,
            "examples": [str(e) for e in list(examples)[:10]], "fix": {"text": fix_text, "auto_fix_id": auto_fix_id},
            "acknowledgeable": severity == "warning" if ackable is None else ackable, "acknowledged": False,
            "details": details}


def _accuracy_band(n: int) -> str:
    return "roughly 60-80%" if n < 20 else "roughly 70-85%"


def _phash_pairs(samples) -> set[tuple]:
    """Near-duplicate image pairs via 4 x 16-bit chunk buckets (pigeonhole: <= 3 bit flips share a chunk)."""
    hashes = [(s, int(s.phash, 16)) for s in samples if s.phash]
    buckets: dict = defaultdict(list)
    for s, h in hashes:
        for k in range(4):
            buckets[(k, (h >> (16 * k)) & 0xFFFF)].append((s, h))
    pairs = set()
    for members in buckets.values():
        if len(members) > 500:  # degenerate bucket (e.g. blank images): exact dupes already cover it
            continue
        for (a, ha), (b, hb) in combinations(members, 2):
            if a.sha256 != b.sha256 and (ha ^ hb).bit_count() <= NEAR_DUP_HAMMING:
                pairs.add(tuple(sorted((a.id, b.id), key=str)))
    return pairs


def _text_pairs(samples) -> set[tuple]:
    from datasketch import MinHash, MinHashLSH

    lsh, mh = MinHashLSH(threshold=0.9, num_perm=64), {}
    for s in samples:
        words = (s.meta or {}).get("text") or ""
        shingles = {" ".join(words.lower().split()[i:i + 3]) for i in range(max(len(words.split()) - 2, 1))}
        if len(words.split()) < 4:
            continue
        h = MinHash(num_perm=64)
        for sh in shingles:
            h.update(sh.encode())
        mh[s.id] = (s, h)
        lsh.insert(str(s.id), h)
    pairs = set()
    for sid, (s, h) in mh.items():
        for other in lsh.query(h):
            o = mh[next(k for k in mh if str(k) == other)][0]
            if o.id != sid and o.sha256 != s.sha256:
                pairs.add(tuple(sorted((sid, o.id), key=str)))
    return pairs


def check_samples(db, v: m.DatasetVersion, project: m.Project) -> list[dict]:
    org, task = v.organization_id, project.task_type
    samples = db.scalars(select(m.Sample).where(m.Sample.dataset_version_id == v.id,
                                                m.Sample.organization_id == org)).all()
    labels = db.scalars(select(m.Label).join(m.Sample, m.Sample.id == m.Label.sample_id).where(
        m.Sample.dataset_version_id == v.id, m.Label.organization_id == org)).all()
    by_sample: dict = defaultdict(list)
    for lb in labels:
        by_sample[lb.sample_id].append(lb)
    flags: dict = defaultdict(set)
    items: list[dict] = []
    name = {s.id: s.original_name for s in samples}

    corrupt = [s for s in samples if (s.meta or {}).get("corrupt")]
    if corrupt:
        for s in corrupt:
            flags[s.id].add("VAL-1")
        items.append(_item("VAL-1", "error", "Corrupt or unreadable files",
                           "These files can't be decoded, so they would crash or silently skew training.",
                           len(corrupt), [name[s.id] for s in corrupt], "Exclude them", "exclude_corrupt",
                           reasons=dict(Counter(s.meta["corrupt"] for s in corrupt))))
    good = [s for s in samples if not (s.meta or {}).get("corrupt")]

    if task in ("image_classification", "text_classification"):
        per_class = Counter(lb.class_name for s in good for lb in by_sample[s.id][:1])
        too_few = sorted(c for c, n in per_class.items() if n < MIN_PER_CLASS)
        low = sorted(c for c, n in per_class.items() if MIN_PER_CLASS <= n < LOW_PER_CLASS)
        if too_few:
            items.append(_item("VAL-2", "error", "Too few samples in some classes",
                               f"Every class needs at least {MIN_PER_CLASS} examples to train and test honestly.",
                               len(too_few), too_few, "Add more data, or merge rare classes",
                               per_class={c: per_class[c] for c in too_few}))
        if low:
            items.append(_item("VAL-3", "warning", "Low sample count in some classes",
                               f"Under {LOW_PER_CLASS} examples per class usually limits accuracy.", len(low), low,
                               "Add data where you can", per_class={c: per_class[c] for c in low},
                               estimated_accuracy={c: _accuracy_band(per_class[c]) for c in low}))
        if len(per_class) >= 2 and min(per_class.values()) > 0:
            big, small = per_class.most_common()[0], per_class.most_common()[-1]
            ratio = big[1] / small[1]
            if ratio > IMBALANCE:
                items.append(_item("VAL-4", "warning", "Class imbalance",
                                   "The model will favour the large class; per-class metrics matter more here.", 1,
                                   [big[0], small[0]], "Class weights are applied automatically; collect more of "
                                   f"'{small[0]}'", ratio=round(ratio, 2), largest=big[0], smallest=small[0]))
        unlabeled = [s for s in good if not by_sample[s.id]]
        if unlabeled:
            items.append(_item("VAL-15", "warning", "Unlabeled samples",
                               "Samples without a class are ignored by training.", len(unlabeled),
                               [name[s.id] for s in unlabeled], "Label them or remove them"))

    # exact duplicates (VAL-5) + conflicting labels (VAL-8)
    by_sha: dict = defaultdict(list)
    for s in good:
        by_sha[s.sha256].append(s)
    dup_groups = [g for g in by_sha.values() if len(g) > 1]
    if dup_groups:
        for g in dup_groups:
            for s in g[1:]:
                flags[s.id].add("VAL-5")
        items.append(_item("VAL-5", "warning", "Exact duplicates", "Duplicates over-weight some examples.",
                           len(dup_groups), [" = ".join(name[s.id] for s in g[:3]) for g in dup_groups],
                           "Keep one of each", "dedupe_exact", duplicate_files=sum(len(g) - 1 for g in dup_groups)))
    near = _phash_pairs(good) if task in ("image_classification", "object_detection") else \
        _text_pairs(good) if task == "text_classification" else set()
    if near:
        for a, b in near:
            flags[a].add("VAL-6")
            flags[b].add("VAL-6")
        items.append(_item("VAL-6", "warning", "Near-duplicates",
                           "Almost identical samples inflate test scores if they end up on both sides of the split.",
                           len(near), [f"{name[a]} ~ {name[b]}" for a, b in sorted(near, key=str)],
                           "Review them and keep one"))

    def classes(s):
        return frozenset(lb.class_name for lb in by_sample[s.id])

    if task != "object_detection":
        conflicts = [g for g in dup_groups if len({classes(s) for s in g}) > 1]
        near_conf = [(a, b) for a, b in near if classes(next(x for x in good if x.id == a)) !=
                     classes(next(x for x in good if x.id == b))]
        if conflicts or near_conf:
            items.append(_item("VAL-8", "warning", "Conflicting labels",
                               "The same (or almost the same) sample has different labels.",
                               len(conflicts) + len(near_conf),
                               [PurePath(name[g[0].id]) for g in conflicts] + [name[a] for a, _ in near_conf],
                               "Review the conflicts"))

    # leakage across splits (VAL-7): exact groups never straddle (grouped split); check near-dupes
    split = {s.id: s.split for s in good}
    leaks = [(a, b) for a, b in near if {split.get(a), split.get(b)} >= {"train", "test"}]
    leaks += [(g[0].id, g[1].id) for g in dup_groups if {"train", "test"} <= {split.get(s.id) for s in g}]
    if leaks:
        items.append(_item("VAL-7", "error", "Leakage across splits",
                           "Near-identical samples in train and test make the test score look better than reality.",
                           len(leaks), [f"{name[a]} ~ {name[b]}" for a, b in leaks],
                           "Re-split with duplicates kept together", "resplit_groups"))

    if task in ("image_classification", "object_detection") and good:
        div, warn = diversity(good, by_sample, near, dup_groups)
        DIVERSITY[v.id] = div
        if warn:
            items.append(_item("VAL-18", "warning", "Low diversity: mostly near-identical burst frames",
                               "Most images are near-identical burst frames from a few sessions. Test scores will be "
                               "optimistic and the model may learn the background instead of the object.",
                               div["samples"] - div["effective_samples"], [],
                               "Add photos in different places, lighting and angles", **div))
    if task == "object_detection":
        items += _detection_checks(good, by_sample, name, flags, project)
    if task in ("image_classification", "object_detection"):
        items += _image_quality(good, name, flags)
    if task == "text_classification":
        items += _text_checks(good, name, flags)

    for s in samples:  # per-sample flags power ?flag= in the browser (derived data; not part of content_hash)
        new = sorted(flags.get(s.id, ()))
        if (s.meta or {}).get("flags", []) != new:
            s.meta = {**(s.meta or {}), "flags": new}
    return items


DIVERSITY: dict = {}  # version id -> stat, picked up by run() (kept out of the item list)
NEAR_SHARE_WARN, FEW_SESSIONS = 0.20, 2


def diversity(samples, by_sample, near_pairs, dup_groups) -> tuple[dict, bool]:
    """Effective sample count = near-identical clusters; sessions = upload/capture batches per class."""
    parent = {s.id: s.id for s in samples}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for g in dup_groups:
        for s in g[1:]:
            parent[find(s.id)] = find(g[0].id)
    for a, b in near_pairs:
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    sizes = Counter(find(s.id) for s in samples)
    clustered = sum(n for n in sizes.values() if n > 1)
    sessions: dict = defaultdict(set)
    for s in samples:
        cls = by_sample[s.id][0].class_name if by_sample[s.id] else "(unlabeled)"
        sessions[cls].add((s.meta or {}).get("upload_id") or "unknown")
    per_class = {c: len(v) for c, v in sorted(sessions.items())}
    n = len(samples)
    camera = sum(1 for s in samples if (s.meta or {}).get("source") == "camera")
    stat = {"samples": n, "effective_samples": len(sizes), "near_duplicate_share": round(clustered / n, 3) if n else 0,
            "sessions_per_class": per_class, "min_sessions_per_class": min(per_class.values()) if per_class else 0,
            "camera_share": round(camera / n, 3) if n else 0}
    # "few sessions" is a camera-burst signal; a folder upload in one batch is normal
    few = n >= 20 and camera / n >= 0.5 and per_class and max(per_class.values()) <= FEW_SESSIONS
    stat["low"] = bool(stat["near_duplicate_share"] > NEAR_SHARE_WARN or few)
    return stat, stat["low"]


class PurePath(str):  # str subclass so examples render as names
    pass


def _detection_checks(samples, by_sample, name, flags, project) -> list[dict]:
    oob, zero, tiny, empty = [], [], [], []
    per_class = Counter()
    for s in samples:
        boxes = [lb for lb in by_sample[s.id] if lb.bbox]
        if not boxes and not s.no_objects:
            empty.append(s)
        for lb in boxes:
            b, W, H = lb.bbox, s.width or 0, s.height or 0
            per_class[lb.class_name] += 1
            if b["w"] <= 0 or b["h"] <= 0:
                zero.append(s)
            elif b["x"] < 0 or b["y"] < 0 or b["x"] + b["w"] > W + 0.5 or b["y"] + b["h"] > H + 0.5:
                oob.append(s)
            elif b["w"] < TINY_BOX or b["h"] < TINY_BOX:
                tiny.append(s)
    out = []
    for code, rows, title in (("out_of_bounds", oob, "Boxes outside the image"), ("zero_area", zero, "Zero-area boxes"),
                              ("tiny", tiny, f"Tiny boxes (< {TINY_BOX} px)")):
        if rows:
            for s in rows:
                flags[s.id].add("VAL-16")
            out.append(_item("VAL-16", "warning", title, "Bad boxes teach the detector the wrong thing.", len(rows),
                             [name[s.id] for s in rows], "Fix or delete these boxes", kind=code))
    if empty:
        for s in empty:
            flags[s.id].add("VAL-16")
        out.append(_item("VAL-16", "warning", "Images without boxes",
                         "Unlabeled images are skipped unless marked 'no objects'.", len(empty),
                         [name[s.id] for s in empty], "Draw boxes or mark them 'no objects'", kind="unlabeled_images"))
    from app.services.specs import train_labels

    classes = train_labels(project.current_spec) or sorted(per_class)
    hybrid = project.current_spec.get("mode") == "pretrained"  # boxes on ready-made objects aren't the user's job
    for cls in sorted(set(classes) | (set() if hybrid else set(per_class))):
        n = per_class.get(cls, 0)
        if n < BOX_ERROR:
            out.append(_item("VAL-16", "error", f"Too few boxes for '{cls}'",
                             f"Each class needs at least {BOX_ERROR} boxes ({BOX_WARN}+ recommended).", n, [cls],
                             "Label more", kind="class_count", class_name=cls, boxes=n))
        elif n < BOX_WARN:
            out.append(_item("VAL-16", "warning", f"Few boxes for '{cls}'",
                             f"Under {BOX_WARN} boxes per class usually limits detection quality.", n, [cls],
                             "Label more", kind="class_count", class_name=cls, boxes=n))
    return out


def _image_quality(samples, name, flags) -> list[dict]:
    imgs = [s for s in samples if s.width and s.height]
    tiny = [s for s in imgs if max(s.width, s.height) < 32]  # small in both dimensions (thin strips are "aspect")
    aspect = [s for s in imgs if max(s.width, s.height) / max(min(s.width, s.height), 1) > 10]
    modes = Counter((s.meta or {}).get("mode") for s in imgs)
    gray = [s for s in imgs if (s.meta or {}).get("mode") in ("L", "LA", "1", "I;16")]
    out = []
    if tiny:
        out.append(_item("VAL-13", "warning", "Very small images (< 32 px)",
                         "They are upscaled for training and carry little detail.", len(tiny),
                         [name[s.id] for s in tiny], "Replace with larger photos", kind="tiny"))
    if aspect:
        out.append(_item("VAL-13", "info", "Extreme aspect ratios", "Resizing distorts them.", len(aspect),
                         [name[s.id] for s in aspect], "Crops/letterboxing are applied automatically",
                         kind="extreme_aspect"))
    if gray and len(gray) < len(imgs) and modes.most_common(1)[0][0] not in ("L", "LA", "1"):
        out.append(_item("VAL-13", "info", "Grayscale images in a colour set",
                         "They are converted to RGB; colour cues are missing for them.", len(gray),
                         [name[s.id] for s in gray], "No action needed", kind="grayscale_in_rgb"))
    for s in tiny + aspect:
        flags[s.id].add("VAL-13")
    return out


SCRIPTS = (("Devanagari", 0x0900, 0x097F), ("Bengali", 0x0980, 0x09FF), ("Gurmukhi", 0x0A00, 0x0A7F),
           ("Gujarati", 0x0A80, 0x0AFF), ("Odia", 0x0B00, 0x0B7F), ("Tamil", 0x0B80, 0x0BFF),
           ("Telugu", 0x0C00, 0x0C7F), ("Kannada", 0x0C80, 0x0CFF), ("Malayalam", 0x0D00, 0x0D7F))


def _script(text: str) -> str:
    counts = Counter()
    for ch in text:
        cp = ord(ch)
        for script, lo, hi in SCRIPTS:
            if lo <= cp <= hi:
                counts[script] += 1
                break
        else:
            if ch.isalpha() and unicodedata.name(ch, "").startswith("LATIN"):
                counts["Latin"] += 1
    return counts.most_common(1)[0][0] if counts else "emoji_only" if text.strip() else "empty"


def _text_checks(samples, name, flags) -> list[dict]:
    texts = [(s, ((s.meta or {}).get("text") or "")) for s in samples]
    empty = [s for s, t in texts if not t.strip()]
    short = [s for s, t in texts if t.strip() and len(t.strip()) < 4]
    scripts = Counter(_script(t) for _, t in texts if t.strip())
    hinglish = [s for s, t in texts if _script(t) == "Latin" and HINGLISH.search(t)]
    non_english = sum(n for sc, n in scripts.items() if sc not in ("Latin", "emoji_only")) + len(hinglish)
    total = len([1 for _, t in texts if t.strip()]) + len(empty)
    out = []
    if empty or short:
        for s in empty + short:
            flags[s.id].add("VAL-14")
        out.append(_item("VAL-14", "info", "Empty or very short texts", "They carry almost no signal.",
                         len(empty) + len(short), [name[s.id] for s in empty + short], "Remove or fix them",
                         empty_or_whitespace=len(empty), very_short=len(short)))
    share = non_english / total if total else 0
    out.append(_item("VAL-14", "info", "Language / script mix",
                     "Model choice depends on the languages in your data.", len(scripts), sorted(scripts),
                     "An Indic/multilingual backbone (MuRIL / XLM-R) will be used" if share > 0.10 else
                     "An English backbone is fine", scripts=dict(scripts), hinglish_romanised=len(hinglish),
                     non_english_share=round(share, 3), suggest_multilingual=share > 0.10))
    return out


def check_table(v: m.DatasetVersion, project: m.Project) -> list[dict]:
    df = pd.read_parquet(io.BytesIO(storage.get_bytes(v.manifest_uri or v.table_uri)))
    split = df.pop("__split__") if "__split__" in df.columns else None
    cm = v.column_mapping or {}
    target = cm.get("target")
    roles = {c["name"]: c["role"] for c in cm.get("columns", [])}
    features = [c for c in df.columns if roles.get(c, "feature") == "feature"]
    items: list[dict] = []
    if len(df) < MIN_TABULAR_ROWS:
        items.append(_item("VAL-2", "error", "Too few rows", f"At least {MIN_TABULAR_ROWS} rows are needed.", len(df),
                           fix_text="Add more rows"))
    dupes = int(df.drop(columns=[c for c, r in roles.items() if r == "id" and c in df.columns]).duplicated().sum())
    if dupes:
        items.append(_item("VAL-5", "warning", "Duplicate rows", "Duplicates over-weight some rows.", dupes,
                           fix_text="Drop duplicate rows", auto_fix_id="dedupe_rows"))
    missing = {c: round(float(df[c].isna().mean()), 3) for c in features if df[c].isna().any()}
    if missing:
        heavy = sorted(c for c, f in missing.items() if f > 0.5)
        items.append(_item("VAL-9", "warning", "Missing values", "Models need a strategy for gaps.", len(missing),
                           heavy or sorted(missing), "Impute (default) or drop the column", missing_fraction=missing,
                           over_half=heavy, choices=["impute", "drop"]))
    n = len(df)
    id_like = []
    for c in features:
        s = df[c].dropna()
        if n and s.nunique() / n > 0.95 and (pd.api.types.is_integer_dtype(s) or s.astype(str).str.fullmatch(
                r"[0-9a-fA-F-]{8,}|(?=.*\d)[A-Za-z0-9_-]{6,}|[A-Za-z]*\d+").mean() > 0.9):
            id_like.append(c)
    if id_like:
        items.append(_item("VAL-10", "warning", "ID-like columns", "Unique identifiers can't generalise; they "
                           "memorise rows.", len(id_like), id_like, "Excluded from features (auto-fix)",
                           "exclude_id_columns", columns=id_like))
    if target and project.task_type == "tabular_classification":
        y = df[target].astype(str)
        leaky = []
        for c in [c for c in features if c not in id_like]:
            reason = None
            if LEAKY_NAMES.search(c):
                reason = "name"
            x = df[c]
            presence = x.notna().astype(int)
            if 0 < presence.mean() < 1 and y.nunique() == 2 and (presence.groupby(y).mean().round(3).isin([0, 1]).all()):
                reason = (reason + "+" if reason else "") + "missingness_matches_target"
            elif x.nunique(dropna=True) / max(n, 1) <= 0.5 and _single_feature_auc(x, y) > 0.98:
                # (near-unique columns trivially "predict" via per-value means: that's VAL-10/12, not leakage)
                reason = (reason + "+" if reason else "") + "auc"
            if reason:
                leaky.append((c, reason))
        if leaky:
            items.append(_item("VAL-11", "error", "Possible target leakage",
                               "A feature that predicts the target almost perfectly usually encodes the answer.",
                               len(leaky), [c for c, _ in leaky], "Review; exclude or acknowledge", ackable=True,
                               columns={c: r for c, r in leaky}))
    const = [c for c in features if df[c].nunique(dropna=True) <= 1 or
             (n and df[c].value_counts(dropna=False).iloc[0] / n >= 0.99)]
    high_card = [c for c in features if c not in id_like and not pd.api.types.is_numeric_dtype(df[c])
                 and df[c].nunique() > 50
                 and df[c].nunique() / max(n, 1) > 0.5]
    if const:
        items.append(_item("VAL-12", "info", "Constant columns", "They carry no information.", len(const), const,
                           "Dropped automatically", kind="constant", columns=const))
    if high_card:
        items.append(_item("VAL-12", "warning", "High-cardinality categoricals", "Too many categories to learn from.",
                           len(high_card), high_card, "Drop or hash-encode", kind="high_cardinality",
                           columns=high_card))
    _ = split
    return items


def _single_feature_auc(x: pd.Series, y: pd.Series) -> float:
    from sklearn.metrics import roc_auc_score

    if y.nunique() != 2 or x.nunique(dropna=True) < 2:
        return 0.0
    pos = sorted(y.unique())[1]
    yy = (y == pos).astype(int)
    if pd.api.types.is_numeric_dtype(x):
        xs = x.fillna(x.median())
    else:
        means = yy.groupby(x.astype(str)).mean()
        xs = x.astype(str).map(means)
    try:
        auc = roc_auc_score(yy, xs)
    except ValueError:
        return 0.0
    return max(auc, 1 - auc)


def summarise(items: list[dict], acknowledged: list[str]) -> dict[str, Any]:
    for it in items:
        it["acknowledged"] = it["acknowledgeable"] and it["code"] in acknowledged
    counts = Counter(it["severity"] for it in items)
    blocking = [it for it in items if (it["severity"] == "error" and not it["acknowledged"])]
    pending = [it for it in items if it["severity"] == "warning" and not it["acknowledged"]]
    status = "failed" if blocking else "warnings" if pending else "passed"
    return {"status": status, "summary": {"errors": counts["error"], "warnings": counts["warning"],
                                          "info": counts["info"]}, "items": items}


def run(db, v: m.DatasetVersion, project: m.Project) -> None:
    items = check_table(v, project) if v.table_uri and project.task_type != "text_classification" else \
        check_samples(db, v, project)
    report = summarise(items, v.acknowledgements or [])
    report["diversity"] = DIVERSITY.pop(v.id, None)  # structured stat for the UI (null for tabular/text)
    v.validation_report = report
    v.validation_status = report["status"]
    db.flush()
    refresh_status(db, project)
