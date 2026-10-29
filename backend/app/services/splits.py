"""Train/val/test splits (DAT-6): stratified by class, grouped (a group never straddles splits), deterministic by seed.

Exact duplicates (same SHA-256) are always one group, so identical files can't leak across splits (VAL-7).
"""

import random
from collections import Counter, defaultdict
from typing import Any

MIN_TEST_PER_CLASS = 2


def assign(items: list[tuple[Any, str | None, Any]], train: float, val: float, test: float, seed: int,
           stratify: bool = True) -> tuple[dict[Any, str], dict[str, Any]]:
    """items: (item_key, class_or_None, group_key). Returns ({item_key: split}, summary)."""
    groups: dict[Any, list[tuple[Any, str | None]]] = defaultdict(list)
    for key, cls, grp in items:
        groups[grp].append((key, cls))
    strata: dict[str, list[Any]] = defaultdict(list)
    for grp, members in groups.items():
        label = Counter(c or "__none__" for _, c in members).most_common(1)[0][0] if stratify else "__all__"
        strata[label].append(grp)

    out: dict[Any, str] = {}
    per_class: dict[str, dict[str, int]] = {}
    warnings: list[dict[str, Any]] = []
    for label in sorted(strata, key=str):
        grps = sorted(strata[label], key=str)
        random.Random(f"{seed}:{label}").shuffle(grps)
        n = sum(len(groups[g]) for g in grps)
        n_test, n_val = round(n * test), round(n * val)
        if n >= 3 and test > 0:
            n_test = max(n_test, 1)
        counts = {"train": 0, "val": 0, "test": 0}
        for g in grps:
            size = len(groups[g])
            split = "test" if counts["test"] < n_test else "val" if counts["val"] < n_val else "train"
            counts[split] += size
            for key, _ in groups[g]:
                out[key] = split
        per_class[label] = counts
        if label not in ("__none__", "__all__") and counts["test"] < MIN_TEST_PER_CLASS:
            warnings.append({"code": "min_test_count_not_met", "class": label, "test": counts["test"],
                             "required": MIN_TEST_PER_CLASS, "total": n})
    totals = Counter(out.values())
    summary = {"counts": {s: totals.get(s, 0) for s in ("train", "val", "test")}, "per_class": per_class,
               "seed": seed, "ratios": {"train": train, "val": val, "test": test}, "warnings": warnings}
    return out, summary
