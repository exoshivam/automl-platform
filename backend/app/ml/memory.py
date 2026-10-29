"""Fail fast instead of thrashing a memory-starved machine."""

import psutil

from app.settings import get_settings


class NotEnoughMemory(Exception):
    pass


def check(stage: str) -> None:
    s = get_settings()
    rss = psutil.Process().memory_info().rss / 1e9
    free = psutil.virtual_memory().available / 1e9
    if rss > s.train_max_memory_gb:
        raise NotEnoughMemory(f"training used {rss:.1f} GB (limit {s.train_max_memory_gb} GB) at {stage}")
    if free < s.train_min_free_gb:
        raise NotEnoughMemory(f"only {free:.2f} GB RAM free at {stage} (need {s.train_min_free_gb} GB)")


def is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "mps backend out of memory" in text or isinstance(exc, MemoryError)
