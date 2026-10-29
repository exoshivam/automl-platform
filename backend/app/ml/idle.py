"""Idle unload for in-process models (face ONNX sessions, the predict cache): they stay loaded only while used.

A daemon thread checks every few seconds and calls each registered unload(idle_for) hook, which frees whatever
has been unused for longer than its own limit. A request after an unload simply reloads (callers report load_ms).
"""

import gc
import threading
import time

import structlog

log = structlog.get_logger()
_HOOKS: dict[str, object] = {}
_LOCK = threading.Lock()
_THREAD: threading.Thread | None = None
TICK_S = 5.0


def register(name: str, hook) -> None:
    """hook(now: float) -> int (items freed). Starts the reaper thread on first registration."""
    global _THREAD
    with _LOCK:
        _HOOKS[name] = hook
        if _THREAD is None or not _THREAD.is_alive():
            _THREAD = threading.Thread(target=_loop, name="model-idle-unload", daemon=True)
            _THREAD.start()


def sweep(now: float | None = None) -> int:
    freed = 0
    for name, hook in list(_HOOKS.items()):
        try:
            freed += hook(time.monotonic() if now is None else now)
        except Exception as e:  # noqa: BLE001
            log.warning("model_idle_unload_failed", cache=name, error=str(e)[:200])
    if freed:
        gc.collect()  # drop the ONNX / torch buffers now, not at some later collection
    return freed


def _loop() -> None:
    while True:
        time.sleep(TICK_S)
        sweep()
