"""Device choice: cuda -> mps -> cpu (TRAIN_DEVICE forces one). Imports torch lazily."""

from app.settings import get_settings


def pick_device() -> str:
    import torch

    forced = get_settings().train_device
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def free_memory(device: str) -> None:
    import gc

    import torch

    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
