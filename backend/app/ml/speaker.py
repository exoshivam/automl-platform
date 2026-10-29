"""Speaker recognition pipeline (gallery pattern, never trained): decode (soundfile: WAV/FLAC/OGG, no ffmpeg) ->
16 kHz mono -> energy-based VAD trims silence -> quality checks -> WavLM-base-plus-SV x-vector (512-d, L2-normalised).
Lazy load + idle unload. Licence of the weights: CC BY-SA 3.0 (per the model card -> microsoft/UniSpeech LICENSE).
"""

from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass

import numpy as np
import structlog

from app.ml import idle

log = structlog.get_logger()
MODEL_ID, REVISION = "microsoft/wavlm-base-plus-sv", "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e"
MODEL = {"id": MODEL_ID, "name": "WavLM-base-plus-SV (x-vector)", "revision": REVISION,
         "licence": "CC BY-SA 3.0 (per model card -> microsoft/UniSpeech LICENSE): attribution + share-alike",
         "commercial_use": True}
SR = 16_000
FRAME = 480  # 30 ms
MIN_ENROL_S, MIN_IDENTIFY_S = 3.0, 1.5
QUIET_RMS, CLIP_LEVEL, CLIP_FRACTION = 0.005, 0.99, 0.01
_LOCK = threading.Lock()
_STATE: dict = {"model": None, "last_used": 0.0, "load_ms": 0}


class AudioError(ValueError):
    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason, self.detail = reason, detail


@dataclass
class Clip:
    speech: np.ndarray  # 16 kHz mono float32, silence trimmed
    seconds: float
    quality: float  # 0..1
    rms: float
    clipped: float  # fraction of samples at full scale


def decode(data: bytes) -> np.ndarray:
    """-> 16 kHz mono float32 in [-1, 1]. WAV/FLAC/OGG via soundfile (bundled libsndfile, no ffmpeg)."""
    import soundfile as sf

    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except sf.LibsndfileError as e:
        raise AudioError("unsupported_format", "Use WAV, FLAC or OGG audio") from e
    except Exception as e:
        raise AudioError("not_audio", "Not a readable audio file") from e
    x = x.mean(axis=1)
    if sr != SR:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    return x


def vad(x: np.ndarray) -> np.ndarray:
    """Energy VAD: keep 30 ms frames above max(-40 dB of the loudest frame, an absolute floor), with 2-frame hangover."""
    n = len(x) // FRAME
    if n == 0:
        return x[:0]
    frames = x[: n * FRAME].reshape(n, FRAME)
    e = np.sqrt((frames ** 2).mean(axis=1))
    voiced = e > max(e.max() * 0.01, 0.003)
    keep = voiced.copy()
    for shift in (1, 2):  # hangover so word endings aren't chopped
        keep[shift:] |= voiced[:-shift]
        keep[:-shift] |= voiced[shift:]
    return frames[keep].reshape(-1)


def analyse(data: bytes, min_seconds: float) -> Clip:
    x = decode(data)
    clipped = float((np.abs(x) >= CLIP_LEVEL).mean()) if len(x) else 0.0
    speech = vad(x)
    seconds = len(speech) / SR
    rms = float(np.sqrt((speech ** 2).mean())) if len(speech) else 0.0
    if rms < QUIET_RMS and len(x) and float(np.sqrt((x ** 2).mean())) < QUIET_RMS:
        raise AudioError("too_quiet", "The recording is almost silent; speak closer to the microphone")
    if seconds < min_seconds:
        raise AudioError("too_short", f"{seconds:.1f} s of speech; at least {min_seconds:g} s is needed")
    if clipped > CLIP_FRACTION:
        raise AudioError("clipping", "The recording is distorted (too loud); move back from the microphone")
    quality = round(min(1.0, seconds / 6.0) * min(1.0, rms / 0.05) * (1.0 - min(1.0, clipped * 20)), 3)
    return Clip(speech, round(seconds, 2), quality, round(rms, 4), round(clipped, 4))


def _model():
    """Call with _LOCK held."""
    if _STATE["model"] is None:
        from transformers import AutoFeatureExtractor, WavLMForXVector

        t0 = time.perf_counter()
        fx = AutoFeatureExtractor.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
        m = WavLMForXVector.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True).eval()  # weights_only
        _STATE["model"], _STATE["load_ms"] = (fx, m), round((time.perf_counter() - t0) * 1000)
        log.info("speaker_model_loaded", load_ms=_STATE["load_ms"])
        idle.register("speaker", _unload_if_idle)
    _STATE["last_used"] = time.monotonic()
    return _STATE["model"]


def _unload_if_idle(now: float) -> int:
    from app.settings import get_settings

    with _LOCK:
        if _STATE["model"] is None or now - _STATE["last_used"] < get_settings().predict_idle_unload_s:
            return 0
        _STATE["model"] = None
    log.info("speaker_model_unloaded")
    return 1


def loaded() -> bool:
    return _STATE["model"] is not None


def embed(speech: np.ndarray) -> tuple[np.ndarray, int]:
    """(L2-normalised 512-d x-vector, model_load_ms). Clips are capped at 20 s of speech."""
    import torch

    with _LOCK:
        cold = _STATE["model"] is None
        fx, m = _model()
        inp = fx(speech[: 20 * SR], sampling_rate=SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            v = m(**inp).embeddings[0].numpy().astype(np.float64)
        load_ms = _STATE["load_ms"] if cold else 0
    return v / np.linalg.norm(v), load_ms
