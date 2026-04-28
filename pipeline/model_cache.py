"""Model-weight cache awareness.

Several adapters lazily download weights on first use (HuggingFace
`from_pretrained`, Ultralytics auto-download, etc.). For UX we want to
distinguish "cached, fast load" from "first-time, multi-hundred-MB
download" before the user kicks off a run — both for an accurate progress
message and for an explicit consent prompt.

This module is intentionally narrow: per-impl metadata + a cache-status
probe that doesn't require loading the model. As we wire more adapters,
add their metadata to KNOWN_MODELS so the cache-status endpoint reports
on them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelInfo:
    """Per-impl info used by the cache-status endpoint and progress UI."""

    impl: str
    model_id: str  # huggingface repo id (or other registry key)
    estimated_bytes: int  # rough size of the lazily-fetched assets
    backend: str = "huggingface"  # which probe to use


# Add new entries as adapters land. `estimated_bytes` is approximate — the
# exact size isn't worth tracking precisely; this is just to surface "this
# is a chunky download" to the user before we kick off.
KNOWN_MODELS: dict[str, ModelInfo] = {
    "groundingdino": ModelInfo(
        impl="groundingdino",
        model_id="IDEA-Research/grounding-dino-tiny",
        estimated_bytes=700_000_000,
        backend="huggingface",
    ),
    # Future:
    # "sam2-tiny": ModelInfo(... model_id="facebook/sam2-hiera-tiny", ...)
    # "dinov3-vits16": ModelInfo(... model_id="facebook/dinov3-vits16", ...)
    # "yolov8n": ModelInfo(... ultralytics, backend="ultralytics", ...)
}


def _hf_is_cached(model_id: str) -> bool:
    """Probe the HuggingFace cache without loading the model.

    Asks for a small, always-present file (config.json) and reports True
    iff that file is on disk. `try_to_load_from_cache` returns:
        * an `str` path  — file is cached
        * `None`         — never seen this file
        * a sentinel obj — file is *known not to exist* in the repo
    We treat only the first case as cached. Any error (ImportError,
    network, etc.) is treated as 'not cached' so we err toward asking the
    user before a download.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception as e:  # pragma: no cover — should never fire
        log.warning("huggingface_hub not importable: %s", e)
        return False
    try:
        path = try_to_load_from_cache(model_id, "config.json")
    except Exception as e:
        log.warning("try_to_load_from_cache(%s) raised: %s", model_id, e)
        return False
    return isinstance(path, str) and bool(path)


@dataclass
class CacheStatus:
    impl: str
    known: bool
    cached: bool
    estimated_bytes: int = 0
    model_id: Optional[str] = None


def cache_status(impl: str) -> CacheStatus:
    """Return cache info for one impl. Unknown impls are treated as
    'not requiring a download' (cached=True, bytes=0) — they're either
    pure-Python (ByteTrack) or ship-with-Ultralytics or otherwise
    non-blocking from the user's perspective.
    """
    info = KNOWN_MODELS.get(impl)
    if info is None:
        return CacheStatus(impl=impl, known=False, cached=True, estimated_bytes=0)

    if info.backend == "huggingface":
        cached = _hf_is_cached(info.model_id)
    else:  # pragma: no cover — future backends
        cached = True

    return CacheStatus(
        impl=info.impl,
        known=True,
        cached=cached,
        estimated_bytes=info.estimated_bytes,
        model_id=info.model_id,
    )
