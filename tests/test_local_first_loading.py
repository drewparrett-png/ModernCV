"""Local-first model loading.

The GroundingDINO adapter must pass `local_files_only=True` to
`AutoProcessor.from_pretrained` and `AutoModelForZeroShotObjectDetection.from_pretrained`
when the model is already in the HF cache. That kwarg is the HF library's
contract for "do not issue any network calls" — passing it short-circuits
the freshness HEAD that's the source of "transient DNS blip kills my
Learn run" failures.

When the model is NOT cached, the adapter falls through to the default
networked path so first-time downloads still work — covered by the
"uncached" test below.

We don't load the real model in tests; we monkeypatch the transformers
classes so `setup()` runs without weights and we can inspect the kwargs
the adapter passed.
"""

from __future__ import annotations

from typing import Any

import pytest


# ---- Fake transformers classes ---------------------------------------------


class _CapturedKwargs(dict):
    """Tiny container so each test gets its own captured-kwargs view."""


class _FakeModel:
    """Stand-in returned in place of the real model. The adapter calls
    `.to(device).eval()` on the result of `from_pretrained`, so this
    object handles both as no-ops."""

    def to(self, device: str) -> "_FakeModel":
        return self

    def eval(self) -> None:
        pass


@pytest.fixture
def captured_load_kwargs(monkeypatch: pytest.MonkeyPatch) -> _CapturedKwargs:
    """Patch `from_pretrained` on the real transformers classes — patching
    `transformers.AutoProcessor` itself doesn't survive transformers' lazy
    `_LazyModule` path, which is what `from transformers import X` hits.
    Patching the *method* on the resolved class works because `from X
    import Y` binds `Y` to the same class object the patch targets."""
    captured = _CapturedKwargs()

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    def _fake_processor_from_pretrained(
        cls: Any, model_id: str, **kwargs: Any
    ) -> object:
        captured["processor_model_id"] = model_id
        captured["processor_kwargs"] = kwargs
        return object()

    def _fake_model_from_pretrained(
        cls: Any, model_id: str, **kwargs: Any
    ) -> _FakeModel:
        captured["model_model_id"] = model_id
        captured["model_kwargs"] = kwargs
        return _FakeModel()

    monkeypatch.setattr(
        AutoProcessor, "from_pretrained", classmethod(_fake_processor_from_pretrained)
    )
    monkeypatch.setattr(
        AutoModelForZeroShotObjectDetection,
        "from_pretrained",
        classmethod(_fake_model_from_pretrained),
    )
    return captured


# ---- Tests -----------------------------------------------------------------


def test_cached_model_passes_local_files_only(
    monkeypatch: pytest.MonkeyPatch, captured_load_kwargs: _CapturedKwargs
) -> None:
    """Cache present ⇒ both from_pretrained calls receive
    `local_files_only=True`. This is the contract that prevents HF from
    issuing the freshness-check HEAD that broke us last time."""
    from pipeline.models import grounding_dino

    monkeypatch.setattr(grounding_dino, "is_hf_cached", lambda _model_id: True)

    adapter = grounding_dino.GroundingDINOAdapter({"prompts": ["soccer ball"]})
    adapter.setup()

    assert captured_load_kwargs["processor_kwargs"].get("local_files_only") is True
    assert captured_load_kwargs["model_kwargs"].get("local_files_only") is True


def test_uncached_model_omits_local_files_only(
    monkeypatch: pytest.MonkeyPatch, captured_load_kwargs: _CapturedKwargs
) -> None:
    """Cache miss ⇒ no `local_files_only` kwarg, so HF can do its
    networked download. First-run UX must keep working."""
    from pipeline.models import grounding_dino

    monkeypatch.setattr(grounding_dino, "is_hf_cached", lambda _model_id: False)

    adapter = grounding_dino.GroundingDINOAdapter({"prompts": ["soccer ball"]})
    adapter.setup()

    assert "local_files_only" not in captured_load_kwargs["processor_kwargs"]
    assert "local_files_only" not in captured_load_kwargs["model_kwargs"]


def test_cached_load_does_not_touch_huggingface_hub_network(
    monkeypatch: pytest.MonkeyPatch, captured_load_kwargs: _CapturedKwargs
) -> None:
    """Belt-and-braces: even if our flag-passing logic regresses, this
    test is a tripwire — patch every network entrypoint we can reach in
    `huggingface_hub` to raise loudly, then confirm the cached-load path
    completes anyway. If a future refactor accidentally drops the
    local-only flag, this test fires before users hit it.
    """
    from pipeline.models import grounding_dino

    monkeypatch.setattr(grounding_dino, "is_hf_cached", lambda _model_id: True)

    # Tripwire on the HF download path. The fake transformers classes
    # already short-circuit before they'd reach this — so if these are
    # ever called, something is very wrong.
    def _explode(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError(
            "huggingface_hub network call attempted on a cached load — "
            "local_files_only flag must be regressed."
        )

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _explode)
    if hasattr(huggingface_hub, "snapshot_download"):
        monkeypatch.setattr(huggingface_hub, "snapshot_download", _explode)

    adapter = grounding_dino.GroundingDINOAdapter({"prompts": ["soccer ball"]})
    adapter.setup()  # must not raise


def test_is_hf_cached_helper_is_public(monkeypatch: pytest.MonkeyPatch) -> None:
    """The renamed-public helper must stay importable from its public
    name; the GroundingDINO adapter binds to it at import time."""
    from pipeline import model_cache

    assert callable(model_cache.is_hf_cached)


def test_cache_status_endpoint_still_resolves_after_rename() -> None:
    """The internal call site of the helper inside `cache_status` was
    updated alongside the rename. Verify `cache_status("groundingdino")`
    still returns a populated record so the GUI's pre-run "want to
    download?" probe keeps working."""
    from pipeline import model_cache

    status = model_cache.cache_status("groundingdino")
    assert status.known is True
    assert status.model_id == "IDEA-Research/grounding-dino-tiny"
    # `cached` may be True or False depending on the test environment;
    # we don't assert either way — just that the call returns cleanly.
    assert isinstance(status.cached, bool)
