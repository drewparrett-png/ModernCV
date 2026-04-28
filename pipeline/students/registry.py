"""Student trainer registry.

Mirrors `pipeline.models.registry` exactly — same decorator pattern, same
shape — so anyone familiar with the model-adapter side of the codebase
can read this in seconds.

Two halves:

  • `TRAINERS` — what's actually wired right now. Trainers register
    themselves at import time via `@register("yolov8n")` and the
    orchestration layer / GUI dropdown read from this dict.
  • `make_trainer(name)` — the single hook `pipeline.optimize` uses.
    Raises a clear `ValueError` for unknown architectures so the API
    layer can convert it to a 400/422 with a useful message instead of
    a `KeyError` traceback.

The registry is populated as a *side effect* of importing the trainer
modules. `pipeline.students.__init__` imports them; `pipeline.optimize`
imports `pipeline.students` so the registrations are guaranteed to have
run before `make_trainer` is called.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, TypeVar

if TYPE_CHECKING:
    from pipeline.students.base import StudentTrainer


# Architecture name → trainer class. Keys are the strings the GUI sends
# in `OptimizeRequest.architecture` and that get persisted on
# `StudentManifest.architecture`.
TRAINERS: dict[str, type["StudentTrainer"]] = {}


T = TypeVar("T")


def register(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator: associate a `StudentTrainer` class with a name.

    Duplicate registrations raise — silently shadowing a name would make
    the bug "I added yolov8n in two places and the wrong one wins" a
    pain to track down. Loud failure at import time is the right move.
    """

    def deco(cls: type[T]) -> type[T]:
        if name in TRAINERS:
            raise ValueError(
                f"student trainer already registered for {name!r}: "
                f"{TRAINERS[name]} (new: {cls})"
            )
        # Stamp the registered name onto the class so instances satisfy the
        # `StudentTrainer` Protocol's `name: str` attribute without each
        # subclass having to remember to set it manually.
        cls.name = name  # type: ignore[attr-defined]
        TRAINERS[name] = cls  # type: ignore[assignment]
        return cls

    return deco


def make_trainer(
    name: str, params: Optional[dict] = None
) -> "StudentTrainer":
    """Instantiate the registered trainer for `name`.

    Unknown names raise `ValueError` with the list of available
    architectures so the API layer's 400/422 message is immediately
    useful — the user sees "valid options are: yolov8n, yolov8s, yolov8m"
    rather than an opaque KeyError.

    `params` is forwarded to the trainer's constructor; trainers that
    don't take any (the common case for Phase 1) ignore it.
    """
    cls = TRAINERS.get(name)
    if cls is None:
        available = sorted(TRAINERS.keys())
        raise ValueError(
            f"unknown student architecture {name!r} — "
            f"available: {available or '<none registered>'}"
        )
    if params:
        return cls(**params)  # type: ignore[call-arg]
    return cls()  # type: ignore[call-arg]


def list_trainers() -> list[str]:
    """Names of every registered trainer, sorted alphabetically.

    Used by the `GET /students/architectures` endpoint and by the
    `OptimizeRequest` validator to reject unknown names at request time.
    Sorting is purely cosmetic — it stabilises the GUI dropdown order
    so the same impl always lives in the same place.
    """
    return sorted(TRAINERS.keys())
