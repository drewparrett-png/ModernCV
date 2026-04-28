"""Round-trip tests for `pipeline.students.registry` (Phase 1.5).

The registry is the seam every other phase plugs new architectures into,
so the contract that matters is small but load-bearing:

  • `register(name)` adds the class to `TRAINERS`.
  • `make_trainer(name)` returns an instance.
  • Unknown names raise `ValueError` with the available list in the
    message — the API layer relies on this to produce a useful 422.
  • Duplicate registrations raise at import time (loud failure beats
    silently shadowing a name).
  • `list_trainers()` returns a sorted list of every registered name.

The fixture below uses a clearly test-only name (`__test_stub__`) and
pops it from `TRAINERS` after each test so a failed assertion can't leak
state into the next test or into a subsequent pytest run that shares the
same interpreter (e.g. `pytest -k`).
"""

from __future__ import annotations

import pytest

# Importing the package guarantees the `@register("yolov8n")` etc.
# side effects have fired before any test runs.
import pipeline.students  # noqa: F401
from pipeline.students.registry import (
    TRAINERS,
    list_trainers,
    make_trainer,
    register,
)


STUB_NAME = "__test_stub__"


class _StubTrainer:
    """Minimal stand-in for the StudentTrainer protocol.

    We don't exercise train/eval/time_inference here — the registry tests
    only care that the class round-trips through `register` →
    `make_trainer`. Defining the methods would need real fixtures and
    drag in YOLO weights for no extra coverage.
    """

    name = STUB_NAME

    def __init__(self) -> None:
        self.constructed = True


@pytest.fixture
def clean_stub():
    """Register `_StubTrainer` under STUB_NAME for the duration of one
    test, then unconditionally remove it. Skipping the cleanup would
    poison subsequent tests with a duplicate-registration error or a
    bogus entry in the live `list_trainers()` output."""
    # Defensive: if a previous run aborted between register and pop,
    # clear any stale entry before we try to register fresh.
    TRAINERS.pop(STUB_NAME, None)
    register(STUB_NAME)(_StubTrainer)
    try:
        yield
    finally:
        TRAINERS.pop(STUB_NAME, None)


def test_register_and_make_trainer_roundtrip(clean_stub) -> None:
    """The happy path: a registered class is constructible via
    `make_trainer`, and the instance is the same shape we registered."""
    trainer = make_trainer(STUB_NAME)
    assert isinstance(trainer, _StubTrainer)
    assert trainer.constructed is True
    # `register` stamps the registered name onto the class so the
    # Protocol's `name: str` attribute is satisfied without each
    # subclass setting it manually.
    assert trainer.name == STUB_NAME


def test_make_trainer_unknown_name_raises_with_available_list(clean_stub) -> None:
    """Unknown architectures must raise ValueError, and the message must
    include the list of valid options — the API layer's 422 surfaces
    this directly to the user."""
    with pytest.raises(ValueError) as exc:
        make_trainer("does-not-exist")
    msg = str(exc.value)
    assert "does-not-exist" in msg
    # The stub should be in the available list; so should the real
    # yolov8n that always ships with the package.
    assert STUB_NAME in msg
    assert "yolov8n" in msg


def test_duplicate_registration_raises(clean_stub) -> None:
    """Re-registering the same name is the import-time guard — silently
    shadowing would make 'I added yolov8n in two places' a debugging
    nightmare. The error message must name the conflicting name."""
    with pytest.raises(ValueError) as exc:
        register(STUB_NAME)(_StubTrainer)
    assert STUB_NAME in str(exc.value)


def test_list_trainers_is_sorted_and_includes_registered(clean_stub) -> None:
    """`list_trainers()` is what the GUI dropdown reads; it must be
    deterministic across reloads. Sorting also makes this test stable
    against future trainers being added."""
    names = list_trainers()
    assert names == sorted(names)
    assert STUB_NAME in names
    # The real yolov8n must always be present — it's the default
    # architecture and disappearing from the list would silently break
    # the OptimizeRequest validator.
    assert "yolov8n" in names


def test_stub_does_not_leak_between_tests() -> None:
    """Sanity check: outside the `clean_stub` fixture the stub must NOT
    be registered. If this fails, our teardown is broken and other
    tests are running with a polluted TRAINERS dict."""
    assert STUB_NAME not in TRAINERS
    assert STUB_NAME not in list_trainers()
