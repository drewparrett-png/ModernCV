"""Validation tests for `server.schemas.OptimizeRequest` (Phase 0.6).

The threshold rule (`t_low <= t_high`) needs to fire at the schema layer
so the GUI gets a 422 before any worker thread is spawned. Catching it
later (in `prepare_yolo_dataset` for instance) would still raise, but
the user sees an opaque "Student failed" row instead of an inline
form error.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from server.schemas import OptimizeRequest


def test_defaults_are_the_spec_values() -> None:
    """The defaults are documented in `docs/student-training.md`. If we
    ever bump them, this test forces an explicit decision rather than a
    silent drift in serialised payloads."""
    req = OptimizeRequest(train_teacher_ids=["t1"])
    assert req.t_high == 0.35
    assert req.t_low == 0.15
    assert req.treat_empty_as_negative is False


def test_t_low_equal_t_high_is_allowed() -> None:
    """The boundary case — equal thresholds collapse the uncertain band
    to nothing, which is a valid (if unusual) configuration. Accept it
    rather than forcing a strict-less-than."""
    req = OptimizeRequest(
        train_teacher_ids=["t1"], t_high=0.4, t_low=0.4,
    )
    assert req.t_high == 0.4
    assert req.t_low == 0.4


def test_t_low_above_t_high_rejected() -> None:
    """Inverted thresholds → ValidationError. The error message must
    name both fields so the GUI can highlight them."""
    with pytest.raises(ValidationError) as exc:
        OptimizeRequest(
            train_teacher_ids=["t1"], t_high=0.10, t_low=0.50,
        )
    msg = str(exc.value)
    assert "t_low" in msg and "t_high" in msg


def test_t_low_just_above_t_high_rejected() -> None:
    """Fence-post — even a tiny margin should fail."""
    with pytest.raises(ValidationError):
        OptimizeRequest(
            train_teacher_ids=["t1"], t_high=0.35, t_low=0.36,
        )


def test_treat_empty_as_negative_explicit_true() -> None:
    """The escape-hatch flag must round-trip cleanly — flipping it on
    is the user's only path to reproducing the old training set, so we
    care that the schema doesn't drop it."""
    req = OptimizeRequest(
        train_teacher_ids=["t1"], treat_empty_as_negative=True,
    )
    assert req.treat_empty_as_negative is True
