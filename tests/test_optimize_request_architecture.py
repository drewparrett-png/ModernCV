"""Architecture-validation tests for `OptimizeRequest` (Phase 1.5).

The schema's `_architecture_is_registered` validator (added in 1.3) is
the seam that turns "user typed a typo" into an inline 422 instead of a
"Student failed" row showing up a few seconds later. The contract:

  • `architecture="yolov8n"` round-trips with the explicit value preserved.
  • Omitted defaults to `"yolov8n"` — the only architecture that existed
    before the dispatcher refactor.
  • Unknown architectures raise `ValidationError`, and the message lists
    the valid options (so the GUI/API can surface a useful error).

The test imports `pipeline.students` first so the `@register(...)` side
effects have fired by the time the validator runs `list_trainers()`. If
that import line is removed, the validator runs against an empty
registry and every architecture (including `yolov8n`) gets rejected —
the assertion at the bottom guards against that regression too.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

# IMPORTANT: importing the students package fires the trainer modules'
# `@register(...)` side effects, populating `TRAINERS` before the
# OptimizeRequest validator queries it. Without this import the
# validator's `list_trainers()` call returns [] and even valid names
# get rejected.
import pipeline.students  # noqa: F401
from pipeline.students.registry import list_trainers
from server.schemas import OptimizeRequest


def test_explicit_yolov8n_roundtrips() -> None:
    """The default architecture passes validation when sent explicitly
    and survives serialisation unchanged."""
    req = OptimizeRequest(
        train_teacher_ids=["t1"], architecture="yolov8n",
    )
    assert req.architecture == "yolov8n"


def test_omitted_architecture_defaults_to_yolov8n() -> None:
    """An OptimizeRequest with no architecture key must default to
    yolov8n — keeps old GUI builds (which don't send the field)
    producing the same Student as before the dispatcher landed."""
    req = OptimizeRequest(train_teacher_ids=["t1"])
    assert req.architecture == "yolov8n"


def test_unknown_architecture_rejected_at_validation_time() -> None:
    """Bogus architecture names must 422 at the schema, not bubble up
    later from `make_trainer`. The message must list the valid options
    so the user can self-correct without reading the source."""
    with pytest.raises(ValidationError) as exc:
        OptimizeRequest(
            train_teacher_ids=["t1"], architecture="not-a-thing",
        )
    msg = str(exc.value)
    assert "not-a-thing" in msg
    # The real yolov8n must appear in the available list rendered in
    # the error — that's the user's hint about what they could have
    # typed instead. If this assertion fails, either the validator
    # stopped including the available list, or the registry is empty
    # (the import-side-effect contract is broken).
    assert "yolov8n" in msg


def test_validator_reads_registry_live() -> None:
    """Sanity check on the import-time contract: by the time this test
    file is imported, `pipeline.students` has been imported (above),
    so the registry must be non-empty. If this fails the rest of the
    architecture tests are meaningless."""
    names = list_trainers()
    assert "yolov8n" in names
    assert len(names) > 0


def test_other_registered_yolo_sizes_pass() -> None:
    """The Phase 1.2 trainer module registers `yolov8n` / `yolov8s` /
    `yolov8m`. Any of them must validate cleanly — confirms the
    validator isn't accidentally pinned to the default name."""
    for name in ["yolov8s", "yolov8m"]:
        if name in list_trainers():
            req = OptimizeRequest(
                train_teacher_ids=["t1"], architecture=name,
            )
            assert req.architecture == name
