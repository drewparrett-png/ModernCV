"""Tests for the `POST /students/preview-buckets` endpoint (Phase 0.4).

The endpoint is *the* live-preview surface for the New Student form: every
checkbox toggle and threshold tweak in the GUI debounces into a call here,
and the user trusts the numbers it returns to predict what the trainer
will see. Bugs that hurt:

  • Aggregate counts not matching the per-teacher rows (the user catches this
    visually — the per-teacher numbers should sum to the aggregate).
  • `treat_empty_as_negative=True` not collapsing uncertain → true_negative
    in the *response* (the trainer would do the right thing, but the preview
    would lie).
  • t_low > t_high accepted silently (would either crash the
    classify_frames helper or return inverted buckets).

The endpoint reads each teacher's `coco.json` from disk via
`pipeline.runs.run_dir(...)`; we lay out a couple of fake teacher dirs in a
tmp tree and point `RUNS_DIR` at it via monkeypatch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pipeline import runs as runs_mod
from server import main as server_main
from server.main import app


# ---- Fixtures -------------------------------------------------------------


def _write_teacher(
    runs_root: Path,
    teacher_id: str,
    *,
    images: list[dict],
    annotations: list[dict],
    categories: list[dict] | None = None,
) -> None:
    """Lay out a fake completed Teacher dir with a coco.json under it."""
    tdir = runs_root / teacher_id
    (tdir / runs_mod.LABELS_DIR).mkdir(parents=True, exist_ok=True)
    manifest = runs_mod.RunManifest(
        id=teacher_id,
        task="detection",
        prompt="test",
        video_path="/fake/video.mp4",
        started_at="2026-04-28T12:00:00Z",
        ended_at="2026-04-28T12:01:00Z",
        status="completed",
        models={"detect": "groundingdino"},
    )
    (tdir / runs_mod.MANIFEST_NAME).write_text(manifest.to_json())
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories
        or [{"id": 0, "name": "ball"}, {"id": 1, "name": "player"}],
    }
    (tdir / runs_mod.LABELS_DIR / runs_mod.COCO_NAME).write_text(json.dumps(coco))


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient bound to a `runs/` dir under tmp_path.

    `pipeline.runs.run_dir(tid)` defaults `runs_root=RUNS_DIR`, where
    `RUNS_DIR = Path("runs")` is relative — so `chdir`-ing into tmp_path
    makes `Path("runs")` resolve under it. Cleaner than monkeypatching the
    constant (which wouldn't update the captured default arg anyway).
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    return TestClient(app)


def _coco_classifiable(*, n_positive: int, n_uncertain: int, n_true_negative: int) -> tuple[list[dict], list[dict]]:
    """Build a (images, annotations) pair that bucket exactly as named at
    the default thresholds (t_high=0.35, t_low=0.15).

    Frame ids are unique across the three buckets so each call produces a
    self-consistent COCO chunk. Score values:
      • positive      → 0.9 (well above t_high)
      • uncertain     → 0.20 (in [t_low, t_high))
      • true_negative → no annotations
    """
    images: list[dict] = []
    anns: list[dict] = []
    nxt_frame = 0
    nxt_ann = 0
    for _ in range(n_positive):
        images.append({"id": nxt_frame, "width": 100, "height": 100})
        anns.append({"id": nxt_ann, "image_id": nxt_frame, "category_id": 0,
                     "bbox": [0, 0, 10, 10], "score": 0.9})
        nxt_frame += 1
        nxt_ann += 1
    for _ in range(n_uncertain):
        images.append({"id": nxt_frame, "width": 100, "height": 100})
        anns.append({"id": nxt_ann, "image_id": nxt_frame, "category_id": 1,
                     "bbox": [0, 0, 10, 10], "score": 0.20})
        nxt_frame += 1
        nxt_ann += 1
    for _ in range(n_true_negative):
        images.append({"id": nxt_frame, "width": 100, "height": 100})
        nxt_frame += 1
    return images, anns


# ---- Tests ----------------------------------------------------------------


def test_single_teacher_strict_mode(client: TestClient, tmp_path: Path) -> None:
    """One teacher, default thresholds, strict mode — bucket counts come back
    matching what `classify_frames` would compute on the same COCO."""
    images, anns = _coco_classifiable(n_positive=3, n_uncertain=2, n_true_negative=1)
    _write_teacher(tmp_path / "runs", "teacher_a", images=images, annotations=anns)

    res = client.post(
        "/students/preview-buckets",
        json={"teacher_ids": ["teacher_a"]},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["aggregate"] == {
        "positive": 3,
        "uncertain": 2,
        "true_negative": 1,
        "n_classes": 2,
        "class_names": ["ball", "player"],
    }
    assert body["per_teacher"] == [
        {"teacher_id": "teacher_a", "positive": 3, "uncertain": 2, "true_negative": 1}
    ]


def test_aggregation_across_teachers(client: TestClient, tmp_path: Path) -> None:
    """Two teachers — aggregate must equal the per-teacher sum, and the
    class-name union dedupes overlapping vocabularies."""
    a_imgs, a_anns = _coco_classifiable(n_positive=2, n_uncertain=1, n_true_negative=0)
    b_imgs, b_anns = _coco_classifiable(n_positive=1, n_uncertain=2, n_true_negative=3)
    _write_teacher(
        tmp_path / "runs", "teacher_a",
        images=a_imgs, annotations=a_anns,
        categories=[{"id": 0, "name": "ball"}, {"id": 1, "name": "player"}],
    )
    _write_teacher(
        tmp_path / "runs", "teacher_b",
        images=b_imgs, annotations=b_anns,
        # Overlap on "ball", new "ref" class — verify dedup + ordering.
        categories=[{"id": 0, "name": "ball"}, {"id": 1, "name": "ref"}],
    )

    res = client.post(
        "/students/preview-buckets",
        json={"teacher_ids": ["teacher_a", "teacher_b"]},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["aggregate"]["positive"] == 3
    assert body["aggregate"]["uncertain"] == 3
    assert body["aggregate"]["true_negative"] == 3
    # Class union: ball appears once (first), player from a, ref from b.
    assert body["aggregate"]["class_names"] == ["ball", "player", "ref"]
    assert body["aggregate"]["n_classes"] == 3
    # Per-teacher rows preserved in input order.
    assert [t["teacher_id"] for t in body["per_teacher"]] == [
        "teacher_a", "teacher_b",
    ]


def test_treat_empty_as_negative_reclassifies_in_response(
    client: TestClient, tmp_path: Path,
) -> None:
    """With `treat_empty_as_negative=True`, the response must collapse
    uncertain→true_negative — not just at the trainer."""
    images, anns = _coco_classifiable(n_positive=2, n_uncertain=4, n_true_negative=1)
    _write_teacher(tmp_path / "runs", "teacher_a", images=images, annotations=anns)

    res = client.post(
        "/students/preview-buckets",
        json={"teacher_ids": ["teacher_a"], "treat_empty_as_negative": True},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    # Strict mode would have shown 2/4/1; escape-hatch shows 2/0/5.
    assert body["aggregate"]["positive"] == 2
    assert body["aggregate"]["uncertain"] == 0
    assert body["aggregate"]["true_negative"] == 5
    assert body["per_teacher"][0] == {
        "teacher_id": "teacher_a",
        "positive": 2,
        "uncertain": 0,
        "true_negative": 5,
    }


def test_custom_thresholds_change_buckets(client: TestClient, tmp_path: Path) -> None:
    """Loosening t_high should let the 0.20-scored anns become positive
    instead of uncertain — confirms the request thresholds reach
    classify_frames untouched."""
    images, anns = _coco_classifiable(n_positive=1, n_uncertain=3, n_true_negative=0)
    _write_teacher(tmp_path / "runs", "teacher_a", images=images, annotations=anns)

    # Drop t_high below 0.20 → previously-uncertain frames are now positive.
    res = client.post(
        "/students/preview-buckets",
        json={"teacher_ids": ["teacher_a"], "t_high": 0.10, "t_low": 0.05},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["aggregate"]["positive"] == 4
    assert body["aggregate"]["uncertain"] == 0
    assert body["aggregate"]["true_negative"] == 0


def test_t_low_above_t_high_rejected(client: TestClient, tmp_path: Path) -> None:
    """Pydantic validator on PreviewBucketsRequest fires before any
    teacher I/O happens — should be a 422 with t_low/t_high in the message."""
    res = client.post(
        "/students/preview-buckets",
        json={"teacher_ids": [], "t_high": 0.10, "t_low": 0.50},
    )
    assert res.status_code == 422
    detail = json.dumps(res.json())
    assert "t_low" in detail and "t_high" in detail


def test_missing_teacher_returns_404(client: TestClient, tmp_path: Path) -> None:
    """Caller passes a teacher_id that doesn't exist — surface as 404
    with the offending id in the body."""
    res = client.post(
        "/students/preview-buckets",
        json={"teacher_ids": ["does-not-exist"]},
    )
    assert res.status_code == 404
    assert "does-not-exist" in res.text


def test_teacher_without_coco_returns_400(
    client: TestClient, tmp_path: Path,
) -> None:
    """Teacher dir exists but Learn run never produced a coco.json (e.g.
    aborted run) — surface as 400, not 500."""
    runs_root = tmp_path / "runs"
    tdir = runs_root / "teacher_no_coco"
    (tdir / runs_mod.LABELS_DIR).mkdir(parents=True)
    manifest = runs_mod.RunManifest(
        id="teacher_no_coco",
        task="detection",
        prompt="test",
        video_path="/fake/video.mp4",
        started_at="2026-04-28T12:00:00Z",
        status="running",  # never finished
    )
    (tdir / runs_mod.MANIFEST_NAME).write_text(manifest.to_json())

    res = client.post(
        "/students/preview-buckets",
        json={"teacher_ids": ["teacher_no_coco"]},
    )
    assert res.status_code == 400
    assert "coco.json" in res.text


def test_empty_teacher_list_returns_zero_aggregate(client: TestClient) -> None:
    """No teachers selected → all-zero aggregate, empty per_teacher.
    The GUI uses this as the initial state before the user picks anything."""
    res = client.post(
        "/students/preview-buckets",
        json={"teacher_ids": []},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["aggregate"]["positive"] == 0
    assert body["aggregate"]["uncertain"] == 0
    assert body["aggregate"]["true_negative"] == 0
    assert body["aggregate"]["n_classes"] == 0
    assert body["aggregate"]["class_names"] == []
    assert body["per_teacher"] == []
