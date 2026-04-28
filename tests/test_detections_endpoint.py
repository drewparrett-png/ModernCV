"""Phase 4: crop-flip review — detections list + detection_crop coverage.

`iter_detection_rows` and the `/detections` endpoint flatten a run's
per-frame.jsonl into one ordered list, attach an `accepted` flag derived
from `frame_states.json`, and sort score-asc by default. The crop endpoint
caches a single (frame, det, pad) JPEG on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pipeline import runs as runs_mod
from server.main import app


# ---- Fixtures --------------------------------------------------------------


@pytest.fixture
def runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "runs"
    root.mkdir()
    return root


@pytest.fixture
def project(runs_root: Path) -> runs_mod.Project:
    return runs_mod.create_project(
        name="Soccer ball test",
        task="detection",
        prompts=["soccer ball"],
    )


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _seed_per_frame_with_dets(
    rdir: Path,
    frames: dict[int, list[dict]],
) -> None:
    """Write `per_frame.jsonl` from a `{frame_idx: [det_dict, ...]}` map.

    Each det dict can override `score`, `class_name`, `class_id`, `bbox_xyxy`.
    Defaults give the test a stable distribution to reason about ordering.
    """
    pf_path = rdir / runs_mod.LABELS_DIR / runs_mod.PER_FRAME_NAME
    pf_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for fi, dets in frames.items():
        rec = {
            "frame_idx": fi,
            "detections": [
                {
                    "score": d.get("score", 0.5),
                    "class_name": d.get("class_name", "x"),
                    "class_id": d.get("class_id", 0),
                    "bbox_xyxy": d.get("bbox_xyxy", [10.0, 10.0, 50.0, 50.0]),
                }
                for d in dets
            ],
        }
        lines.append(json.dumps(rec))
    pf_path.write_text("\n".join(lines) + "\n")


def _make_run_with_dets(
    project_id: str,
    frames: dict[int, list[dict]],
) -> Path:
    rdir, _ = runs_mod.create_run(
        project_id=project_id,
        task="detection",
        prompt="t",
        video_path="data/x.mp4",
        models={"detect": "groundingdino"},
    )
    _seed_per_frame_with_dets(rdir, frames)
    runs_mod.mark_completed(rdir)
    return rdir


# ---- iter_detection_rows --------------------------------------------------


def test_iter_detection_rows_flattens_in_order(
    project: runs_mod.Project,
) -> None:
    rdir = _make_run_with_dets(
        project.id,
        {
            0: [{"score": 0.9, "class_name": "ball"}],
            1: [
                {"score": 0.2, "class_name": "ball"},
                {"score": 0.7, "class_name": "ball"},
            ],
            2: [],
        },
    )
    rows = runs_mod.iter_detection_rows(rdir)
    # Natural order: frame asc, det_idx asc. Empty frames contribute nothing.
    assert [(r.frame_idx, r.det_idx) for r in rows] == [(0, 0), (1, 0), (1, 1)]
    assert [r.score for r in rows] == [0.9, 0.2, 0.7]
    assert all(r.accepted for r in rows)


def test_iter_detection_rows_marks_curated_rejections(
    project: runs_mod.Project,
) -> None:
    rdir = _make_run_with_dets(
        project.id,
        {0: [{"score": 0.9}, {"score": 0.4}, {"score": 0.1}]},
    )
    runs_mod.set_frame_state(rdir, 0, "curated", rejected_dets=[1])
    rows = runs_mod.iter_detection_rows(rdir)
    accepted = {r.det_idx: r.accepted for r in rows}
    assert accepted == {0: True, 1: False, 2: True}


def test_iter_detection_rows_ignores_whole_frame_states(
    project: runs_mod.Project,
) -> None:
    """`confirmed_empty` and `marked_missed` are whole-frame verdicts and
    don't change per-detection accept state — that's the inspector pills'
    job, not crop-flip's."""
    rdir = _make_run_with_dets(
        project.id,
        {0: [{"score": 0.9}], 1: [{"score": 0.5}]},
    )
    runs_mod.set_frame_state(rdir, 0, "confirmed_empty")
    runs_mod.set_frame_state(rdir, 1, "marked_missed")
    rows = runs_mod.iter_detection_rows(rdir)
    assert all(r.accepted for r in rows)


# ---- GET /detections ------------------------------------------------------


def test_get_detections_score_asc_default(
    project: runs_mod.Project, client: TestClient
) -> None:
    rdir = _make_run_with_dets(
        project.id,
        {
            0: [{"score": 0.9, "class_name": "a"}, {"score": 0.1, "class_name": "b"}],
            1: [{"score": 0.5, "class_name": "c"}],
        },
    )
    res = client.get(f"/projects/{project.id}/runs/{rdir.name}/detections")
    assert res.status_code == 200
    body = res.json()
    assert body["total"] == 3
    scores = [d["score"] for d in body["detections"]]
    assert scores == sorted(scores)
    assert scores[0] == pytest.approx(0.1)


def test_get_detections_carries_accepted_flag(
    project: runs_mod.Project, client: TestClient
) -> None:
    rdir = _make_run_with_dets(
        project.id, {0: [{"score": 0.9}, {"score": 0.5}]}
    )
    runs_mod.set_frame_state(rdir, 0, "curated", rejected_dets=[0])
    body = client.get(
        f"/projects/{project.id}/runs/{rdir.name}/detections"
    ).json()
    by_det = {(d["frame_idx"], d["det_idx"]): d["accepted"] for d in body["detections"]}
    assert by_det[(0, 0)] is False
    assert by_det[(0, 1)] is True


def test_get_detections_pagination(
    project: runs_mod.Project, client: TestClient
) -> None:
    # 5 detections, all distinct scores so the asc order is deterministic.
    rdir = _make_run_with_dets(
        project.id,
        {0: [{"score": 0.05 + 0.1 * i} for i in range(5)]},
    )
    body = client.get(
        f"/projects/{project.id}/runs/{rdir.name}/detections?limit=2&offset=1"
    ).json()
    assert body["total"] == 5
    assert len(body["detections"]) == 2
    # Skipped the lowest (0.05); next two are 0.15 and 0.25.
    scores = [d["score"] for d in body["detections"]]
    assert scores[0] == pytest.approx(0.15)
    assert scores[1] == pytest.approx(0.25)


def test_get_detections_404_for_missing_run(
    project: runs_mod.Project, client: TestClient
) -> None:
    res = client.get(
        f"/projects/{project.id}/runs/teacher_does_not_exist/detections"
    )
    assert res.status_code == 404


def test_get_detections_empty_run(
    project: runs_mod.Project, client: TestClient
) -> None:
    rdir = _make_run_with_dets(project.id, {0: [], 1: []})
    body = client.get(
        f"/projects/{project.id}/runs/{rdir.name}/detections"
    ).json()
    assert body == {"detections": [], "total": 0}


# ---- GET /detection_crop --------------------------------------------------


def _write_test_video(path: Path, n_frames: int = 3, w: int = 96, h: int = 64) -> None:
    """Render a tiny mp4 the crop endpoint can seek into.

    Each frame is a flat color so we can easily verify the green outline is
    drawn on top. Uses mp4v which OpenCV ships out of the box on macOS.
    """
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (w, h))
    try:
        for i in range(n_frames):
            frame = np.full((h, w, 3), fill_value=20 * i + 30, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_detection_crop_returns_jpeg_and_caches(
    project: runs_mod.Project, client: TestClient, tmp_path: Path
) -> None:
    video = tmp_path / "x.mp4"
    _write_test_video(video, n_frames=3)
    rdir, _ = runs_mod.create_run(
        project_id=project.id,
        task="detection",
        prompt="t",
        video_path=str(video),
        models={"detect": "groundingdino"},
    )
    _seed_per_frame_with_dets(
        rdir,
        {
            0: [
                {
                    "score": 0.9,
                    "class_name": "ball",
                    "bbox_xyxy": [20.0, 20.0, 40.0, 40.0],
                }
            ]
        },
    )
    runs_mod.mark_completed(rdir)

    url = (
        f"/projects/{project.id}/runs/{rdir.name}/detection_crop/0/0.jpg?pad=8"
    )
    res = client.get(url)
    assert res.status_code == 200, res.text
    assert res.headers["content-type"] == "image/jpeg"
    assert len(res.content) > 0

    cached = runs_mod.detection_crop_path(rdir, 0, 0, pad=8)
    assert cached.exists()
    assert cached.read_bytes() == res.content

    # Second hit serves from cache (same bytes, no error).
    res2 = client.get(url)
    assert res2.status_code == 200
    assert res2.content == res.content


def test_detection_crop_pad_changes_cache_path(
    project: runs_mod.Project, client: TestClient, tmp_path: Path
) -> None:
    video = tmp_path / "x.mp4"
    _write_test_video(video)
    rdir, _ = runs_mod.create_run(
        project_id=project.id,
        task="detection",
        prompt="t",
        video_path=str(video),
        models={"detect": "groundingdino"},
    )
    _seed_per_frame_with_dets(
        rdir,
        {
            0: [
                {
                    "score": 0.9,
                    "bbox_xyxy": [20.0, 20.0, 40.0, 40.0],
                }
            ]
        },
    )
    runs_mod.mark_completed(rdir)

    base = f"/projects/{project.id}/runs/{rdir.name}/detection_crop/0/0.jpg"
    client.get(f"{base}?pad=4")
    client.get(f"{base}?pad=24")
    crops = sorted(p.name for p in runs_mod.crops_dir(rdir).iterdir())
    assert crops == ["0_0_p24.jpg", "0_0_p4.jpg"]


def test_detection_crop_404_for_missing_detection(
    project: runs_mod.Project, client: TestClient, tmp_path: Path
) -> None:
    video = tmp_path / "x.mp4"
    _write_test_video(video)
    rdir, _ = runs_mod.create_run(
        project_id=project.id,
        task="detection",
        prompt="t",
        video_path=str(video),
        models={"detect": "groundingdino"},
    )
    _seed_per_frame_with_dets(rdir, {0: [{"score": 0.9}]})
    runs_mod.mark_completed(rdir)

    res = client.get(
        f"/projects/{project.id}/runs/{rdir.name}/detection_crop/0/99.jpg"
    )
    assert res.status_code == 404


def test_detection_crop_404_for_missing_run(
    project: runs_mod.Project, client: TestClient
) -> None:
    res = client.get(
        f"/projects/{project.id}/runs/nope/detection_crop/0/0.jpg"
    )
    assert res.status_code == 404
