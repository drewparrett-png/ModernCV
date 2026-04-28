# Student Training — Implementation Spec

Two streams of work, sequenced so that the multi-architecture comparison
sits on top of clean, honest training data rather than baking in a known
pseudo-label pathology.

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Confidence-band filtering & visibility | not started |
| 1 | Trainer dispatch refactor (no behaviour change) | not started |
| 2 | RT-DETR as second architecture | not started |
| 3 | Compare view in the GUI | not started |
| 4 | DINOv3 + detection head (needs sub-spec before handoff) | not started |
| 5 | Label-efficiency sweep | not started |

Each phase is intended to be **one PR** (Phase 0 may split into two —
backend then GUI — at the implementer's discretion).

Defaults that this spec commits to (override in PR if better numbers
land in testing):
- `t_high = 0.35`, `t_low = 0.15`
- Default architecture: `yolov8n`
- Default `imgsz = 640`, default `epochs = 50`
- Compare view: new tab inside Optimize mode (not a separate top-level mode)

---

## Phase 0 — Confidence-band filtering & visibility

**Why first.** Every architecture comparison inherits the same training
data. Today, every frame the teacher returns no detections on becomes a
"true negative" training example (`pipeline/distill.py:285-308`,
explicit comment). With a high teacher threshold this propagates false
negatives into the student, training it to suppress detections it
*should* be making. That noise corrupts the inter-architecture signal
the rest of this spec is trying to surface. Fix the data before
measuring models on it.

### 0.1 Plumb confidence scores through `_coco_to_yolo_lines`

File: `pipeline/distill.py`

`runs.py:572` already serialises `det.score` into each COCO annotation.
The distill pipeline currently ignores it.

Change `_coco_to_yolo_lines()`:
- Add a `min_score: float = 0.0` parameter.
- Drop annotations where `ann.get("score", 1.0) < min_score`. (The
  `1.0` default keeps non-scored annotations — e.g. user-added labels
  in the future — passing through.)

This is a pure mechanical change with no behaviour change at the
default value of 0.0.

### 0.2 New helper: `classify_frames`

```python
@dataclass
class FrameBuckets:
    positive: list[int]       # ≥1 annotation with score ≥ t_high
    uncertain: list[int]      # any annotation in [t_low, t_high) and none ≥ t_high
    true_negative: list[int]  # no annotations, or all annotations < t_low

def classify_frames(coco: dict, *, t_high: float, t_low: float) -> FrameBuckets:
    ...
```

Pure function on a parsed COCO dict — no I/O. Reused by the trainer
(0.3) and the GUI preview endpoint (0.4).

Edge cases:
- A frame whose only annotations are below `t_low` is `true_negative`
  (the teacher tried, came up with nothing convincing — strongest
  possible "really empty" signal).
- A frame with both a high-scoring annotation and a low-scoring one is
  `positive` (we keep it, and 0.1's filtering drops the low-score box
  from the YOLO label).

### 0.3 Wire bucketing into `prepare_yolo_dataset`

New signature:

```python
def prepare_yolo_dataset(
    *,
    student_dir: Path,
    train_teacher_ids: list[str],
    t_high: float = 0.35,
    t_low: float = 0.15,
    treat_empty_as_negative: bool = False,
    split_ratio: float = DEFAULT_TRAIN_VAL_SPLIT,
    seed: int = 42,
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> DatasetSummary:
```

Behaviour:
- Per teacher: `classify_frames(coco, t_high=…, t_low=…)`.
- Skip uncertain frames entirely — no frame extraction, no label file.
- Extract positive + true_negative frames. Positive frames get filtered
  YOLO labels (`min_score=t_high` in `_coco_to_yolo_lines`). True
  negative frames get an empty `.txt` (genuine background).
- If `treat_empty_as_negative=True`, the bucket logic still runs (so
  the user sees the breakdown), but uncertain frames are reclassified
  to `true_negative` before extraction. This reproduces the **old**
  training set exactly — kept as an opt-in escape hatch for users who
  trust their teacher.

Extend `DatasetSummary`:

```python
@dataclass
class DatasetSummary:
    data_yaml: Path
    n_train_images: int
    n_train_annotations: int
    n_val_images: int
    class_names: list[str]
    # NEW
    n_positive_frames: int
    n_uncertain_dropped: int
    n_true_negative_frames: int
    per_teacher_buckets: list[dict]
        # [{"teacher_id": str, "positive": int, "uncertain": int, "true_negative": int}]
```

These flow into `StudentStats` (0.5).

### 0.4 GUI: live preview in the New Student form

Files: `pipeline/optimize.py`, `server/main.py`, `gui/src/modes/Optimize.tsx`,
`gui/src/api.ts`, `gui/src/types.ts`

New backend endpoint:
```
POST /students/preview-buckets
body: { teacher_ids: string[], t_high: number, t_low: number,
        treat_empty_as_negative: bool }
→ {
    aggregate: { positive: int, uncertain: int, true_negative: int,
                 n_classes: int, class_names: string[] },
    per_teacher: [
      { teacher_id, positive, uncertain, true_negative }
    ]
  }
```

Cheap — reads each teacher's `coco.json` and runs `classify_frames`. No
frame extraction. Should respond in tens of ms even for large COCOs.

GUI in `Optimize.tsx`:
- Below the teacher picker, above the toolchain rows, render a live
  preview line:
  > **Training data preview** — 432 positive · 87 uncertain (excluded) · 156 true negatives · 12 classes from 3 teachers
- Add a collapsible "Advanced" panel containing:
  - `t_high` number input (default 0.35, step 0.05, range 0–1)
  - `t_low` number input (default 0.15, step 0.05, range 0–1, must be ≤ t_high)
  - `treat_empty_as_negative` checkbox (default off)
- Debounce preview fetches at 250ms so checkbox toggles and slider
  drags don't hammer the server.
- The same `t_high`, `t_low`, `treat_empty_as_negative` values are sent
  with `POST /optimize` (see 0.6).

Friendly hover/help text on each:
- `t_high` — "Detections at or above this confidence become labels"
- `t_low` — "Frames with detections only between t_low and t_high are dropped (teacher was unsure)"
- `treat_empty_as_negative` — "Treat every zero-detection frame as a true negative. Reproduces the old behaviour. Off by default."

### 0.5 Surface bucket counts on the Student detail card

Files: `pipeline/runs.py` (`StudentStats`), `gui/src/types.ts`,
`gui/src/modes/Optimize.tsx` (`SelectedStudent`).

New fields on `StudentStats`:
```python
n_positive_frames: int = 0
n_uncertain_dropped: int = 0
n_true_negative_frames: int = 0
per_teacher_buckets: list[dict] = field(default_factory=list)
t_high: float = 0.35
t_low: float = 0.15
treat_empty_as_negative: bool = False
```

All defaulted to safe values so existing manifests keep loading.

In `SelectedStudent`, add a row:

> **Frame buckets** 432 positive · 87 uncertain (dropped) · 156 true negatives  *(t_high=0.35, t_low=0.15)*

Below the existing per-eval-teacher table, add a per-train-teacher
bucket table reusing the same row styling.

### 0.6 Plumb thresholds through the orchestration

File: `pipeline/optimize.py`, `server/main.py`, `server/schemas.py`

Add to `OptimizeRequest` (existing schema):
```python
t_high: float = 0.35
t_low: float = 0.15
treat_empty_as_negative: bool = False
```

`run_optimize_in_background()` accepts and forwards them to
`prepare_yolo_dataset()`.

Persist them on the `StudentManifest` so the run is reproducible from
the manifest alone (also feeds the detail card).

### 0.7 Acceptance — Phase 0

- Old completed Students still load; their detail card shows the new
  fields with placeholder dashes (no fake numbers).
- Default-threshold run on an existing teacher produces strictly fewer
  training frames than before (uncertain band dropped).
- `treat_empty_as_negative=True` with default thresholds reproduces the
  **previous** training set exactly — same image count, same
  annotation count. Acts as a regression check.
- Preview endpoint round-trips in < 200ms for the largest existing
  teacher coco.json.
- `t_low > t_high` is rejected at validation time with a clear error.
- New unit tests in `tests/`:
  - `test_classify_frames` — buckets fence-post cases, no-annotation
    frames, mixed-score frames, empty COCOs.
  - `test_coco_to_yolo_lines_score_filter` — round-trip with and
    without `min_score`.

---

## Phase 1 — Trainer dispatch refactor

No behaviour change for `yolov8n`. Pulls the trainer out of
`pipeline/distill.py` into a small registry so the next two phases can
just register new architectures.

### 1.1 Create `pipeline/students/` package

```
pipeline/students/
    __init__.py
    base.py          # StudentTrainer protocol, TrainResult
    registry.py      # TRAINERS, register(), make_trainer(), list_trainers()
    yolo.py          # @register("yolov8n"), @register("yolov8s")
```

`base.py`:
```python
@dataclass
class TrainResult:
    weights_path: Path
    train_seconds: float
    epoch_metrics: list[dict] = field(default_factory=list)
        # one dict per epoch — at minimum {"epoch": int, "map50": float},
        # used by the future learning-curve plot. May be empty for
        # architectures whose framework doesn't expose it cleanly.

class StudentTrainer(Protocol):
    name: str

    def train(
        self,
        *,
        data_yaml: Path,
        student_dir: Path,
        epochs: int,
        imgsz: int,
        progress: Callable[[int, int], None] | None,
    ) -> TrainResult: ...

    def eval(self, *, weights: Path, data_yaml: Path) -> tuple[float, float]:
        """Returns (map50, map50_95)."""

    def time_inference(
        self,
        *,
        weights: Path,
        sample_image_dir: Path,
        n_samples: int = INFERENCE_TIMING_SAMPLES,
    ) -> tuple[float, float, float]:
        """Returns (avg_ms, p50_ms, p95_ms). Warmup pass MUST be discarded."""
```

`registry.py` mirrors `pipeline/models/registry.py` exactly — same
decorator pattern.

### 1.2 First implementation: YOLO

`pipeline/students/yolo.py` — port the existing `train_yolo`, `eval_yolo`,
`time_inference` bodies from `distill.py` into a `YoloTrainer` class.
Each class is parameterised by `base_model` so registering different
sizes is one-liners:

```python
@register("yolov8n")
class YoloV8Nano(YoloTrainer): base_model = "yolov8n.pt"

@register("yolov8s")
class YoloV8Small(YoloTrainer): base_model = "yolov8s.pt"

@register("yolov8m")
class YoloV8Medium(YoloTrainer): base_model = "yolov8m.pt"
```

Leave `pipeline/distill.py` as a thin re-export shim during the
transition so any in-flight callers don't break, then remove it once
`optimize.py` is fully migrated.

### 1.3 Plumb through `pipeline/optimize.py`

`run_optimize_in_background()` accepts `architecture: str = "yolov8n"`.
The worker:
```python
trainer = make_trainer(architecture)
result = trainer.train(...)
map50, map50_95 = trainer.eval(...)
avg_ms, p50, p95 = trainer.time_inference(...)
```

`StudentManifest`: add `architecture: str = "yolov8n"` (default
preserves old runs at load time).

### 1.4 GUI: architecture selector

- New endpoint `GET /students/architectures` → `list_trainers()`.
- In `Optimize.tsx`, add an "Architecture" `<select>` above the toolchain
  rows. Default `yolov8n`. Use the same dropdown styling as toolchain
  impls.
- The selected value is sent on `POST /optimize`.
- The Student detail card gains an "Architecture" row.

### 1.5 Acceptance — Phase 1

- An old completed Student loads and renders with `architecture =
  "yolov8n"` (default-fill).
- A new `yolov8n` student through the dispatcher produces numerically
  identical results (within run-to-run variance) to the old code path
  on the same teacher set with the same `t_high`/`t_low`.
- Registering a no-op stub trainer makes it appear in the dropdown
  without any GUI changes.
- Tests: registry round-trip, architecture validation in `OptimizeRequest`.

---

## Phase 2 — RT-DETR

### 2.1 Trainer

`pipeline/students/rtdetr.py`:
```python
@register("rtdetr-l")
class RTDETRLarge(YoloTrainer):
    base_model = "rtdetr-l.pt"
```

Ultralytics dispatches RT-DETR through the same `YOLO()` class — the
existing trainer body works as-is. The only thing that may need
attention: RT-DETR does **not** support all of YOLO's `model.predict()`
keyword arguments. Make sure `time_inference` doesn't pass unsupported
kwargs.

Skip `rtdetr-x` for now — too heavy for a laptop test loop.

### 2.2 Comparability fields

To make Phase 3's compare view honest, persist the eval protocol on
`StudentStats`:
```python
imgsz: int = 640
device: str = ""        # "mps" | "cuda" | "cpu"
inference_warmup_discarded: bool = True
```

Phase 3 surfaces them and badges any cross-student mismatch.

### 2.3 Acceptance — Phase 2

- Train an RT-DETR student on the same teacher set as a YOLO student;
  finished run with non-zero mAP and reasonable inference timings.
- Both students share `imgsz`, `device` recorded the same way.
- "rtdetr-l" appears in the architecture dropdown without code changes
  to the GUI.

---

## Phase 3 — Compare view in the GUI

A new tab inside Optimize mode (left-rail item: **Compare**). Renders a
multi-select of completed Students plus three views.

### 3.1 Side-by-side details table

Columns:
| Student | Arch | Train (pos / unc / neg) | Epochs | Train time | Mean mAP@0.5 | mAP@0.5:0.95 | Latency p50 / p95 | Size MB |

Sortable by every numeric column. Highlight the best cell in each
column (green).

### 3.2 Per-eval-teacher matrix

Rows: union of held-out eval teachers across the selected students.
Cols: the selected students.
Cells: `mAP@0.5` colour-graded with the existing `map-good / map-okay /
map-poor` thresholds.

If a student didn't include a given teacher in its eval set, render a
hatched "—" cell with a tooltip — visually obvious so the user
notices comparison mismatches.

Add a "Filter to shared eval teachers" toggle — collapses the matrix
to only teachers all selected students were evaluated against. Useful
once eval sets diverge.

### 3.3 Pareto plot: mAP vs. latency

Recharts `<ScatterChart>`. Already a dep — no new package.

- X: `p50_inference_ms` (lower is better)
- Y: mean `map50` (higher is better)
- Colour: architecture (categorical palette)
- Size: `model_size_mb` (small range — mostly cosmetic)
- Hover: full StudentStats summary

Top-right corner is the desirable region. Frontier students stand out
naturally; this is the chart for the NVIDIA conversation.

### 3.4 Comparability badges

If selected students differ in `imgsz`, `device`, or `epochs`, render a
small warning badge above the table:

> ⚠ Comparing students with different `imgsz` (640, 1280) — latency numbers aren't directly comparable.

Doesn't block — just informs.

### 3.5 Acceptance — Phase 3

- Pick 2+ students of any architecture mix; all three views render.
- Eval-teacher matrix correctly handles partial overlap.
- Pareto plot is intuitive at a glance.
- Comparability warnings fire correctly when `imgsz`/`device`/`epochs`
  diverge.

---

## Phase 4 — DINOv3 + detection head

**Do not hand to Claude Code as-is.** Needs a separate sub-spec
(`docs/dinov3-student.md`) before implementation. Things to nail down
first:

- Head architecture: DETR-style decoder vs. anchor-free YOLO-style
  head vs. simple linear classifier+box regressor on patch tokens.
  Likely a small DETR decoder is the cleanest, well-studied option.
- DINOv3 weights: HuggingFace `facebook/dinov3-vits16` to start
  (~22M params, tractable on MPS). Pin a revision SHA in code.
- Freezing strategy: backbone fully frozen for the first iteration.
  Unfreezing last-N blocks is a knob to add later, not first.
- Mixed-precision: enable for ViT pass on MPS if numerically stable.
- Eval pipeline: `eval_yolo` won't work — needs a custom evaluator
  computing COCO-style mAP from this model's outputs against the
  YOLO-format eval data. Reuse `pycocotools` via a small adapter.
- Inference timing: ViT pass dominates. Same warmup-discard
  methodology as 1.1; document expected p50 range so the user isn't
  surprised by 50–200 ms numbers vs. yolov8n's 5–15 ms.

Recommend prototyping in a Jupyter notebook (`notebooks/dinov3_proto.ipynb`)
before integrating, so the architecture decision is visible and
reviewable in isolation from registry/orchestration plumbing.

---

## Phase 5 — Label-efficiency sweep

Once three architectures are working end-to-end. Demonstrates the
foundation-model thesis: DINOv3 should dominate at small training
fractions and YOLO should catch up at full data.

### 5.1 Add `data_fraction` to `prepare_yolo_dataset`

```python
def prepare_yolo_dataset(
    ...,
    data_fraction: float = 1.0,
) -> DatasetSummary: ...
```

After bucketing and before frame extraction, subsample
`positive` and `true_negative` lists by `data_fraction`. Use
`random.Random(seed)` so 0.5 is a deterministic subset of 1.0 (helpful
for the sweep — you want the 25% set to be a subset of the 50% set).

Persist `data_fraction` on `StudentStats`.

### 5.2 Sweep runner

New endpoint:
```
POST /students/sweep
body: { train_teacher_ids, eval_teacher_ids,
        architectures: string[], fractions: number[],
        epochs?: int, imgsz?: int,
        t_high?: number, t_low?: number,
        treat_empty_as_negative?: bool }
→ { sweep_id, student_ids: string[] }
```

Worker: spawns `len(architectures) × len(fractions)` Student runs in
sequence (parallel would saturate the GPU on a laptop). Each is a
normal Student in `runs/`, plus a small `runs/sweep_<id>/manifest.json`
pointing at the spawned student IDs.

### 5.3 GUI: sweep view

In the Compare tab, a "Sweeps" sub-section listing existing sweeps.
Selecting a sweep renders:

- A single line chart per architecture: x = `data_fraction`, y = mean
  `map50`, line per architecture, dot per student.
- Below: the same details table from 3.1, filtered to sweep students,
  grouped by architecture.

### 5.4 Acceptance — Phase 5

- Kick off a sweep across `yolov8n`, `rtdetr-l`, `dinov3-detr` at
  `[0.25, 0.5, 1.0]` — get 9 students, all completed.
- Sweep view renders the 3-line plot with a sensible legend.
- 25% subset frames are a strict subset of the 50% subset, which is a
  strict subset of the 100% set. Test this directly.

---

## Conventions

- One PR per phase. Small phases (0.4-only, etc.) can be split if it
  helps review.
- Every new field on `StudentManifest` / `StudentStats` is defaulted
  in the dataclass and `Optional[...]`-friendly in the Pydantic
  schemas, so old `manifest.json` and `stats.json` files keep loading
  without migration scripts.
- Tests on pure functions (`classify_frames`,
  `_coco_to_yolo_lines`, registry); training loops stay manually
  exercised via the GUI.
- Avoid breaking changes to existing API endpoints — extend rather
  than rename. The GUI sends the new fields; old GUI versions just
  won't send them and the server applies defaults.

---

## Suggested handoff order

A reasonable sequence for handing off to Claude Code:

1. Phase 0.1 + 0.2 + 0.3 — backend bucketing, no GUI yet. Verifiable
   via existing CLI/API and the regression check (0.7 acceptance #2).
2. Phase 0.4 + 0.5 + 0.6 — wire it through to the GUI.
3. Phase 1 — dispatch refactor, no behaviour change.
4. Phase 2 — RT-DETR. Small.
5. Phase 3 — Compare view. The biggest GUI change in this spec.
6. Sub-spec for Phase 4. Review + prototype before implementing.
7. Phase 4 — DINOv3.
8. Phase 5 — sweep.

If we're confident, 0.1-0.6 can land as a single PR — they're tightly
coupled and the regression check is the natural acceptance gate.
