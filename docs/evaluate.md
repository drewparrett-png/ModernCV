# Evaluate — Implementation Spec

A new top-level mode that lets you pick a dataset and rank your Students
on it side-by-side. Speed and accuracy in one view. Replaces the (now
removed) Graph Editor as the third tab.

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Dataset review state — `approved_at` on Teacher manifests + approve/unapprove API | not started |
| 1 | Review-status pill, surfaced in existing UI + approve action on Run Inspector | not started |
| 2 | Evaluate mode — dataset header + Students-only leaderboard | not started |

Each phase is one PR. Phases 0 and 1 are deliberately small so the
state plumbing and the cross-app pill can land before any of the
leaderboard work.

This spec assumes Phase 1 of `student-training.md` (the trainer
dispatch refactor adding `architecture` to `StudentManifest`) is in
flight. Evaluate renders whatever fields exist on `StudentStats`; the
comparability-badge logic in Phase 2.4 below is a no-op if those fields
aren't there yet.

Design decisions this spec commits to (push back in PR if you disagree):

- **The Teacher is dataset metadata, not a leaderboard row.** Putting
  the Teacher in the same table as Students forces a measurement that
  doesn't fit the column — Student mAP is "agreement with the
  dataset's GT," and the Teacher trivially scores against itself.
  Instead, the Teacher's identity, inference speed, and review status
  live in a header band at the top of the page, above a Students-only
  leaderboard.
- **`review_status` is derived, not stored.** The single persisted
  field is `approved_at: datetime | None` on `RunManifest`. Display
  state is computed at read time:
  - `approved` — `approved_at is not None`
  - `reviewed` — `approved_at is None` and at least one rejection on disk
  - `unreviewed` — neither
- **Approval is reversible but asymmetric.** Approve is a one-click
  action available wherever the dataset is listed (Evaluate header,
  Run Inspector). Un-approve is only on Run Inspector — keeps a
  destructive-feeling action out of the casual flow.
- **Leaderboard subjects = Students whose `per_eval_teacher` includes
  the picked dataset.** The Student must have a real measured score
  against this Teacher. Students that only trained on it (no eval) are
  excluded from this dataset's leaderboard — train-set scores aren't
  comparable to held-out scores.

---

## Phase 0 — Dataset review state

**Why first.** Phase 1's pill and Phase 2's header both need
`review_status` to exist on the wire. Land the state plumbing on its
own so any UI work can mock against the real shape.

### 0.1 Add `approved_at` to `RunManifest`

File: `pipeline/runs.py` (`RunManifest` dataclass), `server/schemas.py`
(`RunManifest` Pydantic mirror).

```python
approved_at: Optional[str] = None  # ISO 8601, set by /approve, cleared by /unapprove
```

Default `None` so existing manifest.json files keep loading without
migration.

### 0.2 Approve / unapprove endpoints

File: `server/main.py`, `server/schemas.py`.

```
POST /runs/:id/approve
→ { manifest: RunManifest }   # updated, with approved_at populated
```

```
POST /runs/:id/unapprove
→ { manifest: RunManifest }   # approved_at cleared back to None
```

Validation:
- Run must exist (404 otherwise).
- Run's `status` must be `"completed"` (400 otherwise — a still-running
  or failed run has nothing meaningful to approve).
- Approving an already-approved run is a no-op (returns the manifest
  unchanged, 200). Same for unapproving an unreviewed/reviewed run.

Persistence: rewrite the run's `manifest.json` with the new field,
reuse whatever pattern the existing rejection writes use.

### 0.3 Derived `review_status` on the wire

`RunManifest` Pydantic schema gains a computed field:

```python
@computed_field
@property
def review_status(self) -> Literal["unreviewed", "reviewed", "approved"]:
    if self.approved_at is not None:
        return "approved"
    if _has_any_rejections(self.id):  # cheap — checks rejections.json existence + non-empty
        return "reviewed"
    return "unreviewed"
```

Helper lives wherever `read_rejections` lives (`pipeline/runs.py`).
Don't read the full rejection map for this — short-circuit on file
existence and a single-key check, since this serialises with every
`/runs` listing.

If the cost shows up on large run lists, cache the boolean in
`RunManifest` itself at write time. Out of scope for v1; profile first.

### 0.4 Acceptance — Phase 0

- Old completed runs load with `approved_at = None`,
  `review_status = "unreviewed"` (or `"reviewed"` if rejections exist).
- `POST /runs/:id/approve` flips the wire field and persists across
  server restart.
- `POST /runs/:id/unapprove` reverses it cleanly.
- Approving a non-completed run returns 400 with a clear error.
- `/runs` listing serves `review_status` for every run without a
  noticeable latency regression on a 50-run directory.
- New unit test: `test_review_status_derivation` covering all three
  states + transitions.

---

## Phase 1 — Review-status pill across the existing UI

The pill is a small reusable component. It earns its keep by appearing
everywhere a Teacher run is named — Learn sidebar, Optimize teacher
selector, Evaluate dataset picker (Phase 2). Same component, same
visual language across the app.

### 1.1 `<ReviewStatusPill />`

File: `gui/src/components/ReviewStatusPill.tsx` (new).

```tsx
type Props = { status: "unreviewed" | "reviewed" | "approved" };
```

Visuals:
- `unreviewed` — gray dot or empty circle, no text by default. Tooltip:
  "No human review yet."
- `reviewed` — amber half-circle / partial check. Tooltip: "Partially
  reviewed — some detections rejected."
- `approved` — green check ✓. Tooltip: "Approved as ground truth on
  {date}."

Accept an optional `compact?: boolean` for places that want only the
icon (sidebars), default to icon + short label ("Approved",
"Reviewed", or no label for unreviewed).

### 1.2 Surface it in existing lists

Files: `gui/src/modes/Learn.tsx` (teacher sidebar list),
`gui/src/modes/Optimize.tsx` (train-teacher and eval-teacher selectors),
`gui/src/components/RunInspector.tsx` (next to the run title).

In each list, the pill goes inline with the run's name. Compact mode
in dense lists (sidebars), full mode where there's room (Run Inspector
title bar, Evaluate header).

### 1.3 Approve action on Run Inspector

File: `gui/src/components/RunInspector.tsx`.

Near the rejection summary (the existing block that shows total
rejections), add an action row:

- When `review_status === "unreviewed"` or `"reviewed"`:
  > **[ ✓ Approve as ground truth ]**
  > Marks this dataset as fully human-reviewed. Students evaluated
  > against it will be scored against your kept labels.
- When `review_status === "approved"`:
  > ✓ Approved on {date} · *[Un-approve]*
  > The un-approve link is small and de-emphasized; clicking shows a
  > confirm dialog ("Un-approve dataset? Students' scores will be
  > recomputed against partial-review GT.").

Approve calls `POST /runs/:id/approve`, refreshes the local
`teacherDetails` from the response. Un-approve calls
`POST /runs/:id/unapprove`. Both should be optimistic — flip the pill
immediately, roll back on error.

### 1.4 Acceptance — Phase 1

- The pill renders correctly for all three states in Learn sidebar,
  Optimize selectors, and Run Inspector title bar.
- Approving from Run Inspector flips the pill across all surfaces
  (state lives in `teacherDetails`, so a single store update
  propagates).
- Un-approving requires a confirm dialog.
- Tooltip dates are formatted in the user's locale.
- Unit test: `<ReviewStatusPill />` snapshot for each state.

---

## Phase 2 — Evaluate mode

A new top-level mode. Tab order: Learn · Optimize · Evaluate.

### 2.1 Add the mode

Files: `gui/src/types.ts`, `gui/src/components/ModeShell.tsx`,
`gui/src/store.ts`.

```ts
export type Mode = "learn" | "optimize" | "evaluate";
```

Add the tab to `TABS` in ModeShell:

```ts
{ mode: "evaluate", label: "Evaluate", blurb: "Compare Students on a dataset" },
```

Render `<Evaluate />` from `gui/src/modes/Evaluate.tsx` (new).

Locking behaviour: tab is disabled with the same lock pattern used for
Optimize, with the lock condition being "no completed Teacher runs
exist." Tooltip when locked:

> 🔒 Run something in Learn first — Evaluate needs a completed
> dataset.

### 2.2 Dataset header

Top of the Evaluate panel.

Layout:
```
┌────────────────────────────────────────────────────────────────┐
│ Dataset:  [ ▼ teacher-run-2026-04-22-soccer    ]  ✓ Approved   │
│           1,247 frames · 3,891 annotations · 4 classes         │
│           Teacher: GroundingDINO + ByteTrack — 142 ms p50      │
│           [ ✓ Approve as ground truth → ]   ← only when amber  │
└────────────────────────────────────────────────────────────────┘
```

Components:
- Dataset picker — `<select>` listing all `completed` Teacher runs,
  most-recent first. Each option label is the run name with the
  `<ReviewStatusPill compact />` icon prefixed. Default selection is
  the last-touched approved run, falling back to most-recent
  completed.
- Counts row — pulled from `RunStats` on the selected run
  (`frames_with_detections`, `n_detections_total`) plus class count
  from the COCO file (cheap to count, no need for a new field).
- Teacher speed line — `RunStats.p50_ms_per_frame` /
  `p95_ms_per_frame`. Gives the user the "what does my Student need to
  beat" anchor.
- Inline approve action when `review_status === "reviewed"` or
  `"unreviewed"`. Same call as Phase 1.3, just placed in the Evaluate
  flow.

### 2.3 Students-only leaderboard

Below the header.

Source data: every Student where `selected_dataset_id ∈ student.eval_teacher_ids`.
The mAP numbers come from `student.stats.per_eval_teacher` filtered to
the selected dataset. Speed/size/architecture come from
`student.stats` directly.

Columns:

| Student | Arch | Trained on | mAP@50 | mAP@50-95 | p50 ms | p95 ms | Size MB |

- **Student** — name, status pill (running/completed). Click to open
  Run Inspector for that Student (existing flow).
- **Arch** — `student.manifest.architecture` (defaulted "yolov8n" for
  pre-Phase-1 students).
- **Trained on** — collapsed-by-default badge: "3 teachers". Hover
  expands to the list of teacher run names (with their compact review
  pills inline). Wide column otherwise.
- **mAP@50, mAP@50-95** — colour-graded with the existing
  `map-good / map-okay / map-poor` thresholds (already in the codebase
  for the per-eval-teacher table).
- **p50 ms, p95 ms** — raw numbers. Render a faint horizontal line
  across both speed columns at the Teacher's p50 / p95 — gives a
  visual anchor for "your Student needs to beat this." Implementation
  hint: a CSS `::before` on the column header band, or an absolutely
  positioned div over the table body. Recharts is overkill here.
- **Size MB** — `student.stats.model_size_mb`.

Sortable on every numeric column. Default sort: `mAP@50` desc.

Highlight the best cell in each numeric column with the existing green
"best" pill style from the Phase-3 spec in `student-training.md` (if
that's landed). Pre-Phase-3 it's just a green background colour on
the cell.

Empty / single states:
- Zero qualifying Students for this dataset:
  > **No Students yet for this dataset.** Train one in Optimize → it
  > will appear here once it completes.
  > [ Open Optimize → ] (deep-links into Optimize with this dataset
  > pre-selected as a train teacher).
- Single qualifying Student: still show the table — a one-row table
  tells the truth.

### 2.4 Comparability badges

If the displayed Students differ on `imgsz`, `device`, or `epochs`
(fields added in `student-training.md` Phase 2.2), render a single
warning row above the table:

> ⚠ Comparing Students with different `imgsz` (640, 1280) — latency
> isn't directly comparable.

If those fields don't exist yet on stats (pre-Phase-2.2), this block
renders nothing. No errors.

### 2.5 Wiring

Files: `gui/src/store.ts`, `gui/src/api.ts`, `gui/src/types.ts`.

Store additions:

```ts
// Evaluate state
selectedDatasetId: string | null;
selectDataset: (id: string | null) => void;
```

No new fetch endpoints — Evaluate is a pure projection over already-
loaded `teacherDetails` and `studentDetails`. The reactive selectors
do the filtering and joining at render time.

API additions for Phase 0/1:

```ts
export async function approveRun(id: string): Promise<RunManifest>;
export async function unapproveRun(id: string): Promise<RunManifest>;
```

### 2.6 Acceptance — Phase 2

- Evaluate tab is locked until at least one completed Teacher exists;
  unlocks once one does.
- Dataset picker lists only `completed` Teachers; status pill renders
  inline next to each name.
- Header counts and Teacher speed are correct against the chosen
  dataset's `RunStats`.
- Leaderboard correctly filters Students to those with the dataset in
  `eval_teacher_ids`; mAP values match the `per_eval_teacher` numbers
  shown in Optimize for the same Student / Teacher pair.
- Sorting on every numeric column works; default sort is mAP@50 desc.
- Speed-baseline marker visually aligns with the Teacher's p50/p95.
- Empty state CTA deep-links into Optimize with the dataset
  preselected as train teacher.
- Approving from the header flips the pill in the picker, in Learn,
  and in Optimize selectors all at once (one store mutation).

---

## Out of scope (good v2 candidates)

- **Side-by-side frame viewer.** Pick a frame; see Teacher GT and each
  Student's overlay tiled. This is the most fun thing to build but
  has real complexity (multi-Student inference on demand vs. cached
  predictions, overlay rendering per Student). Save for after v1
  proves the leaderboard is useful.
- **Cross-dataset matrix.** Rows = Students, cols = datasets, cells =
  mAP. Useful for transferability stories but a different page.
- **External datasets.** A user-supplied COCO folder as the dataset,
  not derived from a Teacher run. Sensible follow-up; needs its own
  upload/path-picker flow.
- **Approving with corrections.** Adding missed boxes during review,
  not just rejecting wrong ones. Big feature — would change the
  rejection-map data model.

---

## Conventions

- One PR per phase. Phase 0 must land before either UI phase; Phase 1
  and Phase 2 can land in either order, but Phase 1 first is cleaner
  (Evaluate header reuses the Phase 1 pill).
- All new fields default-fill in dataclasses and are
  `Optional[...]`-friendly in Pydantic — old `manifest.json` /
  `stats.json` keep loading without migration.
- No new top-level files in `pipeline/` — review-status logic lives
  alongside `read_rejections` in `pipeline/runs.py`.
- Evaluate is a pure projection over existing store state. No new
  polling, no new server-side aggregation — joins happen in selectors.
- Tests on derived state (`test_review_status_derivation`) and on the
  pill component snapshot. The leaderboard view is exercised
  manually via the GUI.

---

## Suggested handoff order

1. Phase 0 — backend state plumbing. Verifiable via curl alone.
2. Phase 1 — pill component + approve action. Visible improvement
   to the existing app even before Evaluate exists.
3. Phase 2 — the Evaluate mode itself.

If Phases 0 and 1 are tightly reviewed they can land in a single PR;
the cross-cutting risk is low and the pill is meaningless without the
state.
