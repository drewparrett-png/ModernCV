# Labeling UX Improvements

**Date:** 2026-04-29  
**Status:** Approved by user

## Context

Three pain points reported with the human-labeling workflow in the crop-review UI:

1. **No progress visibility** — when returning to an in-progress labeling session there is no visual indicator of what has been done vs. what remains, and the crop viewer always starts at crop 0.
2. **No resume** — the viewer doesn't jump to the first unlabeled crop on open, making it hard to continue where you left off.
3. **Stale reviewed count** — after closing `CropReview` and returning to the Teacher run list, the "N / total reviewed" badge on the run card shows the old count until the full page is reloaded.

Additionally, there is no UI signal that partial human labels feed into student training (even though the pipeline already handles this correctly via `distill._apply_frame_state_overrides`).

## What is NOT changing

- The pipeline already reads `frame_states.json` and prioritises curated/confirmed_empty/marked_missed frames during student training — no backend changes needed.
- The student training endpoint (`POST /projects/{pid}/students`) already accepts teachers with `review_status == "in_progress"` — no gating change needed.
- The `preview-buckets` endpoint is not being modified (it classifies by confidence, human-override state is surfaced separately in the teacher picker).

---

## Design

### 1. Progress strip in `CropReview.tsx`

A full-width colored strip is rendered above the crop image inside the `CropReview` component. One segment per entry in `liveDetections` (same order as navigation, `score_asc`).

**Segment colors:**
| State | Color |
|---|---|
| Frame `curated`, `det_idx` not in `rejected_dets` | Green (accepted) |
| Frame `curated`, `det_idx` in `rejected_dets` | Red (rejected) |
| Frame `confirmed_empty` or `marked_missed` | Red |
| No frame state (unreviewed) | Gray |

**Current position:** a thin white vertical cursor rule inside the strip segment at `index`.

**Click to navigate:** `onClick` computes `Math.floor((clickX / stripWidth) * total)` and calls `setIndex(i)`. No image loading required — the strip colors come from `liveDetections` and `frameStates` which are already in local state.

**Rendering at scale:** each segment uses `flex: 1 0 auto; min-width: 2px` in a `display: flex` container so segments compress proportionally at hundreds of crops without a scrollbar.

**Implementation location:** new `<LabelStrip>` sub-component inside `CropReview.tsx`, rendered just above the main crop image. Uses `useMemo` keyed on `[liveDetections, frameStates, index]` to compute the color array.

---

### 2. Auto-jump + resume toast in `CropReview.tsx`

**Auto-jump:** In the `Promise.all([fetchDetections, fetchFrameStates])` `.then()` handler (currently line 74–79 of `CropReview.tsx`), replace the hard-coded `setIndex(0)` with:

```ts
const dets = resp.detections.filter(d => d.score >= threshold);
const firstUnlabeled = dets.findIndex(d => !fs[String(d.frame_idx)]);
const startIdx = firstUnlabeled >= 0 ? firstUnlabeled : 0;
setIndex(startIdx);
```

If `startIdx > 0`, also set a resume toast.

**Resume toast:** New `const [resumeToast, setResumeToast] = useState<string | null>(null)`. Message format:

> "Resumed from crop N · X unlabeled remaining"

where `N = firstUnlabeled + 1` (1-indexed) and `X = dets.filter(d => !fs[String(d.frame_idx)]).length`.

Toast renders as a small overlay inside the crop area, fades out after 3 seconds via `useEffect(() => { const t = setTimeout(() => setResumeToast(null), 3000); return () => clearTimeout(t); }, [resumeToast])`.

Toast only appears when `firstUnlabeled > 0` (there's actually something to resume from).

---

### 3. Stale stats fix in `RunInspector.tsx`

**Root cause:** When `CropReview` closes, `RunInspector` already calls `fetchRunDetail` and `fetchFrameStates` to refresh its own local state (lines 347–361 of `RunInspector.tsx`). But the global Zustand `teacherDetails` store — which drives the Teacher run cards in `Learn.tsx` — is never updated, because polling stopped when the run reached `"completed"`.

**Fix:** In the existing `onClose` callback's `fetchRunDetail(...).then(d => setDetail(d))` chain, also push `d` into the global store:

```ts
.then((d) => {
  setDetail(d);
  useStore.setState(s => ({
    teacherDetails: { ...s.teacherDetails, [runId]: d },
  }));
})
```

No new store action needed — the `useStore.setState` pattern is already used elsewhere in the store.

---

### 4. Human-label badge in `Optimize.tsx` teacher picker

**Context:** The student creation dialog in `Optimize.tsx` renders a `TeacherPicker` list of `RunDetail[]` (from the `teacherDetails` store). Each row already shows task pill and prompt. No API change is needed — `RunDetail.manifest` already carries `n_frames_reviewed`, `n_frames_total`, and `review_status`.

**Change:** Inside the teacher-picker row metadata block (around lines 532–542), add a label badge alongside the existing task pill:

```tsx
{t.manifest.review_status !== "unreviewed" && (
  <span className="teacher-picker-review-badge">
    {t.manifest.review_status === "approved"
      ? "Fully labeled"
      : `${t.manifest.n_frames_reviewed} / ${t.manifest.n_frames_total} labeled`}
  </span>
)}
```

This makes it immediately visible how much human-curated data each teacher contributes to the upcoming training run.

The `Learn.tsx` run cards already display the same information (lines 230–237) via the existing `curation-pill` — no change needed there; the stale-data fix in §3 is sufficient.

---

## Files to modify

| File | Change |
|---|---|
| `gui/src/components/CropReview.tsx` | Add `LabelStrip` sub-component; change `setIndex(0)` to auto-jump; add resume toast state + display |
| `gui/src/components/RunInspector.tsx` | In CropReview `onClose`, push refreshed `RunDetail` to global `teacherDetails` store |
| `gui/src/modes/Optimize.tsx` | Add `teacher-picker-review-badge` span to teacher picker rows |
| `gui/src/App.css` (or component CSS) | Add styles for `.crop-label-strip`, `.strip-seg`, `.strip-seg-accepted`, `.strip-seg-rejected`, `.strip-cursor`, `.crop-resume-toast`, `.teacher-picker-review-badge` |

No backend changes. No new API endpoints.

---

## Verification

1. **Progress strip:**
   - Open CropReview on a run with some frames already labeled.
   - Confirm strip shows green/red/gray segments proportional to detection count.
   - Click a gray segment — confirm viewer jumps to that crop.
   - Accept a crop — confirm the corresponding strip segment turns green without page reload.

2. **Auto-jump + toast:**
   - Close CropReview mid-session, reopen.
   - Confirm viewer lands on the first unlabeled crop (not crop 0).
   - Confirm toast "Resumed from crop N · X unlabeled remaining" appears and fades after ~3 s.
   - Open a fully-labeled run — confirm no toast, starts at 0.

3. **Stats refresh:**
   - Label a few crops, close CropReview.
   - Without reloading, confirm the Teacher run card in Learn.tsx immediately shows the updated "N / total reviewed" count.

4. **Human-label badge (Optimize):**
   - Navigate to Optimize → select a teacher run that has partial labeling.
   - Confirm the teacher-picker row shows "N / total labeled" badge.
   - Confirm a fully-labeled run shows "Fully labeled".
