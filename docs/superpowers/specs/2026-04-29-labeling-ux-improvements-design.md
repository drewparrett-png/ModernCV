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

**Segment colors — tri-state:** `liveDetections[i].accepted` collapses two distinct cases into `true` — a frame with no state entry (unreviewed) and a frame with a `curated` entry where this detection is accepted. To distinguish them, compute a `stripColors` array in the parent before passing to `LabelStrip`:

```ts
type SegColor = 'accepted' | 'rejected' | 'unreviewed';

const stripColors = useMemo<SegColor[]>(
  () =>
    liveDetections.map(d => {
      const entry = frameStates[String(d.frame_idx)];
      if (!entry) return 'unreviewed';
      return d.accepted ? 'accepted' : 'rejected';
    }),
  [liveDetections, frameStates],
);
```

| `SegColor` | Display |
|---|---|
| `'unreviewed'` | Gray |
| `'accepted'` | Green |
| `'rejected'` | Red |

Frames in `confirmed_empty` or `marked_missed` states have no detections and therefore no entries in `liveDetections`; they never appear in the strip.

**Current position:** a thin white vertical cursor rule inside the strip segment at `index`.

**Click to navigate:** `onClick` computes the target index and calls `setIndex`:

```ts
const i = Math.min(total - 1, Math.floor((clickX / stripWidth) * total));
setIndex(i);
```

The `Math.min(total - 1, ...)` clamp prevents an off-by-one on right-edge clicks.

**Rendering at scale:** each segment uses `flex: 1 0 auto; min-width: 2px` in a `display: flex` container so segments compress proportionally at hundreds of crops without a scrollbar.

**Implementation:** new `LabelStrip` component defined in `CropReview.tsx` using `React.memo`. Props: `colors: SegColor[]`, `index: number`, `onJump: (i: number) => void`. The parent passes `stripColors` and `index`; `LabelStrip` does not need `frameStates` or `liveDetections` directly.

```tsx
const LabelStrip = React.memo(function LabelStrip({
  colors, index, onJump,
}: { colors: SegColor[]; index: number; onJump: (i: number) => void }) {
  // render one <span> per color, highlight segment at `index`
});
```

---

### 2. Auto-jump + resume toast in `CropReview.tsx`

**Auto-jump:** In the `Promise.all([fetchDetections, fetchFrameStates])` `.then()` handler (currently lines 74–79 of `CropReview.tsx`), replace the hard-coded `setIndex(0)` with the snippet below. Also add `threshold` to the `useEffect` dependency array (currently `[projectId, runId]`) so the effect re-runs if the threshold prop changes after mount.

```ts
const dets = resp.detections.filter(d => d.score >= threshold);
const firstUnlabeled = dets.findIndex(d => !fs[String(d.frame_idx)]);
// findIndex returns -1 when all crops are labeled → fall back to 0
const startIdx = firstUnlabeled > 0 ? firstUnlabeled : 0;
setIndex(startIdx);
```

Toast is only shown when `firstUnlabeled > 0` — this correctly covers three cases:
- `findIndex === -1`: all crops are labeled, `startIdx` falls back to 0, no toast (nothing to resume).
- `findIndex === 0`: the very first crop is unlabeled, meaning the session was never advanced at all — the viewer lands at 0 and no toast is needed because there is nothing to resume *from*.
- `findIndex > 0`: the user had labeled some crops and stopped — auto-jump + toast fires.

**Resume toast:** New `const [resumeToast, setResumeToast] = useState<string | null>(null)`. Message format:

> "Resumed from crop N · X unlabeled remaining"

where `N = firstUnlabeled + 1` (1-indexed) and `X = dets.filter(d => !fs[String(d.frame_idx)]).length`.

Toast renders as a small overlay inside the crop area. It disappears after 3 seconds — instant removal (no CSS fade), set via:

```ts
useEffect(() => {
  if (!resumeToast) return;
  const t = setTimeout(() => setResumeToast(null), 3000);
  return () => clearTimeout(t);
}, [resumeToast]);
```

Add `.crop-resume-toast` styles: `position: absolute; bottom: 12px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.7); color: #fff; padding: 4px 10px; border-radius: 4px; font-size: 12px; pointer-events: none; z-index: 10`.

---

### 3. Stale stats fix in `RunInspector.tsx`

**Root cause:** When `CropReview` closes, `RunInspector` already calls `fetchRunDetail` and `fetchFrameStates` to refresh its own local state (lines 347–361 of `RunInspector.tsx`). The calls are fire-and-forget with separate `.then`/`.catch` chains — not chained together. The global Zustand `teacherDetails` store — which drives the Teacher run cards in `Learn.tsx` — is never updated, because polling stopped when the run reached `"completed"`.

**Fix:** Restructure the `fetchRunDetail` block inside `onClose` to also push `d` to the global store:

```ts
// Replace the existing fetchRunDetail block:
fetchRunDetail(projectId, runId)
  .then((d) => {
    setDetail(d);
    useStore.setState(s => ({
      teacherDetails: { ...s.teacherDetails, [runId]: d },
    }));
  })
  .catch(() => { /* non-fatal */ });
```

The `fetchFrameStates` block in the same `onClose` is unchanged. `useStore` is already imported in `RunInspector.tsx`.

---

### 4. Human-label badge in `Optimize.tsx` teacher picker

**Context:** The student creation dialog in `Optimize.tsx` renders a `TeacherPicker` list of `RunDetail[]` (from the `teacherDetails` store). Each row already shows task pill and prompt. No API change is needed — `RunDetail.manifest` already carries `n_frames_reviewed`, `n_frames_total`, and `review_status`.

**Change:** Inside the teacher-picker row metadata block (around lines 532–542 of `Optimize.tsx`), add a review badge that uses the same wording as `Learn.tsx` ("reviewed", not "labeled") for UI consistency:

```tsx
{t.manifest.review_status !== "unreviewed" && (
  <span className="teacher-picker-review-badge">
    {t.manifest.review_status === "approved"
      ? "Fully curated"
      : `${t.manifest.n_frames_reviewed}/${t.manifest.n_frames_total} reviewed`}
  </span>
)}
```

The wording mirrors `Learn.tsx` lines 233–235 exactly ("Fully curated" / "N/total reviewed").

The `Learn.tsx` run cards already display the same information via the existing `curation-pill` — no change needed there; the stale-data fix in §3 is sufficient.

---

## Files to modify

| File | Change |
|---|---|
| `gui/src/components/CropReview.tsx` | Add `LabelStrip` sub-component; replace `setIndex(0)` with auto-jump; add resume toast state + display |
| `gui/src/components/RunInspector.tsx` | Restructure `fetchRunDetail` block in `onClose` to also push fresh `RunDetail` to global `teacherDetails` store |
| `gui/src/modes/Optimize.tsx` | Add `teacher-picker-review-badge` span to teacher picker rows |
| `gui/src/index.css` | Add styles for `.crop-label-strip`, `.strip-seg`, `.strip-seg-accepted`, `.strip-seg-rejected`, `.strip-cursor`, `.crop-resume-toast`, `.teacher-picker-review-badge` |

No backend changes. No new API endpoints.

---

## Verification

1. **Progress strip:**
   - Open CropReview on a run with some frames already labeled (perform a few accept/reject actions first to create server-side frame state).
   - Confirm strip shows green/red/gray segments in the same order as detection navigation.
   - Click a gray segment — confirm viewer jumps to that crop.
   - Accept a crop — confirm the corresponding strip segment turns green without page reload.

2. **Auto-jump + toast:**
   - Label a few crops, close CropReview, reopen it.
   - Confirm viewer lands on the first unlabeled crop (not crop 0).
   - Confirm toast "Resumed from crop N · X unlabeled remaining" appears and disappears after ~3 s.
   - Open a fully-labeled run — confirm no toast, viewer starts at 0.
   - Open an unlabeled run (crop 0 is unlabeled) — confirm no toast, viewer starts at 0.

3. **Stats refresh:**
   - Label a few crops via CropReview, then close it.
   - Without reloading the page, confirm the Teacher run card in `Learn.tsx` immediately shows the updated "N / total reviewed" count.

4. **Human-label badge (Optimize):**
   - Navigate to Optimize → teacher picker.
   - Select a teacher run with partial labeling — confirm row shows "N / total reviewed" badge.
   - Select a fully-labeled run — confirm row shows "Fully curated".
   - Select an unlabeled run — confirm no badge appears.
