# Labeling UX Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three labeling UX gaps — add a progress strip with auto-resume to `CropReview`, refresh the global store when the inspector closes, and surface review counts in the student teacher picker.

**Architecture:** Pure frontend changes across three React components and one CSS file. No new API endpoints, no backend changes. The strip is a `React.memo` component that takes a pre-computed `SegColor[]` array, avoiding any coupling to `frameStates` internals. The stale-stats fix is a one-line addition to an existing `fetchRunDetail` chain.

**Tech Stack:** React 18, TypeScript, Zustand, Vite (no test framework — verification is via `npm run build` for type safety and manual testing in the dev server)

---

## File Map

| File | Action | What changes |
|---|---|---|
| `gui/src/index.css` | Modify | Add CSS for strip segments, cursor, toast overlay, and badge |
| `gui/src/components/CropReview.tsx` | Modify | Add `SegColor` type, `LabelStrip` component, `stripColors` memo, `resumeToast` state, auto-jump logic |
| `gui/src/components/RunInspector.tsx` | Modify | Push refreshed `RunDetail` into global `teacherDetails` on CropReview close |
| `gui/src/modes/Optimize.tsx` | Modify | Add review badge to teacher-picker rows |

---

## Task 1: Add CSS for all new UI elements

**Files:**
- Modify: `gui/src/index.css`

Add styles at two locations:
1. After the existing `/* ---- Crop review ---- */` block (around line 3214, after `.crop-review-actions button:disabled`)
2. After the existing `.teacher-picker-warn` block (around line 1558)

- [ ] **Step 1: Add crop label strip and toast CSS**

Find the end of the `.crop-review-actions button:disabled` rule and add immediately after:

```css
/* ---- Label progress strip -------------------------------------------- */

.crop-label-strip {
  display: flex;
  height: 8px;
  width: 100%;
  cursor: pointer;
  flex-shrink: 0;
  overflow: hidden;
}

.strip-seg {
  flex: 1 0 auto;
  min-width: 2px;
}

.strip-seg-unreviewed { background: #d1d5db; }
.strip-seg-accepted   { background: #10b981; }
.strip-seg-rejected   { background: #ef4444; }

.strip-cursor {
  outline: 2px solid white;
  outline-offset: -2px;
  position: relative;
  z-index: 1;
}

.crop-resume-toast {
  position: absolute;
  bottom: 12px;
  left: 50%;
  transform: translateX(-50%);
  background: rgba(0, 0, 0, 0.7);
  color: #fff;
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 12px;
  pointer-events: none;
  z-index: 10;
  white-space: nowrap;
}
```

- [ ] **Step 2: Add teacher-picker review badge CSS**

Find `.teacher-picker-warn` rule and add after it:

```css
.teacher-picker-review-badge {
  font-size: 11px;
  padding: 1px 6px;
  border-radius: 10px;
  background: #eff6ff;
  color: #1e40af;
  white-space: nowrap;
}
```

- [ ] **Step 3: Verify build passes**

```bash
cd gui && npm run build
```

Expected: exits 0, no TypeScript errors.

- [ ] **Step 4: Commit**

```bash
git add gui/src/index.css
git commit -m "style: add crop label strip, resume toast, and review badge CSS"
```

---

## Task 2: LabelStrip component and stripColors memo

**Files:**
- Modify: `gui/src/components/CropReview.tsx`

Add the `SegColor` type, the `LabelStrip` component (above `CropReview`), the `stripColors` useMemo (inside `CropReview`), and render the strip between the header and the body.

- [ ] **Step 1: Add `memo` to the React import**

The file currently imports:
```ts
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
```

Replace with:
```ts
import {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
```

- [ ] **Step 2: Add `SegColor` type and `LabelStrip` component**

Add after the existing `const CROP_PAD = 24;` line and before the `interface UndoEntry` block:

```ts
type SegColor = "accepted" | "rejected" | "unreviewed";

const LabelStrip = memo(function LabelStrip({
  colors,
  index,
  onJump,
}: {
  colors: SegColor[];
  index: number;
  onJump: (i: number) => void;
}) {
  const total = colors.length;
  function handleClick(e: React.MouseEvent<HTMLDivElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const i = Math.min(
      total - 1,
      Math.floor(((e.clientX - rect.left) / rect.width) * total),
    );
    onJump(i);
  }
  return (
    <div
      className="crop-label-strip"
      onClick={handleClick}
      title="Click to navigate to a crop"
    >
      {colors.map((c, i) => (
        <span
          key={i}
          className={`strip-seg strip-seg-${c}${i === index ? " strip-cursor" : ""}`}
        />
      ))}
    </div>
  );
});
```

Note: `React.MouseEvent` requires adding `React` to the import — either add `import React from "react"` or import `{ MouseEvent }` from `"react"` and use `MouseEvent<HTMLDivElement>`. The cleaner approach is to use `MouseEvent<HTMLDivElement>` from React's named exports. Update the import:

```ts
import {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type MouseEvent,
} from "react";
```

And use `MouseEvent<HTMLDivElement>` in `LabelStrip`.

- [ ] **Step 3: Add `stripColors` useMemo inside `CropReview`**

Add after the existing `reviewedCount` useMemo (around line 107):

```ts
const stripColors = useMemo<SegColor[]>(
  () =>
    liveDetections.map((d) => {
      const entry = frameStates[String(d.frame_idx)];
      if (!entry) return "unreviewed";
      return d.accepted ? "accepted" : "rejected";
    }),
  [liveDetections, frameStates],
);
```

- [ ] **Step 4: Render `<LabelStrip>` in the JSX**

Find the block that currently reads (around line 288–295):
```tsx
{error && <div className="inspector-error">Failed to load: {error}</div>}
{!error && !loaded && <div className="inspector-loading">Loading…</div>}
{loaded && total === 0 && (
  <div className="inspector-empty">
    No detections in this run. Crop review is empty.
  </div>
)}
```

Add the strip render between the error/loading blocks and the body:
```tsx
{error && <div className="inspector-error">Failed to load: {error}</div>}
{!error && !loaded && <div className="inspector-loading">Loading…</div>}
{loaded && total > 0 && (
  <LabelStrip colors={stripColors} index={index} onJump={setIndex} />
)}
{loaded && total === 0 && (
  <div className="inspector-empty">
    No detections in this run. Crop review is empty.
  </div>
)}
```

- [ ] **Step 5: Verify build passes**

```bash
cd gui && npm run build
```

Expected: exits 0, no TypeScript errors.

- [ ] **Step 6: Start dev server and verify strip renders**

```bash
cd gui && npm run dev
```

Open the app, navigate to a Teacher run, open Crop Review. Confirm:
- A thin horizontal bar appears below the header and above the main crop image
- All segments are gray (unreviewed) if no crops have been labeled yet
- Clicking a segment jumps to that crop position

- [ ] **Step 7: Verify strip colors update when labeling**

While in Crop Review, press `A` to accept the current crop. Confirm:
- The corresponding strip segment turns green
- The cursor rule moves to the next segment

- [ ] **Step 8: Commit**

```bash
git add gui/src/components/CropReview.tsx
git commit -m "feat: add LabelStrip progress bar to CropReview"
```

---

## Task 3: Auto-jump to first unlabeled crop and resume toast

**Files:**
- Modify: `gui/src/components/CropReview.tsx`

- [ ] **Step 1: Add `resumeToast` state**

Add after the existing `const [loaded, setLoaded] = useState(false);` line:

```ts
const [resumeToast, setResumeToast] = useState<string | null>(null);
```

- [ ] **Step 2: Replace the load useEffect with auto-jump logic**

Find the existing `useEffect` (lines 66–81):
```ts
useEffect(() => {
  if (!projectId || !runId) return;
  setError(null);
  setLoaded(false);
  Promise.all([
    fetchDetections(projectId, runId, { sort: "score_asc" }),
    fetchFrameStates(projectId, runId),
  ])
    .then(([resp, fs]) => {
      setDetections(resp.detections.filter((d) => d.score >= threshold));
      setFrameStates(fs);
      setIndex(0);
      setLoaded(true);
    })
    .catch((e) => setError(e instanceof Error ? e.message : String(e)));
}, [projectId, runId]);
```

Replace with:
```ts
useEffect(() => {
  if (!projectId || !runId) return;
  setError(null);
  setLoaded(false);
  setResumeToast(null);
  Promise.all([
    fetchDetections(projectId, runId, { sort: "score_asc" }),
    fetchFrameStates(projectId, runId),
  ])
    .then(([resp, fs]) => {
      const dets = resp.detections.filter((d) => d.score >= threshold);
      const firstUnlabeled = dets.findIndex((d) => !fs[String(d.frame_idx)]);
      // firstUnlabeled === -1: all labeled → start at 0, no toast
      // firstUnlabeled === 0:  never started → start at 0, no toast
      // firstUnlabeled > 0:   mid-session → jump + show toast
      const startIdx = firstUnlabeled > 0 ? firstUnlabeled : 0;
      setDetections(dets);
      setFrameStates(fs);
      setIndex(startIdx);
      setLoaded(true);
      if (firstUnlabeled > 0) {
        const unlabeledCount = dets.filter(
          (d) => !fs[String(d.frame_idx)],
        ).length;
        setResumeToast(
          `Resumed from crop ${firstUnlabeled + 1} · ${unlabeledCount} unlabeled remaining`,
        );
      }
    })
    .catch((e) => setError(e instanceof Error ? e.message : String(e)));
}, [projectId, runId, threshold]);
```

Key changes:
- `threshold` added to the dependency array so the effect re-runs if the prop changes
- `setResumeToast(null)` in the reset block clears any prior toast on re-open
- `startIdx = firstUnlabeled > 0 ? firstUnlabeled : 0` — both the "all labeled" and "never started" cases land at 0 with no toast

- [ ] **Step 3: Add toast auto-dismiss useEffect**

Add after the load useEffect:

```ts
useEffect(() => {
  if (!resumeToast) return;
  const t = setTimeout(() => setResumeToast(null), 3000);
  return () => clearTimeout(t);
}, [resumeToast]);
```

- [ ] **Step 4: Render toast inside the crop image area**

Find the `.crop-review-main` div (around line 298):
```tsx
<div className="crop-review-main">
  <img ... />
  <div className="crop-review-verdict ...">
    {cur.accepted ? "Accepted" : "Rejected"}
  </div>
</div>
```

Add the toast inside, after the verdict div:
```tsx
<div className="crop-review-main">
  <img ... />
  <div className="crop-review-verdict ...">
    {cur.accepted ? "Accepted" : "Rejected"}
  </div>
  {resumeToast && (
    <div className="crop-resume-toast">{resumeToast}</div>
  )}
</div>
```

- [ ] **Step 5: Verify build passes**

```bash
cd gui && npm run build
```

Expected: exits 0.

- [ ] **Step 6: Test auto-jump and toast in dev server**

```bash
cd gui && npm run dev
```

1. Open a Teacher run and accept a few crops (A key). Close CropReview.
2. Re-open CropReview. Confirm:
   - Viewer opens at the first unlabeled crop (not crop 0)
   - Toast "Resumed from crop N · X unlabeled remaining" appears at the bottom of the image
   - Toast disappears after ~3 seconds without any click

3. Open a run with no labels at all. Confirm:
   - Viewer opens at crop 0
   - No toast appears

4. Label every single crop. Close and reopen. Confirm:
   - Viewer opens at crop 0
   - No toast appears

- [ ] **Step 7: Commit**

```bash
git add gui/src/components/CropReview.tsx
git commit -m "feat: auto-jump to first unlabeled crop + resume toast in CropReview"
```

---

## Task 4: Fix stale reviewed count after CropReview closes

**Files:**
- Modify: `gui/src/components/RunInspector.tsx`

When CropReview closes, `RunInspector` already refreshes its own local `detail` and `frameStates` state. But the global Zustand `teacherDetails` store (which drives the run cards in `Learn.tsx`) is never updated — so the "N / total reviewed" badge on the run card stays stale until the page reloads. The fix: push the fresh `RunDetail` to the store inside the existing `fetchRunDetail` chain.

- [ ] **Step 1: Update the `fetchRunDetail` block in the `onClose` callback**

Find the existing `onClose` callback in `RunInspector.tsx` (around line 346):
```ts
fetchRunDetail(projectId, runId)
  .then((d) => setDetail(d))
  .catch(() => {
    /* non-fatal */
  });
```

Replace with:
```ts
fetchRunDetail(projectId, runId)
  .then((d) => {
    setDetail(d);
    useStore.setState((s) => ({
      teacherDetails: { ...s.teacherDetails, [runId]: d },
    }));
  })
  .catch(() => {
    /* non-fatal */
  });
```

`useStore` is already imported at the top of the file (`import { useStore } from "../store";`). No new imports needed.

- [ ] **Step 2: Verify build passes**

```bash
cd gui && npm run build
```

Expected: exits 0.

- [ ] **Step 3: Test in dev server**

```bash
cd gui && npm run dev
```

1. Navigate to Learn mode. Note the "N / total reviewed" badge on a Teacher run card.
2. Open that run's inspector, then open Crop Review.
3. Accept or reject a few crops.
4. Close Crop Review (Esc or Close button).
5. Without reloading the page, check the run card in Learn mode. Confirm the badge count updated immediately.

- [ ] **Step 4: Commit**

```bash
git add gui/src/components/RunInspector.tsx
git commit -m "fix: refresh global teacherDetails store when CropReview closes"
```

---

## Task 5: Add review badge to Optimize teacher picker

**Files:**
- Modify: `gui/src/modes/Optimize.tsx`

The Optimize teacher picker already receives full `RunDetail[]` objects from the global store, which means `t.manifest.review_status`, `t.manifest.n_frames_reviewed`, and `t.manifest.n_frames_total` are already available — no API change needed.

- [ ] **Step 1: Add the review badge to the teacher-picker row**

Find the `teacher-picker-row-top` div (around line 533 in `Optimize.tsx`):
```tsx
<div className="teacher-picker-row-top">
  <span className={`task-pill task-${t.manifest.task}`}>
    {t.manifest.task}
  </span>
  <span className="teacher-picker-prompt">
    {t.manifest.prompt}
  </span>
</div>
```

Add the badge inside `teacher-picker-row-top`, after the prompt span:
```tsx
<div className="teacher-picker-row-top">
  <span className={`task-pill task-${t.manifest.task}`}>
    {t.manifest.task}
  </span>
  <span className="teacher-picker-prompt">
    {t.manifest.prompt}
  </span>
  {t.manifest.review_status !== "unreviewed" && (
    <span className="teacher-picker-review-badge">
      {t.manifest.review_status === "approved"
        ? "Fully curated"
        : `${t.manifest.n_frames_reviewed}/${t.manifest.n_frames_total} reviewed`}
    </span>
  )}
</div>
```

Wording matches `Learn.tsx` exactly ("Fully curated" / "N/total reviewed").

- [ ] **Step 2: Verify build passes**

```bash
cd gui && npm run build
```

Expected: exits 0, no TypeScript errors.

- [ ] **Step 3: Test in dev server**

```bash
cd gui && npm run dev
```

1. Navigate to Optimize mode.
2. Open or scroll to the teacher picker panel.
3. Find a teacher run that has partial labeling (review_status = "in_progress"). Confirm a badge like "12/47 reviewed" appears next to the prompt.
4. Find a fully-labeled teacher (review_status = "approved"). Confirm badge shows "Fully curated".
5. Find an unreviewed teacher. Confirm no badge appears.

- [ ] **Step 4: Commit**

```bash
git add gui/src/modes/Optimize.tsx
git commit -m "feat: show review progress badge in Optimize teacher picker"
```

---

## Final Verification

After all four tasks are committed, do a full end-to-end check:

- [ ] `cd gui && npm run build` — clean build, no errors
- [ ] Run dev server and walk through the spec verification steps:
  1. Strip shows green/red/gray — click a segment, confirm navigation jumps correctly
  2. Close and reopen CropReview mid-session — confirm auto-jump + toast; confirm no toast on fresh or fully-labeled runs
  3. Label crops, close CropReview, confirm Learn page badge updates without reload
  4. Open Optimize teacher picker — confirm review badges are present and wording is consistent with Learn page
