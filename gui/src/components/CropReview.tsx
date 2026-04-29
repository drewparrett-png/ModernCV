/**
 * CropReview — keyboard-driven per-detection accept/reject.
 *
 * Walks every detection in a Teacher run, lowest-score first by default,
 * with hotkeys for navigate / accept / reject / undo. Sits alongside the
 * existing inspector — the user opens it from a button on the inspector
 * header and exits with Escape.
 *
 * State model: every accept/reject is a PUT to `frame_states/{frame_idx}`
 * with `state="curated"` and the updated `rejected_dets` list — exactly
 * the same write the inspector's per-detection toggle issues. We hold the
 * full ordered detection list locally and a `frameStates` map that mirrors
 * what's on disk; navigation uses local state, persistence is async.
 */

import {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type MouseEvent,
} from "react";
import { useStore } from "../store";
import {
  deleteFrameState,
  detectionCropUrl,
  fetchDetections,
  fetchFrameStates,
  putFrameState,
  runFrameUrl,
} from "../api";
import { useHotkeys } from "../hooks/useHotkeys";
import type { DetectionRow, FrameStatesMap } from "../types";

const PREFETCH_AHEAD = 3;
const CROP_PAD = 24;

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
  function handleClick(e: MouseEvent<HTMLDivElement>) {
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

interface UndoEntry {
  /** Position in the detection list at the time of the action — restored
   *  when undo runs so the user lands back on the affected detection. */
  index: number;
  /** Full prior frame state for the affected frame, or `null` if the
   *  frame had no entry before. */
  prior: { state: "curated"; rejected_dets: number[] } | null;
}

interface Props {
  onClose: () => void;
  threshold?: number;
}

export function CropReview({ onClose, threshold = 0 }: Props) {
  const projectId = useStore((s) => s.currentProjectId);
  const runId = useStore((s) => s.inspectingRunId);

  const [detections, setDetections] = useState<DetectionRow[]>([]);
  const [frameStates, setFrameStates] = useState<FrameStatesMap>({});
  const [index, setIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);

  // Action history for U-key undo. Bounded so a long session doesn't
  // grow unbounded — 50 is plenty for "I just hit the wrong key" recovery.
  const undoStack = useRef<UndoEntry[]>([]);

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

  // Live `accepted` view: derived from `frameStates`, not from the static
  // server response. The server snapshot is the initial state; subsequent
  // accept/reject hotkey presses mutate `frameStates` and we re-render.
  const liveDetections = useMemo<DetectionRow[]>(() => {
    return detections.map((d) => {
      const entry = frameStates[String(d.frame_idx)];
      const accepted =
        entry?.state === "curated" && entry.rejected_dets.includes(d.det_idx)
          ? false
          : true;
      return { ...d, accepted };
    });
  }, [detections, frameStates]);

  const total = liveDetections.length;
  const cur = liveDetections[index];

  const reviewedCount = useMemo(() => {
    let n = 0;
    for (const d of liveDetections) {
      const entry = frameStates[String(d.frame_idx)];
      if (entry?.state === "curated") n += 1;
    }
    return n;
  }, [liveDetections, frameStates]);

  const stripColors = useMemo<SegColor[]>(
    () =>
      liveDetections.map((d) => {
        const entry = frameStates[String(d.frame_idx)];
        if (!entry) return "unreviewed";
        return d.accepted ? "accepted" : "rejected";
      }),
    [liveDetections, frameStates],
  );

  // Persist the current frame's intended `rejected_dets` for the given
  // detection. Returns the prior frame entry (for undo) or null if the
  // frame was unreviewed.
  const writeAccept = useCallback(
    async (
      frame_idx: number,
      det_idx: number,
      shouldReject: boolean,
    ): Promise<UndoEntry["prior"]> => {
      if (!projectId || !runId) return null;
      const key = String(frame_idx);
      const prevEntry = frameStates[key];
      const prevRejected = new Set(
        prevEntry?.state === "curated" ? prevEntry.rejected_dets : [],
      );
      const nextRejected = new Set(prevRejected);
      if (shouldReject) nextRejected.add(det_idx);
      else nextRejected.delete(det_idx);
      const nextList = [...nextRejected].sort((a, b) => a - b);

      // Optimistic.
      setFrameStates((p) => ({
        ...p,
        [key]: { state: "curated", rejected_dets: nextList },
      }));

      try {
        const updated = await putFrameState(projectId, runId, frame_idx, {
          state: "curated",
          rejected_dets: nextList,
        });
        setFrameStates((p) => ({ ...p, [key]: updated }));
      } catch (e) {
        console.error("accept/reject failed", e);
        // Roll back to pre-action local state.
        setFrameStates((p) => {
          const next = { ...p };
          if (prevEntry) next[key] = prevEntry;
          else delete next[key];
          return next;
        });
        throw e;
      }

      return prevEntry?.state === "curated"
        ? {
            state: "curated",
            rejected_dets: [...prevEntry.rejected_dets],
          }
        : null;
    },
    [projectId, runId, frameStates],
  );

  const accept = useCallback(async () => {
    if (!cur) return;
    try {
      const prior = await writeAccept(cur.frame_idx, cur.det_idx, false);
      undoStack.current.push({ index, prior });
      if (undoStack.current.length > 50) undoStack.current.shift();
      if (index < total - 1) setIndex(index + 1);
    } catch {
      /* writeAccept handled rollback */
    }
  }, [cur, writeAccept, index, total]);

  const reject = useCallback(async () => {
    if (!cur) return;
    try {
      const prior = await writeAccept(cur.frame_idx, cur.det_idx, true);
      undoStack.current.push({ index, prior });
      if (undoStack.current.length > 50) undoStack.current.shift();
      if (index < total - 1) setIndex(index + 1);
    } catch {
      /* writeAccept handled rollback */
    }
  }, [cur, writeAccept, index, total]);

  const undo = useCallback(async () => {
    const last = undoStack.current.pop();
    if (!last || !projectId || !runId) return;
    const target = liveDetections[last.index];
    if (!target) return;
    const key = String(target.frame_idx);

    // Optimistic restore.
    setFrameStates((p) => {
      const next = { ...p };
      if (last.prior) next[key] = last.prior;
      else delete next[key];
      return next;
    });
    setIndex(last.index);

    try {
      if (last.prior) {
        await putFrameState(projectId, runId, target.frame_idx, {
          state: "curated",
          rejected_dets: last.prior.rejected_dets,
        });
      } else {
        // Restoring to "no entry" requires the dedicated DELETE endpoint.
        await deleteFrameState(projectId, runId, target.frame_idx);
      }
    } catch (e) {
      console.error("undo failed", e);
      // Best-effort: refetch the canonical state on failure.
      try {
        setFrameStates(await fetchFrameStates(projectId, runId));
      } catch {
        /* user can retry */
      }
    }
  }, [projectId, runId, liveDetections]);

  const next = useCallback(() => {
    if (index < total - 1) setIndex(index + 1);
  }, [index, total]);
  const prev = useCallback(() => {
    if (index > 0) setIndex(index - 1);
  }, [index]);

  useHotkeys(
    {
      ArrowRight: next,
      ArrowLeft: prev,
      a: accept,
      y: accept,
      r: reject,
      n: reject,
      u: undo,
      Escape: onClose,
    },
    loaded,
  );

  // Prefetch the next few crops into the browser cache so navigation
  // feels instant. `Image()` requests bypass React; the browser cache
  // serves the same URL when our <img> mounts.
  useEffect(() => {
    if (!projectId || !runId) return;
    for (let i = 1; i <= PREFETCH_AHEAD; i++) {
      const target = liveDetections[index + i];
      if (!target) break;
      const img = new Image();
      img.src = detectionCropUrl(
        projectId,
        runId,
        target.frame_idx,
        target.det_idx,
        CROP_PAD,
      );
    }
  }, [projectId, runId, liveDetections, index]);

  if (!projectId || !runId) return null;

  return (
    <div className="inspector-overlay" role="dialog" aria-modal="true">
      <div className="inspector">
        <header className="inspector-header">
          <div>
            <h2>Crop review</h2>
            <div className="inspector-subtitle">
              ←/→ navigate · A/Y accept · R/N reject · U undo · Esc exit
            </div>
          </div>
          <div className="inspector-header-actions">
            {loaded && total > 0 && (
              <span className="crop-review-pos mono">
                {index + 1} / {total} · {reviewedCount} reviewed
              </span>
            )}
            <button type="button" className="close-button" onClick={onClose}>
              Close
            </button>
          </div>
        </header>

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

        {loaded && cur && (
          <div className="crop-review-body">
            <div className="crop-review-main">
              <img
                key={`${cur.frame_idx}-${cur.det_idx}`}
                className="crop-review-img"
                src={detectionCropUrl(
                  projectId,
                  runId,
                  cur.frame_idx,
                  cur.det_idx,
                  CROP_PAD,
                )}
                alt={`crop frame=${cur.frame_idx} det=${cur.det_idx}`}
              />
              <div
                className={
                  "crop-review-verdict " +
                  (cur.accepted ? "verdict-accepted" : "verdict-rejected")
                }
              >
                {cur.accepted ? "Accepted" : "Rejected"}
              </div>
            </div>
            <aside className="crop-review-sidebar">
              <div className="crop-review-meta">
                <div className="crop-review-meta-row">
                  <span className="crop-review-meta-label">Class</span>
                  <span className="mono">{cur.class_name}</span>
                </div>
                <div className="crop-review-meta-row">
                  <span className="crop-review-meta-label">Score</span>
                  <span className="mono">{cur.score.toFixed(3)}</span>
                </div>
                <div className="crop-review-meta-row">
                  <span className="crop-review-meta-label">Frame</span>
                  <span className="mono">{cur.frame_idx}</span>
                </div>
                <div className="crop-review-meta-row">
                  <span className="crop-review-meta-label">Det idx</span>
                  <span className="mono">{cur.det_idx}</span>
                </div>
              </div>
              <div className="crop-review-thumb">
                <img
                  className="crop-review-thumb-img"
                  src={runFrameUrl(projectId, runId, cur.frame_idx, "raw")}
                  alt={`source frame ${cur.frame_idx}`}
                />
                <div className="crop-review-thumb-caption">Source frame</div>
              </div>
              <div className="crop-review-actions">
                <button type="button" onClick={prev} disabled={index === 0}>
                  ←
                </button>
                <button
                  type="button"
                  onClick={next}
                  disabled={index >= total - 1}
                >
                  →
                </button>
                <button type="button" onClick={accept}>
                  Accept (A)
                </button>
                <button type="button" onClick={reject}>
                  Reject (R)
                </button>
                <button
                  type="button"
                  onClick={undo}
                  disabled={undoStack.current.length === 0}
                >
                  Undo (U)
                </button>
              </div>
            </aside>
          </div>
        )}
      </div>
    </div>
  );
}
