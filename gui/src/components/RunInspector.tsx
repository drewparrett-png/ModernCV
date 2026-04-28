/**
 * Run Inspector — frame-by-frame drill-in for a Teacher run.
 *
 * Layout: source frame on the left, source frame + live-rendered detection
 * overlay on the right. Both panes share a *single* zoom/pan view so they
 * stay locked to the same region — scroll-zoom or drag-pan on either pane
 * and the other follows. That's how you actually pick out individual
 * detections in dense scenes (a soccer huddle with 19 boxes).
 *
 * The right canvas is interactive: click any box to toggle reject. Numbered
 * badges on boxes match numbered rows in the side label panel; hover either
 * side highlights the other.
 *
 * Rendering note: image uses `object-fit: contain`, so the pixels live
 * inside a smaller-than-the-element rectangle (with letterbox bars). The
 * canvas paints at the *unscaled* layout dimensions; the wrapper's CSS
 * transform handles the visual zoom. Hit-test math converts event coords
 * back to layout pixels by dividing by the bounding rect (which reflects
 * the transform).
 */

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type WheelEvent as ReactWheelEvent,
  type MouseEvent as ReactMouseEvent,
} from "react";
import { useStore } from "../store";
import {
  fetchRejections,
  fetchRunDetail,
  fetchRunLabels,
  runFrameUrl,
  toggleRejection as apiToggleRejection,
  type RejectionMap,
} from "../api";
import type { PerFrameLabels, RunDetail } from "../types";
import { colorForClass } from "../classColors";

interface Layout {
  /** Top-left of the *rendered image* inside its element box. */
  offsetX: number;
  offsetY: number;
  /** Pixel-space → element-space scaling. */
  sx: number;
  sy: number;
}

/** "#22c55e" + 0.85 → "rgba(34,197,94,0.85)". Used for badges/score chips
 *  whose fill color is derived from a class color but needs alpha. */
function hexToRgba(hex: string, alpha: number): string {
  const h = hex.replace("#", "");
  const v =
    h.length === 3
      ? h.split("").map((c) => parseInt(c + c, 16))
      : [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  return `rgba(${v[0]}, ${v[1]}, ${v[2]}, ${alpha})`;
}

/** Compute the contain-fit rectangle: where the image actually paints
 *  inside its element, given object-fit: contain. */
function computeLayout(
  containerW: number,
  containerH: number,
  naturalW: number,
  naturalH: number,
): Layout {
  if (!naturalW || !naturalH || !containerW || !containerH) {
    return { offsetX: 0, offsetY: 0, sx: 1, sy: 1 };
  }
  const containerAspect = containerW / containerH;
  const imageAspect = naturalW / naturalH;
  let renderedW: number;
  let renderedH: number;
  let offsetX: number;
  let offsetY: number;
  if (imageAspect > containerAspect) {
    renderedW = containerW;
    renderedH = containerW / imageAspect;
    offsetX = 0;
    offsetY = (containerH - renderedH) / 2;
  } else {
    renderedW = containerH * imageAspect;
    renderedH = containerH;
    offsetX = (containerW - renderedW) / 2;
    offsetY = 0;
  }
  return {
    offsetX,
    offsetY,
    sx: renderedW / naturalW,
    sy: renderedH / naturalH,
  };
}

function hitTestBox(
  x: number,
  y: number,
  detections: PerFrameLabels["detections"],
  layout: Layout,
): number | null {
  const imgX = (x - layout.offsetX) / layout.sx;
  const imgY = (y - layout.offsetY) / layout.sy;
  let best: number | null = null;
  let bestArea = Infinity;
  detections.forEach((d, i) => {
    const [x1, y1, x2, y2] = d.bbox_xyxy;
    if (imgX < x1 || imgX > x2 || imgY < y1 || imgY > y2) return;
    const area = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    if (area < bestArea) {
      best = i;
      bestArea = area;
    }
  });
  return best;
}

// ---- Shared zoom/pan view state -------------------------------------------

interface View {
  zoom: number;
  panX: number;
  panY: number;
}

const IDENTITY_VIEW: View = { zoom: 1, panX: 0, panY: 0 };
const MIN_ZOOM = 1;
const MAX_ZOOM = 10;

function clampZoom(z: number): number {
  return Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z));
}

// ---- Inspector ------------------------------------------------------------

export function RunInspector() {
  const runId = useStore((s) => s.inspectingRunId);
  const close = useStore((s) => s.closeInspector);

  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [labels, setLabels] = useState<PerFrameLabels[] | null>(null);
  const [rejections, setRejections] = useState<RejectionMap>({});
  const [error, setError] = useState<string | null>(null);
  const [frameIdx, setFrameIdx] = useState(0);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  // Shared view across both panes — wheel/drag on either updates this and
  // both re-render in lockstep.
  const [view, setView] = useState<View>(IDENTITY_VIEW);

  useEffect(() => {
    if (!runId) return;
    setError(null);
    setDetail(null);
    setLabels(null);
    setRejections({});
    setFrameIdx(0);
    setHoveredIdx(null);
    setView(IDENTITY_VIEW);

    Promise.all([
      fetchRunDetail(runId),
      fetchRunLabels(runId),
      fetchRejections(runId),
    ])
      .then(([d, l, r]) => {
        setDetail(d);
        setLabels(l);
        setRejections(r);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [runId]);

  // Reset the zoom every time the user steps to a different frame — keeping
  // a 5x zoom across an entire scrub would be more disorienting than useful.
  useEffect(() => {
    setView(IDENTITY_VIEW);
  }, [frameIdx]);

  const labelByFrame = useMemo(() => {
    if (!labels) return new Map<number, PerFrameLabels>();
    const m = new Map<number, PerFrameLabels>();
    for (const l of labels) m.set(l.frame_idx, l);
    return m;
  }, [labels]);

  const kept_total = useMemo(() => {
    if (!labels) return [0, 0] as const;
    let total = 0;
    let rejected = 0;
    for (const l of labels) {
      total += l.detections.length;
      const dropped = rejections[String(l.frame_idx)];
      if (dropped) rejected += dropped.length;
    }
    return [total - rejected, total] as const;
  }, [labels, rejections]);

  const handleToggle = useCallback(
    async (detIdx: number) => {
      if (!runId) return;
      setRejections((prev) => {
        const key = String(frameIdx);
        const cur = new Set(prev[key] ?? []);
        if (cur.has(detIdx)) cur.delete(detIdx);
        else cur.add(detIdx);
        const next = { ...prev };
        if (cur.size === 0) delete next[key];
        else next[key] = [...cur].sort((a, b) => a - b);
        return next;
      });
      try {
        const canonical = await apiToggleRejection(runId, frameIdx, detIdx);
        setRejections(canonical);
      } catch (e) {
        console.error("toggle rejection failed", e);
        try {
          setRejections(await fetchRejections(runId));
        } catch {
          /* user can retry */
        }
      }
    },
    [runId, frameIdx],
  );

  if (!runId) return null;

  const totalFrames = detail?.stats?.frames_processed ?? labels?.length ?? 0;
  const maxIdx = Math.max(0, totalFrames - 1);
  const currentLabels = labelByFrame.get(frameIdx);
  const rejectedHere = new Set(rejections[String(frameIdx)] ?? []);

  return (
    <div className="inspector-overlay" role="dialog" aria-modal="true">
      <div className="inspector">
        <header className="inspector-header">
          <div>
            <h2>Run Inspector</h2>
            {detail && (
              <div className="inspector-subtitle mono">
                {detail.manifest.id} · {detail.manifest.task} ·{" "}
                {detail.manifest.prompt}
              </div>
            )}
            {labels && (
              <div className="inspector-keepcount">
                {kept_total[0]} kept / {kept_total[1]} total detections
              </div>
            )}
          </div>
          <div className="inspector-header-actions">
            <span className="zoom-indicator mono">
              {view.zoom.toFixed(1)}×
            </span>
            <button
              type="button"
              className="zoom-reset"
              onClick={() => setView(IDENTITY_VIEW)}
              disabled={view.zoom === 1 && view.panX === 0 && view.panY === 0}
              title="Reset zoom"
            >
              Reset zoom
            </button>
            <button type="button" className="close-button" onClick={close}>
              Close
            </button>
          </div>
        </header>

        {error && <div className="inspector-error">Failed to load: {error}</div>}
        {!error && !detail && <div className="inspector-loading">Loading…</div>}
        {detail && totalFrames === 0 && (
          <div className="inspector-empty">
            This run has no frames recorded yet.
          </div>
        )}

        {detail && totalFrames > 0 && (
          <div className="inspector-body">
            <div className="inspector-grid">
              <FrameView
                title="Source frame"
                src={runFrameUrl(runId, frameIdx, "raw")}
                view={view}
                onViewChange={setView}
              />
              <FrameView
                title="Detections — click a box to reject · scroll to zoom · drag to pan"
                src={runFrameUrl(runId, frameIdx, "raw")}
                detections={currentLabels?.detections ?? []}
                rejected={rejectedHere}
                hoveredIdx={hoveredIdx}
                onHover={setHoveredIdx}
                onToggle={handleToggle}
                view={view}
                onViewChange={setView}
              />
              <LabelPanel
                labels={currentLabels}
                rejected={rejectedHere}
                hoveredIdx={hoveredIdx}
                onHover={setHoveredIdx}
                onToggle={handleToggle}
              />
            </div>

            {labels && labels.length > 0 && (
              <PerFrameCountChart
                labels={labels}
                rejections={rejections}
                totalFrames={totalFrames}
                currentFrame={frameIdx}
                onSeek={setFrameIdx}
              />
            )}

            <div className="scrubber-row">
              <button
                type="button"
                onClick={() => setFrameIdx(Math.max(0, frameIdx - 10))}
              >
                −10
              </button>
              <button
                type="button"
                onClick={() => setFrameIdx(Math.max(0, frameIdx - 1))}
              >
                ←
              </button>
              <input
                type="range"
                min={0}
                max={maxIdx}
                value={frameIdx}
                onChange={(e) => setFrameIdx(Number(e.target.value))}
                className="scrubber"
              />
              <button
                type="button"
                onClick={() => setFrameIdx(Math.min(maxIdx, frameIdx + 1))}
              >
                →
              </button>
              <button
                type="button"
                onClick={() => setFrameIdx(Math.min(maxIdx, frameIdx + 10))}
              >
                +10
              </button>
              <span className="scrubber-pos mono">
                frame {frameIdx} / {maxIdx}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

interface FrameViewProps {
  title: string;
  src: string;
  detections?: PerFrameLabels["detections"];
  rejected?: Set<number>;
  hoveredIdx?: number | null;
  onHover?: (idx: number | null) => void;
  onToggle?: (idx: number) => void;
  view: View;
  onViewChange: (v: View) => void;
}

function FrameView({
  title,
  src,
  detections,
  rejected,
  hoveredIdx,
  onHover,
  onToggle,
  view,
  onViewChange,
}: FrameViewProps) {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const layoutRef = useRef<Layout>({ offsetX: 0, offsetY: 0, sx: 1, sy: 1 });

  // Drag state — kept in a ref so handlers don't need to be wrapped in
  // setState/closures. `moved` lets us suppress the click that fires after
  // a drag (browsers fire click on mouseup even if dragging happened).
  const dragRef = useRef<
    null | {
      startClientX: number;
      startClientY: number;
      startPanX: number;
      startPanY: number;
      moved: boolean;
    }
  >(null);

  // The canvas now lives *outside* the zoom-transformed wrapper (the image
  // is the only thing that scales via CSS). The paint code multiplies
  // layout.sx/sy by view.zoom and adds view.panX/Y to the offsets so the
  // boxes still wrap the right pixels — but badges (fixed 11px radius) and
  // score labels (fixed 12px font) stay at constant screen size, so a 5×
  // zoom doesn't make a single label swallow the whole frame.
  useEffect(() => {
    if (!detections) return;
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return;

    const paint = () => {
      // Canvas is sized to its *parent stage* (it's a sibling of the zoom
      // wrapper, not inside it), so we use the canvas's own client rect.
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      if (!img.naturalWidth) return;

      // Base layout: where the contain-fit image *would* paint at zoom=1
      // inside this canvas-sized rectangle. Then we apply the user's
      // zoom/pan to that, manually, so we control which parts grow.
      const base = computeLayout(
        w,
        h,
        img.naturalWidth,
        img.naturalHeight,
      );
      const layout: Layout = {
        offsetX: base.offsetX * view.zoom + view.panX,
        offsetY: base.offsetY * view.zoom + view.panY,
        sx: base.sx * view.zoom,
        sy: base.sy * view.zoom,
      };
      layoutRef.current = layout;

      ctx.font = "12px ui-monospace, Menlo, monospace";
      ctx.textBaseline = "alphabetic";

      detections.forEach((d, i) => {
        const isRejected = rejected?.has(i) ?? false;
        const isHovered = hoveredIdx === i;
        const [x1, y1, x2, y2] = d.bbox_xyxy;
        const X = layout.offsetX + x1 * layout.sx;
        const Y = layout.offsetY + y1 * layout.sy;
        const W = (x2 - x1) * layout.sx;
        const H = (y2 - y1) * layout.sy;

        // Kept detections take their class's color so you can pick
        // "balls vs players" at a glance in dense scenes. Rejected stays
        // red + dashed regardless of class — the action overrides identity.
        const classColor = colorForClass(d.class_name);
        const stroke = isRejected ? "#ef4444" : classColor;
        ctx.lineWidth = isHovered ? 3 : 2;
        ctx.setLineDash(isRejected ? [5, 4] : []);
        ctx.strokeStyle = stroke;
        if (isHovered) {
          ctx.shadowColor = stroke;
          ctx.shadowBlur = 10;
        } else {
          ctx.shadowBlur = 0;
        }
        ctx.strokeRect(X, Y, W, H);
        ctx.shadowBlur = 0;
        ctx.setLineDash([]);

        const badgeR = 11;
        const badgeX = X + badgeR;
        const badgeY = Y + badgeR;
        ctx.fillStyle = isRejected
          ? "rgba(239,68,68,0.95)"
          : hexToRgba(classColor, 0.95);
        ctx.beginPath();
        ctx.arc(badgeX, badgeY, badgeR, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#fff";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(i + 1), badgeX, badgeY + 0.5);
        ctx.textAlign = "start";
        ctx.textBaseline = "alphabetic";

        if (!isRejected) {
          const label = `${d.score.toFixed(2)}`;
          const textW = ctx.measureText(label).width + 8;
          const textH = 16;
          ctx.fillStyle = hexToRgba(classColor, 0.85);
          ctx.fillRect(
            X + W - textW,
            Math.max(0, Y - textH),
            textW,
            textH,
          );
          ctx.fillStyle = "#fff";
          ctx.fillText(label, X + W - textW + 4, Math.max(textH - 4, Y - 4));
        }
      });
    };

    if (img.complete) paint();
    img.addEventListener("load", paint);
    const ro = new ResizeObserver(paint);
    ro.observe(canvas);
    ro.observe(img);
    return () => {
      img.removeEventListener("load", paint);
      ro.disconnect();
    };
  }, [detections, rejected, hoveredIdx, src, view]);

  const interactive = Boolean(detections && (onHover || onToggle));

  /** Mouse event → canvas-pixel coordinates. The canvas is no longer inside
   *  the zoom-transformed wrapper, so a plain rect-relative subtraction is
   *  correct. The hit-test then uses `layoutRef.current`, which already has
   *  zoom/pan baked in. */
  function eventToLayoutXY(e: { clientX: number; clientY: number }): {
    x: number;
    y: number;
  } {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  /** Wheel zoom anchored at the cursor — keeps the same world point under
   *  the pointer fixed across the zoom. */
  function handleWheel(e: ReactWheelEvent<HTMLDivElement>) {
    const stage = stageRef.current;
    if (!stage) return;
    e.preventDefault();
    const stageRect = stage.getBoundingClientRect();
    const cx = e.clientX - stageRect.left;
    const cy = e.clientY - stageRect.top;
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const newZoom = clampZoom(view.zoom * factor);
    if (newZoom === view.zoom) return;
    // World point under the cursor in the un-zoomed wrapper:
    const worldX = (cx - view.panX) / view.zoom;
    const worldY = (cy - view.panY) / view.zoom;
    let newPanX = cx - worldX * newZoom;
    let newPanY = cy - worldY * newZoom;
    if (newZoom === 1) {
      // Snap to origin when fully zoomed out so a stray pan doesn't leave
      // the image off-center.
      newPanX = 0;
      newPanY = 0;
    }
    onViewChange({ zoom: newZoom, panX: newPanX, panY: newPanY });
  }

  function handleMouseDown(e: ReactMouseEvent<HTMLDivElement>) {
    // Only start a drag when zoomed in — at 1× there's nothing to pan.
    if (view.zoom <= 1) return;
    if (e.button !== 0) return; // primary button only
    dragRef.current = {
      startClientX: e.clientX,
      startClientY: e.clientY,
      startPanX: view.panX,
      startPanY: view.panY,
      moved: false,
    };
  }

  function handleMouseMove(e: ReactMouseEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (drag) {
      const dx = e.clientX - drag.startClientX;
      const dy = e.clientY - drag.startClientY;
      if (!drag.moved && Math.hypot(dx, dy) > 3) drag.moved = true;
      onViewChange({
        zoom: view.zoom,
        panX: drag.startPanX + dx,
        panY: drag.startPanY + dy,
      });
      return;
    }
    if (!interactive) return;
    const { x, y } = eventToLayoutXY(e);
    const idx = hitTestBox(x, y, detections!, layoutRef.current);
    onHover?.(idx);
  }

  function handleMouseUp(_e: ReactMouseEvent<HTMLDivElement>) {
    // We let onClick fire below; it'll consult dragRef.moved to decide
    // whether to actually toggle.
  }

  function handleMouseLeave() {
    onHover?.(null);
    // If the user dragged off the canvas, end the drag — otherwise they'd
    // never get a mouseup if their mouse leaves the window mid-drag.
    if (dragRef.current) dragRef.current = null;
  }

  function handleClick(e: ReactMouseEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    dragRef.current = null;
    if (drag?.moved) return; // suppress click that came after a pan
    if (!interactive) return;
    const { x, y } = eventToLayoutXY(e);
    const idx = hitTestBox(x, y, detections!, layoutRef.current);
    if (idx !== null) onToggle?.(idx);
  }

  const transform = `translate(${view.panX}px, ${view.panY}px) scale(${view.zoom})`;
  const cursor =
    view.zoom > 1
      ? dragRef.current
        ? "grabbing"
        : "grab"
      : interactive
        ? "pointer"
        : "default";

  return (
    <div className="frame-view">
      <div className="frame-title">{title}</div>
      <div
        className="frame-stage"
        ref={stageRef}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
        style={{ cursor }}
      >
        {/* The image is the only thing that scales via CSS transform.
            The canvas (sibling, not child) stays at the stage's natural
            size, and the paint code applies the same zoom/pan to box
            geometry while keeping badges + text at fixed screen size. */}
        <div
          className="frame-zoom-wrapper"
          style={{ transform, transformOrigin: "0 0" }}
        >
          <img
            ref={imgRef}
            key={src}
            src={src}
            alt={title}
            className="frame-img"
            draggable={false}
          />
        </div>
        {detections && (
          <canvas
            ref={canvasRef}
            className={`frame-canvas ${interactive ? "interactive" : ""}`}
          />
        )}
      </div>
    </div>
  );
}

function LabelPanel({
  labels,
  rejected,
  hoveredIdx,
  onHover,
  onToggle,
}: {
  labels: PerFrameLabels | undefined;
  rejected: Set<number>;
  hoveredIdx: number | null;
  onHover: (idx: number | null) => void;
  onToggle: (detIdx: number) => void;
}) {
  if (!labels || labels.detections.length === 0) {
    return (
      <div className="label-panel empty">
        No detections on this frame.
      </div>
    );
  }
  const keptCount = labels.detections.length - rejected.size;
  return (
    <div className="label-panel">
      <div className="label-panel-title">
        {keptCount} kept / {labels.detections.length} on this frame
      </div>
      <ul className="label-list">
        {labels.detections.map((d, i) => {
          const isRejected = rejected.has(i);
          const isHovered = hoveredIdx === i;
          return (
            <li
              key={i}
              className={`label-item ${isRejected ? "rejected" : ""} ${
                isHovered ? "hovered" : ""
              }`}
              onMouseEnter={() => onHover(i)}
              onMouseLeave={() => onHover(null)}
              onClick={() => onToggle(i)}
              role="button"
              title={
                isRejected
                  ? "Click to restore this detection"
                  : "Click to reject this detection"
              }
            >
              <span
                className={`label-badge ${isRejected ? "rejected" : ""}`}
                style={
                  isRejected
                    ? undefined
                    : { backgroundColor: colorForClass(d.class_name) }
                }
              >
                {i + 1}
              </span>
              <span className="label-class">
                <span
                  className="label-swatch"
                  style={{ backgroundColor: colorForClass(d.class_name) }}
                  aria-hidden
                />
                {d.class_name}
              </span>
              <span className="label-score">{d.score.toFixed(2)}</span>
              <span
                className={`label-action ${isRejected ? "restore" : ""}`}
              >
                {isRejected ? "↶ restore" : "✕ reject"}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

/**
 * Per-frame detection-count chart.
 *
 * One line per class, x-axis is frame index, y-axis is **kept** count of
 * detections of that class in that frame (rejections filtered out). A
 * vertical marker tracks the current frame; clicking anywhere on the chart
 * seeks to the corresponding frame.
 *
 * Why this lives in the Inspector (not the Teachers card): the chart is
 * useful precisely *because* you can scrub to interesting points — spikes
 * (cluster of players) or drops (ball off-screen). It's the "step through
 * to understand" affordance.
 *
 * Why kept-only (not raw): during curation you want to see what your
 * dataset will look like after rejections, not what the detector originally
 * proposed. Toggling a reject in the side panel makes the chart redraw
 * immediately, which is useful for spotting "I just nuked the only ball
 * detection in frame 17" — a draftsman's view of curation impact.
 *
 * Implementation notes:
 *   - Counts are computed client-side from the labels array we already
 *     loaded for box rendering — no extra fetch.
 *   - Frames missing from labels are treated as count=0 (matches learn.py:
 *     every processed frame produces a JSONL line, even empty ones).
 *   - Classes that exist in raw labels but are *fully rejected* in every
 *     frame still render in the legend (all-zero line). Keeping them in
 *     the legend matches expectations — curating away a class shouldn't
 *     make it vanish from the chart you're using to make decisions.
 *   - Total frame width may exceed canvas width — we map frame_idx →
 *     fractional pixel. For 50 frames in a 700px-wide chart this is
 *     generous; for thousands, the line is a smooth densogram.
 */
function PerFrameCountChart({
  labels,
  rejections,
  totalFrames,
  currentFrame,
  onSeek,
}: {
  labels: PerFrameLabels[];
  rejections: RejectionMap;
  totalFrames: number;
  currentFrame: number;
  onSeek: (frame: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Build per-class series indexed by frame_idx, plus the legend list. We
  // keep classes from the *raw* label set so the legend is stable across
  // curation; series values are kept-only counts.
  const { classes, series, total, maxCount, totalRejected } = useMemo(() => {
    const allClasses = new Set<string>();
    for (const l of labels) {
      for (const d of l.detections) allClasses.add(d.class_name);
    }
    const classes = [...allClasses].sort();
    const n = Math.max(totalFrames, ...labels.map((l) => l.frame_idx + 1));
    const series = new Map<string, number[]>();
    for (const c of classes) series.set(c, new Array(n).fill(0));
    const total = new Array(n).fill(0);
    let maxCount = 0;
    let totalRejected = 0;
    for (const l of labels) {
      const idx = l.frame_idx;
      if (idx < 0 || idx >= n) continue;
      const rejectedHere = new Set(rejections[String(idx)] ?? []);
      l.detections.forEach((d, detIdx) => {
        if (rejectedHere.has(detIdx)) {
          totalRejected += 1;
          return; // kept-only series
        }
        const arr = series.get(d.class_name);
        if (arr) arr[idx] += 1;
        total[idx] += 1;
      });
      if (total[idx] > maxCount) maxCount = total[idx];
    }
    return { classes, series, total, maxCount, totalRejected };
  }, [labels, rejections, totalFrames]);

  // Repaint when state changes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth;
    const cssH = canvas.clientHeight;
    if (canvas.width !== cssW * dpr || canvas.height !== cssH * dpr) {
      canvas.width = cssW * dpr;
      canvas.height = cssH * dpr;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    // Layout: left padding for the y-axis tick, bottom padding for x ticks
    const PADL = 24;
    const PADR = 8;
    const PADT = 6;
    const PADB = 14;
    const W = cssW - PADL - PADR;
    const H = cssH - PADT - PADB;
    const n = total.length;
    if (n === 0 || W <= 0 || H <= 0) return;
    const yMax = Math.max(1, maxCount); // avoid div/0
    const xAt = (i: number) => PADL + (i * W) / Math.max(1, n - 1);
    const yAt = (v: number) => PADT + H - (v / yMax) * H;

    // Faint baseline + max gridlines
    ctx.strokeStyle = "#e5e7eb";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PADL, PADT + H);
    ctx.lineTo(PADL + W, PADT + H);
    ctx.moveTo(PADL, PADT);
    ctx.lineTo(PADL + W, PADT);
    ctx.stroke();
    ctx.fillStyle = "#9ca3af";
    ctx.font = "10px ui-monospace, Menlo, monospace";
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "right";
    ctx.fillText(String(yMax), PADL - 4, PADT + 8);
    ctx.fillText("0", PADL - 4, PADT + H);
    ctx.textAlign = "start";

    // Per-class lines
    classes.forEach((cls) => {
      const arr = series.get(cls);
      if (!arr) return;
      ctx.strokeStyle = colorForClass(cls);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < arr.length; i++) {
        const x = xAt(i);
        const y = yAt(arr[i]);
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    });

    // Current-frame marker
    if (currentFrame >= 0 && currentFrame < n) {
      const x = xAt(currentFrame);
      ctx.strokeStyle = "#111827";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x, PADT);
      ctx.lineTo(x, PADT + H);
      ctx.stroke();
      ctx.setLineDash([]);

      // Tooltip near the marker: frame N · per-class counts
      const tipParts: string[] = [`f${currentFrame}`];
      for (const cls of classes) {
        const v = series.get(cls)?.[currentFrame] ?? 0;
        if (v > 0) tipParts.push(`${cls}:${v}`);
      }
      const text = tipParts.join("  ");
      const tw = ctx.measureText(text).width + 10;
      const tipX = Math.min(PADL + W - tw, Math.max(PADL, x + 4));
      ctx.fillStyle = "rgba(17,24,39,0.85)";
      ctx.fillRect(tipX, PADT, tw, 16);
      ctx.fillStyle = "#fff";
      ctx.fillText(text, tipX + 5, PADT + 12);
    }
  }, [classes, series, total, maxCount, currentFrame]);

  const handleClick = useCallback(
    (e: ReactMouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const PADL = 24;
      const PADR = 8;
      const W = rect.width - PADL - PADR;
      const x = e.clientX - rect.left - PADL;
      const n = total.length;
      if (n === 0 || W <= 0) return;
      const frac = Math.max(0, Math.min(1, x / W));
      onSeek(Math.round(frac * (n - 1)));
    },
    [onSeek, total.length],
  );

  return (
    <div className="frame-chart-row">
      <div className="frame-chart-legend">
        <span
          className="frame-chart-mode"
          title="Chart shows kept counts — rejected detections are excluded. Toggle a reject in the panel and watch the line move."
        >
          kept
          {totalRejected > 0 && (
            <span className="frame-chart-rejected-note">
              {" "}
              · {totalRejected} rejected hidden
            </span>
          )}
        </span>
        {classes.map((c) => (
          <span key={c} className="frame-chart-legend-item">
            <span
              className="frame-chart-swatch"
              style={{ backgroundColor: colorForClass(c) }}
            />
            <span className="frame-chart-legend-label">{c}</span>
          </span>
        ))}
      </div>
      <canvas
        ref={canvasRef}
        className="frame-chart"
        onClick={handleClick}
        title="Click to jump to that frame"
      />
    </div>
  );
}
