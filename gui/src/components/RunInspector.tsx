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
  patchRunDisplayThreshold,
  runFrameUrl,
  toggleRejection as apiToggleRejection,
  type RejectionMap,
} from "../api";
import type { PerFrameLabels, RunDetail } from "../types";

interface Layout {
  /** Top-left of the *rendered image* inside its element box. */
  offsetX: number;
  offsetY: number;
  /** Pixel-space → element-space scaling. */
  sx: number;
  sy: number;
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
  const projectId = useStore((s) => s.currentProjectId);
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
  // Phase 2: post-hoc display threshold. Initialized from the manifest
  // when the run loads; the slider drives it live (client-side filter)
  // and a debounced effect PATCHes it back to the backend.
  const [threshold, setThreshold] = useState<number>(0.3);

  useEffect(() => {
    if (!runId || !projectId) return;
    setError(null);
    setDetail(null);
    setLabels(null);
    setRejections({});
    setFrameIdx(0);
    setHoveredIdx(null);
    setView(IDENTITY_VIEW);

    Promise.all([
      fetchRunDetail(projectId, runId),
      fetchRunLabels(projectId, runId),
      fetchRejections(projectId, runId),
    ])
      .then(([d, l, r]) => {
        setDetail(d);
        setLabels(l);
        setRejections(r);
        setThreshold(d.manifest.display_threshold);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [runId, projectId]);

  // PATCH the manifest after the slider settles. Skips when the value
  // already matches what's on disk (initial load, or the user dragged
  // back to the persisted value). Refresh the detail after the PATCH so
  // any server-recomputed stats land in the local copy.
  useEffect(() => {
    if (!runId || !projectId || !detail) return;
    if (threshold === detail.manifest.display_threshold) return;
    const handle = setTimeout(async () => {
      try {
        const updated = await patchRunDisplayThreshold(
          projectId,
          runId,
          threshold,
        );
        setDetail(updated);
      } catch (e) {
        console.error("PATCH display_threshold failed", e);
      }
    }, 300);
    return () => clearTimeout(handle);
  }, [threshold, projectId, runId, detail]);

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

  // Counts respect both the rejection state and the live threshold —
  // matches what the canvas + label panel actually paint, so the header
  // tally never disagrees with the visible boxes.
  const kept_total = useMemo(() => {
    if (!labels) return [0, 0] as const;
    let total = 0;
    let kept = 0;
    for (const l of labels) {
      const dropped = new Set(rejections[String(l.frame_idx)] ?? []);
      l.detections.forEach((d, i) => {
        if (d.score < threshold) return;
        total += 1;
        if (!dropped.has(i)) kept += 1;
      });
    }
    return [kept, total] as const;
  }, [labels, rejections, threshold]);

  const handleToggle = useCallback(
    async (detIdx: number) => {
      if (!runId || !projectId) return;
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
        const canonical = await apiToggleRejection(
          projectId,
          runId,
          frameIdx,
          detIdx,
        );
        setRejections(canonical);
      } catch (e) {
        console.error("toggle rejection failed", e);
        try {
          setRejections(await fetchRejections(projectId, runId));
        } catch {
          /* user can retry */
        }
      }
    },
    [projectId, runId, frameIdx],
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
            <label className="threshold-slider" title="Hide detections below this score. Persists to the run's manifest.">
              <span className="threshold-slider-label mono">
                ≥ {threshold.toFixed(2)}
              </span>
              <input
                type="range"
                min={0.05}
                max={1}
                step={0.01}
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
              />
            </label>
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
                src={runFrameUrl(projectId ?? "", runId, frameIdx, "raw")}
                view={view}
                onViewChange={setView}
              />
              <FrameView
                title="Detections — click a box to reject · scroll to zoom · drag to pan"
                src={runFrameUrl(projectId ?? "", runId, frameIdx, "raw")}
                detections={currentLabels?.detections ?? []}
                rejected={rejectedHere}
                hoveredIdx={hoveredIdx}
                onHover={setHoveredIdx}
                onToggle={handleToggle}
                view={view}
                onViewChange={setView}
                threshold={threshold}
              />
              <LabelPanel
                labels={currentLabels}
                rejected={rejectedHere}
                hoveredIdx={hoveredIdx}
                onHover={setHoveredIdx}
                onToggle={handleToggle}
                threshold={threshold}
              />
            </div>

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
  /** Phase 2: hide detections strictly below this score. Index identity
   *  is preserved (rejection toggles still address the original list);
   *  hidden boxes simply don't paint. */
  threshold?: number;
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
  threshold = 0,
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
        // Phase 2 post-hoc filter: skip painting low-confidence boxes.
        // Index identity is preserved so the hit-test + rejection toggle
        // still address the original detection list.
        if (d.score < threshold) return;
        const isRejected = rejected?.has(i) ?? false;
        const isHovered = hoveredIdx === i;
        const [x1, y1, x2, y2] = d.bbox_xyxy;
        const X = layout.offsetX + x1 * layout.sx;
        const Y = layout.offsetY + y1 * layout.sy;
        const W = (x2 - x1) * layout.sx;
        const H = (y2 - y1) * layout.sy;

        const stroke = isRejected ? "#ef4444" : "#22c55e";
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
          : "rgba(34,197,94,0.95)";
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
          ctx.fillStyle = "rgba(34,197,94,0.85)";
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
  }, [detections, rejected, hoveredIdx, src, view, threshold]);

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
  threshold = 0,
}: {
  labels: PerFrameLabels | undefined;
  rejected: Set<number>;
  hoveredIdx: number | null;
  onHover: (idx: number | null) => void;
  onToggle: (detIdx: number) => void;
  threshold?: number;
}) {
  // Mirror the canvas: skip below-threshold rows. Index identity is
  // preserved so the badge number lines up with the canvas badge.
  const visible = (labels?.detections ?? []).flatMap((d, i) =>
    d.score < threshold ? [] : [{ d, i }],
  );
  if (!labels || visible.length === 0) {
    return (
      <div className="label-panel empty">
        No detections on this frame.
      </div>
    );
  }
  const keptCount = visible.filter(({ i }) => !rejected.has(i)).length;
  return (
    <div className="label-panel">
      <div className="label-panel-title">
        {keptCount} kept / {visible.length} on this frame
      </div>
      <ul className="label-list">
        {visible.map(({ d, i }) => {
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
              >
                {i + 1}
              </span>
              <span className="label-class">{d.class_name}</span>
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
