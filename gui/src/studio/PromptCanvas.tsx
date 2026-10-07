/**
 * PromptCanvas — the image surface shared by Label, Test and Pallet.
 *
 * Draws an image plus overlay shapes (annotations, suggestions, SAM
 * previews, predictions, pallet results) and turns pointer input into
 * prompts:
 *
 *   Box tool     drag → box prompt            (⌥-drag → negative box)
 *   Brush tool   paint → positive stroke      (⌥ or right-drag → negative)
 *   Erase tool   paint → negative stroke
 *   Select tool  click → select shape         (⇧ adds to selection)
 *   Pan          space-drag / middle-drag / Pan tool; wheel zooms at cursor;
 *                double-click empty space re-fits.
 *
 * Geometry is kept in image pixels; the view transform (scale + offset)
 * only affects drawing. Brush radius is chosen in *screen* pixels so the
 * brush feels the same at any zoom, and converted to image pixels when a
 * stroke starts.
 */

import { useCallback, useEffect, useRef, type ReactNode } from "react";
import type { BBox, Prompt, Pt } from "./types";
import { isTypingTarget } from "./widgets";

export type CanvasTool = "select" | "box" | "brush" | "erase" | "pan";

export interface Shape {
  id: string;
  color: string;
  polygon?: Pt[] | null;
  bbox?: BBox;
  obbPoints?: Pt[];
  keypoints?: [number, number, number][];
  label?: string;
  variant: "annotation" | "suggestion" | "preview" | "prediction" | "result";
  dimmed?: boolean;
}

interface Props {
  imageUrl: string | null;
  imageWidth: number;
  imageHeight: number;
  shapes: Shape[];
  prompts?: Prompt[];
  colorForClass?: (id: number) => string;
  tool: CanvasTool;
  brushRadius?: number;
  activeClassId?: number | null;
  selectedIds?: string[];
  hoveredId?: string | null;
  overlayUrl?: string | null;
  overlayOpacity?: number;
  showLabels?: boolean;
  keypointEdges?: [number, number][];
  fitKey?: string | number;
  onPrompt?: (p: Prompt) => void;
  onSelect?: (id: string | null, additive: boolean) => void;
  onHover?: (id: string | null) => void;
  onBrushRadius?: (r: number) => void;
  children?: ReactNode;
}

interface Drag {
  kind: "pan" | "box" | "stroke";
  startScreen: Pt;
  startImage: Pt;
  current: Pt;
  points: Pt[];
  polarity: 1 | -1;
  radiusImg: number;
  pointerId: number;
}

const NEG = "#ef4444";

function pointInPolygon(p: Pt, poly: Pt[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > p[1] !== yj > p[1] && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi + 1e-12) + xi) inside = !inside;
  }
  return inside;
}

function shapeContains(s: Shape, p: Pt): boolean {
  if (s.polygon && s.polygon.length >= 3) return pointInPolygon(p, s.polygon);
  if (s.obbPoints && s.obbPoints.length >= 3) return pointInPolygon(p, s.obbPoints);
  if (s.bbox) return p[0] >= s.bbox[0] && p[0] <= s.bbox[2] && p[1] >= s.bbox[1] && p[1] <= s.bbox[3];
  return false;
}

export function PromptCanvas(props: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const overlayRef = useRef<HTMLImageElement | null>(null);
  const view = useRef({ scale: 1, ox: 0, oy: 0, fit: true });
  const size = useRef({ w: 0, h: 0, dpr: 1 });
  const drag = useRef<Drag | null>(null);
  const mouse = useRef<{ x: number; y: number; inside: boolean }>({ x: 0, y: 0, inside: false });
  const space = useRef(false);
  const hoverRef = useRef<string | null>(null);
  const propsRef = useRef(props);
  propsRef.current = props;
  const raf = useRef(0);

  const toImage = useCallback((sx: number, sy: number): Pt => {
    const v = view.current;
    return [(sx - v.ox) / v.scale, (sy - v.oy) / v.scale];
  }, []);

  const fitView = useCallback(() => {
    const { w, h } = size.current;
    const iw = propsRef.current.imageWidth || imgRef.current?.naturalWidth || 1;
    const ih = propsRef.current.imageHeight || imgRef.current?.naturalHeight || 1;
    if (!w || !h) return;
    const s = Math.min(w / iw, h / ih) * 0.97;
    view.current = { scale: s, ox: (w - iw * s) / 2, oy: (h - ih * s) / 2, fit: true };
  }, []);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const p = propsRef.current;
    const { w, h, dpr } = size.current;
    const v = view.current;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#14161a";
    ctx.fillRect(0, 0, w, h);

    ctx.setTransform(dpr * v.scale, 0, 0, dpr * v.scale, dpr * v.ox, dpr * v.oy);
    const img = imgRef.current;
    if (img && img.complete) {
      ctx.imageSmoothingEnabled = v.scale < 2;
      ctx.drawImage(img, 0, 0, p.imageWidth, p.imageHeight);
    }
    const ov = overlayRef.current;
    if (ov && ov.complete && p.overlayUrl && (p.overlayOpacity ?? 0) > 0) {
      ctx.globalAlpha = p.overlayOpacity ?? 0.6;
      ctx.drawImage(ov, 0, 0, p.imageWidth, p.imageHeight);
      ctx.globalAlpha = 1;
    }
    const px = 1 / v.scale; // one screen pixel in image units
    const selected = new Set(p.selectedIds ?? []);

    // ---- shapes -----------------------------------------------------------
    for (const s of p.shapes) {
      const isSel = selected.has(s.id);
      const isHover = p.hoveredId === s.id;
      const path = new Path2D();
      const pts = s.polygon && s.polygon.length >= 3 ? s.polygon : s.obbPoints && s.obbPoints.length >= 3 ? s.obbPoints : null;
      if (pts) {
        path.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < pts.length; i++) path.lineTo(pts[i][0], pts[i][1]);
        path.closePath();
      } else if (s.bbox) {
        path.rect(s.bbox[0], s.bbox[1], s.bbox[2] - s.bbox[0], s.bbox[3] - s.bbox[1]);
      } else continue;

      ctx.globalAlpha = s.dimmed ? 0.3 : 1;
      let fill = 0.2;
      let lw = 2;
      let dash: number[] = [];
      if (s.variant === "suggestion") {
        fill = 0.1;
        lw = 1.6;
        dash = [6, 4];
      } else if (s.variant === "preview") {
        fill = 0.38;
        lw = 2.2;
        dash = [4, 3];
      } else if (s.variant === "prediction") {
        fill = 0.16;
      } else if (s.variant === "result") {
        fill = 0.32;
      }
      if (isSel || isHover) {
        fill += 0.15;
        lw += 1.2;
      }
      ctx.fillStyle = s.color;
      ctx.globalAlpha = (s.dimmed ? 0.3 : 1) * fill;
      ctx.fill(path);
      ctx.globalAlpha = s.dimmed ? 0.3 : 1;
      ctx.strokeStyle = s.variant === "preview" ? "#ffffff" : s.color;
      ctx.lineWidth = lw * px;
      ctx.setLineDash(dash.map((d) => d * px));
      ctx.stroke(path);
      if (s.variant === "preview") {
        ctx.strokeStyle = s.color;
        ctx.setLineDash(dash.map((d) => d * px));
        ctx.lineDashOffset = dash[0] * px;
        ctx.stroke(path);
        ctx.lineDashOffset = 0;
      }
      ctx.setLineDash([]);
      if (isSel && s.bbox) {
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1 * px;
        ctx.setLineDash([3 * px, 3 * px]);
        ctx.strokeRect(s.bbox[0], s.bbox[1], s.bbox[2] - s.bbox[0], s.bbox[3] - s.bbox[1]);
        ctx.setLineDash([]);
      }
      if (s.keypoints) {
        const kp = s.keypoints;
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 2 * px;
        for (const [a, b] of p.keypointEdges ?? []) {
          if (!kp[a] || !kp[b] || kp[a][2] < 0.3 || kp[b][2] < 0.3) continue;
          ctx.beginPath();
          ctx.moveTo(kp[a][0], kp[a][1]);
          ctx.lineTo(kp[b][0], kp[b][1]);
          ctx.stroke();
        }
        ctx.fillStyle = "#ffffff";
        for (const [x, y, c] of kp) {
          if (c < 0.3) continue;
          ctx.beginPath();
          ctx.arc(x, y, 3 * px, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.globalAlpha = 1;
    }

    // ---- prompts (committed + in-progress) --------------------------------
    const colorFor = (cid: number, pol: number) => (pol < 0 ? NEG : p.colorForClass?.(cid) ?? "#22d3ee");
    const drawPrompt = (pr: Prompt) => {
      const col = colorFor(pr.class_id, pr.polarity);
      if (pr.type === "box") {
        const [x1, y1, x2, y2] = pr.bbox;
        ctx.globalAlpha = 0.1;
        ctx.fillStyle = col;
        ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
        ctx.globalAlpha = 1;
        ctx.strokeStyle = col;
        ctx.lineWidth = 2 * px;
        ctx.setLineDash([7 * px, 4 * px]);
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.setLineDash([]);
      } else {
        ctx.globalAlpha = pr.polarity < 0 ? 0.5 : 0.45;
        ctx.strokeStyle = col;
        ctx.fillStyle = col;
        if (pr.points.length === 1) {
          ctx.beginPath();
          ctx.arc(pr.points[0][0], pr.points[0][1], pr.radius, 0, Math.PI * 2);
          ctx.fill();
        } else {
          ctx.lineWidth = pr.radius * 2;
          ctx.lineCap = "round";
          ctx.lineJoin = "round";
          ctx.beginPath();
          ctx.moveTo(pr.points[0][0], pr.points[0][1]);
          for (let i = 1; i < pr.points.length; i++) ctx.lineTo(pr.points[i][0], pr.points[i][1]);
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }
    };
    for (const pr of p.prompts ?? []) drawPrompt(pr);
    const d = drag.current;
    if (d && d.kind === "box") {
      const [ax, ay] = d.startImage;
      const [bx, by] = d.current;
      drawPrompt({
        type: "box",
        bbox: [Math.min(ax, bx), Math.min(ay, by), Math.max(ax, bx), Math.max(ay, by)],
        class_id: p.activeClassId ?? -1,
        polarity: d.polarity,
      });
    } else if (d && d.kind === "stroke") {
      drawPrompt({ type: "stroke", points: d.points, radius: d.radiusImg, class_id: p.activeClassId ?? -1, polarity: d.polarity });
    }

    // ---- screen-space decorations ----------------------------------------
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (p.showLabels !== false) {
      ctx.font = "11px -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.textBaseline = "top";
      for (const s of p.shapes) {
        if (!s.label || !s.bbox) continue;
        const sx = s.bbox[0] * v.scale + v.ox;
        const sy = s.bbox[1] * v.scale + v.oy;
        const tw = ctx.measureText(s.label).width + 8;
        const ty = sy - 16 < 0 ? sy + 1 : sy - 16;
        ctx.globalAlpha = s.dimmed ? 0.35 : 0.92;
        ctx.fillStyle = s.color;
        ctx.fillRect(sx, ty, tw, 15);
        ctx.fillStyle = "#ffffff";
        ctx.globalAlpha = s.dimmed ? 0.5 : 1;
        ctx.fillText(s.label, sx + 4, ty + 2);
        ctx.globalAlpha = 1;
      }
    }
    const m = mouse.current;
    if (m.inside && (p.tool === "brush" || p.tool === "erase") && !space.current) {
      const r = p.brushRadius ?? 12;
      ctx.beginPath();
      ctx.arc(m.x, m.y, r, 0, Math.PI * 2);
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(0,0,0,0.55)";
      ctx.stroke();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = p.tool === "erase" ? NEG : "#ffffff";
      ctx.stroke();
    }
    if (m.inside && p.tool === "box" && !space.current && !drag.current) {
      ctx.strokeStyle = "rgba(255,255,255,0.35)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(m.x, 0);
      ctx.lineTo(m.x, h);
      ctx.moveTo(0, m.y);
      ctx.lineTo(w, m.y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, []);

  const requestDraw = useCallback(() => {
    cancelAnimationFrame(raf.current);
    raf.current = requestAnimationFrame(draw);
  }, [draw]);

  // Redraw whenever props change.
  useEffect(() => {
    requestDraw();
  });

  // Size tracking.
  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;
    const ro = new ResizeObserver(() => {
      const rect = wrap.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      size.current = { w: rect.width, h: rect.height, dpr };
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      if (view.current.fit) fitView();
      requestDraw();
    });
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [fitView, requestDraw]);

  // Image + overlay loading.
  useEffect(() => {
    imgRef.current = null;
    if (!props.imageUrl) {
      requestDraw();
      return;
    }
    const img = new Image();
    img.onload = () => {
      imgRef.current = img;
      fitView();
      requestDraw();
    };
    img.src = props.imageUrl;
    return () => {
      img.onload = null;
    };
  }, [props.imageUrl, fitView, requestDraw]);

  useEffect(() => {
    overlayRef.current = null;
    if (!props.overlayUrl) return;
    const img = new Image();
    img.onload = () => {
      overlayRef.current = img;
      requestDraw();
    };
    img.src = props.overlayUrl;
  }, [props.overlayUrl, requestDraw]);

  useEffect(() => {
    if (props.fitKey === undefined) return;
    fitView();
    requestDraw();
  }, [props.fitKey, fitView, requestDraw]);

  // Space-to-pan.
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code === "Space" && !isTypingTarget(e) && mouse.current.inside) {
        space.current = true;
        e.preventDefault();
        requestDraw();
      }
    };
    const up = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        space.current = false;
        requestDraw();
      }
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, [requestDraw]);

  // Wheel zoom (non-passive so we can preventDefault page scroll).
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      if (e.shiftKey && propsRef.current.onBrushRadius && (propsRef.current.tool === "brush" || propsRef.current.tool === "erase")) {
        const r = propsRef.current.brushRadius ?? 12;
        propsRef.current.onBrushRadius(Math.max(2, Math.min(120, r * Math.exp(-e.deltaY * 0.003))));
        return;
      }
      const [ix, iy] = toImage(sx, sy);
      const factor = Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0018));
      const v = view.current;
      const scale = Math.max(0.05, Math.min(40, v.scale * factor));
      view.current = { scale, ox: sx - ix * scale, oy: sy - iy * scale, fit: false };
      requestDraw();
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, [toImage, requestDraw]);

  const screenPt = (e: React.PointerEvent | React.MouseEvent): Pt => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top];
  };

  const hitTest = (ip: Pt): string | null => {
    const shapes = propsRef.current.shapes;
    for (let i = shapes.length - 1; i >= 0; i--) {
      const s = shapes[i];
      if (s.variant === "preview") continue;
      if (shapeContains(s, ip)) return s.id;
    }
    return null;
  };

  const clampPt = (pt: Pt): Pt => [
    Math.max(0, Math.min(propsRef.current.imageWidth, pt[0])),
    Math.max(0, Math.min(propsRef.current.imageHeight, pt[1])),
  ];

  const onPointerDown = (e: React.PointerEvent) => {
    const p = propsRef.current;
    const sp = screenPt(e);
    const ip = toImage(sp[0], sp[1]);
    const canvas = canvasRef.current!;
    canvas.setPointerCapture(e.pointerId);
    const base = { startScreen: sp, startImage: ip, current: ip, points: [] as Pt[], radiusImg: 0, pointerId: e.pointerId };
    if (e.button === 1 || space.current || p.tool === "pan") {
      drag.current = { ...base, kind: "pan", polarity: 1 };
      return;
    }
    if (p.tool === "select") {
      if (e.button !== 0) return;
      p.onSelect?.(hitTest(ip), e.shiftKey || e.metaKey);
      return;
    }
    if (!p.onPrompt) return;
    if (p.tool === "box") {
      drag.current = { ...base, kind: "box", startImage: clampPt(ip), current: clampPt(ip), polarity: e.altKey ? -1 : 1 };
      return;
    }
    if (p.tool === "brush" || p.tool === "erase") {
      const negative = p.tool === "erase" || e.altKey || e.button === 2;
      const radiusImg = (p.brushRadius ?? 12) / view.current.scale;
      drag.current = { ...base, kind: "stroke", points: [clampPt(ip)], polarity: negative ? -1 : 1, radiusImg };
      requestDraw();
    }
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const sp = screenPt(e);
    mouse.current = { x: sp[0], y: sp[1], inside: true };
    const d = drag.current;
    if (!d) {
      const p = propsRef.current;
      if (p.onHover) {
        const id = hitTest(toImage(sp[0], sp[1]));
        if (id !== hoverRef.current) {
          hoverRef.current = id;
          p.onHover(id);
        }
      }
      requestDraw();
      return;
    }
    if (d.kind === "pan") {
      const v = view.current;
      const dx = sp[0] - d.startScreen[0];
      const dy = sp[1] - d.startScreen[1];
      view.current = { ...v, ox: v.ox + dx, oy: v.oy + dy, fit: false };
      d.startScreen = sp;
    } else if (d.kind === "box") {
      d.current = clampPt(toImage(sp[0], sp[1]));
    } else if (d.kind === "stroke") {
      const ip = clampPt(toImage(sp[0], sp[1]));
      const last = d.points[d.points.length - 1];
      if (Math.hypot(ip[0] - last[0], ip[1] - last[1]) >= Math.max(1, d.radiusImg * 0.3)) d.points.push(ip);
    }
    requestDraw();
  };

  const onPointerUp = (e: React.PointerEvent) => {
    const d = drag.current;
    drag.current = null;
    try {
      canvasRef.current?.releasePointerCapture(e.pointerId);
    } catch {
      /* already released */
    }
    const p = propsRef.current;
    if (!d || !p.onPrompt) {
      requestDraw();
      return;
    }
    const cid = p.activeClassId ?? -1;
    if (d.kind === "box") {
      const [ax, ay] = d.startImage;
      const [bx, by] = d.current;
      const bbox: BBox = [Math.min(ax, bx), Math.min(ay, by), Math.max(ax, bx), Math.max(ay, by)];
      if (bbox[2] - bbox[0] >= 3 && bbox[3] - bbox[1] >= 3) {
        p.onPrompt({ type: "box", bbox: bbox.map((x) => Math.round(x * 10) / 10) as BBox, class_id: cid, polarity: d.polarity });
      }
    } else if (d.kind === "stroke") {
      p.onPrompt({
        type: "stroke",
        points: d.points.map(([x, y]) => [Math.round(x * 10) / 10, Math.round(y * 10) / 10] as Pt),
        radius: Math.round(d.radiusImg * 10) / 10,
        class_id: cid,
        polarity: d.polarity,
      });
    }
    requestDraw();
  };

  const cursor =
    drag.current?.kind === "pan"
      ? "grabbing"
      : space.current || props.tool === "pan"
        ? "grab"
        : props.tool === "box"
          ? "crosshair"
          : props.tool === "brush" || props.tool === "erase"
            ? "none"
            : "default";

  return (
    <div ref={wrapRef} className="pc-wrap">
      <canvas
        ref={canvasRef}
        className="pc-canvas"
        style={{ cursor }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onPointerLeave={() => {
          mouse.current.inside = false;
          if (hoverRef.current !== null) {
            hoverRef.current = null;
            propsRef.current.onHover?.(null);
          }
          requestDraw();
        }}
        onDoubleClick={(e) => {
          const sp = screenPt(e);
          if (propsRef.current.tool === "pan" || !hitTest(toImage(sp[0], sp[1]))) {
            fitView();
            requestDraw();
          }
        }}
        onContextMenu={(e) => e.preventDefault()}
      />
      {props.children}
    </div>
  );
}
