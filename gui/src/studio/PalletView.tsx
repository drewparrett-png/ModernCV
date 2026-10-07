/**
 * Pallet — the Pal/DePal question: given carton masks and depth, how tall
 * is each carton, which layer is it on, and what would a robot pick first?
 *
 * Cartons come from labels, suggestions or a model run; depth from an
 * uploaded RGB-D map or monocular Depth Anything V2. The server fits the
 * pallet/floor plane, measures each carton's top face above it, clusters
 * layers and proposes a pick order (pipeline/studio/pallet.py).
 */

import { useEffect, useMemo, useRef, useState } from "react";
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import * as api from "./api";
import { ImageList } from "./ImageList";
import { PromptCanvas, type CanvasTool, type Shape } from "./PromptCanvas";
import type { ModelSpec, PalletBox, PalletRequest, PalletResult, Prompt, Pt, StrokePrompt } from "./types";
import { classColor, useStudio } from "./useStudio";
import { ModelPicker, Section, Segmented, Slider, Toggle } from "./widgets";

const LAYER_COLORS = ["#ef4444", "#f59e0b", "#22c55e", "#3b82f6", "#a855f7", "#ec4899", "#14b8a6", "#84cc16"];
const HIDDEN = "#9ca3af";
const layerColor = (l?: number) => (l ? LAYER_COLORS[(l - 1) % LAYER_COLORS.length] : HIDDEN);
const cm = (m: number | null | undefined, digits = 1) => (m === null || m === undefined ? "—" : (m * 100).toFixed(digits));

type Overlay = "none" | "height" | "depth";
type InstanceKind = PalletRequest["instances"]["kind"];
type DepthSource = PalletRequest["depth"]["source"];
type PlaneMode = PalletRequest["plane"]["mode"];

interface Settings {
  instKind: InstanceKind;
  modelSpec: ModelSpec;
  modelConf: number;
  depthSource: DepthSource;
  depthModel: string;
  useIntrinsics: boolean;
  hfov: number;
  knownHeight: string;
  planeMode: PlaneMode;
  layerTol: number;
  autoRun: boolean;
  overlay: Overlay;
  overlayOpacity: number;
  showLabels: boolean;
}

const DEFAULTS: Settings = {
  instKind: "annotations",
  modelSpec: { kind: "yolo", family: "26", task: "segment", size: "n" },
  modelConf: 0.25,
  depthSource: "auto",
  depthModel: "da2-metric-indoor-b",
  useIntrinsics: true,
  hfov: 60,
  knownHeight: "",
  planeMode: "auto",
  layerTol: 0.05,
  autoRun: true,
  overlay: "height",
  overlayOpacity: 0.55,
  showLabels: true,
};

/** Survives sub-tab switches (the view unmounts); reset when the project changes. */
const memory: { pid: string | null; settings: Settings; results: Record<string, PalletResult> } = {
  pid: null,
  settings: DEFAULTS,
  results: {},
};
const MAX_REMEMBERED_RESULTS = 12; // each carries two base64 overlays

export function PalletView() {
  const pid = useStudio((s) => s.projectId);
  const images = useStudio((s) => s.images);
  const classes = useStudio((s) => s.classes);
  const models = useStudio((s) => s.models);
  const catalog = useStudio((s) => s.catalog);
  const currentId = useStudio((s) => s.currentImageId);
  const selectImage = useStudio((s) => s.selectImage);
  const annotationsMap = useStudio((s) => s.annotations);
  const applyImage = useStudio((s) => s.applyImage);
  const run = useStudio((s) => s.run);
  const toast = useStudio((s) => s.toast);

  const saved = memory.pid === pid ? memory : { settings: DEFAULTS, results: {} as Record<string, PalletResult> };
  const [instKind, setInstKind] = useState<InstanceKind>(saved.settings.instKind);
  const [modelSpec, setModelSpec] = useState<ModelSpec>(saved.settings.modelSpec);
  const [modelConf, setModelConf] = useState(saved.settings.modelConf);
  const [depthSource, setDepthSource] = useState<DepthSource>(saved.settings.depthSource);
  const [depthModel, setDepthModel] = useState(saved.settings.depthModel);
  const [useIntrinsics, setUseIntrinsics] = useState(saved.settings.useIntrinsics);
  const [hfov, setHfov] = useState(saved.settings.hfov);
  const [knownHeight, setKnownHeight] = useState(saved.settings.knownHeight);
  const [planeMode, setPlaneMode] = useState<PlaneMode>(saved.settings.planeMode);
  const [planeStrokes, setPlaneStrokes] = useState<StrokePrompt[]>([]);
  const [layerTol, setLayerTol] = useState(saved.settings.layerTol);
  const [autoRun, setAutoRun] = useState(saved.settings.autoRun);
  const [results, setResults] = useState<Record<string, PalletResult>>(saved.results);
  const [overlay, setOverlay] = useState<Overlay>(saved.settings.overlay);
  const [overlayOpacity, setOverlayOpacity] = useState(saved.settings.overlayOpacity);
  const [showLabels, setShowLabels] = useState(saved.settings.showLabels);

  useEffect(() => {
    const keys = Object.keys(results);
    const kept = keys.length > MAX_REMEMBERED_RESULTS
      ? Object.fromEntries(keys.slice(-MAX_REMEMBERED_RESULTS).map((k) => [k, results[k]]))
      : results;
    memory.pid = pid;
    memory.results = kept;
    memory.settings = {
      instKind, modelSpec, modelConf, depthSource, depthModel, useIntrinsics, hfov, knownHeight,
      planeMode, layerTol, autoRun, overlay, overlayOpacity, showLabels,
    };
  }, [pid, results, instKind, modelSpec, modelConf, depthSource, depthModel, useIntrinsics, hfov, knownHeight,
      planeMode, layerTol, autoRun, overlay, overlayOpacity, showLabels]);
  const [tool, setTool] = useState<CanvasTool>("select");
  const [brush, setBrush] = useState(24);
  const [hovered, setHovered] = useState<string | null>(null);
  const depthRef = useRef<HTMLInputElement>(null);

  const image = images.find((i) => i.id === currentId);
  const result = currentId ? results[currentId] : undefined;
  const anns = (currentId && annotationsMap[currentId]) || [];

  // Prefer the newest trained segmentation model over COCO weights (which
  // have no carton class) as soon as one exists.
  const newestSeg = models.find((m) => m.status === "completed" && m.task === "segment");
  useEffect(() => {
    if (newestSeg && modelSpec.kind === "yolo") setModelSpec({ kind: "trained", model_id: newestSeg.id });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [newestSeg?.id]);

  // Painted reference strokes belong to one image.
  useEffect(() => {
    setPlaneStrokes([]);
    setHovered(null);
  }, [currentId]);
  useEffect(() => {
    setTool(planeMode === "painted" ? "brush" : "select");
  }, [planeMode]);

  const buildRequest = (): PalletRequest => ({
    instances:
      instKind === "predict"
        ? { kind: "predict", model: modelSpec, params: { conf: modelConf } }
        : { kind: instKind },
    depth: { source: depthSource, model: depthModel },
    use_image_intrinsics: useIntrinsics,
    hfov_deg: hfov,
    known_camera_height_m: knownHeight.trim() ? Number(knownHeight) : null,
    plane: { mode: planeMode, strokes: planeStrokes.map((s) => ({ points: s.points, radius: s.radius })) },
    layer_tol_m: layerTol,
  });

  const analyse = async (iid = currentId) => {
    if (!pid || !iid) return;
    if (planeMode === "painted" && planeStrokes.length === 0) {
      toast("Paint a patch of pallet deck or floor first (reference plane = painted).", "error");
      return;
    }
    const mono = depthSource === "mono" || (depthSource === "auto" && !images.find((i) => i.id === iid)?.has_depth);
    const out = await run(mono ? "Estimating depth + analysing" : "Analysing pallet", () =>
      api.analyzePallet(pid, iid, buildRequest()),
    );
    if (out) setResults((r) => ({ ...r, [iid]: out }));
  };

  // Auto-analyse when switching to an image that has labels.
  const lastAuto = useRef<string | null>(null);
  useEffect(() => {
    if (!autoRun || !currentId || results[currentId] || lastAuto.current === currentId) return;
    const img = images.find((i) => i.id === currentId);
    if (!img) return;
    const ready = instKind === "predict" || (instKind === "annotations" ? img.n_annotations > 0 : img.n_suggestions > 0);
    if (!ready || planeMode === "painted") return;
    lastAuto.current = currentId;
    analyse(currentId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentId, autoRun, images]);

  const sortedBoxes = useMemo(() => {
    if (!result) return [];
    return [...result.boxes].sort((a, b) => (a.pick_order ?? 1e9) - (b.pick_order ?? 1e9) || (b.height_m ?? 0) - (a.height_m ?? 0));
  }, [result]);

  const boxLabel = (b: PalletBox) => {
    if (b.height_m === null) return "no depth";
    if (b.height_is_lower_bound) return `≥${cm(b.height_m, 0)} cm (top hidden)`;
    return `#${b.pick_order} · L${b.layer} · ${cm(b.height_m, 1)} cm${b.blocked ? " · blocked" : ""}`;
  };

  const shapes: Shape[] = useMemo(() => {
    if (result) {
      return result.boxes
        .filter((b) => b.polygon)
        .map((b) => ({
          id: `b:${b.id}`,
          color: b.height_is_lower_bound || b.height_m === null ? HIDDEN : layerColor(b.layer),
          polygon: b.polygon ?? null,
          bbox: bboxOf(b.polygon!),
          label: showLabels ? boxLabel(b) : undefined,
          variant: "result" as const,
          dimmed: hovered !== null && hovered !== `b:${b.id}`,
        }));
    }
    return anns.map((a) => ({
      id: `a:${a.id}`,
      color: classColor(classes, a.class_id),
      polygon: a.polygon,
      bbox: a.bbox,
      variant: "annotation" as const,
      dimmed: true,
    }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result, anns, classes, hovered, showLabels]);

  const overlayUrl = result ? (overlay === "height" ? result.height_vis : overlay === "depth" ? result.depth_vis : null) : null;

  const scatter = useMemo(() => {
    if (!result) return [];
    const byLayer = new Map<string, { x: number; y: number; z: number; pick?: number; id: string; hidden: boolean }[]>();
    for (const b of result.boxes) {
      if (b.height_m === null || !b.plane_xy_m) continue;
      const key = b.height_is_lower_bound ? "hidden" : `L${b.layer}`;
      const arr = byLayer.get(key) ?? [];
      arr.push({ x: b.plane_xy_m[0], y: b.height_m, z: (b.dims_m?.[0] ?? 0.2) * 100, pick: b.pick_order, id: b.id, hidden: !!b.height_is_lower_bound });
      byLayer.set(key, arr);
    }
    return [...byLayer.entries()].sort();
  }, [result]);

  if (!pid) return null;

  const onPrompt = (p: Prompt) => {
    if (p.type === "stroke" && p.polarity > 0) setPlaneStrokes((cur) => [...cur, p]);
    else if (p.type === "stroke") setPlaneStrokes([]);
  };

  return (
    <div className="st-pallet">
      <ImageList selectedId={currentId} onSelect={(id) => selectImage(id)} compact />

      <div className="st-stage">
        <div className="st-toolbar">
          <Segmented
            value={overlay}
            onChange={setOverlay}
            options={[
              { value: "none", label: "Image" },
              { value: "height", label: "Height map", disabled: !result },
              { value: "depth", label: "Depth", disabled: !result },
            ]}
          />
          {overlay !== "none" && result && (
            <Slider label="Opacity" value={overlayOpacity} min={0} max={1} step={0.05} onChange={setOverlayOpacity} format={(v) => `${Math.round(v * 100)}%`} />
          )}
          <Toggle label="Labels" checked={showLabels} onChange={setShowLabels} />
          {planeMode === "painted" && (
            <>
              <Segmented
                size="sm"
                value={tool}
                onChange={setTool}
                options={[
                  { value: "brush", label: "✎ Paint floor/deck" },
                  { value: "pan", label: "✋" },
                ]}
              />
              <Slider label="Brush" value={brush} min={4} max={120} step={1} onChange={setBrush} format={(v) => `${v}px`} />
              <button type="button" className="st-btn sm ghost" onClick={() => setPlaneStrokes([])} disabled={!planeStrokes.length}>
                Clear paint
              </button>
            </>
          )}
        </div>
        <div className="st-canvas-area">
          {image ? (
            <PromptCanvas
              imageUrl={api.imageUrl(pid, image.id)}
              imageWidth={image.width}
              imageHeight={image.height}
              shapes={shapes}
              prompts={planeStrokes}
              colorForClass={() => "#14b8a6"}
              tool={tool}
              brushRadius={brush}
              onBrushRadius={(r) => setBrush(Math.round(r))}
              activeClassId={-1}
              onPrompt={planeMode === "painted" ? onPrompt : undefined}
              hoveredId={hovered}
              onHover={setHovered}
              onSelect={(id) => setHovered(id)}
              overlayUrl={overlayUrl}
              overlayOpacity={overlay === "none" ? 0 : overlayOpacity}
              showLabels={showLabels}
              fitKey={image.id}
            >
              {result && overlay === "height" && (
                <div className="st-canvas-legend">
                  Height above {result.plane.mode === "boxes" ? "lowest carton" : "reference plane"}
                  <div className="st-legend-bar" />
                  <div className="st-legend-ticks">
                    <span>0</span>
                    <span>{cm(result.height_range_m[1], 0)} cm</span>
                  </div>
                </div>
              )}
              {result && overlay === "depth" && (
                <div className="st-canvas-legend">
                  Depth (near → far)
                  <div className="st-legend-bar" style={{ transform: "scaleX(-1)" }} />
                  <div className="st-legend-ticks">
                    <span>{result.depth_range_m[0].toFixed(2)} m</span>
                    <span>{result.depth_range_m[1].toFixed(2)} m</span>
                  </div>
                </div>
              )}
            </PromptCanvas>
          ) : (
            <div className="st-canvas-empty">
              <h2>Pal / DePal analysis</h2>
              <p>Pick an image with carton labels. For real heights attach a depth map from an RGB-D camera; otherwise monocular depth estimates relative heights.</p>
              <p className="st-muted">No pallet photos yet? Use <b>＋ → Synthetic pallets</b> for scenes with exact depth and ground truth.</p>
            </div>
          )}
        </div>
      </div>

      <aside className="st-panel">
        <Section title="Cartons from">
          <Segmented
            size="sm"
            value={instKind}
            onChange={setInstKind}
            options={[
              { value: "annotations", label: `Labels${image ? ` (${image.n_annotations})` : ""}` },
              { value: "suggestions", label: `Suggestions${image ? ` (${image.n_suggestions})` : ""}` },
              { value: "predict", label: "Run a model" },
            ]}
          />
          {instKind === "predict" && (
            <>
              <ModelPicker value={modelSpec} onChange={setModelSpec} models={models} catalog={catalog} kinds={["trained", "yolo", "yoloe-text"]} tasks={["segment", "detect"]} />
              <Slider label="Conf" value={modelConf} min={0.02} max={0.9} step={0.01} onChange={setModelConf} format={(v) => v.toFixed(2)} />
            </>
          )}
          <p className="st-muted st-help">Masks give clean heights; plain boxes mix in background pixels.</p>
        </Section>

        <Section title="Depth">
          <Segmented
            size="sm"
            value={depthSource}
            onChange={setDepthSource}
            options={[
              { value: "auto", label: "Auto", title: "Sensor depth if attached, else monocular" },
              { value: "sensor", label: "RGB-D upload", disabled: !image?.has_depth },
              { value: "mono", label: "Monocular" },
            ]}
          />
          {image && (
            <div className="st-row wrap">
              {image.has_depth ? <span className="st-badge depth">sensor depth attached</span> : <span className="st-muted">No depth map on this image.</span>}
              <button type="button" className="st-btn sm" onClick={() => depthRef.current?.click()}>
                {image.has_depth ? "Replace" : "Upload depth"}
              </button>
              <input
                ref={depthRef}
                type="file"
                accept=".png,.tif,.tiff,.npy"
                hidden
                onChange={async (e) => {
                  const f = e.target.files?.[0];
                  e.target.value = "";
                  if (!f || !image) return;
                  const out = await run("Uploading depth", () => api.uploadDepth(pid, image.id, f));
                  if (out) {
                    applyImage(out);
                    toast("Depth attached (16-bit PNG = millimetres, .npy = metres).", "success");
                  }
                }}
              />
            </div>
          )}
          {(depthSource === "mono" || (depthSource === "auto" && !image?.has_depth)) && (
            <select className="st-input" value={depthModel} onChange={(e) => setDepthModel(e.target.value)}>
              {(catalog?.depth ?? []).map((d) => (
                <option key={d.id} value={d.id}>
                  {d.label}
                </option>
              ))}
            </select>
          )}
        </Section>

        <Section title="Camera & reference">
          {image?.intrinsics ? (
            <Toggle
              label={`Use image intrinsics (fx ${image.intrinsics.fx.toFixed(0)})`}
              checked={useIntrinsics}
              onChange={setUseIntrinsics}
            />
          ) : null}
          {(!image?.intrinsics || !useIntrinsics) && (
            <Slider label="Horizontal FOV" value={hfov} min={30} max={120} step={1} onChange={setHfov} format={(v) => `${v}°`} hint="Used to back-project depth when the image has no intrinsics" />
          )}
          <label className="st-mini" title="If the camera is mounted at a known height above the floor/deck, monocular depth is rescaled to match it">
            Known camera height
            <input className="st-input num" placeholder="m" value={knownHeight} onChange={(e) => setKnownHeight(e.target.value)} />
            m (optional)
          </label>
          <div className="st-row wrap">
            <span className="st-field-label">Reference plane</span>
            <Segmented
              size="sm"
              value={planeMode}
              onChange={setPlaneMode}
              options={[
                { value: "auto", label: "Auto", title: "Largest plane among non-carton pixels (deck / floor)" },
                { value: "painted", label: "Painted", title: "Paint the deck or floor on the canvas" },
                { value: "boxes", label: "Lowest carton", title: "Heights relative to the lowest carton top" },
              ]}
            />
          </div>
          {planeMode === "painted" && (
            <p className="st-muted st-help">
              Paint a patch of pallet deck or floor on the image ({planeStrokes.length} stroke{planeStrokes.length === 1 ? "" : "s"}).
            </p>
          )}
          <Slider label="Layer tolerance" value={layerTol} min={0.01} max={0.2} step={0.005} onChange={setLayerTol} format={(v) => `${(v * 100).toFixed(1)} cm`} />
          <div className="st-row">
            <button type="button" className="st-btn primary" disabled={!image} onClick={() => analyse()}>
              Analyse pallet
            </button>
            <Toggle label="Auto on image change" checked={autoRun} onChange={setAutoRun} />
          </div>
        </Section>

        {result && <ResultPanel result={result} boxes={sortedBoxes} hovered={hovered} setHovered={setHovered} scatter={scatter} />}
      </aside>
    </div>
  );
}

function bboxOf(poly: Pt[]): [number, number, number, number] {
  let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
  for (const [x, y] of poly) {
    x1 = Math.min(x1, x);
    y1 = Math.min(y1, y);
    x2 = Math.max(x2, x);
    y2 = Math.max(y2, y);
  }
  return [x1, y1, x2, y2];
}

type ScatterPoint = { x: number; y: number; z: number; pick?: number; id: string; hidden: boolean };

function ResultPanel({
  result,
  boxes,
  hovered,
  setHovered,
  scatter,
}: {
  result: PalletResult;
  boxes: PalletBox[];
  hovered: string | null;
  setHovered: (id: string | null) => void;
  scatter: [string, ScatterPoint[]][];
}) {
  const s = result.summary;
  const gt = result.ground_truth;
  return (
    <>
      <Section title="Result">
        <div className="st-grid-metrics">
          <div className="st-metric">
            <div className="st-metric-label">Cartons measured</div>
            <div className="st-metric-value">
              {s.n_measured}
              <span className="st-metric-sub"> / {s.n_boxes}</span>
            </div>
          </div>
          <div className="st-metric">
            <div className="st-metric-label">Layers</div>
            <div className="st-metric-value">{s.n_layers}</div>
          </div>
          <div className="st-metric">
            <div className="st-metric-label">Tallest top</div>
            <div className="st-metric-value">
              {cm(s.max_height_m, 0)}
              <span className="st-metric-sub"> cm</span>
            </div>
          </div>
          {gt && gt.mean_abs_err_m !== null && (
            <div className="st-metric" title={`Synthetic ground truth: ${gt.n_matched} cartons matched`}>
              <div className="st-metric-label">Error vs truth</div>
              <div className="st-metric-value">
                {cm(gt.mean_abs_err_m, 1)}
                <span className="st-metric-sub"> cm avg</span>
              </div>
              <div className="st-metric-sub">max {cm(gt.max_abs_err_m, 1)} cm</div>
            </div>
          )}
        </div>
        <div className="st-kv">
          <span>Depth</span>
          <span>
            {result.depth_source === "sensor" ? "RGB-D upload" : result.depth_source.replace("mono:", "monocular · ")}
            {result.calibration ? ` · scaled ×${result.calibration.depth_scale}` : ""}
          </span>
          <span>Plane</span>
          <span title={result.plane.note}>
            {result.plane.mode} · {(result.plane.inlier_ratio * 100).toFixed(0)}% inliers · rms {cm(result.plane.rms_m, 1)} cm
          </span>
          <span>Camera</span>
          <span>
            {result.plane.mode === "boxes" ? "—" : `${result.plane.camera_height_m.toFixed(2)} m above plane`} · tilt {result.plane.camera_tilt_deg}°
          </span>
          <span>Intrinsics</span>
          <span>{result.intrinsics.source}</span>
          {s.n_top_hidden > 0 && (
            <>
              <span>Hidden tops</span>
              <span>{s.n_top_hidden} (side face only)</span>
            </>
          )}
          {s.n_blocked > 0 && (
            <>
              <span>Blocked</span>
              <span>{s.n_blocked} under a higher carton</span>
            </>
          )}
        </div>
        {result.warnings.map((w, i) => (
          <div key={i} className="st-warn-box">
            {w}
          </div>
        ))}
      </Section>

      <Section title="Layers">
        <table className="st-table">
          <thead>
            <tr>
              <th>Layer</th>
              <th className="num">Cartons</th>
              <th className="num">Mean (cm)</th>
              <th className="num">Range (cm)</th>
            </tr>
          </thead>
          <tbody>
            {result.layers.map((l) => (
              <tr key={l.layer}>
                <td>
                  <span className="st-layer-chip" style={{ background: layerColor(l.layer) }}>
                    L{l.layer}
                  </span>
                </td>
                <td className="num">{l.n}</td>
                <td className="num">{cm(l.mean_height_m)}</td>
                <td className="num">
                  {cm(l.min_height_m)}–{cm(l.max_height_m)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {scatter.length > 0 && (
          <div style={{ height: 180 }}>
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 8, right: 8, bottom: 4, left: -12 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eef0f3" />
                <XAxis type="number" dataKey="x" name="across" unit=" m" tick={{ fontSize: 10 }} domain={["auto", "auto"]} />
                <YAxis type="number" dataKey="y" name="height" unit=" m" tick={{ fontSize: 10 }} domain={[0, "auto"]} />
                <ZAxis type="number" dataKey="z" range={[40, 160]} />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  formatter={(v: unknown, name: unknown) => [typeof v === "number" ? v.toFixed(3) : String(v), String(name)]}
                />
                {scatter.map(([key, pts]) => (
                  <Scatter
                    key={key}
                    name={key}
                    data={pts}
                    fill={key === "hidden" ? HIDDEN : layerColor(Number(key.slice(1)))}
                    onMouseEnter={(p: { payload?: ScatterPoint }) => p?.payload && setHovered(`b:${p.payload.id}`)}
                    onMouseLeave={() => setHovered(null)}
                  />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        )}
        <p className="st-muted st-help">Side elevation: each dot is a carton top, x = position across the pallet plane.</p>
      </Section>

      <Section title={`Cartons · pick order`}>
        <div style={{ overflowX: "auto" }}>
          <table className="st-table">
            <thead>
              <tr>
                <th>#</th>
                <th>L</th>
                <th className="num">Height</th>
                {result.ground_truth && <th className="num">Truth</th>}
                <th className="num">Top L×W (cm)</th>
                <th className="num">Yaw</th>
                <th>Flags</th>
              </tr>
            </thead>
            <tbody>
              {boxes.map((b) => (
                <tr
                  key={b.id}
                  className={hovered === `b:${b.id}` ? "hover" : ""}
                  onMouseEnter={() => setHovered(`b:${b.id}`)}
                  onMouseLeave={() => setHovered(null)}
                >
                  <td>{b.pick_order ?? "—"}</td>
                  <td>
                    {b.layer ? (
                      <span className="st-layer-chip" style={{ background: layerColor(b.layer) }}>
                        {b.layer}
                      </span>
                    ) : (
                      <span className="st-muted">—</span>
                    )}
                  </td>
                  <td className="num">
                    {b.height_is_lower_bound ? "≥" : ""}
                    {cm(b.height_m)}
                  </td>
                  {result.ground_truth && <td className="num">{cm(b.gt_height_m)}</td>}
                  <td className="num">{b.dims_m ? `${cm(b.dims_m[0], 0)}×${cm(b.dims_m[1], 0)}` : "—"}</td>
                  <td className="num">{b.yaw_deg !== null && b.yaw_deg !== undefined ? `${b.yaw_deg.toFixed(0)}°` : "—"}</td>
                  <td>
                    {b.blocked && <span className="st-badge warn">blocked</span>}{" "}
                    {b.flags
                      .filter((f) => f !== "few_points")
                      .map((f) => (
                        <span key={f} className={`st-badge ${f === "top_hidden" ? "" : "bad"}`} title={FLAG_HELP[f] ?? f}>
                          {FLAG_SHORT[f] ?? f.replace("_", " ")}
                        </span>
                      ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </>
  );
}

const FLAG_SHORT: Record<string, string> = { top_hidden: "hidden", touches_border: "edge", box_only: "box", no_depth: "no depth" };

const FLAG_HELP: Record<string, string> = {
  top_hidden: "Only a side face is visible — the top is under another carton; height is a lower bound.",
  tilted: "Top surface tilted more than 10° from the reference plane (crushed, leaning, or noisy depth).",
  touches_border: "Cut off by the image edge — dimensions are underestimated.",
  box_only: "Box label without a mask — height mixes in background pixels.",
  no_depth: "No valid depth inside the mask.",
};
