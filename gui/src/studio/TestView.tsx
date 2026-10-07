/**
 * Test — Ultralytics Predict and Track modes.
 *
 *   Predict  any model Studio can run (YOLO26 detect / segment / classify /
 *            pose / OBB, trained models, YOLOE text or prompt-free) on a
 *            dataset image, with the NMS / head / TTA knobs exposed. Results
 *            draw on the canvas and list in a linked table, and can be sent
 *            to Label as suggestions.
 *   Track    ByteTrack / BoT-SORT over a video in data/ as a background job,
 *            with an optional counting line; "Watch" swaps the canvas for
 *            the annotated overlay video.
 *
 * Auto-run re-predicts 300 ms after any change. A sequence counter drops
 * stale responses, and a small cache (predictions are deterministic for
 * image + model + params) makes flipping back to an image instant.
 */

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode, type RefObject } from "react";
import { useStore } from "../store";
import * as api from "./api";
import { PromptCanvas, type Shape } from "./PromptCanvas";
import type { Detection, ModelSpec, Prediction, PredictParams, StudioImage, TrackJob, TrainedModel, YoloTask } from "./types";
import { useStudio } from "./useStudio";
import { COCO_SKELETON, ModelPicker, Section, Segmented, Slider, Toggle, colorForIndex, fmtSecs, pct, specLabel } from "./widgets";
import "./test.css";

// ---- Settings ----------------------------------------------------------------

const DEFAULT_MODEL: ModelSpec = { kind: "yolo", family: "26", task: "segment", size: "n" };
const IMGSZ = [224, 320, 416, 480, 512, 640, 768, 896, 1024, 1280];
const DEBOUNCE_MS = 300;
const POLL_MS = 2000;
const CACHE_MAX = 32;
/** PromptCanvas doesn't draw keypoints below this confidence — count them as hidden. */
const KP_VISIBLE = 0.3;

type HeadMode = "auto" | "e2e" | "nms";
const HEAD_END2END: Record<HeadMode, boolean | null> = { auto: null, e2e: true, nms: false };

interface PredictOpts {
  conf: number;
  iou: number;
  imgsz: number;
  maxDet: number;
  agnostic: boolean;
  augment: boolean;
  retina: boolean;
  half: boolean;
  head: HeadMode;
  /** Comma-separated class ids; empty = every class. */
  classes: string;
}

const DEFAULT_OPTS: PredictOpts = {
  conf: 0.25,
  iou: 0.7,
  imgsz: 640,
  maxDet: 300,
  agnostic: false,
  augment: false,
  retina: false,
  half: false,
  head: "auto",
  classes: "",
};

type LineAxis = "none" | "x" | "y";

interface TrackOpts {
  video: string;
  tracker: string;
  stride: number;
  maxFrames: number;
  conf: number;
  line: LineAxis;
  linePos: number;
}

const DEFAULT_TRACK: TrackOpts = { video: "", tracker: "bytetrack", stride: 1, maxFrames: 300, conf: 0.25, line: "none", linePos: 0.5 };

/** Settings outlive the view, which unmounts whenever another Studio tab is open. */
let remembered: { pid: string; model: ModelSpec; opts: PredictOpts; autoRun: boolean; showLabels: boolean } | null = null;
let rememberedTrack: TrackOpts = DEFAULT_TRACK;

function recall() {
  return remembered && remembered.pid === useStudio.getState().projectId ? remembered : null;
}

/** Request key → prediction (LRU). */
const predCache = new Map<string, Prediction>();

function cachePut(key: string, pred: Prediction) {
  predCache.delete(key);
  predCache.set(key, pred);
  for (const k of predCache.keys()) {
    if (predCache.size <= CACHE_MAX) break;
    predCache.delete(k);
  }
}

// ---- Helpers -----------------------------------------------------------------

const NO_DETS: Detection[] = [];
const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));
const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;
const fmtMs = (v: number | undefined) => (v === undefined ? "—" : v.toFixed(1));
const baseName = (path: string) => path.split("/").pop() || path;
const numOrEmpty = (v: number) => (Number.isNaN(v) ? "" : v);
const isImageFile = (f: File) => f.type.startsWith("image/") || /\.(jpe?g|png|webp|bmp|tiff?)$/i.test(f.name);
const TRACKER_LABEL: Record<string, string> = { bytetrack: "ByteTrack", botsort: "BoT-SORT" };
const trackerLabel = (t: string) => TRACKER_LABEL[t] ?? t;

function isTyping(e: KeyboardEvent): boolean {
  const t = e.target as HTMLElement | null;
  if (!t) return false;
  return ["INPUT", "TEXTAREA", "SELECT", "VIDEO"].includes(t.tagName) || t.isContentEditable;
}

/** The task a spec runs, when knowable up front (YOLOE checkpoints are -seg models). */
function specTask(spec: ModelSpec, models: TrainedModel[]): YoloTask | null {
  switch (spec.kind) {
    case "yolo":
      return spec.task;
    case "trained":
      return models.find((m) => m.id === spec.model_id)?.task ?? null;
    case "yoloe-text":
    case "yoloe-pf":
      return "segment";
  }
}

function predictIssue(spec: ModelSpec): string | null {
  if (spec.kind === "trained" && !spec.model_id) return "Pick a trained model — or train one in the Train tab.";
  if (spec.kind === "yoloe-text" && !spec.classes.length) return "Type at least one class name for the text prompt.";
  return null;
}

/** Only YOLO26 checkpoints (pretrained or trained from YOLO26) have the end-to-end head. */
function isYolo26(spec: ModelSpec, models: TrainedModel[]): boolean {
  if (spec.kind === "yolo") return spec.family === "26";
  if (spec.kind === "trained") return (models.find((m) => m.id === spec.model_id)?.family ?? "26") === "26";
  return false;
}

/** Track mode needs boxes, and the tracker job loads YOLO, trained or YOLOE text models only. */
function trackIssue(spec: ModelSpec): string | null {
  if (spec.kind === "yoloe-pf") return "Prompt-free YOLOE can't track — pick YOLO26 / YOLO11, a trained model or a YOLOE text prompt above.";
  if (spec.kind === "yolo" && spec.task === "classify") return "Classifiers don't output boxes to track — pick Detect, Segment, Pose or OBB above.";
  return predictIssue(spec);
}

function parseClassFilter(text: string): { ids: number[]; bad: string[] } {
  const ids: number[] = [];
  const bad: string[] = [];
  for (const tok of text.split(/[\s,;]+/)) {
    if (!tok) continue;
    if (/^\d+$/.test(tok)) {
      const id = Number(tok);
      if (!ids.includes(id)) ids.push(id);
    } else bad.push(tok);
  }
  return { ids, bad };
}

function toShape(d: Detection, i: number, dimmed: boolean): Shape {
  return {
    id: `d:${i}`,
    color: colorForIndex(d.class_id),
    polygon: d.polygon ?? null,
    obbPoints: d.polygon ? undefined : d.obb?.points,
    bbox: d.bbox, // the outline falls back to it; it also anchors the label
    keypoints: d.keypoints,
    label: `${d.class_name} ${Math.floor(d.score * 100)}${d.track_id !== undefined ? ` #${d.track_id}` : ""}`,
    variant: "prediction",
    dimmed,
  };
}

const EXTRA_HEAD: Partial<Record<YoloTask, string>> = { segment: "Mask", obb: "Angle", pose: "Keypoints" };

function extraCell(d: Detection): string {
  if (d.obb) return `${d.obb.angle_deg.toFixed(1)}°`;
  if (d.keypoints) return `${d.keypoints.filter((k) => k[2] >= KP_VISIBLE).length}/${d.keypoints.length}`;
  if (d.polygon) return `${d.polygon.length} pts`;
  return "—";
}

/** Width × height in px — of the rotated box for OBB, the axis-aligned box otherwise. */
function boxSize(d: Detection): string {
  const w = d.obb ? d.obb.w : d.bbox[2] - d.bbox[0];
  const h = d.obb ? d.obb.h : d.bbox[3] - d.bbox[1];
  return `${Math.round(w)}×${Math.round(h)}`;
}

function classCounts(dets: Detection[]): { id: number; name: string; n: number }[] {
  const byId = new Map<number, { id: number; name: string; n: number }>();
  for (const d of dets) {
    const c = byId.get(d.class_id);
    if (c) c.n++;
    else byId.set(d.class_id, { id: d.class_id, name: d.class_name, n: 1 });
  }
  return [...byId.values()].sort((a, b) => b.n - a.n || a.name.localeCompare(b.name));
}

// ---- View --------------------------------------------------------------------

interface PredictResult {
  key: string;
  modelKey: string;
  imageId: string;
  pred: Prediction;
}

interface Hud {
  kind: "busy" | "info" | "warn" | "err";
  text: string;
}

export function TestView() {
  const pid = useStudio((s) => s.projectId);
  const images = useStudio((s) => s.images);
  const currentId = useStudio((s) => s.currentImageId);
  const selectImage = useStudio((s) => s.selectImage);
  const loadImageData = useStudio((s) => s.loadImageData);
  const refresh = useStudio((s) => s.refresh);
  const models = useStudio((s) => s.models);
  const catalog = useStudio((s) => s.catalog);
  const testModel = useStudio((s) => s.testModel);
  const setTestModel = useStudio((s) => s.setTestModel);
  const setTab = useStudio((s) => s.setTab);
  const run = useStudio((s) => s.run);
  const toast = useStudio((s) => s.toast);

  const [model, setModel] = useState<ModelSpec>(() => useStudio.getState().testModel ?? recall()?.model ?? DEFAULT_MODEL);
  const [opts, setOpts] = useState<PredictOpts>(() => recall()?.opts ?? DEFAULT_OPTS);
  const [autoRun, setAutoRun] = useState(() => recall()?.autoRun ?? true);
  const [showLabels, setShowLabels] = useState(() => recall()?.showLabels ?? true);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [failure, setFailure] = useState<{ key: string; message: string } | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [hovered, setHovered] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [focusClass, setFocusClass] = useState<number | null>(null);
  const [fitN, setFitN] = useState(0);
  const [sent, setSent] = useState<{ imageId: string; n: number } | null>(null);
  const [watchId, setWatchId] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLElement>(null);
  const { jobs, reload: reloadJobs } = useTrackJobs(pid);

  const setOpt = <K extends keyof PredictOpts>(k: K, v: PredictOpts[K]) => setOpts((o) => ({ ...o, [k]: v }));

  // Train → "Test it" hands a model over through the store.
  useEffect(() => {
    if (!testModel) return;
    setModel(testModel);
    setTestModel(null);
  }, [testModel, setTestModel]);

  useEffect(() => {
    if (pid) remembered = { pid, model, opts, autoRun, showLabels };
  }, [pid, model, opts, autoRun, showLabels]);

  // ---- Request ---------------------------------------------------------------
  const image: StudioImage | null = images.find((i) => i.id === currentId) ?? null;
  const imageId = image?.id ?? null;
  const imageIdx = image ? images.indexOf(image) : -1;
  const task = specTask(model, models);
  const boxTask = task !== "classify";
  const headApplies = isYolo26(model, models) && boxTask;
  const nmsOnly = !isYolo26(model, models) && (model.kind === "yolo" || model.kind === "trained") && boxTask;
  const iouIgnored = headApplies && opts.head !== "nms";
  const filter = useMemo(() => parseClassFilter(opts.classes), [opts.classes]);
  const modelKey = useMemo(() => JSON.stringify(model), [model]);

  const params = useMemo<PredictParams>(() => {
    // Send only what the panel shows: classifiers have no box / NMS knobs.
    if (!boxTask) return { imgsz: opts.imgsz, half: opts.half };
    const p: PredictParams = {
      conf: opts.conf,
      iou: opts.iou,
      imgsz: opts.imgsz,
      max_det: opts.maxDet >= 1 ? Math.floor(opts.maxDet) : DEFAULT_OPTS.maxDet,
      agnostic_nms: opts.agnostic,
      augment: opts.augment,
      half: opts.half,
    };
    if (task === "segment") p.retina_masks = opts.retina;
    if (filter.ids.length) p.classes = filter.ids;
    const end2end = HEAD_END2END[opts.head];
    if (headApplies && end2end !== null) p.end2end = end2end;
    return p;
  }, [opts, task, boxTask, headApplies, filter.ids]);

  const issue = predictIssue(model);
  const request = useMemo(
    () => (pid && imageId && !issue ? { pid, imageId, model, params } : null),
    [pid, imageId, issue, model, params],
  );
  const requestKey = useMemo(() => (request ? JSON.stringify(request) : ""), [request]);

  // ---- Predict ---------------------------------------------------------------
  const seq = useRef(0);
  const latest = useRef({ request, requestKey, modelKey });
  latest.current = { request, requestKey, modelKey };

  const predictNow = useCallback(
    async (manual: boolean) => {
      const { request: req, requestKey: key, modelKey: mkey } = latest.current;
      if (!req) return;
      const my = ++seq.current;
      setPredicting(true);
      try {
        const pred = await api.predict(req.pid, { model: req.model, image_id: req.imageId, params: req.params });
        cachePut(key, pred);
        if (my !== seq.current) return;
        setResult({ key, modelKey: mkey, imageId: req.imageId, pred });
        setFailure(null);
      } catch (e) {
        if (my !== seq.current) return;
        setFailure({ key, message: errText(e) });
        if (manual) toast(errText(e), "error");
      } finally {
        if (my === seq.current) setPredicting(false);
      }
    },
    [toast],
  );

  // Cached → show at once; otherwise auto-run after a short debounce.
  useEffect(() => {
    const { request: req, modelKey: mkey } = latest.current;
    if (!req) {
      seq.current++; // nothing valid to run — drop anything in flight
      setPredicting(false);
      return;
    }
    const hit = predCache.get(requestKey);
    if (hit) {
      seq.current++; // drop whatever is still in flight
      cachePut(requestKey, hit);
      setPredicting(false);
      setResult({ key: requestKey, modelKey: mkey, imageId: req.imageId, pred: hit });
      return;
    }
    if (!autoRun) return;
    const t = window.setTimeout(() => void predictNow(false), DEBOUNCE_MS);
    return () => window.clearTimeout(t);
  }, [requestKey, autoRun, predictNow]);

  // Responses that land after unmount are ignored.
  useEffect(() => {
    const s = seq;
    return () => {
      s.current++;
    };
  }, []);

  const failed = failure && failure.key === requestKey ? failure.message : null;
  const shown = result && result.imageId === imageId ? result : null;
  // An invalid model or a failed run shows no (stale) results.
  const pred = shown && !issue && !failed ? shown.pred : null;
  const outdated = !!shown && shown.key !== requestKey;
  const dets = pred?.detections ?? NO_DETS;
  const top = pred?.classification?.top ?? null;
  const focus = focusClass !== null && dets.some((d) => d.class_id === focusClass) ? focusClass : null;
  const shapes = useMemo(() => dets.map((d, i) => toShape(d, i, focus !== null && d.class_id !== focus)), [dets, focus]);

  useEffect(() => {
    setHovered(null);
    setSelected(null);
  }, [pred]);

  // ---- Images ----------------------------------------------------------------
  const watchJob = useMemo(
    () => jobs.find((j) => j.id === watchId && j.status === "completed" && j.has_video) ?? null,
    [jobs, watchId],
  );

  const pick = useCallback(
    (id: string) => {
      setWatchId(null);
      selectImage(id).catch((e) => toast(errText(e), "error"));
    },
    [selectImage, toast],
  );

  const firstId = images[0]?.id ?? null;
  useEffect(() => {
    if (!imageId && firstId) selectImage(firstId).catch(() => {});
  }, [imageId, firstId, selectImage]);

  const step = (delta: number) => {
    if (!images.length) return;
    const from = Math.max(0, imageIdx);
    pick(images[(from + delta + images.length) % images.length].id);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (watchJob || isTyping(e) || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "ArrowRight") step(1);
      else if (e.key === "ArrowLeft") step(-1);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const upload = async (files: File[]) => {
    if (!pid || !files.length) return;
    const imgs = files.filter(isImageFile);
    if (!imgs.length) {
      toast("Drop image files (jpg, png, webp, bmp, tiff).", "error");
      return;
    }
    const out = await run(`Uploading ${plural(imgs.length, "image")}`, async () => {
      const res = await api.uploadImages(pid, imgs);
      await refresh();
      return res;
    });
    if (!out) return;
    if (out.errors.length) toast(out.errors.join("\n"), "error");
    if (out.images[0]) pick(out.images[0].id);
  };

  // ---- Actions ---------------------------------------------------------------
  const canSend = !!request && task !== "classify";
  const sendToLabel = async () => {
    if (!request) return;
    const { pid: p, imageId: iid, model: spec, params: prm } = request;
    const out = await run("Sending predictions to Label", async () => {
      const res = await api.detectToSuggestions(p, { model: spec, scope: "image", image_id: iid, params: prm });
      await refresh();
      await loadImageData(iid);
      return res;
    });
    if (!out) return;
    const n = out.counts[iid] ?? 0;
    setSent({ imageId: iid, n });
    if (n) toast(`${plural(n, "suggestion")} added to this image — review them in Label.`, "success");
    else toast("Nothing to suggest: no detections, or they all overlap existing labels.", "info");
  };

  const revealRow = (id: string) =>
    resultsRef.current?.querySelector<HTMLElement>(`[data-sid="${CSS.escape(id)}"]`)?.scrollIntoView({ block: "nearest" });

  const onCanvasHover = (id: string | null) => {
    setHovered(id);
    if (id) revealRow(id);
  };

  const onCanvasSelect = (id: string | null) => {
    setSelected((cur) => (id && id !== cur ? id : null));
    if (id) revealRow(id);
  };

  // ---- Derived UI ------------------------------------------------------------
  let hud: Hud | null = null;
  if (issue) hud = { kind: "warn", text: issue };
  else if (predicting) hud = { kind: "busy", text: `Running ${specLabel(model, models)}…` };
  else if (failed) hud = { kind: "err", text: "Prediction failed — see Results" };
  else if (!autoRun && !pred) hud = { kind: "info", text: "Press Run to predict this image" };
  else if (!autoRun && outdated) hud = { kind: "info", text: "Settings changed — press Run" };

  const names = result && result.modelKey === modelKey ? result.pred.names : undefined;
  const classHint = [
    filter.ids.length ? `Only ${filter.ids.map((id) => names?.[String(id)] ?? `class ${id}`).join(", ")}` : "",
    filter.bad.length ? `ignoring ${filter.bad.map((b) => `“${b}”`).join(", ")} (ids only)` : "",
  ]
    .filter(Boolean)
    .join(" · ");

  const nRunning = jobs.filter((j) => j.status === "running").length;
  const nQueued = jobs.filter((j) => j.status === "queued").length;
  const trackBadge =
    nRunning || nQueued ? (
      <span className="st-badge warn">{nRunning ? `${nRunning} running` : `${nQueued} queued`}</span>
    ) : jobs.length ? (
      <span className="st-muted">{plural(jobs.length, "job")}</span>
    ) : undefined;

  const imgszSelect = (
    <label className="st-mini">
      imgsz
      <select className="st-input slim" value={opts.imgsz} onChange={(e) => setOpt("imgsz", Number(e.target.value))}>
        {IMGSZ.map((s) => (
          <option key={s} value={s}>
            {s}
          </option>
        ))}
      </select>
    </label>
  );
  const halfToggle = (
    <Toggle label="Half (FP16)" checked={opts.half} onChange={(v) => setOpt("half", v)} hint="Half-precision inference — faster on a GPU, no gain on CPU" />
  );

  if (!pid) return null;

  return (
    <div className="st-page">
      <aside className="st-page-side st-test-side">
        <Section
          title="Model"
          right={catalog ? <span className="st-badge" title="Inference device">{catalog.device}</span> : undefined}
        >
          <ModelPicker value={model} onChange={setModel} models={models} catalog={catalog} />
          {headApplies && (
            <div className="st-test-field">
              <div className="st-row">
                <span className="st-field-label">Head</span>
                <Segmented
                  size="sm"
                  value={opts.head}
                  onChange={(v) => setOpt("head", v)}
                  options={[
                    { value: "auto", label: "Auto", title: "The checkpoint's default head" },
                    { value: "e2e", label: "NMS-free", title: "One-to-one head, no NMS (end2end=True)" },
                    { value: "nms", label: "NMS", title: "One-to-many head + NMS (end2end=False)" },
                  ]}
                />
              </div>
              <p className="st-test-hint">YOLO26 predicts end-to-end without NMS; switch to the one-to-many head + NMS to compare.</p>
            </div>
          )}
          {nmsOnly && (
            <p className="st-test-hint">YOLO11 has a single one-to-many head and always uses NMS (the IoU slider applies).</p>
          )}
        </Section>

        <Section title="Predict">
          {boxTask ? (
            <>
              <Slider
                label="Conf"
                value={opts.conf}
                min={0.01}
                max={0.95}
                step={0.01}
                onChange={(v) => setOpt("conf", v)}
                format={(v) => v.toFixed(2)}
                hint="Minimum confidence to keep a detection"
              />
              <div className={iouIgnored ? "st-test-off" : undefined}>
                <Slider
                  label="IoU"
                  value={opts.iou}
                  min={0.1}
                  max={0.95}
                  step={0.05}
                  onChange={(v) => setOpt("iou", v)}
                  format={(v) => v.toFixed(2)}
                  hint={iouIgnored ? "Unused by YOLO26's NMS-free head — set Head to NMS to use it" : "NMS overlap threshold"}
                />
              </div>
              <div className="st-test-inline">
                {imgszSelect}
                <label className="st-mini" title="Maximum detections per image">
                  max det
                  <input
                    className="st-input num"
                    type="number"
                    min={1}
                    max={3000}
                    value={numOrEmpty(opts.maxDet)}
                    onChange={(e) => setOpt("maxDet", e.target.valueAsNumber)}
                  />
                </label>
              </div>
              <div className="st-test-toggles">
                <Toggle label="Agnostic NMS" checked={opts.agnostic} onChange={(v) => setOpt("agnostic", v)} hint="Suppress overlapping boxes across classes, not just within one" />
                <Toggle
                  label="TTA (augment)"
                  checked={opts.augment}
                  onChange={(v) => setOpt("augment", v)}
                  hint="Test-time augmentation (flips + 3 scales). Ultralytics only applies it to detect models running NMS; others fall back to single-scale"
                />
                {task === "segment" && (
                  <Toggle label="Retina masks" checked={opts.retina} onChange={(v) => setOpt("retina", v)} hint="Masks at full image resolution — crisper edges, slower" />
                )}
                {halfToggle}
              </div>
              <label className="st-test-field">
                <span className="st-field-label">Class filter</span>
                <input
                  className="st-input"
                  placeholder="class ids, e.g. 0, 2, 7 — empty = all"
                  value={opts.classes}
                  onChange={(e) => setOpt("classes", e.target.value)}
                />
              </label>
              {classHint && <p className="st-test-hint">{classHint}</p>}
            </>
          ) : (
            <>
              <p className="st-test-hint">Classifiers return top-5 class probabilities — box and NMS settings don't apply.</p>
              <div className="st-test-inline">
                {imgszSelect}
                {halfToggle}
              </div>
              {opts.imgsz !== 224 && (
                <p className="st-test-hint">
                  YOLO26-cls models are trained at 224 px —{" "}
                  <button type="button" className="st-test-linkbtn" onClick={() => setOpt("imgsz", 224)}>
                    use 224
                  </button>
                </p>
              )}
            </>
          )}

          <div className="st-test-run">
            <button
              type="button"
              className={`st-btn ${autoRun ? "" : "primary"}`}
              disabled={!request || predicting}
              onClick={() => void predictNow(true)}
              title="Predict the current image"
            >
              ▶ Run
            </button>
            <Toggle label="Auto-run" checked={autoRun} onChange={setAutoRun} hint="Re-run 300 ms after any model, setting or image change" />
          </div>
          {pred && <RunSummary pred={pred} stale={outdated && !autoRun} />}
          <div>
            <button
              type="button"
              className="st-btn"
              disabled={!canSend}
              onClick={sendToLabel}
              title="Run this model with these settings and store the detections as suggestions to review in Label"
            >
              Send to Label as suggestions
            </button>
          </div>
          {sent && sent.imageId === imageId && (
            <div className="st-test-sent">
              ✓ {plural(sent.n, "suggestion")} sent
              <button type="button" className="st-btn sm ghost" onClick={() => setTab("label")}>
                Review in Label →
              </button>
            </div>
          )}
        </Section>

        <Section title="Track video" collapsible defaultOpen={false} right={trackBadge}>
          <TrackPanel
            model={model}
            iou={opts.iou}
            imgsz={opts.imgsz}
            jobs={jobs}
            reload={reloadJobs}
            watchId={watchJob?.id ?? null}
            onWatch={setWatchId}
          />
        </Section>
      </aside>

      <div
        className={`st-test-main ${dragging ? "drop" : ""}`}
        onDragOver={(e) => {
          if (!e.dataTransfer.types.includes("Files")) return;
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={(e) => {
          if (!(e.relatedTarget instanceof Node && e.currentTarget.contains(e.relatedTarget))) setDragging(false);
        }}
        onDrop={(e) => {
          if (!e.dataTransfer.types.includes("Files")) return;
          e.preventDefault();
          setDragging(false);
          void upload(Array.from(e.dataTransfer.files));
        }}
      >
        <ThumbStrip
          pid={pid}
          images={images}
          selectedId={watchJob ? null : imageId}
          onSelect={pick}
          onUpload={() => fileRef.current?.click()}
        />
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          multiple
          hidden
          onChange={(e) => {
            void upload(Array.from(e.target.files ?? []));
            e.target.value = "";
          }}
        />

        <div className="st-test-stage">
          {watchJob ? (
            <TrackViewer pid={pid} job={watchJob} onClose={() => setWatchId(null)} />
          ) : (
            <>
              <div className="st-test-viewer">
                {image && (
                  <div className="st-toolbar">
                    <span className="st-test-name" title={image.filename}>
                      {image.filename}
                    </span>
                    <span className="st-muted">
                      {image.width}×{image.height}
                    </span>
                    <div className="st-toolbar-spacer" />
                    <Toggle label="Labels" checked={showLabels} onChange={setShowLabels} hint="Class + score tag on each detection" />
                    <button
                      type="button"
                      className="st-btn sm ghost"
                      onClick={() => setFitN((n) => n + 1)}
                      title="Fit to view. Scroll to zoom, hold Space and drag to pan, double-click empty space to fit"
                    >
                      Fit
                    </button>
                    <span className="st-muted st-nav">
                      <button type="button" className="st-btn sm ghost" onClick={() => step(-1)} title="Previous image (←)">
                        ←
                      </button>
                      {imageIdx + 1}/{images.length}
                      <button type="button" className="st-btn sm ghost" onClick={() => step(1)} title="Next image (→)">
                        →
                      </button>
                    </span>
                  </div>
                )}
                <div className="st-canvas-area">
                  {image ? (
                    <PromptCanvas
                      imageUrl={api.imageUrl(pid, image.id)}
                      imageWidth={image.width}
                      imageHeight={image.height}
                      shapes={shapes}
                      tool="select"
                      showLabels={showLabels}
                      keypointEdges={COCO_SKELETON}
                      hoveredId={hovered}
                      selectedIds={selected ? [selected] : undefined}
                      onHover={onCanvasHover}
                      onSelect={onCanvasSelect}
                      fitKey={`${image.id}:${fitN}`}
                    >
                      {top && <ClassifyCard top={top} />}
                      {hud && (
                        <div className={`st-canvas-hint st-test-hud ${hud.kind}`}>
                          {hud.kind === "busy" && <span className="st-spinner" />}
                          <span className="st-ellipsis">{hud.text}</span>
                        </div>
                      )}
                    </PromptCanvas>
                  ) : images.length === 0 ? (
                    <div className="st-canvas-empty">
                      <h2>Test a model</h2>
                      <p>
                        Run YOLO26 (detect, segment, classify, pose, OBB), YOLOE or your own trained models on an image and inspect
                        every detection.
                      </p>
                      <div className="st-test-empty-actions">
                        <button type="button" className="st-btn primary" onClick={() => fileRef.current?.click()}>
                          Upload images
                        </button>
                        <button type="button" className="st-btn" onClick={() => setTab("label")}>
                          Go to Label
                        </button>
                      </div>
                      <p className="st-muted">…or drop images anywhere here. Images you add in Label show up here too.</p>
                    </div>
                  ) : null}
                </div>
              </div>
              {image && (
                <ResultsPanel
                  pred={pred}
                  predicting={predicting}
                  failed={failed}
                  issue={issue}
                  autoRun={autoRun}
                  stale={outdated && !autoRun}
                  focus={focus}
                  onFocus={setFocusClass}
                  hovered={hovered}
                  onHover={setHovered}
                  selected={selected}
                  onSelect={(id) => setSelected((cur) => (cur === id ? null : id))}
                  onRetry={() => void predictNow(true)}
                  panelRef={resultsRef}
                />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ---- Thumbnail strip -----------------------------------------------------------

function ThumbStrip({
  pid,
  images,
  selectedId,
  onSelect,
  onUpload,
}: {
  pid: string;
  images: StudioImage[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onUpload: () => void;
}) {
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!selectedId) return;
    listRef.current
      ?.querySelector<HTMLElement>(`[data-id="${CSS.escape(selectedId)}"]`)
      ?.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "smooth" });
  }, [selectedId]);

  // A plain mouse wheel scrolls the strip sideways.
  useEffect(() => {
    const el = listRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if (Math.abs(e.deltaY) <= Math.abs(e.deltaX) || el.scrollWidth <= el.clientWidth) return;
      e.preventDefault();
      el.scrollLeft += e.deltaY;
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  return (
    <div className="st-test-strip">
      <button type="button" className="st-btn sm" onClick={onUpload} title="Upload images (or drop them anywhere here)">
        ⤒ Upload
      </button>
      <div className="st-test-thumbs" ref={listRef}>
        {images.length === 0 && <span className="st-muted">No images yet.</span>}
        {images.map((img) => (
          <button
            key={img.id}
            type="button"
            data-id={img.id}
            className={`st-test-thumb ${img.id === selectedId ? "on" : ""}`}
            onClick={() => onSelect(img.id)}
            title={`${img.filename} · ${img.width}×${img.height}`}
            aria-label={img.filename}
          >
            <img src={api.thumbUrl(pid, img.id)} loading="lazy" alt="" />
          </button>
        ))}
      </div>
    </div>
  );
}

// ---- Prediction summaries --------------------------------------------------------

function RunSummary({ pred, stale }: { pred: Prediction; stale: boolean }) {
  const parts = [
    { key: "pre", label: "pre", ms: pred.speed.preprocess },
    { key: "inf", label: "inference", ms: pred.speed.inference },
    { key: "post", label: "post", ms: pred.speed.postprocess },
  ];
  const total = parts.reduce((sum, p) => sum + (p.ms ?? 0), 0);
  const best = pred.classification?.top[0];
  return (
    <div className={`st-subtle-block st-test-summary ${stale ? "stale" : ""}`} title={stale ? "From the previous settings" : undefined}>
      <div className="st-row">
        <span className="st-grow st-test-summary-model" title={pred.model_label}>
          {pred.model_label}
        </span>
        <span className="st-test-count">
          {best ? `${best.class_name} · ${pct(best.score)}%` : plural(pred.detections.length, "object")}
        </span>
      </div>
      {total > 0 && (
        <div className="st-test-speed" title="Per-image time: preprocess / inference / postprocess">
          {parts.map((p) => (p.ms ? <span key={p.key} className={p.key} style={{ flexGrow: p.ms }} /> : null))}
        </div>
      )}
      <div className="st-test-legend">
        {parts.map((p) => (
          <span key={p.key}>
            <i className={`st-test-dot ${p.key}`} />
            {p.label} {fmtMs(p.ms)}
          </span>
        ))}
        <b>{total.toFixed(1)} ms</b>
      </div>
    </div>
  );
}

function ClassifyCard({ top }: { top: { class_id: number; class_name: string; score: number }[] }) {
  return (
    <div className="st-test-cls">
      <div className="st-test-cls-title">Top-{top.length} classes</div>
      {top.map((t, i) => (
        <div key={t.class_id} className={`st-test-cls-row ${i === 0 ? "top" : ""}`}>
          <span className="st-ellipsis">{t.class_name}</span>
          <span className="st-test-cls-score">{pct(t.score)}%</span>
          <div className="st-test-cls-bar">
            <div style={{ width: `${Math.max(1, t.score * 100)}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

interface ResultsPanelProps {
  pred: Prediction | null;
  predicting: boolean;
  failed: string | null;
  issue: string | null;
  autoRun: boolean;
  stale: boolean;
  focus: number | null;
  onFocus: (classId: number | null) => void;
  hovered: string | null;
  onHover: (id: string | null) => void;
  selected: string | null;
  onSelect: (id: string) => void;
  onRetry: () => void;
  panelRef: RefObject<HTMLElement>;
}

function ResultsPanel(props: ResultsPanelProps) {
  const { pred, focus, hovered, selected } = props;
  const dets = pred?.detections ?? NO_DETS;
  const top = pred?.classification?.top ?? null;
  const chips = useMemo(() => classCounts(dets), [dets]);
  const rows = useMemo(
    () => dets.map((d, i) => ({ d, i })).filter(({ d }) => focus === null || d.class_id === focus),
    [dets, focus],
  );
  const extraHead = pred?.task ? EXTRA_HEAD[pred.task] : undefined;

  let body: ReactNode;
  if (props.issue) {
    body = <div className="st-warn-box">{props.issue}</div>;
  } else if (props.failed) {
    body = (
      <>
        <div className="st-err-box">{props.failed}</div>
        <div>
          <button type="button" className="st-btn sm" onClick={props.onRetry}>
            Retry
          </button>
        </div>
      </>
    );
  } else if (!pred) {
    body = (
      <p className="st-muted st-help">
        {props.predicting ? (
          <>
            <span className="st-spinner dark" /> Running…
          </>
        ) : props.autoRun ? (
          "Waiting for the first prediction…"
        ) : (
          "Press Run to predict this image."
        )}
      </p>
    );
  } else if (top) {
    body = (
      <table className="st-table st-test-table">
        <colgroup>
          <col className="st-test-col-idx" />
          <col />
          <col className="st-test-col-score" />
        </colgroup>
        <thead>
          <tr>
            <th className="num">#</th>
            <th>Class</th>
            <th className="num">Score</th>
          </tr>
        </thead>
        <tbody>
          {top.map((t, i) => (
            <tr key={t.class_id}>
              <td className="num st-muted">{i + 1}</td>
              <td>
                <span className="st-test-cname">
                  <span className="st-ellipsis">{t.class_name}</span>
                  <span className="st-test-cid">{t.class_id}</span>
                </span>
              </td>
              <td className="num">{pct(t.score)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  } else if (!dets.length) {
    body = <p className="st-muted st-help">No objects above the confidence threshold. Lower Conf, clear the class filter or try another model.</p>;
  } else {
    body = (
      <>
        <div className="st-test-chips">
          <button type="button" className={`st-test-chip ${focus === null ? "on" : ""}`} onClick={() => props.onFocus(null)}>
            All <b>{dets.length}</b>
          </button>
          {chips.map((c) => (
            <button
              key={c.id}
              type="button"
              className={`st-test-chip ${focus === c.id ? "on" : ""}`}
              title={`Class ${c.id} — click to isolate`}
              onClick={() => props.onFocus(focus === c.id ? null : c.id)}
            >
              <span className="st-swatch" style={{ background: colorForIndex(c.id) }} />
              <span className="st-ellipsis">{c.name}</span>
              <b>{c.n}</b>
            </button>
          ))}
        </div>
        <table className="st-table st-test-table">
          <colgroup>
            <col className="st-test-col-idx" />
            <col />
            <col className="st-test-col-score" />
            <col className="st-test-col-box" />
            {extraHead && <col className="st-test-col-extra" />}
          </colgroup>
          <thead>
            <tr>
              <th className="num">#</th>
              <th>Class</th>
              <th className="num">Score</th>
              <th className="num" title="Box width × height in pixels (the rotated box for OBB)">
                Box px
              </th>
              {extraHead && <th className="num">{extraHead}</th>}
            </tr>
          </thead>
          <tbody>
            {rows.map(({ d, i }) => {
              const sid = `d:${i}`;
              return (
                <tr
                  key={sid}
                  data-sid={sid}
                  className={`${hovered === sid ? "hover" : ""} ${selected === sid ? "on" : ""}`}
                  onMouseEnter={() => props.onHover(sid)}
                  onMouseLeave={() => props.onHover(null)}
                  onClick={() => props.onSelect(sid)}
                >
                  <td className="num st-muted">{i + 1}</td>
                  <td title={`${d.class_name} (class ${d.class_id})`}>
                    <span className="st-test-cname">
                      <span className="st-swatch" style={{ background: colorForIndex(d.class_id) }} />
                      <span className="st-ellipsis">{d.class_name}</span>
                      <span className="st-test-cid">{d.class_id}</span>
                    </span>
                  </td>
                  <td className="num">{pct(d.score)}</td>
                  <td className="num">{boxSize(d)}</td>
                  {extraHead && <td className="num">{extraCell(d)}</td>}
                </tr>
              );
            })}
          </tbody>
        </table>
      </>
    );
  }

  return (
    <aside className={`st-test-results ${props.stale ? "stale" : ""}`} ref={props.panelRef}>
      <header className="st-test-results-head">
        <h3>Results</h3>
        {pred?.task && <span className="st-pill">{pred.task}</span>}
        {pred && <span className="st-test-count">{top ? `top-${top.length}` : plural(dets.length, "object")}</span>}
      </header>
      <div className="st-test-results-body">{body}</div>
    </aside>
  );
}

// ---- Tracking --------------------------------------------------------------------

/** Track jobs for the project; polls while any job is queued or running. */
function useTrackJobs(pid: string | null) {
  const toast = useStudio((s) => s.toast);
  const [jobs, setJobs] = useState<TrackJob[]>([]);
  const seen = useRef(new Map<string, TrackJob["status"]>());

  const reload = useCallback(async () => {
    if (!pid) return;
    const { tracks } = await api.fetchTracks(pid);
    if (useStudio.getState().projectId !== pid) return;
    for (const j of tracks) {
      const was = seen.current.get(j.id);
      if (was !== "queued" && was !== "running") continue;
      if (j.status === "completed") {
        toast(`Tracking done · ${baseName(j.video_path)} · ${plural(j.stats?.unique_tracks ?? 0, "track")}`, "success");
      } else if (j.status === "failed") {
        toast(`Tracking failed · ${baseName(j.video_path)}${j.error ? `: ${j.error}` : ""}`, "error");
      }
    }
    seen.current = new Map(tracks.map((j) => [j.id, j.status]));
    setJobs(tracks);
  }, [pid, toast]);

  useEffect(() => {
    setJobs([]);
    seen.current = new Map();
    reload().catch((e) => toast(`Couldn't load tracking jobs: ${errText(e)}`, "error"));
  }, [reload, toast]);

  const active = jobs.some((j) => j.status === "queued" || j.status === "running");
  useEffect(() => {
    if (!active) return;
    let timer = 0;
    let stopped = false;
    const tick = () => {
      reload()
        .catch(() => {
          /* transient — keep polling */
        })
        .finally(() => {
          if (!stopped) timer = window.setTimeout(tick, POLL_MS);
        });
    };
    timer = window.setTimeout(tick, POLL_MS);
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [active, reload]);

  return { jobs, reload };
}

function TrackPanel({
  model,
  iou,
  imgsz,
  jobs,
  reload,
  watchId,
  onWatch,
}: {
  model: ModelSpec;
  iou: number;
  imgsz: number;
  jobs: TrackJob[];
  reload: () => Promise<void>;
  watchId: string | null;
  onWatch: (id: string | null) => void;
}) {
  const pid = useStudio((s) => s.projectId);
  const models = useStudio((s) => s.models);
  const catalog = useStudio((s) => s.catalog);
  const run = useStudio((s) => s.run);
  const toast = useStudio((s) => s.toast);
  const videos = useStore((s) => s.videos);
  const [o, setO] = useState<TrackOpts>(rememberedTrack);

  useEffect(() => {
    rememberedTrack = o;
  }, [o]);

  const set = <K extends keyof TrackOpts>(k: K, v: TrackOpts[K]) => setO((cur) => ({ ...cur, [k]: v }));
  const trackers = catalog?.trackers?.length ? catalog.trackers : ["bytetrack", "botsort"];
  const tracker = trackers.includes(o.tracker) ? o.tracker : trackers[0];
  const issue = trackIssue(model);

  const start = async () => {
    if (!pid || !o.video || issue) return;
    const job = await run("Starting tracking", () =>
      api.startTrack(pid, {
        video_path: o.video,
        model,
        tracker,
        params: { conf: o.conf, iou, imgsz },
        stride: Math.max(1, Math.floor(o.stride) || 1),
        max_frames: Math.max(0, Math.floor(o.maxFrames) || 0),
        count_line: o.line === "none" ? null : { axis: o.line, pos: o.linePos },
      }),
    );
    if (!job) return;
    reload().catch(() => {});
    toast(`Tracking ${baseName(o.video)} in the background — the overlay video appears below when it's done.`, "success");
  };

  const remove = async (job: TrackJob) => {
    if (!pid) return;
    const active = job.status === "queued" || job.status === "running";
    if (!active && !window.confirm(`Delete the tracking result for ${baseName(job.video_path)}?`)) return;
    await run(active ? "Cancelling tracking" : "Deleting tracking result", () => api.deleteTrack(pid, job.id));
    if (job.id === watchId) onWatch(null);
    reload().catch(() => {});
    // A running job is cleaned up by the worker once it notices the cancel.
    if (active) window.setTimeout(() => reload().catch(() => {}), 2500);
  };

  return (
    <>
      <p className="st-test-hint">
        Ultralytics Track mode: the model above runs on every frame and the tracker links detections into IDs. Jobs run in the
        background.
      </p>
      <label className="st-test-field">
        <span className="st-field-label">Video</span>
        <select className="st-input" value={o.video} onChange={(e) => set("video", e.target.value)} title={o.video || undefined}>
          <option value="">{videos.length ? "Choose a video in data/…" : "No videos found in data/"}</option>
          {videos.map((v) => (
            <option key={v} value={v}>
              {v.replace(/^data\//, "")}
            </option>
          ))}
        </select>
      </label>
      <div className="st-row">
        <span className="st-field-label">Tracker</span>
        <Segmented size="sm" value={tracker} onChange={(v) => set("tracker", v)} options={trackers.map((t) => ({ value: t, label: trackerLabel(t) }))} />
      </div>
      <div className="st-test-inline">
        <label className="st-mini" title="Process every Nth frame">
          every
          <input className="st-input num" type="number" min={1} value={numOrEmpty(o.stride)} onChange={(e) => set("stride", e.target.valueAsNumber)} />
          frame{o.stride === 1 ? "" : "s"}
        </label>
        <label className="st-mini" title="Stop after this many processed frames (0 = the whole video)">
          max
          <input className="st-input num" type="number" min={0} value={numOrEmpty(o.maxFrames)} onChange={(e) => set("maxFrames", e.target.valueAsNumber)} />
          {o.maxFrames === 0 ? "(all)" : "frames"}
        </label>
      </div>
      <Slider label="Conf" value={o.conf} min={0.01} max={0.95} step={0.01} onChange={(v) => set("conf", v)} format={(v) => v.toFixed(2)} />
      <div className="st-row wrap">
        <span className="st-field-label">Counting line</span>
        <Segmented
          size="sm"
          value={o.line}
          onChange={(v) => set("line", v)}
          options={[
            { value: "none", label: "None" },
            { value: "y", label: "Horizontal" },
            { value: "x", label: "Vertical" },
          ]}
        />
      </div>
      {o.line !== "none" && (
        <div className="st-test-line">
          <div className={`st-test-line-box ${o.line}`}>
            <i style={o.line === "y" ? { top: `${o.linePos * 100}%` } : { left: `${o.linePos * 100}%` }} />
          </div>
          <div className="st-test-line-ctl">
            <Slider
              label="Position"
              value={o.linePos}
              min={0.05}
              max={0.95}
              step={0.01}
              onChange={(v) => set("linePos", v)}
              format={(v) => `${Math.round(v * 100)}%`}
            />
            <p className="st-test-hint">
              Counts each track once when its centre crosses the line (e.g. cartons on a conveyor), per direction:{" "}
              {o.line === "y" ? "↓ down / ↑ up" : "→ right / ← left"}.
            </p>
          </div>
        </div>
      )}
      <p className="st-test-hint">
        Model <b>{specLabel(model, models)}</b> · IoU {iou.toFixed(2)} · {imgsz} px — set in Predict above.
      </p>
      {issue && <div className="st-warn-box">{issue}</div>}
      <div>
        <button type="button" className="st-btn primary" disabled={!pid || !o.video || !!issue} onClick={start}>
          Start tracking
        </button>
      </div>
      {jobs.length > 0 && (
        <div className="st-test-jobs">
          {jobs.map((j) => (
            <JobCard key={j.id} job={j} watching={j.id === watchId} onWatch={() => onWatch(j.id)} onRemove={() => remove(j)} />
          ))}
        </div>
      )}
    </>
  );
}

function JobCard({ job, watching, onWatch, onRemove }: { job: TrackJob; watching: boolean; onWatch: () => void; onRemove: () => void }) {
  const models = useStudio((s) => s.models);
  const active = job.status === "queued" || job.status === "running";
  const p = job.progress;
  const frac = p && p.total > 0 ? Math.min(1, p.done / p.total) : null;
  const s = job.stats;
  const line = job.count_line;
  const [fwd, bwd] = line?.axis === "x" ? ["→", "←"] : ["↓", "↑"];
  const classes = s ? Object.entries(s.unique_by_class) : [];

  return (
    <div className={`st-test-job ${watching ? "on" : ""}`}>
      <div className="st-test-job-head">
        <span className={`st-pill ${job.status}`}>{job.status}</span>
        <span className="st-test-job-name" title={job.video_path}>
          {baseName(job.video_path)}
        </span>
        {job.status === "completed" && job.has_video && (
          <button type="button" className={`st-btn sm ${watching ? "on" : "primary"}`} disabled={watching} onClick={onWatch}>
            {watching ? "Watching" : "▶ Watch"}
          </button>
        )}
        <button type="button" className="st-btn sm ghost" onClick={onRemove} title={active ? "Stop this job" : "Delete this result"}>
          {active ? "Cancel" : "Delete"}
        </button>
      </div>
      <div className="st-muted st-test-job-meta">
        {specLabel(job.model, models)} · {trackerLabel(job.tracker)}
        {job.stride > 1 ? ` · every ${job.stride} frames` : ""}
        {job.max_frames ? ` · ≤${job.max_frames} frames` : ""}
        {line ? ` · ${line.axis === "y" ? "horizontal" : "vertical"} line at ${Math.round(line.pos * 100)}%` : ""}
      </div>

      {active && (
        <>
          <div className={`st-progress ${frac === null && job.status === "running" ? "st-test-indet" : ""}`}>
            <div style={frac === null ? undefined : { width: `${frac * 100}%` }} />
          </div>
          <div className="st-muted">
            {job.status === "queued"
              ? "Queued — waiting for the tracker worker"
              : p
                ? `${p.done}${p.total ? ` / ${p.total}` : ""} frames${frac !== null ? ` · ${Math.round(frac * 100)}%` : ""}`
                : "Loading the model…"}
          </div>
        </>
      )}

      {job.status === "failed" && job.error && <div className="st-err-box">{job.error}</div>}

      {job.status === "completed" && s && (
        <>
          <div className="st-test-jobstats">
            <div title="Frames processed">
              <b>{s.frames}</b>
              <span>frames</span>
            </div>
            <div title="Processing speed (frames per second)">
              <b>{s.fps ?? "—"}</b>
              <span>proc fps</span>
            </div>
            <div title="Mean model time per frame">
              <b>{s.mean_infer_ms ?? "—"}</b>
              <span>ms / frame</span>
            </div>
            <div title="Unique track IDs">
              <b>{s.unique_tracks}</b>
              <span>tracks</span>
            </div>
          </div>
          {classes.length > 0 && (
            <div className="st-test-chips" title="Unique tracks per class">
              {classes.map(([name, n]) => (
                <span key={name} className="st-test-chip static">
                  {name} <b>{n}</b>
                </span>
              ))}
            </div>
          )}
          {s.line_counts && (
            <div className="st-test-linecount">
              <b>{s.line_counts.forward + s.line_counts.backward}</b> crossed the line
              <span className="st-muted">
                {fwd} {s.line_counts.forward} · {bwd} {s.line_counts.backward}
              </span>
            </div>
          )}
          <div className="st-muted">Took {fmtSecs(s.seconds)}</div>
        </>
      )}
    </div>
  );
}

function TrackViewer({ pid, job, onClose }: { pid: string; job: TrackJob; onClose: () => void }) {
  const models = useStudio((s) => s.models);
  const url = api.trackVideoUrl(pid, job.id);
  const s = job.stats;
  return (
    <div className="st-test-viewer">
      <div className="st-toolbar">
        <span className="st-pill completed">Track</span>
        <span className="st-test-name" title={job.video_path}>
          {baseName(job.video_path)}
        </span>
        <span className="st-muted st-ellipsis st-test-shrink">
          {trackerLabel(job.tracker)} · {specLabel(job.model, models)}
          {s ? ` · ${plural(s.unique_tracks, "track")} · ${s.frames} frames` : ""}
          {s?.line_counts ? ` · ${s.line_counts.forward + s.line_counts.backward} crossed the line` : ""}
        </span>
        <div className="st-toolbar-spacer" />
        <a className="st-btn sm ghost" href={url} target="_blank" rel="noreferrer" title="Open the video in a new tab">
          Open ↗
        </a>
        <button type="button" className="st-btn sm" onClick={onClose}>
          ← Back to image
        </button>
      </div>
      <div className="st-test-video">
        <video key={job.id} src={url} controls autoPlay muted playsInline />
      </div>
    </div>
  );
}
