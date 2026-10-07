/**
 * Train — YOLO26 detect / segment / OBB on this project's labels.
 *
 *   Left   a new run: task, size, starting weights, schedule, and a collapsed
 *          "Advanced" block (optimiser, LR, freeze, val split, augmentation).
 *   Right  dataset health → model list with live progress → the selected
 *          model: metrics, curves, per-class AP, Ultralytics plots, and
 *          validate / export / benchmark once it has weights.
 *
 * Jobs run one at a time in a server subprocess (pipeline/studio/training.py).
 * The Studio shell polls the model list while anything is in flight; the
 * detail panel polls its own model every 3 s while it is queued or running.
 */

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import * as api from "./api";
import type { BenchRow, PerClassMetric, TrainConfig, TrainedModel, TrainTask, ValResult, YoloSize } from "./types";
import { classColor, useStudio } from "./useStudio";
import { Section, Segmented, Slider, Toggle, fmtBytes, fmtSecs, pct } from "./widgets";
import "./train.css";

type Status = TrainedModel["status"];
type BaseKind = "pretrained" | "scratch" | "model";
type ValSplit = "val" | "train" | "test";
type Head = "box" | "mask";
type Metrics = Record<string, number>;

// ---- Constants --------------------------------------------------------------

const TASK_LABEL: Record<TrainTask, string> = { detect: "Detect", segment: "Segment", obb: "OBB" };
const TASK_SUFFIX: Record<TrainTask, string> = { detect: "", segment: "-seg", obb: "-obb" };

const TASK_OPTIONS: RadioOption<TrainTask>[] = [
  { value: "detect", title: "Detect", blurb: "Fastest — axis-aligned boxes only. Counting and coarse picks." },
  { value: "segment", title: "Segment", blurb: "Carton masks — needed for height analysis in the Pallet tab.", tag: "Pallet" },
  { value: "obb", title: "OBB", blurb: "Rotated boxes — each carton's yaw for the gripper." },
];

/** Approximate parameter counts; `cost` is relative training time (1–5). */
const SIZE_INFO: { value: YoloSize; params: string; cost: number; hint: string }[] = [
  { value: "n", params: "2.4M", cost: 1, hint: "fastest to train — ideal for a first run" },
  { value: "s", params: "9.5M", cost: 2, hint: "a good accuracy / speed balance" },
  { value: "m", params: "20M", cost: 3, hint: "noticeably slower to train on a laptop" },
  { value: "l", params: "25M", cost: 4, hint: "slow to train on an M-series Mac" },
  { value: "x", params: "56M", cost: 5, hint: "slowest — the largest, most accurate model" },
];

const IMG_SIZES = [320, 416, 512, 640, 800, 960, 1024, 1280];
const OPTIMIZERS = ["auto", "MuSGD", "SGD", "AdamW", "Adam"];
const ADVANCED_KEYS = ["optimizer", "lr0", "cos_lr", "freeze", "val_pct"] as const;

/** Mirrors training.DEFAULT_CONFIG until the catalog has loaded. */
const FALLBACK_CONFIG: TrainConfig = {
  epochs: 50,
  imgsz: 640,
  batch: 8,
  patience: 25,
  optimizer: "auto",
  lr0: null,
  cos_lr: false,
  freeze: null,
  val_pct: 20,
  seed: 0,
  workers: 2,
  cache: false,
  augment: {},
};

interface AugSpec {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  /** Ultralytics' default — only values that differ from it are sent. */
  def: number;
  hint: string;
  unit?: "deg" | "epochs";
  segmentOnly?: boolean;
}

const AUGS: AugSpec[] = [
  { key: "mosaic", label: "Mosaic", min: 0, max: 1, step: 0.05, def: 1, hint: "Probability of stitching 4 images into one — a strong regulariser for small datasets." },
  { key: "close_mosaic", label: "Close mosaic", min: 0, max: 50, step: 1, def: 10, unit: "epochs", hint: "Turn mosaic off for the last N epochs so training finishes on whole, realistic images." },
  { key: "fliplr", label: "Flip L↔R", min: 0, max: 1, step: 0.05, def: 0.5, hint: "Probability of a horizontal flip." },
  { key: "flipud", label: "Flip U↕D", min: 0, max: 1, step: 0.05, def: 0, hint: "Probability of a vertical flip. Top-down pallet shots have no 'up', so 0.5 is a free win." },
  { key: "degrees", label: "Rotate ±", min: 0, max: 180, step: 5, def: 0, unit: "deg", hint: "Random rotation range. Great for top-down pallets with masks / OBB; with Detect, big rotations loosen the boxes." },
  { key: "scale", label: "Scale ±", min: 0, max: 0.9, step: 0.05, def: 0.5, hint: "Random zoom gain — covers different camera heights." },
  { key: "translate", label: "Translate ±", min: 0, max: 0.5, step: 0.05, def: 0.1, hint: "Random shift as a fraction of the image size." },
  { key: "hsv_v", label: "Brightness", min: 0, max: 0.9, step: 0.05, def: 0.4, hint: "Random brightness (HSV value) gain — covers lighting changes on the line." },
  { key: "mixup", label: "Mixup", min: 0, max: 1, step: 0.05, def: 0, hint: "Probability of blending two images. Rarely helps small datasets." },
  { key: "copy_paste", label: "Copy-paste", min: 0, max: 1, step: 0.05, def: 0, segmentOnly: true, hint: "Paste masked objects between images (segment only) — more cartons per image." },
];

/** Top-down pallet shots have no "up": flips and rotation are free data. */
function topDownPreset(task: TrainTask): Record<string, number> {
  // Rotating axis-aligned boxes makes them loose, so Detect only gets the flip.
  return task === "detect" ? { flipud: 0.5 } : { flipud: 0.5, degrees: 90 };
}

const PREFERRED_PLOTS = ["results.png", "confusion_matrix_normalized.png", "BoxPR_curve.png", "MaskPR_curve.png"];
const LOSS_COLORS = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#0ea5e9", "#a855f7", "#64748b"];
/** YOLO26 reports some losses (e.g. DFL) as exactly zero; hide those. */
const LOSS_EPS = 1e-4;
const POLL_MS = 3000;
const BENCH_IMAGES = 20;

// ---- Helpers ----------------------------------------------------------------

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const num = (v: unknown): number | null => (isNum(v) ? v : null);
const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;
const isLive = (s: Status) => s === "queued" || s === "running";
const weightsName = (task: TrainTask, size: YoloSize) => `yolo26${size}${TASK_SUFFIX[task]}`;
const lossName = (key: string) => key.replace(/^train\//, "").replace(/_loss$/, "");
const firstLine = (s: string | null) => (s ?? "").split("\n").find((l) => l.trim()) ?? "";
const plotLabel = (name: string) => name.replace(/\.png$/i, "").replace(/_/g, " ");

/** Ultralytics metric suffixes: (B) box / rotated-box head, (M) mask head. */
const headsFor = (task: TrainTask): { suffix: "B" | "M"; label: string }[] => [
  { suffix: "B", label: task === "obb" ? "OBB" : "Box" },
  { suffix: "M", label: "Mask" },
];
const mkey = (metric: string, suffix: "B" | "M") => `metrics/${metric}(${suffix})`;

/** fmtSecs, plus hours for long runs. */
function fmtDur(s: number | null | undefined): string {
  if (!isNum(s)) return "—";
  if (s < 3600) return fmtSecs(s);
  return `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m`;
}

function fmtMs(v: number | undefined): string {
  if (!isNum(v)) return "—";
  return v < 10 ? v.toFixed(2) : v < 100 ? v.toFixed(1) : v.toFixed(0);
}

function fmtFps(v: number | undefined): string {
  if (!isNum(v)) return "—";
  return v < 100 ? v.toFixed(1) : v.toFixed(0);
}

const fmtLoss = (v: number) => (Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(3));

function fmtAug(a: AugSpec, v: number): string {
  if (a.unit === "deg") return `${v}°`;
  if (a.unit === "epochs") return `${v} ep`;
  return v.toFixed(2);
}

function fmtWhen(iso: string | null | undefined): string {
  if (!iso) return "—";
  const t = Date.parse(iso);
  return Number.isNaN(t) ? iso : new Date(t).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function timeAgo(iso: string | null | undefined): string {
  if (!iso) return "—";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return iso;
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 45) return "just now";
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  if (s < 7 * 86400) return `${Math.round(s / 86400)}d ago`;
  return new Date(t).toLocaleDateString();
}

/** Jobs that run before `m`: whatever is running, plus older queued jobs. */
function queueAhead(m: TrainedModel, models: TrainedModel[]): number {
  return models.filter(
    (x) => x.id !== m.id && (x.status === "running" || (x.status === "queued" && x.created_at < m.created_at)),
  ).length;
}

function baseLabel(m: TrainedModel, models: TrainedModel[], long: boolean): string {
  if (m.base === "scratch") return long ? `From scratch (${m.base_weights})` : "from scratch";
  if (m.base.startsWith("model:")) {
    const id = m.base.slice("model:".length);
    const parent = models.find((x) => x.id === id)?.name ?? id;
    return long ? `Fine-tuned from ${parent}` : `↳ ${parent}`;
  }
  const src = m.task === "obb" ? "DOTA" : "COCO";
  return long ? `${m.base_weights} · pretrained on ${src}` : `pretrained ${src}`;
}

/** The number Pal/DePal cares about: mask mAP50-95 for segment, box otherwise. */
function keyMetric(m: TrainedModel): number | undefined {
  return m.metrics?.[mkey("mAP50-95", m.task === "segment" ? "M" : "B")];
}

function qClass(v: number | undefined): string {
  if (!isNum(v)) return "";
  return v >= 0.75 ? "st-train-q-good" : v < 0.4 ? "st-train-q-bad" : "";
}

function orderPlots(names: string[]): string[] {
  const rank = (n: string) => {
    const i = PREFERRED_PLOTS.indexOf(n);
    return i === -1 ? PREFERRED_PLOTS.length : i;
  };
  return names.filter((n) => /\.png$/i.test(n)).sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
}

const sizeOptions = (current: number) => Array.from(new Set([...IMG_SIZES, current])).sort((a, b) => a - b);

interface ProgressInfo {
  frac: number;
  busy: boolean;
  short: string;
  long: string;
}

function progressInfo(m: TrainedModel, ahead: number): ProgressInfo {
  const p = m.progress;
  const epochs = p?.epochs || m.config.epochs;
  const done = Math.min(p?.epoch ?? 0, epochs);
  if (m.status === "queued") {
    return ahead
      ? { frac: 0, busy: true, short: `${plural(ahead, "job")} ahead`, long: `Queued — ${plural(ahead, "job")} ahead` }
      : { frac: 0, busy: true, short: "next up", long: "Queued — starting shortly" };
  }
  switch (p?.stage) {
    case "training":
      return done > 0
        ? { frac: done / epochs, busy: false, short: `epoch ${done}/${epochs}`, long: `${done} of ${epochs} epochs done` }
        : { frac: 0, busy: true, short: `epoch 0/${epochs}`, long: "First epoch running" };
    case "validating":
      return { frac: 1, busy: true, short: "final validation", long: "Final validation" };
    case "done":
      return { frac: 1, busy: true, short: "saving", long: "Saving results" };
    case "failed":
      return { frac: done / epochs, busy: false, short: "failing", long: p.message ? `Failing — ${p.message}` : "Failing" };
    default:
      return { frac: 0, busy: true, short: "starting", long: p?.message ? `Starting — ${p.message}` : "Starting" };
  }
}

// ---- View -------------------------------------------------------------------

export function TrainView() {
  const pid = useStudio((s) => s.projectId);
  const models = useStudio((s) => s.models);
  const refresh = useStudio((s) => s.refresh);
  const refreshModels = useStudio((s) => s.refreshModels);
  const [task, setTask] = useState<TrainTask>("segment");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  // Label-tab edits don't update the cached dataset stats — reload on entry.
  useEffect(() => {
    if (!pid) return;
    refresh().catch(() => undefined);
    refreshModels().catch(() => undefined);
  }, [pid, refresh, refreshModels]);

  const sorted = useMemo(() => [...models].sort((a, b) => b.created_at.localeCompare(a.created_at)), [models]);
  const selected = sorted.find((m) => m.id === selectedId) ?? sorted[0] ?? null;

  if (!pid) return null;

  return (
    <div className="st-page st-train">
      <aside className="st-page-side st-train-side">
        <TrainForm task={task} onTask={setTask} onQueued={(m) => setSelectedId(m.id)} />
      </aside>
      <div className="st-page-main">
        <DatasetHealth task={task} />
        <ModelList models={sorted} selectedId={selected?.id ?? null} onSelect={setSelectedId} />
        {selected && <ModelDetail key={selected.id} model={selected} models={sorted} />}
      </div>
    </div>
  );
}

// ---- New-run form -----------------------------------------------------------

function TrainForm({
  task,
  onTask,
  onQueued,
}: {
  task: TrainTask;
  onTask: (t: TrainTask) => void;
  onQueued: (m: TrainedModel) => void;
}) {
  const pid = useStudio((s) => s.projectId);
  const catalog = useStudio((s) => s.catalog);
  const models = useStudio((s) => s.models);
  const stats = useStudio((s) => s.stats);
  const run = useStudio((s) => s.run);
  const toast = useStudio((s) => s.toast);
  const refreshModels = useStudio((s) => s.refreshModels);

  const [size, setSize] = useState<YoloSize>("n");
  const [baseKind, setBaseKind] = useState<BaseKind>("pretrained");
  const [parentId, setParentId] = useState("");
  const [name, setName] = useState("");
  const [overrides, setOverrides] = useState<Partial<TrainConfig>>({});
  const [aug, setAug] = useState<Record<string, number>>({});
  const [submitting, setSubmitting] = useState(false);

  // Defaults come from the server catalog; the form only stores what you changed.
  const defaults = catalog?.default_train_config ?? FALLBACK_CONFIG;
  const cfg: TrainConfig = { ...defaults, ...overrides };
  const patchCfg = (patch: Partial<TrainConfig>) => setOverrides((o) => ({ ...o, ...patch }));

  // Fine-tuning needs a completed model of the same task; its size is fixed.
  const parents = models.filter((m) => m.status === "completed" && m.task === task);
  const parent = parents.find((m) => m.id === parentId) ?? parents[0] ?? null;
  const kind: BaseKind = baseKind === "model" && !parent ? "pretrained" : baseKind;
  const base = kind === "model" && parent ? `model:${parent.id}` : kind === "scratch" ? "scratch" : "pretrained";
  const effSize: YoloSize = kind === "model" && parent ? parent.size : size;
  const weights = weightsName(task, effSize);
  const sizeInfo = SIZE_INFO.find((s) => s.value === effSize) ?? SIZE_INFO[0];
  const isCached = (s: YoloSize) => catalog?.yolo26[task]?.find((x) => x.size === s)?.cached ?? true;

  // Augmentation: sliders show Ultralytics defaults; only changed keys are sent.
  const allowedAug = new Set(catalog?.augment_keys ?? AUGS.map((a) => a.key));
  const augSpecs = AUGS.filter((a) => allowedAug.has(a.key) && (!a.segmentOnly || task === "segment"));
  const augDefault = (a: AugSpec) => defaults.augment?.[a.key] ?? a.def;
  const changedAug = augSpecs.filter((a) => a.key in aug);
  const patchAug = (patch: Record<string, number>) =>
    setAug((cur) => {
      const next = { ...cur };
      for (const [key, value] of Object.entries(patch)) {
        const spec = AUGS.find((a) => a.key === key);
        if (!spec) continue;
        if (value === augDefault(spec)) delete next[key];
        else next[key] = value;
      }
      return next;
    });

  const advChanged = ADVANCED_KEYS.filter((k) => cfg[k] !== defaults[k]).length + changedAug.length;
  const inFlight = models.filter((m) => isLive(m.status)).length;
  const noData = stats !== null && stats.n_labeled === 0;

  const start = async () => {
    if (!pid || submitting) return;
    const config: Partial<TrainConfig> = {
      epochs: cfg.epochs,
      imgsz: cfg.imgsz,
      batch: cfg.batch,
      patience: cfg.patience,
      optimizer: cfg.optimizer,
      lr0: cfg.lr0,
      cos_lr: cfg.cos_lr,
      freeze: cfg.freeze,
      val_pct: cfg.val_pct,
      augment: { ...defaults.augment, ...Object.fromEntries(changedAug.map((a) => [a.key, aug[a.key]])) },
    };
    setSubmitting(true);
    const m = await run("Queuing training", () =>
      api.startTraining(pid, { task, size: effSize, base, name: name.trim() || undefined, config }),
    );
    setSubmitting(false);
    if (!m) return;
    setName("");
    await refreshModels().catch(() => undefined);
    onQueued(m);
    toast(
      inFlight
        ? `Queued “${m.name}” — ${plural(inFlight, "job")} ahead of it.`
        : `Training “${m.name}” — follow along on the right.`,
      "success",
    );
  };

  return (
    <>
      <div className="st-train-side-head">
        <h2>Train a YOLO26 model</h2>
        <p>Fine-tunes Ultralytics YOLO26 on this project's labels. Runs queue up and train one at a time.</p>
      </div>

      <Section title="Task">
        <RadioCards value={task} onChange={onTask} options={TASK_OPTIONS} />
      </Section>

      <Section title="Model size" right={<span className="st-muted">params ≈ approx.</span>}>
        <div className="st-train-sizes" role="radiogroup" aria-label="Model size">
          {SIZE_INFO.map((s) => (
            <button
              key={s.value}
              type="button"
              role="radio"
              aria-checked={effSize === s.value}
              className={`st-train-size${effSize === s.value ? " on" : ""}`}
              disabled={kind === "model"}
              onClick={() => setSize(s.value)}
              title={`${weightsName(task, s.value)} — ≈${s.params} parameters (approximate); ${s.hint}`}
            >
              <span className="st-train-size-letter">{s.value}</span>
              <span className="st-train-size-params">≈{s.params}</span>
              <span className="st-train-size-cost" aria-hidden>
                {[1, 2, 3, 4, 5].map((i) => (
                  <i key={i} className={i <= s.cost ? "on" : ""} />
                ))}
              </span>
              {kind === "pretrained" && catalog && !isCached(s.value) && (
                <span className="st-dl" title="Pretrained weights download on first use">
                  ↓
                </span>
              )}
            </button>
          ))}
        </div>
        <p className="st-train-note">
          {kind === "model" && parent ? (
            <>
              Size follows the parent model (<b>{effSize}</b>).
            </>
          ) : (
            <>
              <b>{weights}</b> · ≈{sizeInfo.params} params — {sizeInfo.hint}. Bars show relative training time.
            </>
          )}
        </p>
      </Section>

      <Section title="Starting weights">
        <RadioCards
          compact
          value={kind}
          onChange={setBaseKind}
          options={[
            {
              value: "pretrained",
              title: `Pretrained (${task === "obb" ? "DOTA" : "COCO"})`,
              blurb: isCached(effSize)
                ? "Ultralytics YOLO26 weights — best for small datasets."
                : `Ultralytics YOLO26 weights — ${weights}.pt downloads on first use.`,
            },
            { value: "scratch", title: "From scratch", blurb: "Random init — needs hundreds of epochs and lots of data." },
            {
              value: "model",
              title: "Fine-tune a trained model",
              blurb: parents.length
                ? `Continue from one of your ${plural(parents.length, `${TASK_LABEL[task]} model`)}.`
                : `No completed ${TASK_LABEL[task]} models yet.`,
              disabled: parents.length === 0,
            },
          ]}
        />
        {kind === "model" && parent && (
          <select className="st-input" value={parent.id} onChange={(e) => setParentId(e.target.value)} title="Model to fine-tune">
            {parents.map((m) => {
              const km = keyMetric(m);
              return (
                <option key={m.id} value={m.id}>
                  {m.name} · {m.size}
                  {isNum(km) ? ` · mAP50-95 ${pct(km)}` : ""}
                </option>
              );
            })}
          </select>
        )}
      </Section>

      <Section title="Schedule">
        <div className="st-train-fields">
          <Field label="Epochs" hint="Full passes over the training images">
            <NumberInput integer min={1} max={1000} value={cfg.epochs} onChange={(v) => v !== null && patchCfg({ epochs: v })} />
          </Field>
          <Field label="Image size" hint="Training resolution — larger finds small cartons but trains slower">
            <select className="st-input" value={cfg.imgsz} onChange={(e) => patchCfg({ imgsz: Number(e.target.value) })}>
              {sizeOptions(cfg.imgsz).map((s) => (
                <option key={s} value={s}>
                  {s} px
                </option>
              ))}
            </select>
          </Field>
          <Field label="Batch" hint="Images per step — lower it if training runs out of memory">
            <NumberInput integer min={1} max={128} value={cfg.batch} onChange={(v) => v !== null && patchCfg({ batch: v })} />
          </Field>
          <Field label="Patience" hint="Stop early after this many epochs without improvement (0 = never)">
            <NumberInput integer min={0} max={1000} value={cfg.patience} onChange={(v) => v !== null && patchCfg({ patience: v })} />
          </Field>
          <Field label="Name (optional)" wide>
            <input
              className="st-input"
              value={name}
              placeholder={`yolo26${effSize}-${task} · ${cfg.epochs}ep`}
              onChange={(e) => setName(e.target.value)}
            />
          </Field>
        </div>
      </Section>

      <Section
        title="Advanced"
        collapsible
        defaultOpen={false}
        right={advChanged > 0 ? <span className="st-badge warn">{advChanged} changed</span> : undefined}
      >
        <div className="st-train-fields">
          <Field label="Optimizer" hint="auto picks AdamW for short runs and MuSGD (YOLO26's Muon + SGD hybrid) for long ones">
            <select className="st-input" value={cfg.optimizer} onChange={(e) => patchCfg({ optimizer: e.target.value })}>
              {Array.from(new Set([...OPTIMIZERS, cfg.optimizer])).map((o) => (
                <option key={o} value={o}>
                  {o}
                </option>
              ))}
            </select>
          </Field>
          <Field label="lr0" hint="Initial learning rate — blank keeps the default">
            <NumberInput nullable min={0.000001} max={1} value={cfg.lr0} placeholder="default" onChange={(v) => patchCfg({ lr0: v })} />
          </Field>
          <Field label="Freeze layers" hint="Freeze the first N layers — faster fine-tuning on tiny datasets">
            <NumberInput nullable integer min={0} max={50} value={cfg.freeze} placeholder="none" onChange={(v) => patchCfg({ freeze: v })} />
          </Field>
          <Field label="LR schedule">
            <Toggle label="Cosine decay" checked={cfg.cos_lr} onChange={(v) => patchCfg({ cos_lr: v })} hint="Cosine instead of linear learning-rate decay" />
          </Field>
        </div>
        {cfg.optimizer === "auto" && isNum(cfg.lr0) && (
          <p className="st-train-note warn">Ultralytics ignores lr0 while the optimizer is “auto” — pick one explicitly to use it.</p>
        )}
        <Slider
          label="Val split"
          value={cfg.val_pct}
          min={0}
          max={50}
          step={5}
          format={(v) => `${v}%`}
          onChange={(v) => patchCfg({ val_pct: v })}
          hint="Share of auto-split images held out for validation. Images you set to train / val / test in Label keep their split."
        />

        <div className="st-train-subhead">
          <span className="st-field-label">Augmentation</span>
          <div className="st-actions">
            <button
              type="button"
              className="st-btn sm"
              onClick={() => patchAug(topDownPreset(task))}
              title={task === "detect" ? "flipud 0.5 (rotation would loosen axis-aligned boxes)" : "flipud 0.5 and ±90° rotation"}
            >
              Top-down preset
            </button>
            <button type="button" className="st-btn sm ghost" disabled={changedAug.length === 0} onClick={() => setAug({})}>
              Reset
            </button>
          </div>
        </div>
        <p className="st-train-note">
          Top-down pallets look the same flipped or rotated, so <b>flipud 0.5</b> and some <b>rotation</b> usually help.
        </p>
        <div className="st-train-augs">
          {augSpecs.map((a) => (
            <div key={a.key} className={`st-train-aug${a.key in aug ? " changed" : ""}`}>
              <Slider
                label={a.label}
                value={aug[a.key] ?? augDefault(a)}
                min={a.min}
                max={a.max}
                step={a.step}
                hint={a.hint}
                format={(v) => fmtAug(a, v)}
                onChange={(v) => patchAug({ [a.key]: v })}
              />
            </div>
          ))}
        </div>
      </Section>

      <div className="st-train-footer">
        <div className="st-train-summary">
          <b>{weights}</b>
          <span>
            · {cfg.epochs} ep · {cfg.imgsz} px · batch {cfg.batch} ·{" "}
            {kind === "model" && parent ? `fine-tune of ${parent.name}` : kind === "scratch" ? "from scratch" : "pretrained"}
          </span>
        </div>
        {noData && <p className="st-train-note warn">Label at least one image in the Label tab first.</p>}
        <button type="button" className="st-btn primary st-train-start" disabled={!pid || submitting || noData} onClick={start}>
          {submitting ? "Queuing…" : inFlight ? `Queue training · ${plural(inFlight, "job")} ahead` : "Start training"}
        </button>
        {pid && (
          <div className="st-train-dl">
            <a className="st-link" href={api.datasetZipUrl(pid, task)} download>
              ⤓ Download dataset (YOLO zip)
            </a>
            <span>{TASK_LABEL[task]} labels, current splits</span>
          </div>
        )}
      </div>
    </>
  );
}

// ---- Form controls ----------------------------------------------------------

interface RadioOption<T extends string> {
  value: T;
  title: ReactNode;
  blurb?: ReactNode;
  tag?: string;
  disabled?: boolean;
}

function RadioCards<T extends string>({
  value,
  options,
  onChange,
  compact = false,
}: {
  value: T;
  options: RadioOption<T>[];
  onChange: (v: T) => void;
  compact?: boolean;
}) {
  return (
    <div className={`st-train-cards${compact ? " compact" : ""}`} role="radiogroup">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          className={`st-train-card${o.value === value ? " on" : ""}`}
          disabled={o.disabled}
          onClick={() => onChange(o.value)}
        >
          <span className="st-train-radio" aria-hidden />
          <span className="st-train-card-text">
            <span className="st-train-card-title">
              {o.title}
              {o.tag && <span className="st-train-card-tag">{o.tag}</span>}
            </span>
            {o.blurb && <span className="st-train-card-blurb">{o.blurb}</span>}
          </span>
        </button>
      ))}
    </div>
  );
}

function Field({ label, hint, wide = false, children }: { label: string; hint?: string; wide?: boolean; children: ReactNode }) {
  return (
    <div className={`st-train-field${wide ? " wide" : ""}`} title={hint}>
      <span className="st-field-label">{label}</span>
      {children}
    </div>
  );
}

interface NumberRules {
  min?: number;
  max?: number;
  integer?: boolean;
  /** Empty input is a valid value (`null`, i.e. "use the default"). */
  nullable?: boolean;
}

type Parsed = { ok: true; value: number | null } | { ok: false };

function parseNumber(text: string, r: NumberRules): Parsed {
  const t = text.trim();
  if (t === "") return r.nullable ? { ok: true, value: null } : { ok: false };
  const v = Number(t);
  if (!Number.isFinite(v) || (r.integer && !Number.isInteger(v))) return { ok: false };
  if ((r.min !== undefined && v < r.min) || (r.max !== undefined && v > r.max)) return { ok: false };
  return { ok: true, value: v };
}

const numText = (v: number | null) => (v === null ? "" : String(v));

/** A number field you can clear and retype freely; only valid values are committed. */
function NumberInput({
  value,
  onChange,
  min,
  max,
  integer = false,
  nullable = false,
  placeholder,
  title,
}: NumberRules & { value: number | null; onChange: (v: number | null) => void; placeholder?: string; title?: string }) {
  const [text, setText] = useState(numText(value));
  // Follow outside changes without reformatting what's being typed ("0.0" stays "0.0").
  useEffect(() => {
    setText((t) => {
      const p = parseNumber(t, { min, max, integer, nullable });
      return p.ok && p.value === value ? t : numText(value);
    });
  }, [value, min, max, integer, nullable]);
  const rules: NumberRules = { min, max, integer, nullable };
  const valid = parseNumber(text, rules).ok;
  return (
    <input
      className="st-input"
      type="number"
      inputMode={integer ? "numeric" : "decimal"}
      min={min}
      max={max}
      step={integer ? 1 : "any"}
      placeholder={placeholder}
      title={title}
      value={text}
      aria-invalid={!valid}
      onChange={(e) => {
        setText(e.target.value);
        const p = parseNumber(e.target.value, rules);
        if (p.ok) onChange(p.value);
      }}
      onBlur={() => {
        if (!valid) setText(numText(value));
      }}
    />
  );
}

// ---- Dataset health ---------------------------------------------------------

const SPLIT_COLORS = { train: "#6366f1", val: "#14b8a6", test: "#94a3b8" } as const;

function DatasetHealth({ task }: { task: TrainTask }) {
  const stats = useStudio((s) => s.stats);
  const classes = useStudio((s) => s.classes);
  const setTab = useStudio((s) => s.setTab);

  const head = (
    <div className="st-card-head">
      <h4>Dataset health</h4>
      <div className="st-row">
        <span className="st-muted">as {TASK_LABEL[task]} labels</span>
        <button type="button" className="st-btn sm ghost" onClick={() => setTab("label")}>
          Open Label →
        </button>
      </div>
    </div>
  );
  if (!stats) {
    return (
      <section className="st-card">
        {head}
        <p className="st-train-note">Loading dataset stats…</p>
      </section>
    );
  }

  const perClass = classes.map((c) => ({ ...c, n: stats.instances_per_class[String(c.id)] ?? 0 }));
  const maxN = Math.max(1, ...perClass.map((c) => c.n));
  const splits = stats.splits;
  const splitTotal = splits.train + splits.val + splits.test;
  const withMasks = stats.n_instances - stats.n_box_only;

  const issues: ReactNode[] = [];
  if (classes.length === 0) issues.push(<>No classes yet — add one in the Label tab.</>);
  if (stats.n_labeled === 0) {
    issues.push(<>No labelled images yet — label a few cartons (or render synthetic pallets) in the Label tab.</>);
  } else {
    if (stats.n_labeled < 10) {
      issues.push(
        <>
          Only <b>{plural(stats.n_labeled, "labelled image")}</b> — expect noisy metrics. 20–50+ varied images is a better start.
        </>,
      );
    }
    if (splits.val === 0) {
      issues.push(
        <>
          No images in the <b>val</b> split — training holds one out if it can; otherwise metrics come from training images and
          read optimistic. Set a few images to <i>val</i> in Label.
        </>,
      );
    }
    const empty = perClass.filter((c) => c.n === 0);
    if (empty.length) {
      issues.push(
        <>
          No labels for {empty.map((c) => `“${c.name}”`).join(", ")} — the model can't learn {empty.length === 1 ? "it" : "them"}.
        </>,
      );
    }
  }
  if (task !== "detect" && stats.n_box_only > 0) {
    issues.push(
      <>
        <b>{plural(stats.n_box_only, "box-only label")}</b> become {task === "obb" ? "axis-aligned rectangles (no yaw)" : "rectangle masks"} —
        refine them with SAM in Label.
      </>,
    );
  }

  return (
    <section className="st-card">
      {head}
      <div className="st-grid-metrics st-train-tiles">
        <Tile label="Labelled images" value={stats.n_labeled} sub={`of ${stats.n_images} in the project`} />
        <Tile
          label="Instances"
          value={stats.n_instances}
          sub={
            <span title={task === "detect" ? "Box-only labels are fine for Detect" : "Box-only labels train as rectangles"}>
              {withMasks} masks · {stats.n_box_only} box-only
            </span>
          }
        />
        <Tile label="Negatives" value={stats.n_negative} sub="background-only images" />
        <Tile label="Unlabelled" value={stats.n_unlabeled} sub="left out of training" />
      </div>
      <div className="st-train-health">
        <div className="st-train-block" title="Images left on 'auto' are counted at the default 20% val share; a run uses its own Val split setting.">
          <span className="st-field-label">Splits</span>
          <div className="st-train-splitbar">
            {(["train", "val", "test"] as const).map((k) =>
              splits[k] > 0 ? (
                <div key={k} style={{ width: `${(splits[k] / splitTotal) * 100}%`, background: SPLIT_COLORS[k] }} title={`${k}: ${splits[k]}`} />
              ) : null,
            )}
          </div>
          <div className="st-train-legend">
            {(["train", "val", "test"] as const).map((k) => (
              <span key={k}>
                <span className="st-train-dot" style={{ background: SPLIT_COLORS[k] }} />
                {k} <b>{splits[k]}</b>
              </span>
            ))}
          </div>
        </div>
        <div className="st-train-block">
          <span className="st-field-label">Instances per class</span>
          {perClass.length ? (
            <div className="st-train-bars">
              {perClass.map((c) => (
                <div key={c.id} className={`st-train-bar-row${c.n === 0 ? " zero" : ""}`}>
                  <span className="st-swatch" style={{ background: c.color }} />
                  <span className="st-train-bar-name" title={c.name}>
                    {c.name}
                  </span>
                  <div className="st-train-bar-track">
                    {c.n > 0 && <div style={{ width: `${(c.n / maxN) * 100}%`, background: c.color }} />}
                  </div>
                  <span className="st-train-bar-n">{c.n}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="st-train-note">No classes yet.</p>
          )}
        </div>
      </div>
      {issues.length > 0 ? (
        <div className="st-warn-box">
          <ul className="st-train-issues">
            {issues.map((issue, i) => (
              <li key={i}>{issue}</li>
            ))}
          </ul>
        </div>
      ) : (
        <div className="st-train-ok">✓ Ready to train — enough labelled images, a val split{task === "detect" ? "" : " and real masks"}.</div>
      )}
    </section>
  );
}

function Tile({ label, value, sub, primary = false }: { label: string; value: ReactNode; sub?: ReactNode; primary?: boolean }) {
  return (
    <div className={`st-metric${primary ? " st-train-primary" : ""}`}>
      <div className="st-metric-label">{label}</div>
      <div className="st-metric-value">{value}</div>
      {sub !== undefined && <div className="st-metric-sub">{sub}</div>}
    </div>
  );
}

function Pct({ v }: { v: number | undefined | null }) {
  if (!isNum(v)) return <>—</>;
  return (
    <>
      {pct(v)}
      <span className="st-train-unit">%</span>
    </>
  );
}

// ---- Model list -------------------------------------------------------------

function useModelActions() {
  const pid = useStudio((s) => s.projectId);
  const run = useStudio((s) => s.run);
  const toast = useStudio((s) => s.toast);
  const refreshModels = useStudio((s) => s.refreshModels);
  const setTab = useStudio((s) => s.setTab);
  const setTestModel = useStudio((s) => s.setTestModel);

  const cancel = async (m: TrainedModel) => {
    if (!pid) return;
    if (m.status === "running" && !window.confirm(`Stop training “${m.name}”? A cancelled run keeps its curves but has no usable weights.`)) {
      return;
    }
    const out = await run("Cancelling", () => api.cancelModel(pid, m.id));
    if (!out) return;
    await refreshModels().catch(() => undefined);
    toast(`Cancelled “${m.name}”.`);
  };

  const remove = async (m: TrainedModel) => {
    if (!pid) return;
    const extra = isLive(m.status) ? " The run is stopped first." : "";
    if (!window.confirm(`Delete “${m.name}”? Its weights, exports, curves and plots are removed.${extra}`)) return;
    const out = await run("Deleting model", () => api.deleteModel(pid, m.id));
    if (!out) return;
    await refreshModels().catch(() => undefined);
    toast(`Deleted “${m.name}”.`);
  };

  const test = (m: TrainedModel) => {
    setTestModel({ kind: "trained", model_id: m.id });
    setTab("test");
  };

  return { cancel, remove, test };
}

function ModelList({
  models,
  selectedId,
  onSelect,
}: {
  models: TrainedModel[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const run = useStudio((s) => s.run);
  const refreshModels = useStudio((s) => s.refreshModels);
  const setTab = useStudio((s) => s.setTab);
  const actions = useModelActions();
  const live = models.filter((m) => isLive(m.status)).length;

  return (
    <section className="st-card">
      <div className="st-card-head">
        <h4>
          Models{models.length > 0 && <span className="st-train-count">{models.length}</span>}
        </h4>
        <div className="st-row">
          {live > 0 && (
            <span className="st-mini">
              <span className="st-spinner dark" /> {live} in progress
            </span>
          )}
          <button type="button" className="st-btn sm ghost" onClick={() => run("Refreshing models", refreshModels)}>
            Refresh
          </button>
        </div>
      </div>
      {models.length === 0 ? (
        <div className="st-train-empty">
          <p>
            <b>No models yet.</b> Label some cartons in the Label tab, pick a task on the left, then hit <b>Start training</b>.
          </p>
          <button type="button" className="st-btn sm" onClick={() => setTab("label")}>
            Go to Label →
          </button>
        </div>
      ) : (
        <div className="st-train-models">
          {models.map((m) => (
            <div
              key={m.id}
              className={`st-train-model${m.id === selectedId ? " on" : ""}`}
              role="button"
              tabIndex={0}
              aria-pressed={m.id === selectedId}
              onClick={() => onSelect(m.id)}
              onKeyDown={(e) => {
                if (e.target === e.currentTarget && (e.key === "Enter" || e.key === " ")) {
                  e.preventDefault();
                  onSelect(m.id);
                }
              }}
            >
              <div className="st-train-model-id">
                <div className="st-train-model-name" title={m.name}>
                  {m.name}
                </div>
                <div className="st-train-model-meta">
                  <TaskBadge task={m.task} />
                  <span>{weightsName(m.task, m.size)}</span>
                  <span>·</span>
                  <span className="st-ellipsis">{baseLabel(m, models, false)}</span>
                  <span>·</span>
                  <span title={fmtWhen(m.created_at)}>{timeAgo(m.created_at)}</span>
                </div>
              </div>
              <div className="st-train-model-state">
                <StatusSummary m={m} ahead={queueAhead(m, models)} />
              </div>
              <div className="st-train-model-actions" onClick={(e) => e.stopPropagation()}>
                {m.status === "completed" && (
                  <button type="button" className="st-btn sm" onClick={() => actions.test(m)}>
                    Test it →
                  </button>
                )}
                {isLive(m.status) && (
                  <button type="button" className="st-btn sm" onClick={() => actions.cancel(m)}>
                    Cancel
                  </button>
                )}
                <button type="button" className="st-btn sm ghost st-train-del" onClick={() => actions.remove(m)}>
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function TaskBadge({ task }: { task: TrainTask }) {
  return <span className={`st-train-task ${task}`}>{TASK_LABEL[task]}</span>;
}

function StatusPill({ status }: { status: Status }) {
  return <span className={`st-pill ${status}`}>{status}</span>;
}

function ProgressBar({ frac, busy }: { frac: number; busy: boolean }) {
  const clamped = Math.max(0, Math.min(1, frac));
  return (
    <div
      className={`st-progress${busy ? " st-train-busy" : ""}`}
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(clamped * 100)}
    >
      <div style={{ width: `${clamped * 100}%` }} />
    </div>
  );
}

function StatusSummary({ m, ahead }: { m: TrainedModel; ahead: number }) {
  if (isLive(m.status)) {
    const info = progressInfo(m, ahead);
    const eta = m.status === "running" && m.progress?.stage === "training" && m.progress.epoch > 0 ? m.progress.eta_s : undefined;
    return (
      <>
        <div className="st-train-state-top">
          <StatusPill status={m.status} />
          <span className="st-ellipsis">{info.short}</span>
        </div>
        <div className="st-train-progress-row">
          <ProgressBar frac={info.frac} busy={info.busy} />
          {isNum(eta) && <span className="st-muted st-train-eta">~{fmtDur(eta)} left</span>}
        </div>
      </>
    );
  }
  if (m.status === "completed") {
    return (
      <>
        <div className="st-train-state-top">
          <StatusPill status={m.status} />
          <span>{isNum(m.train_seconds) ? `trained in ${fmtDur(m.train_seconds)}` : ""}</span>
        </div>
        <Kpis m={m} />
      </>
    );
  }
  return (
    <>
      <div className="st-train-state-top">
        <StatusPill status={m.status} />
        <span title={fmtWhen(m.finished_at)}>{m.finished_at ? timeAgo(m.finished_at) : ""}</span>
      </div>
      {m.status === "failed" ? (
        <div className="st-train-err-line" title={m.error ?? undefined}>
          {firstLine(m.error) || "Training failed"}
        </div>
      ) : (
        <div className="st-muted">Stopped before it finished</div>
      )}
    </>
  );
}

function Kpis({ m }: { m: TrainedModel }) {
  const metrics = m.metrics ?? {};
  const heads = headsFor(m.task).filter((h) => isNum(metrics[mkey("mAP50", h.suffix)]) || isNum(metrics[mkey("mAP50-95", h.suffix)]));
  if (!heads.length) return <div className="st-muted">No metrics recorded</div>;
  return (
    <div className="st-train-kpis" title="mAP50 · mAP50-95 on the val split (%)">
      <span className="st-train-kpi-cap">mAP50 · 50-95</span>
      {heads.map((h) => (
        <span key={h.suffix}>
          {h.label} <b>{pct(metrics[mkey("mAP50", h.suffix)])}</b> · <b>{pct(metrics[mkey("mAP50-95", h.suffix)])}</b>
        </span>
      ))}
    </div>
  );
}

// ---- Selected model ---------------------------------------------------------

/**
 * Full manifest (curve + plots) for one model. Loads on mount, again when the
 * list reports a new status (e.g. the job finished), and every 3 s while live.
 */
function useModelDetail(pid: string | null, mid: string, listStatus: Status) {
  const [detail, setDetail] = useState<TrainedModel | null>(null);
  const [error, setError] = useState<string | null>(null);
  const seq = useRef(0);

  const reload = useCallback(async () => {
    if (!pid) return;
    const mine = ++seq.current;
    try {
      const d = await api.fetchModel(pid, mid);
      if (mine === seq.current) {
        setDetail(d);
        setError(null);
      }
    } catch (e) {
      if (mine === seq.current) setError(e instanceof Error ? e.message : String(e));
    }
  }, [pid, mid]);

  useEffect(() => {
    void reload();
  }, [reload, listStatus]);

  // Drop responses that land after unmount.
  useEffect(
    () => () => {
      seq.current++;
    },
    [],
  );

  const live = isLive(detail?.status ?? listStatus);
  useEffect(() => {
    if (!live) return;
    const t = window.setInterval(() => void reload(), POLL_MS);
    return () => window.clearInterval(t);
  }, [live, reload]);

  return { detail, error, reload };
}

function ModelDetail({ model, models }: { model: TrainedModel; models: TrainedModel[] }) {
  const pid = useStudio((s) => s.projectId);
  const actions = useModelActions();
  const { detail, error, reload } = useModelDetail(pid, model.id, model.status);
  // The list entry renders the header instantly; curves / plots need the detail.
  const m = detail ?? model;

  if (!pid) return null;
  const live = isLive(m.status);
  const completed = m.status === "completed";
  const finalMetrics = m.metrics ?? null;
  const liveMetrics = !finalMetrics && m.progress?.metrics && Object.keys(m.progress.metrics).length ? m.progress.metrics : null;

  return (
    <>
      <section className="st-card">
        <div className="st-card-head st-train-detail-head">
          <div className="st-train-detail-title">
            <h3 title={m.name}>{m.name}</h3>
            <StatusPill status={m.status} />
          </div>
          <div className="st-actions">
            {completed && (
              <a className="st-btn sm" href={api.weightsUrl(pid, m.id)} download title="Download best.pt">
                ⤓ Weights{isNum(m.weights_bytes) ? ` · ${fmtBytes(m.weights_bytes)}` : ""}
              </a>
            )}
            {completed && (
              <button type="button" className="st-btn sm primary" onClick={() => actions.test(m)}>
                Test it →
              </button>
            )}
            {live && (
              <button type="button" className="st-btn sm" onClick={() => actions.cancel(m)}>
                Cancel
              </button>
            )}
            <button type="button" className="st-btn sm danger" onClick={() => actions.remove(m)}>
              Delete
            </button>
          </div>
        </div>

        {live && <LiveProgress m={m} ahead={queueAhead(m, models)} />}
        <Facts m={m} models={models} />

        {m.dataset.warnings.length > 0 && (
          <div className="st-warn-box">
            {m.dataset.warnings.map((w) => (
              <div key={w}>{w}</div>
            ))}
          </div>
        )}
        {m.status === "failed" && m.error && <div className="st-err-box">{m.error}</div>}
        {m.status === "cancelled" && (
          <p className="st-train-note">
            Cancelled {timeAgo(m.finished_at)} — the curves so far are kept, but the run has no weights.
          </p>
        )}
        {error && !detail && (
          <div className="st-row">
            <span className="st-train-err">Couldn't load the full model details: {error}</span>
            <button type="button" className="st-btn sm" onClick={() => void reload()}>
              Retry
            </button>
          </div>
        )}
      </section>

      {(finalMetrics || liveMetrics) && (
        <section className="st-card">
          <div className="st-card-head">
            <h4>{finalMetrics ? "Validation metrics" : "Latest epoch"}</h4>
            <span className="st-muted">{finalMetrics ? "best weights · val split" : `after epoch ${m.progress?.epoch ?? 0}`}</span>
          </div>
          <MetricTiles metrics={finalMetrics ?? liveMetrics ?? {}} task={m.task} />
        </section>
      )}

      <CurvesCard m={m} loaded={detail !== null} />

      {m.per_class && m.per_class.length > 0 && (
        <section className="st-card">
          <div className="st-card-head">
            <h4>Per-class accuracy</h4>
            <span className="st-muted">best weights · val split · %</span>
          </div>
          <PerClassTable rows={m.per_class} model={m} summary={finalMetrics} />
        </section>
      )}

      {m.plots && m.plots.length > 0 && <PlotGallery pid={pid} m={m} />}

      {completed && (
        <>
          <ValidatePanel m={m} />
          <div className="st-train-tools">
            <ExportPanel m={m} onExported={() => void reload()} />
            <BenchmarkPanel m={m} />
          </div>
        </>
      )}
    </>
  );
}

function LiveProgress({ m, ahead }: { m: TrainedModel; ahead: number }) {
  const p = m.progress;
  const info = progressInfo(m, ahead);
  const metrics = p?.metrics ?? {};
  const chips: [string, string][] = [];
  for (const h of headsFor(m.task)) {
    const a = metrics[mkey("mAP50", h.suffix)];
    const b = metrics[mkey("mAP50-95", h.suffix)];
    if (isNum(a) || isNum(b)) chips.push([`${h.label} mAP50 · 50-95`, `${pct(a)} · ${pct(b)}`]);
  }
  for (const [k, v] of Object.entries(p?.loss ?? {})) {
    if (isNum(v) && Math.abs(v) > LOSS_EPS) chips.push([`${lossName(k)} loss`, fmtLoss(v)]);
  }
  const eta = m.status === "running" && p && p.stage === "training" && p.epoch > 0 ? p.eta_s : undefined;

  return (
    <div className="st-train-live">
      <div className="st-train-live-head">
        <span className="st-spinner dark" />
        <b className="st-ellipsis">{info.long}</b>
        <span className="st-muted st-train-live-time">
          {isNum(p?.elapsed_s) ? `${fmtDur(p?.elapsed_s)} elapsed` : ""}
          {isNum(eta) ? ` · ~${fmtDur(eta)} left` : ""}
        </span>
      </div>
      <ProgressBar frac={info.frac} busy={info.busy} />
      {chips.length > 0 && (
        <div className="st-train-chips">
          <span>Last epoch</span>
          {chips.map(([label, value]) => (
            <span key={label}>
              {label} <b>{value}</b>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function Facts({ m, models }: { m: TrainedModel; models: TrainedModel[] }) {
  const c = m.config;
  const aug = Object.entries(c.augment ?? {});
  const { train, val, test } = m.dataset.counts;
  const time = m.status === "completed" ? fmtDur(m.train_seconds) : isLive(m.status) && isNum(m.progress?.elapsed_s) ? `${fmtDur(m.progress?.elapsed_s)} so far` : "—";
  const optimiser = [
    c.optimizer,
    isNum(c.lr0) ? `lr0 ${c.lr0}` : null,
    c.cos_lr ? "cosine LR" : null,
    isNum(c.freeze) && c.freeze > 0 ? `freeze ${c.freeze}` : null,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <div className="st-train-facts">
      <Fact label="Model">
        {TASK_LABEL[m.task]} · {weightsName(m.task, m.size)}
      </Fact>
      <Fact label="Starting weights">{baseLabel(m, models, true)}</Fact>
      <Fact label="Schedule">
        {c.epochs} epochs · {c.imgsz} px · batch {c.batch} · patience {c.patience}
      </Fact>
      <Fact label="Optimizer">{optimiser}</Fact>
      <Fact label="Augmentation">{aug.length ? aug.map(([k, v]) => `${k} ${v}`).join(" · ") : "Ultralytics defaults"}</Fact>
      <Fact label="Dataset snapshot">
        {train} train · {val} val · {test} test images · {plural(m.dataset.instances, "instance")}
      </Fact>
      <Fact label="Classes">{m.dataset.names.join(", ") || "—"}</Fact>
      <Fact label="Created">{fmtWhen(m.created_at)}</Fact>
      <Fact label="Train time">{time}</Fact>
      {m.status === "completed" && isNum(m.speed?.inference) && (
        <Fact label="Val inference">{fmtMs(m.speed?.inference)} ms / image</Fact>
      )}
    </div>
  );
}

function Fact({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="st-train-fact">
      <span className="st-field-label">{label}</span>
      <span>{children}</span>
    </div>
  );
}

const TILE_METRICS = [
  { key: "precision", label: "Precision" },
  { key: "recall", label: "Recall" },
  { key: "mAP50", label: "mAP50" },
  { key: "mAP50-95", label: "mAP50-95" },
];

function MetricTiles({ metrics, task }: { metrics: Metrics; task: TrainTask }) {
  const heads = headsFor(task).filter((h) => TILE_METRICS.some((t) => isNum(metrics[mkey(t.key, h.suffix)])));
  if (!heads.length) return <p className="st-train-note">No box / mask metrics were reported.</p>;
  return (
    <div className="st-train-metric-groups">
      {heads.map((h) => (
        <div key={h.suffix} className="st-train-metric-group">
          <span className="st-field-label">{h.label}</span>
          <div className="st-grid-metrics">
            {TILE_METRICS.map((t) => (
              <Tile key={t.key} label={t.label} value={<Pct v={metrics[mkey(t.key, h.suffix)]} />} primary={t.key === "mAP50-95"} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

// ---- Curves -----------------------------------------------------------------

interface SeriesDef {
  /** Column in results.csv. */
  src: string;
  /** Safe key in the chart rows ("metrics/mAP50(B)" isn't a friendly dataKey). */
  key: string;
  name: string;
  color: string;
  dashed?: boolean;
}

function buildCurves(curve: Record<string, number>[], task: TrainTask) {
  const columns = Array.from(new Set(curve.flatMap((r) => Object.keys(r))));
  const loss: SeriesDef[] = columns
    .filter((k) => k.startsWith("train/") && curve.some((r) => Math.abs(num(r[k]) ?? 0) > LOSS_EPS))
    .map((src, i) => ({ src, key: `l${i}`, name: lossName(src), color: LOSS_COLORS[i % LOSS_COLORS.length] }));
  const box = task === "obb" ? "OBB" : "box";
  const map: SeriesDef[] = [
    { src: mkey("mAP50", "B"), name: `mAP50 ${box}`, color: "#16a34a" },
    { src: mkey("mAP50-95", "B"), name: `mAP50-95 ${box}`, color: "#2563eb" },
    { src: mkey("mAP50", "M"), name: "mAP50 mask", color: "#16a34a", dashed: true },
    { src: mkey("mAP50-95", "M"), name: "mAP50-95 mask", color: "#2563eb", dashed: true },
  ]
    .filter((s) => curve.some((r) => num(r[s.src]) !== null))
    .map((s, i) => ({ ...s, key: `m${i}` }));
  const rows = curve.map((r, i) => {
    const row: Record<string, number> = { epoch: num(r.epoch) ?? i + 1 };
    for (const s of [...loss, ...map]) {
      const v = num(r[s.src]);
      if (v !== null) row[s.key] = v;
    }
    return row;
  });
  return { rows, loss, map };
}

function CurvesCard({ m, loaded }: { m: TrainedModel; loaded: boolean }) {
  const data = useMemo(() => buildCurves(m.curve ?? [], m.task), [m.curve, m.task]);
  return (
    <section className="st-card">
      <div className="st-card-head">
        <h4>Training curves</h4>
        {data.rows.length > 0 && <span className="st-muted">{plural(data.rows.length, "epoch")} logged</span>}
      </div>
      {data.rows.length === 0 ? (
        <p className="st-train-note">
          {!loaded ? "Loading…" : isLive(m.status) ? "Curves appear after the first epoch." : "No training curve was recorded for this run."}
        </p>
      ) : (
        <div className="st-train-charts">
          <CurveChart title="Train loss" rows={data.rows} series={data.loss} />
          <CurveChart title="Validation mAP" rows={data.rows} series={data.map} percent />
        </div>
      )}
    </section>
  );
}

function CurveChart({
  title,
  rows,
  series,
  percent = false,
}: {
  title: string;
  rows: Record<string, number>[];
  series: SeriesDef[];
  percent?: boolean;
}) {
  return (
    <div className="st-train-chart">
      <div className="st-train-chart-title">{title}</div>
      {series.length === 0 ? (
        <p className="st-train-note">Nothing logged yet.</p>
      ) : (
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={rows} margin={{ top: 6, right: 12, bottom: 0, left: -6 }}>
            <CartesianGrid stroke="#eef0f3" strokeDasharray="3 3" />
            <XAxis
              dataKey="epoch"
              type="number"
              domain={["dataMin", "dataMax"]}
              allowDecimals={false}
              tick={{ fontSize: 10, fill: "#6b7280" }}
              stroke="#9ca3af"
            />
            <YAxis
              domain={percent ? [0, 1] : [0, "auto"]}
              tickFormatter={percent ? (v: number) => `${Math.round(v * 100)}` : undefined}
              tick={{ fontSize: 10, fill: "#6b7280" }}
              stroke="#9ca3af"
              width={38}
            />
            <Tooltip
              formatter={(v: unknown) => (isNum(v) ? (percent ? `${pct(v)}%` : fmtLoss(v)) : String(v ?? ""))}
              labelFormatter={(label: unknown) => `Epoch ${String(label)}`}
              contentStyle={{ fontSize: 11, padding: "4px 8px", borderRadius: 6 }}
              labelStyle={{ fontSize: 11, fontWeight: 600 }}
            />
            <Legend iconType="plainline" iconSize={16} wrapperStyle={{ fontSize: 11 }} />
            {series.map((s) => (
              <Line
                key={s.key}
                type="monotone"
                dataKey={s.key}
                name={s.name}
                stroke={s.color}
                strokeWidth={1.75}
                strokeDasharray={s.dashed ? "5 3" : undefined}
                dot={rows.length < 3}
                isAnimationActive={false}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

// ---- Per-class table --------------------------------------------------------

const PC_COLS: {
  short: string;
  long: string;
  /** Summary metric name, for the "All classes" row. */
  sum: string;
  quality: boolean;
  pick: (r: PerClassMetric, head: Head) => number | undefined;
}[] = [
  { short: "mAP50", long: "mAP50", sum: "mAP50", quality: true, pick: (r, h) => (h === "box" ? r.box_map50 : r.mask_map50) },
  { short: "mAP50-95", long: "mAP50-95", sum: "mAP50-95", quality: true, pick: (r, h) => (h === "box" ? r.box_map50_95 : r.mask_map50_95) },
  { short: "P", long: "Precision", sum: "precision", quality: false, pick: (r, h) => (h === "box" ? r.box_precision : r.mask_precision) },
  { short: "R", long: "Recall", sum: "recall", quality: false, pick: (r, h) => (h === "box" ? r.box_recall : r.mask_recall) },
];

function PerClassTable({ rows, model, summary }: { rows: PerClassMetric[]; model: TrainedModel; summary?: Metrics | null }) {
  const classes = useStudio((s) => s.classes);
  const hasMask = rows.some((r) => isNum(r.mask_map50) || isNum(r.mask_map50_95));
  const heads: Head[] = hasMask ? ["box", "mask"] : ["box"];
  const suffix = (h: Head) => (h === "box" ? "B" : "M");
  const missing = model.dataset.names.length - rows.length;

  return (
    <>
      <div className="st-train-table-wrap">
        <table className="st-table st-train-table">
          <thead>
            {hasMask && (
              <tr>
                <th />
                <th colSpan={PC_COLS.length} className="st-train-group">
                  {model.task === "obb" ? "OBB" : "Box"}
                </th>
                <th colSpan={PC_COLS.length} className="st-train-group">
                  Mask
                </th>
              </tr>
            )}
            <tr>
              <th>Class</th>
              {heads.flatMap((h) =>
                PC_COLS.map((c) => (
                  <th key={`${h}-${c.sum}`} className="num" title={c.long}>
                    {hasMask ? c.short : c.long}
                  </th>
                )),
              )}
            </tr>
          </thead>
          <tbody>
            {summary && (
              <tr className="st-train-all">
                <td>All classes</td>
                {heads.flatMap((h) =>
                  PC_COLS.map((c) => (
                    <td key={`${h}-${c.sum}`} className="num">
                      {pct(summary[mkey(c.sum, suffix(h))])}
                    </td>
                  )),
                )}
              </tr>
            )}
            {rows.map((r) => (
              <tr key={r.class_id}>
                <td>
                  <span className="st-train-class">
                    {/* per-class ids are the model's contiguous indices → map back to store ids for colours */}
                    <span className="st-swatch" style={{ background: classColor(classes, model.dataset.class_ids[r.class_id]) }} />
                    {r.class_name}
                  </span>
                </td>
                {heads.flatMap((h) =>
                  PC_COLS.map((c) => {
                    const v = c.pick(r, h);
                    return (
                      <td key={`${h}-${c.sum}`} className={`num ${c.quality ? qClass(v) : ""}`}>
                        {pct(v)}
                      </td>
                    );
                  }),
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {missing > 0 && (
        <p className="st-train-note">
          {plural(missing, "class", "classes")} with no instances in this split {missing === 1 ? "isn't" : "aren't"} listed.
        </p>
      )}
    </>
  );
}

// ---- Plots ------------------------------------------------------------------

function PlotGallery({ pid, m }: { pid: string; m: TrainedModel }) {
  const [showAll, setShowAll] = useState(false);
  const plots = useMemo(() => orderPlots(m.plots ?? []), [m.plots]);
  if (!plots.length) return null;
  const featured = plots.filter((p) => PREFERRED_PLOTS.includes(p));
  const head = featured.length ? featured : plots.slice(0, 4);
  const shown = showAll ? plots : head;
  const hidden = plots.length - head.length;

  return (
    <section className="st-card">
      <div className="st-card-head">
        <h4>Plots</h4>
        {hidden > 0 && (
          <button type="button" className="st-btn sm ghost" onClick={() => setShowAll(!showAll)}>
            {showAll ? "Show fewer" : `Show all ${plots.length}`}
          </button>
        )}
      </div>
      <div className="st-train-plots">
        {shown.map((name) => {
          const url = api.plotUrl(pid, m.id, name);
          const label = plotLabel(name);
          return (
            <a key={name} className="st-train-plot" href={url} target="_blank" rel="noreferrer" title={`Open ${name} in a new tab`}>
              <img src={url} alt={label} loading="lazy" />
              <span>{label}</span>
            </a>
          );
        })}
      </div>
    </section>
  );
}

// ---- Validate ---------------------------------------------------------------

function ValidatePanel({ m }: { m: TrainedModel }) {
  const pid = useStudio((s) => s.projectId);
  const stats = useStudio((s) => s.stats);
  const run = useStudio((s) => s.run);
  const [split, setSplit] = useState<ValSplit>("val");
  const [conf, setConf] = useState<number | null>(null);
  const [iou, setIou] = useState(0.7);
  const [pending, setPending] = useState(false);
  const [result, setResult] = useState<{ split: ValSplit; conf: number | null; iou: number; res: ValResult } | null>(null);
  const noTest = stats !== null && stats.splits.test === 0;

  const validate = async () => {
    if (!pid) return;
    setPending(true);
    const res = await run(`Validating on ${split}`, () => api.validateModel(pid, m.id, { split, conf, iou }));
    setPending(false);
    if (res) setResult({ split, conf, iou, res });
  };

  return (
    <section className="st-card">
      <div className="st-card-head">
        <h4>Validate</h4>
        <span className="st-muted">re-scores the weights against the dataset's current labels</span>
      </div>
      <div className="st-train-controls">
        <div className="st-train-control">
          <span className="st-field-label">Split</span>
          <Segmented
            size="sm"
            value={split}
            onChange={setSplit}
            options={[
              { value: "val", label: "val" },
              { value: "train", label: "train", title: "Training images — expect optimistic numbers" },
              { value: "test", label: "test", disabled: noTest, title: noTest ? "No images in the test split" : undefined },
            ]}
          />
        </div>
        <div className="st-train-control" title="Confidence threshold. Blank keeps Ultralytics' low validation default, which is what mAP expects.">
          <span className="st-field-label">Conf</span>
          <NumberInput nullable min={0} max={1} value={conf} placeholder="auto" onChange={setConf} />
        </div>
        <Slider label="IoU" value={iou} min={0.3} max={0.95} step={0.05} onChange={setIou} format={(v) => v.toFixed(2)} hint="NMS IoU threshold" />
        <button type="button" className="st-btn primary" disabled={pending} onClick={validate}>
          {pending ? "Validating…" : "Run validation"}
        </button>
      </div>

      {result && (
        <div className="st-train-result">
          <div className="st-train-result-head">
            Results on <b>{result.split}</b>
            {result.conf !== null ? ` · conf ${result.conf}` : ""} · IoU {result.iou.toFixed(2)}
            {isNum(result.res.speed.inference) ? ` · ${fmtMs(result.res.speed.inference)} ms / image inference` : ""}
          </div>
          <MetricTiles metrics={result.res.summary} task={m.task} />
          {result.res.per_class.length > 0 && <PerClassTable rows={result.res.per_class} model={m} summary={result.res.summary} />}
          {result.res.confusion && <ConfusionTable labels={result.res.confusion.labels} matrix={result.res.confusion.matrix} />}
        </div>
      )}
    </section>
  );
}

/** Ultralytics layout: rows = predicted class, columns = true class, last = background. */
function ConfusionTable({ labels, matrix }: { labels: string[]; matrix: number[][] }) {
  const colSum = labels.map((_, j) => matrix.reduce((s, row) => s + (row[j] ?? 0), 0));
  const total = colSum.reduce((a, b) => a + b, 0);
  const isBg = (i: number) => i === labels.length - 1 && labels[i] === "background";

  if (!total) {
    return <p className="st-train-note">The confusion matrix came back empty — no matches were counted for this run.</p>;
  }
  return (
    <div className="st-train-cm-wrap">
      <span className="st-field-label">Confusion matrix</span>
      <table className="st-train-cm">
        <thead>
          <tr>
            <th className="st-train-cm-corner">predicted ↓ · true →</th>
            {labels.map((l, j) => (
              <th key={j} className={isBg(j) ? "st-train-cm-bg" : undefined} title={l}>
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th className={isBg(i) ? "st-train-cm-bg" : undefined} title={labels[i]}>
                {labels[i]}
              </th>
              {row.map((v, j) => {
                const share = colSum[j] ? v / colSum[j] : 0;
                return (
                  <td
                    key={j}
                    className={i === j && !isBg(i) ? "diag" : undefined}
                    style={v ? { background: `rgba(79, 70, 229, ${(0.1 + 0.8 * share).toFixed(3)})`, color: share > 0.5 ? "#fff" : undefined } : undefined}
                    title={`${v} × true “${labels[j]}” predicted as “${labels[i]}” (${Math.round(share * 100)}% of true ${labels[j]})`}
                  >
                    {v || ""}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="st-train-note">Shading is each cell's share of its true class; outlined cells are correct predictions.</p>
    </div>
  );
}

// ---- Export -----------------------------------------------------------------

function ExportPanel({ m, onExported }: { m: TrainedModel; onExported: () => void }) {
  const pid = useStudio((s) => s.projectId);
  const catalog = useStudio((s) => s.catalog);
  const run = useStudio((s) => s.run);
  const toast = useStudio((s) => s.toast);
  const formats = catalog?.export_formats ?? [];
  const [format, setFormat] = useState("");
  const [imgsz, setImgsz] = useState(m.config.imgsz);
  const [half, setHalf] = useState(false);
  const [int8, setInt8] = useState(false);
  const [dynamic, setDynamic] = useState(false);
  const [nms, setNms] = useState(false);
  const [pending, setPending] = useState(false);

  const chosen = formats.find((f) => f.id === format && f.available) ?? formats.find((f) => f.available) ?? null;
  const exports = [...m.exports].sort((a, b) => b.created_at.localeCompare(a.created_at));
  const formatLabel = (id: string) => formats.find((f) => f.id === id)?.label ?? id;

  const doExport = async () => {
    if (!pid || !chosen) return;
    setPending(true);
    const entry = await run(`Exporting ${chosen.label}`, () =>
      api.exportModel(pid, m.id, { format: chosen.id, imgsz, half, int8, dynamic, nms }),
    );
    setPending(false);
    if (!entry) return;
    toast(`Exported ${entry.file} · ${fmtBytes(entry.size_bytes)} in ${entry.seconds.toFixed(1)}s`, "success");
    onExported();
  };

  return (
    <section className="st-card">
      <div className="st-card-head">
        <h4>Export</h4>
        <span className="st-muted">converted on CPU</span>
      </div>
      <div className="st-train-fields">
        <Field label="Format">
          <select className="st-input" value={chosen?.id ?? ""} disabled={!formats.length} onChange={(e) => setFormat(e.target.value)}>
            {!formats.length && <option value="">Loading formats…</option>}
            {formats.map((f) => (
              <option key={f.id} value={f.id} disabled={!f.available}>
                {f.label}
                {f.available ? "" : ` — ${f.note}`}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Image size">
          <select className="st-input" value={imgsz} onChange={(e) => setImgsz(Number(e.target.value))}>
            {sizeOptions(m.config.imgsz).map((s) => (
              <option key={s} value={s}>
                {s} px{s === m.config.imgsz ? " (trained)" : ""}
              </option>
            ))}
          </select>
        </Field>
      </div>
      {chosen && <p className="st-train-note">{chosen.note}</p>}
      <div className="st-row wrap st-train-toggles">
        <Toggle label="FP16" checked={half} onChange={setHalf} hint="Half-precision weights — smaller, faster on GPU / Neural Engine" />
        <Toggle label="INT8" checked={int8} onChange={setInt8} hint="Quantise, calibrating on this model's dataset" />
        <Toggle label="Dynamic shapes" checked={dynamic} onChange={setDynamic} hint="Accept any input size (ONNX / OpenVINO / TorchScript)" />
        <Toggle label="Embed NMS" checked={nms} onChange={setNms} hint="Bake NMS into the graph — usually unnecessary for YOLO26's end-to-end head" />
      </div>
      <div>
        <button type="button" className="st-btn primary" disabled={!chosen || pending} onClick={doExport}>
          {pending ? "Exporting…" : `Export${chosen ? ` ${chosen.label}` : ""}`}
        </button>
      </div>
      {exports.length > 0 ? (
        <div className="st-train-table-wrap">
          <table className="st-table">
            <thead>
              <tr>
                <th>Format</th>
                <th>File</th>
                <th className="num">Size</th>
                <th className="num">Time</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {exports.map((e) => (
                <tr key={e.file}>
                  <td>
                    {formatLabel(e.format)}
                    <div className="st-train-opts">
                      {[`${e.imgsz} px`, e.half && "fp16", e.int8 && "int8", e.dynamic && "dynamic", e.nms && "nms"].filter(Boolean).join(" · ")}
                    </div>
                  </td>
                  <td>
                    <span className="st-train-mono st-train-file" title={e.file}>
                      {e.file}
                    </span>
                  </td>
                  <td className="num">{fmtBytes(e.size_bytes)}</td>
                  <td className="num">{e.seconds.toFixed(1)}s</td>
                  <td className="num">
                    {pid && (
                      <a className="st-link" href={api.exportDownloadUrl(pid, m.id, e.file)} download>
                        ⤓ Download
                      </a>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="st-train-note">No exports yet — they're kept with the model and can be benchmarked.</p>
      )}
    </section>
  );
}

// ---- Benchmark --------------------------------------------------------------

function BenchmarkPanel({ m }: { m: TrainedModel }) {
  const pid = useStudio((s) => s.projectId);
  const device = useStudio((s) => s.catalog?.device);
  const run = useStudio((s) => s.run);
  const [rows, setRows] = useState<BenchRow[] | null>(null);
  const [pending, setPending] = useState(false);

  const bench = async () => {
    if (!pid) return;
    setPending(true);
    const out = await run("Benchmarking", () => api.benchmarkModel(pid, m.id, BENCH_IMAGES));
    setPending(false);
    if (out) setRows(out.rows);
  };
  const best = rows ? Math.max(0, ...rows.map((r) => r.fps ?? 0)) : 0;

  return (
    <section className="st-card">
      <div className="st-card-head">
        <h4>Benchmark</h4>
        <button type="button" className="st-btn sm" disabled={pending} onClick={bench}>
          {pending ? "Running…" : rows ? "Run again" : "Run benchmark"}
        </button>
      </div>
      <p className="st-train-note">
        Latency on {BENCH_IMAGES} dataset images at {m.config.imgsz} px — PyTorch on {device ?? "the default device"}
        {device && device !== "cpu" ? " and CPU" : ""}, plus each ONNX / TorchScript / CoreML / OpenVINO export.
      </p>
      {rows &&
        (rows.length ? (
          <div className="st-train-table-wrap">
            <table className="st-table">
              <thead>
                <tr>
                  <th>Runtime</th>
                  <th>Device</th>
                  <th className="num">p50 ms</th>
                  <th className="num">p95 ms</th>
                  <th className="num">FPS</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={`${r.runtime}-${r.device}-${r.artifact}-${i}`}>
                    <td>
                      {r.runtime}
                      {!r.error && best > 0 && r.fps === best && <span className="st-badge ok st-train-fastest">fastest</span>}
                      <div className="st-train-opts st-train-mono" title={r.artifact}>
                        {r.artifact}
                      </div>
                    </td>
                    <td>{r.device}</td>
                    {r.error ? (
                      <td colSpan={3} className="st-train-err">
                        {r.error}
                      </td>
                    ) : (
                      <>
                        <td className="num">{fmtMs(r.p50_ms)}</td>
                        <td className="num">{fmtMs(r.p95_ms)}</td>
                        <td className="num">
                          <b>{fmtFps(r.fps)}</b>
                        </td>
                      </>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="st-train-note">Nothing to benchmark.</p>
        ))}
    </section>
  );
}
