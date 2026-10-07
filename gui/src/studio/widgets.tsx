/** Small shared controls for the Studio views. */

import { useEffect, useState, type ReactNode } from "react";
import type { Catalog, ModelSpec, TrainedModel, YoloSize, YoloTask, YoloeFamily, YoloeSize } from "./types";

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  size = "md",
}: {
  value: T;
  options: { value: T; label: ReactNode; title?: string; disabled?: boolean }[];
  onChange: (v: T) => void;
  size?: "sm" | "md";
}) {
  return (
    <div className={`st-seg st-seg-${size}`} role="radiogroup">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          className={o.value === value ? "on" : ""}
          title={o.title}
          disabled={o.disabled}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
  hint,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  hint?: string;
}) {
  return (
    <label className="st-slider" title={hint}>
      <span className="st-slider-label">{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      <span className="st-slider-value">{format ? format(value) : value}</span>
    </label>
  );
}

export function Toggle({ label, checked, onChange, hint }: { label: ReactNode; checked: boolean; onChange: (v: boolean) => void; hint?: string }) {
  return (
    <label className="st-toggle" title={hint}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span>{label}</span>
    </label>
  );
}

export function Section({
  title,
  children,
  right,
  defaultOpen = true,
  collapsible = false,
}: {
  title: ReactNode;
  children: ReactNode;
  right?: ReactNode;
  defaultOpen?: boolean;
  collapsible?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className={`st-section ${open ? "" : "closed"}`}>
      <header className="st-section-head" onClick={collapsible ? () => setOpen(!open) : undefined} style={collapsible ? { cursor: "pointer" } : undefined}>
        <h3>
          {collapsible && <span className="st-caret">{open ? "▾" : "▸"}</span>}
          {title}
        </h3>
        {right && <div onClick={(e) => e.stopPropagation()}>{right}</div>}
      </header>
      {open && <div className="st-section-body">{children}</div>}
    </section>
  );
}

const TASK_LABEL: Record<YoloTask, string> = {
  detect: "Detect",
  segment: "Segment",
  classify: "Classify",
  pose: "Pose",
  obb: "OBB",
};

type Kind = ModelSpec["kind"];

export const DEFAULT_SPECS: Record<Kind, ModelSpec> = {
  yolo26: { kind: "yolo26", task: "segment", size: "n" },
  trained: { kind: "trained", model_id: "" },
  "yoloe-text": { kind: "yoloe-text", family: "26", size: "s", classes: ["carton"] },
  "yoloe-pf": { kind: "yoloe-pf", family: "26", size: "s" },
};

export function specLabel(spec: ModelSpec, models: TrainedModel[]): string {
  switch (spec.kind) {
    case "yolo26":
      return `yolo26${spec.size}-${spec.task}`;
    case "trained":
      return models.find((m) => m.id === spec.model_id)?.name ?? spec.model_id;
    case "yoloe-text":
      return `YOLOE-${spec.family}${spec.size} · "${spec.classes.join(", ")}"`;
    case "yoloe-pf":
      return `YOLOE-${spec.family}${spec.size} prompt-free`;
  }
}

/**
 * Pick any model Studio can run. `kinds` limits the sources offered and
 * `tasks` limits YOLO26 pretrained tasks (e.g. tracking can't use classify).
 */
export function ModelPicker({
  value,
  onChange,
  models,
  catalog,
  kinds = ["trained", "yolo26", "yoloe-text", "yoloe-pf"],
  tasks = ["detect", "segment", "classify", "pose", "obb"],
}: {
  value: ModelSpec;
  onChange: (m: ModelSpec) => void;
  models: TrainedModel[];
  catalog: Catalog | null;
  kinds?: Kind[];
  tasks?: YoloTask[];
}) {
  const completed = models.filter((m) => m.status === "completed");
  const [classesText, setClassesText] = useState(value.kind === "yoloe-text" ? value.classes.join(", ") : "carton");
  useEffect(() => {
    if (value.kind === "yoloe-text") setClassesText(value.classes.join(", "));
  }, [value]);

  const setKind = (k: Kind) => {
    if (k === "trained") onChange({ kind: "trained", model_id: completed[0]?.id ?? "" });
    else onChange(DEFAULT_SPECS[k]);
  };
  const cachedYolo = (task: YoloTask, size: YoloSize) => catalog?.yolo26[task]?.find((x) => x.size === size)?.cached;
  const cachedYoloe = (f: YoloeFamily, s: YoloeSize, pf: boolean) => {
    const e = catalog?.yoloe.find((x) => x.family === f && x.size === s);
    return pf ? e?.pf_cached : e?.cached;
  };

  return (
    <div className="st-model-picker">
      <select value={value.kind} onChange={(e) => setKind(e.target.value as Kind)}>
        {kinds.includes("trained") && <option value="trained">Your trained models ({completed.length})</option>}
        {kinds.includes("yolo26") && <option value="yolo26">YOLO26 pretrained (COCO / DOTA / ImageNet)</option>}
        {kinds.includes("yoloe-text") && <option value="yoloe-text">YOLOE open-vocab · text prompt</option>}
        {kinds.includes("yoloe-pf") && <option value="yoloe-pf">YOLOE open-vocab · prompt-free</option>}
      </select>

      {value.kind === "trained" &&
        (completed.length ? (
          <select value={value.model_id} onChange={(e) => onChange({ kind: "trained", model_id: e.target.value })}>
            {completed.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} — {m.task}
                {m.metrics?.["metrics/mAP50(B)"] !== undefined ? ` · mAP50 ${(m.metrics["metrics/mAP50(B)"] * 100).toFixed(0)}` : ""}
              </option>
            ))}
          </select>
        ) : (
          <div className="st-muted">No completed models yet — train one in the Train tab.</div>
        ))}

      {value.kind === "yolo26" && (
        <>
          <Segmented
            size="sm"
            value={value.task}
            onChange={(t) => onChange({ ...value, task: t })}
            options={tasks.map((t) => ({ value: t, label: TASK_LABEL[t] }))}
          />
          <Segmented
            size="sm"
            value={value.size}
            onChange={(s) => onChange({ ...value, size: s })}
            options={(["n", "s", "m", "l", "x"] as YoloSize[]).map((s) => ({
              value: s,
              label: (
                <>
                  {s}
                  {!cachedYolo(value.task, s) && <span className="st-dl" title="downloads on first use">↓</span>}
                </>
              ),
            }))}
          />
        </>
      )}

      {(value.kind === "yoloe-text" || value.kind === "yoloe-pf") && (
        <div className="st-row">
          <Segmented
            size="sm"
            value={value.family}
            onChange={(f) => onChange({ ...value, family: f })}
            options={[
              { value: "26", label: "YOLOE-26" },
              { value: "11", label: "YOLOE-11" },
            ]}
          />
          <Segmented
            size="sm"
            value={value.size}
            onChange={(s) => onChange({ ...value, size: s })}
            options={(["s", "m", "l"] as YoloeSize[]).map((s) => ({
              value: s,
              label: (
                <>
                  {s}
                  {!cachedYoloe(value.family, s, value.kind === "yoloe-pf") && <span className="st-dl" title="downloads on first use">↓</span>}
                </>
              ),
            }))}
          />
        </div>
      )}
      {value.kind === "yoloe-text" && (
        <input
          className="st-input"
          placeholder="class names, comma-separated"
          value={classesText}
          onChange={(e) => setClassesText(e.target.value)}
          onBlur={() =>
            onChange({ ...value, classes: classesText.split(",").map((s) => s.trim()).filter(Boolean) })
          }
          onKeyDown={(e) => {
            if (e.key === "Enter") (e.target as HTMLInputElement).blur();
          }}
        />
      )}
    </div>
  );
}

export const COCO_SKELETON: [number, number][] = [
  [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
  [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
];

export function pct(v: number | undefined | null, digits = 1): string {
  return v === undefined || v === null || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}`;
}

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

export function fmtSecs(s: number | undefined | null): string {
  if (s === undefined || s === null) return "—";
  if (s < 60) return `${Math.round(s)}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${Math.round(s % 60)}s`;
}

/** Same palette as the server's class colours (pipeline/studio/store.py). */
export const PALETTE = [
  "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
  "#f032e6", "#bfef45", "#469990", "#dcbeff", "#9a6324", "#800000",
  "#808000", "#000075", "#fabed4", "#aaffc3",
];

export const colorForIndex = (i: number): string => PALETTE[((i % PALETTE.length) + PALETTE.length) % PALETTE.length];

const NON_TEXT_INPUTS = new Set(["checkbox", "radio", "button", "submit", "reset", "file", "color", "range"]);

/**
 * True when a key event belongs to a text field (so global hotkeys should
 * stay out of the way). Checkboxes and buttons keep focus after a click
 * and must not swallow shortcuts; sliders only claim their arrow keys.
 */
export function isTypingTarget(e: KeyboardEvent): boolean {
  const t = e.target as HTMLElement | null;
  if (!t) return false;
  if (t.isContentEditable || t.tagName === "TEXTAREA" || t.tagName === "SELECT") return true;
  if (t.tagName !== "INPUT") return false;
  const type = (t as HTMLInputElement).type;
  if (type === "range") return /^(Arrow|Home|End|Page)/.test(e.key);
  return !NON_TEXT_INPUTS.has(type);
}
