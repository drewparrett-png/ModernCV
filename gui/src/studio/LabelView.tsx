/**
 * Label — prompt, segment and review.
 *
 * The canvas collects a *draft* of prompts (boxes, brush strokes, erase
 * strokes) for the current image. The draft can be:
 *
 *   Segment (S / live)   SAM 2.1 → one precise mask; Enter accepts it.
 *   Find similar (F)     YOLOE visual prompt → suggestions for every
 *                        similar object, here or across the dataset.
 *   Add as label (A)     boxes → box labels, strokes → painted masks.
 *
 * Suggestions from any source (visual / text / prompt-free / a model) are
 * reviewed in the right-hand panel before they become labels.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as api from "./api";
import { ImageList } from "./ImageList";
import { PromptCanvas, type CanvasTool, type Shape } from "./PromptCanvas";
import type { Annotation, ModelSpec, Prompt, SamResult, Scope, StudioImage, YoloeFamily, YoloeSize } from "./types";
import { className, classColor, useStudio } from "./useStudio";
import { isTypingTarget, ModelPicker, Section, Segmented, Slider, Toggle } from "./widgets";

const TOOL_KEYS: Record<string, CanvasTool> = { v: "select", b: "box", p: "brush", e: "erase", h: "pan" };
const GRAN_ORDER: api.Granularity[] = ["auto", "fine", "medium", "coarse"];


export function LabelView() {
  const pid = useStudio((s) => s.projectId);
  const images = useStudio((s) => s.images);
  const classes = useStudio((s) => s.classes);
  const currentId = useStudio((s) => s.currentImageId);
  const selectImage = useStudio((s) => s.selectImage);
  const annotationsMap = useStudio((s) => s.annotations);
  const suggestionsMap = useStudio((s) => s.suggestions);
  const activeClassId = useStudio((s) => s.activeClassId);
  const setActiveClass = useStudio((s) => s.setActiveClass);
  const addClass = useStudio((s) => s.addClass);
  const saveClasses = useStudio((s) => s.saveClasses);
  const setAnnotations = useStudio((s) => s.setAnnotations);
  const setSuggestions = useStudio((s) => s.setSuggestions);
  const applyImage = useStudio((s) => s.applyImage);
  const applyImages = useStudio((s) => s.applyImages);
  const loadImageData = useStudio((s) => s.loadImageData);
  const refresh = useStudio((s) => s.refresh);
  const models = useStudio((s) => s.models);
  const catalog = useStudio((s) => s.catalog);
  const run = useStudio((s) => s.run);
  const toast = useStudio((s) => s.toast);

  const [tool, setTool] = useState<CanvasTool>("box");
  const [brush, setBrush] = useState(14);
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [preview, setPreview] = useState<SamResult | null>(null);
  const [liveSam, setLiveSam] = useState(true);
  const [samModel, setSamModel] = useState("sam2.1_t");
  const [granularity, setGranularity] = useState<api.Granularity>("auto");
  const [samBusy, setSamBusy] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);
  const [hovered, setHovered] = useState<string | null>(null);
  const [minScore, setMinScore] = useState(0.25);
  const [showSugg, setShowSugg] = useState(true);
  const [newClass, setNewClass] = useState("");
  const [fitKey, setFitKey] = useState(0);
  // Find-similar (visual prompt) options.
  const [vpScope, setVpScope] = useState<Scope>("image");
  const [vpFamily, setVpFamily] = useState<YoloeFamily>("11");
  const [vpSize, setVpSize] = useState<YoloeSize>("s");
  const [vpConf, setVpConf] = useState(0.25);
  const [vpUseLabels, setVpUseLabels] = useState(false);
  const [vpRefine, setVpRefine] = useState(false);
  // Auto-label options.
  const [autoScope, setAutoScope] = useState<Scope>("image");
  const [autoRefine, setAutoRefine] = useState(false);
  const [textSpec, setTextSpec] = useState<ModelSpec>({ kind: "yoloe-text", family: "11", size: "s", classes: ["carton"] });
  const [assistSpec, setAssistSpec] = useState<ModelSpec>({ kind: "yolo", family: "26", task: "segment", size: "n" });
  const [assistConf, setAssistConf] = useState(0.25);
  const [assignActive, setAssignActive] = useState(false);

  const image: StudioImage | undefined = images.find((i) => i.id === currentId);
  const anns: Annotation[] = (currentId && annotationsMap[currentId]) || [];
  const sugg = (currentId && suggestionsMap[currentId]) || { source: null, created_at: null, items: [] };
  const visibleSugg = sugg.items.filter((s) => (s.score ?? 1) >= minScore);

  // Reset the draft when the image changes.
  useEffect(() => {
    setPrompts([]);
    setPreview(null);
    setSelected([]);
    setHovered(null);
  }, [currentId]);

  // ---- Live SAM preview ----------------------------------------------------
  const samSeq = useRef(0);
  const runSam = useCallback(
    async (draft: Prompt[]) => {
      if (!pid || !currentId || !draft.some((p) => p.polarity > 0)) {
        setPreview(null);
        return;
      }
      const seq = ++samSeq.current;
      setSamBusy(true);
      try {
        const out = await api.samSegment(pid, currentId, draft, samModel, granularity);
        if (seq === samSeq.current) setPreview(out.result);
      } catch (e) {
        if (seq === samSeq.current) toast(e instanceof Error ? e.message : String(e), "error");
      } finally {
        if (seq === samSeq.current) setSamBusy(false);
      }
    },
    [pid, currentId, samModel, granularity, toast],
  );

  useEffect(() => {
    if (!liveSam || prompts.length === 0) {
      if (prompts.length === 0) setPreview(null);
      return;
    }
    const t = window.setTimeout(() => runSam(prompts), 120);
    return () => window.clearTimeout(t);
  }, [prompts, liveSam, runSam]);

  // ---- Actions ---------------------------------------------------------------
  const ensureClass = async (): Promise<number | null> => {
    if (activeClassId !== null) return activeClassId;
    const c = await addClass(classes.length ? classes[0].name : "object");
    return c?.id ?? null;
  };

  const onPrompt = async (p: Prompt) => {
    const cid = await ensureClass();
    setPrompts((cur) => [...cur, { ...p, class_id: cid ?? -1 }]);
  };

  const clearDraft = () => {
    samSeq.current++;
    setPrompts([]);
    setPreview(null);
    setSamBusy(false);
  };

  const acceptPreview = async () => {
    if (!currentId || !preview) return;
    const cid = activeClassId ?? preview.class_id;
    if (cid === null || cid === undefined) return;
    await run("Saving label", () =>
      setAnnotations(currentId, [
        ...anns,
        { class_id: cid, polygon: preview.polygon, bbox: preview.bbox, source: preview.source, score: preview.score },
      ]),
    );
    clearDraft();
  };

  const addAsLabel = async () => {
    if (!pid || !currentId || !prompts.length) return;
    const out = await run("Adding labels", () => api.paintLabels(pid, currentId, prompts));
    if (!out) return;
    useStudio.setState((st) => ({ annotations: { ...st.annotations, [currentId]: out.annotations } }));
    applyImage(out.image);
    clearDraft();
    toast(`Added ${out.added} label${out.added === 1 ? "" : "s"}.`, "success");
  };

  const summarize = (counts: Record<string, number>, engine: string, visual = false) => {
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    const here = currentId ? counts[currentId] ?? 0 : 0;
    const n = Object.keys(counts).length;
    const base =
      n <= 1
        ? `${engine}: ${here} suggestion${here === 1 ? "" : "s"} on this image.`
        : `${engine}: ${total} suggestions across ${n} images (${here} here).`;
    // Visual-prompt embeddings are scene-specific: strong within an image or
    // a fixed camera setup, weak across different scenes.
    return total === 0 && visual
      ? `${base} Visual prompts transfer best within the same image or camera setup — lower Min conf, add examples from similar images, or train a model.`
      : base;
  };

  /** Drop any in-flight live-SAM preview: a detection run supersedes it. */
  const cancelSam = () => {
    samSeq.current++;
    setPreview(null);
    setSamBusy(false);
  };

  const findSimilar = async () => {
    if (!pid || !currentId) return;
    if (!prompts.some((p) => p.polarity > 0) && !vpUseLabels) {
      toast("Draw a box or paint a stroke on an example first (or tick 'use existing labels').", "error");
      return;
    }
    cancelSam();
    const out = await run(`YOLOE-${vpFamily}${vpSize} visual prompt`, () =>
      api.visualPrompt(pid, {
        refs: [{ image_id: currentId, prompts, use_annotations: vpUseLabels }],
        scope: vpScope,
        image_id: currentId,
        family: vpFamily,
        size: vpSize,
        conf: vpConf,
        refine_with_sam: vpRefine,
        sam_model: samModel,
      }),
    );
    if (!out) return;
    applyImages(out.images);
    await loadImageData(currentId);
    cancelSam();
    setShowSugg(true);
    toast(summarize(out.counts, out.engine, true), "success");
  };

  const learnFromLabels = async () => {
    if (!pid) return;
    const labelled = images.filter((i) => i.n_annotations > 0);
    if (!labelled.length) {
      toast("Label a few objects first — they become the visual examples.", "error");
      return;
    }
    const refs = labelled
      .sort((a, b) => (a.id === currentId ? -1 : b.id === currentId ? 1 : 0))
      .slice(0, 8)
      .map((i) => ({ image_id: i.id, use_annotations: true }));
    const out = await run(`Finding more like your labels (${refs.length} reference image${refs.length === 1 ? "" : "s"})`, () =>
      api.visualPrompt(pid, {
        refs,
        scope: "unlabeled",
        family: vpFamily,
        size: vpSize,
        conf: vpConf,
        refine_with_sam: autoRefine,
        sam_model: samModel,
      }),
    );
    if (!out) return;
    applyImages(out.images);
    if (currentId) await loadImageData(currentId);
    toast(summarize(out.counts, out.engine, true) + " Use the Review filter to step through them.", "success");
  };

  const runDetect = async (spec: ModelSpec, conf: number, assign: boolean, label: string) => {
    if (!pid || !currentId) return;
    if (spec.kind === "trained" && !spec.model_id) {
      toast("Pick a trained model first.", "error");
      return;
    }
    cancelSam();
    const out = await run(label, () =>
      api.detectToSuggestions(pid, {
        model: spec,
        scope: autoScope,
        image_id: currentId,
        params: { conf },
        refine_with_sam: autoRefine,
        sam_model: samModel,
        assign_class_id: assign ? activeClassId : null,
      }),
    );
    if (!out) return;
    applyImages(out.images);
    await loadImageData(currentId);
    setShowSugg(true);
    toast(summarize(out.counts, out.engine), "success");
  };

  const acceptSugg = async (ids?: string[], min = 0) => {
    if (!pid || !currentId) return;
    const out = await run("Accepting", () => api.acceptSuggestions(pid, currentId, { ids, min_score: min }));
    if (!out) return;
    useStudio.setState((st) => ({
      annotations: { ...st.annotations, [currentId]: out.annotations },
      suggestions: { ...st.suggestions, [currentId]: out.suggestions },
      classes: out.classes,
    }));
    applyImage(out.image);
    if (out.accepted === 0 && ids?.length) toast("That suggestion's class doesn't exist yet — pick a class.", "error");
  };

  const rejectSugg = async (ids?: string[]) => {
    if (!pid || !currentId) return;
    const out = await run("Rejecting", () => api.rejectSuggestions(pid, currentId, ids));
    if (!out) return;
    setSuggestions(currentId, out.suggestions);
    applyImage(out.image);
  };

  const deleteSelected = async () => {
    if (!currentId) return;
    const ids = new Set(selected.filter((s) => s.startsWith("a:")).map((s) => s.slice(2)));
    if (!ids.size) return;
    await run("Deleting", () => setAnnotations(currentId, anns.filter((a) => !ids.has(a.id))));
    setSelected([]);
  };

  const reclassSelected = async (cid: number) => {
    if (!currentId) return;
    const ids = new Set(selected.filter((s) => s.startsWith("a:")).map((s) => s.slice(2)));
    if (!ids.size) return;
    await run("Updating", () => setAnnotations(currentId, anns.map((a) => (ids.has(a.id) ? { ...a, class_id: cid } : a))));
  };

  const step = (delta: number) => {
    if (!images.length) return;
    const idx = Math.max(0, images.findIndex((i) => i.id === currentId));
    const next = images[(idx + delta + images.length) % images.length];
    selectImage(next.id);
  };

  // ---- Keyboard ------------------------------------------------------------
  const keyState = useRef({ prompts, preview, selected, liveSam });
  keyState.current = { prompts, preview, selected, liveSam };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTypingTarget(e)) return;
      const k = e.key.toLowerCase();
      const ks = keyState.current;
      if ((e.metaKey || e.ctrlKey) && k === "z") {
        e.preventDefault();
        setPrompts((cur) => cur.slice(0, -1));
        return;
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (TOOL_KEYS[k]) setTool(TOOL_KEYS[k]);
      else if (k === "[") setBrush((r) => Math.max(2, Math.round(r / 1.25)));
      else if (k === "]") setBrush((r) => Math.min(120, Math.round(r * 1.25)));
      else if (/^[1-9]$/.test(k)) {
        const c = classes[Number(k) - 1];
        if (c) {
          setActiveClass(c.id);
          if (ks.selected.length) reclassSelected(c.id);
        }
      } else if (k === "enter") {
        if (ks.preview) acceptPreview();
        else if (ks.prompts.length && !ks.liveSam) runSam(ks.prompts);
      } else if (k === "s") runSam(ks.prompts);
      else if (k === "g") setGranularity((g) => GRAN_ORDER[(GRAN_ORDER.indexOf(g) + 1) % GRAN_ORDER.length]);
      else if (k === "f") findSimilar();
      else if (k === "a") addAsLabel();
      else if (k === "escape") {
        clearDraft();
        setSelected([]);
      } else if (k === "delete" || k === "backspace") deleteSelected();
      else if (k === "arrowright") step(1);
      else if (k === "arrowleft") step(-1);
      else if (k === "0") setFitKey((x) => x + 1);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  // ---- Shapes --------------------------------------------------------------
  const shapes: Shape[] = useMemo(() => {
    // Name tags clutter dense pallets: show them for few labels, or on
    // hover / selection.
    const tagAll = anns.length <= 6;
    const out: Shape[] = anns.map((a) => {
      const sid = `a:${a.id}`;
      return {
        id: sid,
        color: classColor(classes, a.class_id),
        polygon: a.polygon,
        bbox: a.bbox,
        label: tagAll || hovered === sid || selected.includes(sid) ? className(classes, a.class_id) : undefined,
        variant: "annotation" as const,
        dimmed: prompts.length > 0,
      };
    });
    if (showSugg) {
      const tagSugg = visibleSugg.length <= 25;
      for (const s of visibleSugg) {
        const sid = `s:${s.id}`;
        out.push({
          id: sid,
          color: s.class_id >= 0 ? classColor(classes, s.class_id) : "#f59e0b",
          polygon: s.polygon,
          bbox: s.bbox,
          label:
            tagSugg || hovered === sid
              ? `${s.class_id >= 0 ? className(classes, s.class_id) : s.class_name} ${s.score !== null ? (s.score * 100).toFixed(0) : ""}`
              : undefined,
          variant: "suggestion",
        });
      }
    }
    if (preview) {
      out.push({
        id: "preview",
        color: classColor(classes, activeClassId ?? preview.class_id),
        polygon: preview.polygon,
        bbox: preview.bbox,
        label: `SAM ${(preview.score * 100).toFixed(0)} · ⏎ to accept`,
        variant: "preview",
      });
    }
    return out;
  }, [anns, visibleSugg, showSugg, preview, classes, activeClassId, prompts.length, hovered, selected]);

  const onSelectShape = (id: string | null, additive: boolean) => {
    if (!id) {
      setSelected([]);
      return;
    }
    setSelected((cur) => (additive ? (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id]) : [id]));
  };

  const posPrompts = prompts.filter((p) => p.polarity > 0).length;
  const negPrompts = prompts.length - posPrompts;
  const annCounts = useMemo(() => {
    const m = new Map<number, number>();
    for (const a of anns) m.set(a.class_id, (m.get(a.class_id) ?? 0) + 1);
    return m;
  }, [anns]);

  if (!pid) return null;

  return (
    <div className="st-label">
      <ImageList selectedId={currentId} onSelect={(id) => selectImage(id)} />

      <div className="st-stage">
        <div className="st-toolbar">
          <Segmented
            value={tool}
            onChange={setTool}
            options={[
              { value: "select", label: "Select", title: "Select labels (V) — ⇧ to add; Del deletes" },
              { value: "box", label: "▭ Box", title: "Draw a box prompt (B) — ⌥ for negative" },
              { value: "brush", label: "✎ Brush", title: "Paint a positive stroke (P) — ⌥ / right-drag for negative" },
              { value: "erase", label: "⌫ Erase", title: "Paint a negative stroke (E)" },
              { value: "pan", label: "✋", title: "Pan (H, or hold Space)" },
            ]}
          />
          {(tool === "brush" || tool === "erase") && (
            <Slider label="Brush" value={brush} min={2} max={120} step={1} onChange={setBrush} format={(v) => `${v}px`} hint="[ and ] or ⇧+wheel" />
          )}
          <div className="st-toolbar-spacer" />
          <Toggle label="Suggestions" checked={showSugg} onChange={setShowSugg} />
          <span className="st-muted st-nav">
            <button type="button" className="st-btn sm ghost" onClick={() => step(-1)} title="Previous image (←)">
              ←
            </button>
            {images.findIndex((i) => i.id === currentId) + 1}/{images.length}
            <button type="button" className="st-btn sm ghost" onClick={() => step(1)} title="Next image (→)">
              →
            </button>
          </span>
        </div>
        <div className="st-canvas-area">
          {image ? (
            <PromptCanvas
              imageUrl={api.imageUrl(pid, image.id)}
              imageWidth={image.width}
              imageHeight={image.height}
              shapes={shapes}
              prompts={prompts}
              colorForClass={(id) => classColor(classes, id)}
              tool={tool}
              brushRadius={brush}
              onBrushRadius={(r) => setBrush(Math.round(r))}
              activeClassId={activeClassId}
              selectedIds={selected}
              hoveredId={hovered}
              onPrompt={onPrompt}
              onSelect={onSelectShape}
              onHover={setHovered}
              fitKey={`${image.id}:${fitKey}`}
            >
              {(samBusy || prompts.length > 0) && (
                <div className="st-canvas-hint">
                  {samBusy ? "SAM…" : preview ? "⏎ accept · Esc clear · F find similar · A add as label" : "S segment · F find similar · A add as label · ⌘Z undo"}
                </div>
              )}
            </PromptCanvas>
          ) : (
            <div className="st-canvas-empty">
              <h2>YOLO26 Studio</h2>
              <p>Add images on the left — upload your own pallet photos, render synthetic pallets with exact depth, or pull frames from a video.</p>
              <p className="st-muted">Then draw a box or paint a stroke on one carton and press <b>F</b> to find all the others.</p>
            </div>
          )}
        </div>
      </div>

      <aside className="st-panel">
        <Section title="Classes" right={<span className="st-muted">1–9 to pick</span>}>
          <div className="st-classes">
            {classes.map((c, i) => (
              <div
                key={c.id}
                className={`st-class ${c.id === activeClassId ? "on" : ""}`}
                onClick={() => {
                  setActiveClass(c.id);
                  if (selected.length) reclassSelected(c.id);
                }}
                title={selected.length ? "Click to set active class and re-label the selection" : "Set active class"}
              >
                <span className="st-swatch" style={{ background: c.color }} />
                <span className="st-class-name">{c.name}</span>
                <span className="st-class-key">{i < 9 ? i + 1 : ""}</span>
                <span className="st-class-count">{annCounts.get(c.id) ?? 0}</span>
                <button
                  type="button"
                  className="st-x"
                  title="Delete class (and its labels)"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (window.confirm(`Delete class "${c.name}" and all its labels in every image?`)) {
                      run("Deleting class", () => saveClasses(classes.filter((x) => x.id !== c.id)));
                    }
                  }}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
          <form
            className="st-row"
            onSubmit={(e) => {
              e.preventDefault();
              if (newClass.trim()) run("Adding class", () => addClass(newClass)).then(() => setNewClass(""));
            }}
          >
            <input className="st-input" placeholder="New class, e.g. carton" value={newClass} onChange={(e) => setNewClass(e.target.value)} />
            <button type="submit" className="st-btn sm">
              Add
            </button>
          </form>
        </Section>

        <Section
          title="Prompt"
          right={
            <span className="st-muted">
              {posPrompts}+ {negPrompts}−
            </span>
          }
        >
          <div className="st-row wrap">
            <Toggle label="Live SAM" checked={liveSam} onChange={setLiveSam} hint="Re-segment after every stroke or box" />
            <select className="st-input slim" value={samModel} onChange={(e) => setSamModel(e.target.value)} title="SAM model used for segmenting">
              {(catalog?.sam ?? [{ id: "sam2.1_t", label: "SAM 2.1 tiny" }]).map((s) => (
                <option key={s.id} value={s.id}>
                  {s.id}
                </option>
              ))}
            </select>
          </div>
          <div className="st-row wrap" title="SAM returns part / object / whole hypotheses. Auto: a box decides; for strokes, the smallest mask containing the whole stroke. G cycles.">
            <span className="st-field-label">Granularity</span>
            <Segmented
              size="sm"
              value={granularity}
              onChange={setGranularity}
              options={[
                { value: "auto", label: "Auto" },
                { value: "fine", label: "Fine" },
                { value: "medium", label: "Medium" },
                { value: "coarse", label: "Coarse" },
              ]}
            />
          </div>
          {prompts.length === 0 && !preview && (
            <p className="st-muted st-help">
              Draw a <b>box</b> or <b>paint</b> over an object. One object → <b>Segment</b>. An example of many → <b>Find similar</b>.
            </p>
          )}
          {preview && (
            <div className="st-preview-card">
              <span>
                SAM mask · score {(preview.score * 100).toFixed(0)} · {preview.area.toLocaleString()} px
              </span>
              <button type="button" className="st-btn sm primary" onClick={acceptPreview}>
                Accept ⏎
              </button>
            </div>
          )}
          <div className="st-actions">
            <button type="button" className="st-btn" disabled={!prompts.length || samBusy} onClick={() => runSam(prompts)} title="SAM: the whole draft is one object (S)">
              Segment <kbd>S</kbd>
            </button>
            <button type="button" className="st-btn primary" disabled={!prompts.length && !vpUseLabels} onClick={findSimilar} title="YOLOE visual prompt: find every similar object (F)">
              Find similar <kbd>F</kbd>
            </button>
            <button type="button" className="st-btn" disabled={!prompts.length} onClick={addAsLabel} title="Use boxes / painted pixels directly as labels (A)">
              Add as label <kbd>A</kbd>
            </button>
            <button type="button" className="st-btn ghost" disabled={!prompts.length && !preview} onClick={clearDraft}>
              Clear <kbd>Esc</kbd>
            </button>
          </div>
          <div className="st-subtle-block">
            <div className="st-row wrap">
              <span className="st-field-label">Find similar in</span>
              <Segmented
                size="sm"
                value={vpScope}
                onChange={setVpScope}
                options={[
                  { value: "image", label: "this image" },
                  { value: "unlabeled", label: "unlabelled" },
                  { value: "all", label: "all images" },
                ]}
              />
            </div>
            <div className="st-row wrap">
              <span className="st-field-label">Engine</span>
              <Segmented
                size="sm"
                value={vpFamily}
                onChange={setVpFamily}
                options={[
                  { value: "11", label: "YOLOE-11", title: "Best visual prompting on cartons in our tests" },
                  { value: "26", label: "YOLOE-26", title: "YOLO26-based; weaker visual prompts on cartons in our tests" },
                ]}
              />
              <Segmented size="sm" value={vpSize} onChange={setVpSize} options={(["s", "m", "l"] as YoloeSize[]).map((s) => ({ value: s, label: s }))} />
            </div>
            <Slider label="Min conf" value={vpConf} min={0.05} max={0.9} step={0.05} onChange={setVpConf} format={(v) => v.toFixed(2)} />
            <Toggle label="Also use this image's existing labels as examples" checked={vpUseLabels} onChange={setVpUseLabels} />
            <Toggle label="Refine each hit with SAM (crisper masks, slower)" checked={vpRefine} onChange={setVpRefine} />
          </div>
        </Section>

        {sugg.items.length > 0 && (
          <Section
            title={`Suggestions · ${visibleSugg.length}${visibleSugg.length !== sugg.items.length ? ` of ${sugg.items.length}` : ""}`}
            right={<span className="st-muted">{sugg.source}</span>}
          >
            <Slider label="Min score" value={minScore} min={0} max={0.95} step={0.05} onChange={setMinScore} format={(v) => v.toFixed(2)} />
            <div className="st-actions">
              <button type="button" className="st-btn primary sm" onClick={() => acceptSugg(undefined, minScore)}>
                Accept {visibleSugg.length} ≥ {minScore.toFixed(2)}
              </button>
              <button type="button" className="st-btn sm ghost" onClick={() => rejectSugg()}>
                Reject all
              </button>
            </div>
            <ul className="st-list">
              {visibleSugg.map((s) => (
                <li
                  key={s.id}
                  className={hovered === `s:${s.id}` ? "hover" : ""}
                  onMouseEnter={() => setHovered(`s:${s.id}`)}
                  onMouseLeave={() => setHovered(null)}
                >
                  <span className="st-swatch" style={{ background: s.class_id >= 0 ? classColor(classes, s.class_id) : "#f59e0b" }} />
                  <span className="st-grow">
                    {s.class_id >= 0 ? className(classes, s.class_id) : <i title="will create this class">{s.class_name} (new)</i>}
                  </span>
                  <span className="st-score">{s.score !== null ? (s.score * 100).toFixed(0) : "—"}</span>
                  <button type="button" className="st-icon ok" title="Accept" onClick={() => acceptSugg([s.id])}>
                    ✓
                  </button>
                  <button type="button" className="st-icon bad" title="Reject" onClick={() => rejectSugg([s.id])}>
                    ✗
                  </button>
                </li>
              ))}
            </ul>
          </Section>
        )}

        <Section
          title={`Labels · ${anns.length}`}
          right={
            selected.length > 0 ? (
              <button type="button" className="st-btn sm ghost" onClick={deleteSelected}>
                Delete {selected.length}
              </button>
            ) : undefined
          }
        >
          {anns.length === 0 ? (
            <p className="st-muted st-help">No labels on this image yet.</p>
          ) : (
            <ul className="st-list">
              {anns.map((a) => {
                const sid = `a:${a.id}`;
                return (
                  <li
                    key={a.id}
                    className={`${selected.includes(sid) ? "on" : ""} ${hovered === sid ? "hover" : ""}`}
                    onClick={(e) => onSelectShape(sid, e.shiftKey || e.metaKey)}
                    onMouseEnter={() => setHovered(sid)}
                    onMouseLeave={() => setHovered(null)}
                  >
                    <span className="st-swatch" style={{ background: classColor(classes, a.class_id) }} />
                    <span className="st-grow">{className(classes, a.class_id)}</span>
                    <span className="st-muted st-src" title={a.source}>
                      {a.polygon ? "mask" : "box"} · {a.source.split(":")[0]}
                    </span>
                    <button
                      type="button"
                      className="st-icon bad"
                      title="Delete"
                      onClick={(e) => {
                        e.stopPropagation();
                        run("Deleting", () => setAnnotations(currentId!, anns.filter((x) => x.id !== a.id)));
                      }}
                    >
                      ×
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </Section>

        <Section title="Auto-label" collapsible defaultOpen={false}>
          <div className="st-row wrap">
            <span className="st-field-label">Run on</span>
            <Segmented
              size="sm"
              value={autoScope}
              onChange={setAutoScope}
              options={[
                { value: "image", label: "this image" },
                { value: "unlabeled", label: "unlabelled" },
                { value: "all", label: "all" },
              ]}
            />
          </div>
          <Toggle label="Refine boxes into masks with SAM" checked={autoRefine} onChange={setAutoRefine} />

          <div className="st-subtle-block">
            <div className="st-add-title">From your labels</div>
            <p className="st-muted">Uses up to 8 labelled images as visual examples and suggests labels on every unlabelled image.</p>
            <button type="button" className="st-btn sm" onClick={learnFromLabels}>
              Find more like my labels
            </button>
          </div>

          <div className="st-subtle-block">
            <div className="st-add-title">Text prompt (open vocabulary)</div>
            <ModelPicker value={textSpec} onChange={setTextSpec} models={models} catalog={catalog} kinds={["yoloe-text"]} />
            <p className="st-muted">Zero-shot text scored stacked cartons below 0.3 in our tests — lower the conf, or prefer visual prompts.</p>
            <div className="st-row">
              <Slider label="Conf" value={assistConf} min={0.02} max={0.9} step={0.01} onChange={setAssistConf} format={(v) => v.toFixed(2)} />
              <button type="button" className="st-btn sm" onClick={() => runDetect(textSpec, assistConf, false, "Text prompt")}>
                Run
              </button>
            </div>
          </div>

          <div className="st-subtle-block">
            <div className="st-add-title">Model assist / prompt-free</div>
            <ModelPicker value={assistSpec} onChange={setAssistSpec} models={models} catalog={catalog} kinds={["trained", "yolo", "yoloe-pf"]} tasks={["detect", "segment", "obb"]} />
            <Toggle label="Assign every detection to the active class" checked={assignActive} onChange={setAssignActive} hint="e.g. COCO 'suitcase' → your 'carton'" />
            <div className="st-row">
              <Slider label="Conf" value={assistConf} min={0.02} max={0.9} step={0.01} onChange={setAssistConf} format={(v) => v.toFixed(2)} />
              <button type="button" className="st-btn sm" onClick={() => runDetect(assistSpec, assistConf, assignActive, "Model assist")}>
                Run
              </button>
            </div>
          </div>
        </Section>

        {image && <ImageMeta image={image} onChanged={refresh} />}
      </aside>
    </div>
  );
}

function ImageMeta({ image, onChanged }: { image: StudioImage; onChanged: () => Promise<void> }) {
  const pid = useStudio((s) => s.projectId)!;
  const run = useStudio((s) => s.run);
  const applyImage = useStudio((s) => s.applyImage);
  const selectImage = useStudio((s) => s.selectImage);
  const images = useStudio((s) => s.images);
  const toast = useStudio((s) => s.toast);
  const depthRef = useRef<HTMLInputElement>(null);
  const [scale, setScale] = useState("1");

  const patch = async (body: Parameters<typeof api.patchImage>[2]) => {
    const out = await run("Saving", () => api.patchImage(pid, image.id, body));
    if (out) applyImage(out);
  };

  return (
    <Section title="Image" collapsible defaultOpen={false}>
      <div className="st-kv">
        <span>File</span>
        <span className="st-ellipsis" title={image.filename}>
          {image.filename}
        </span>
        <span>Size</span>
        <span>
          {image.width}×{image.height}
        </span>
        <span>Source</span>
        <span className="st-ellipsis" title={image.source}>
          {image.source}
        </span>
        {image.synthetic && (
          <>
            <span>Synthetic</span>
            <span>
              tilt {image.synthetic.tilt_deg}° · cam {image.synthetic.camera_height_m} m
            </span>
          </>
        )}
      </div>
      <div className="st-row wrap">
        <span className="st-field-label">Split</span>
        <Segmented
          size="sm"
          value={image.split}
          onChange={(v) => patch({ split: v })}
          options={[
            { value: "auto", label: "auto", title: "80/20 by hash" },
            { value: "train", label: "train" },
            { value: "val", label: "val" },
            { value: "test", label: "test" },
          ]}
        />
      </div>
      <Toggle
        label="Negative example (no objects — train on it as background)"
        checked={image.negative}
        onChange={(v) => patch({ negative: v })}
      />
      <div className="st-subtle-block">
        <div className="st-add-title">Depth map</div>
        {image.has_depth ? (
          <div className="st-row wrap">
            <span className="st-badge depth">D</span>
            <a className="st-link" href={api.depthPreviewUrl(pid, image.id, image.id)} target="_blank" rel="noreferrer">
              preview
            </a>
            {image.intrinsics && (
              <span className="st-muted">
                fx {image.intrinsics.fx.toFixed(0)} · cx {image.intrinsics.cx.toFixed(0)}
              </span>
            )}
            <button
              type="button"
              className="st-btn sm ghost"
              onClick={async () => {
                const out = await run("Removing depth", () => api.deleteDepth(pid, image.id));
                if (out) applyImage(out);
              }}
            >
              Remove
            </button>
          </div>
        ) : (
          <p className="st-muted">From an RGB-D camera: 16-bit PNG in millimetres (or .npy in metres), aligned to this image.</p>
        )}
        <div className="st-row wrap">
          <label className="st-mini" title="PNG/TIFF only: multiply raw values by this to get millimetres (e.g. 0.25 for quarter-mm units). .npy is always metres.">
            PNG units→mm ×
            <input className="st-input num" value={scale} onChange={(e) => setScale(e.target.value)} />
          </label>
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
              if (!f) return;
              // The scale only applies to PNG/TIFF raw units; .npy is metres.
              const sc = Number(scale);
              const out = await run("Uploading depth", () =>
                api.uploadDepth(pid, image.id, f, { scale_to_mm: sc && sc !== 1 && !/\.npy$/i.test(f.name) ? sc : undefined }),
              );
              if (out) {
                applyImage(out);
                toast("Depth map attached — analyse it in the Pallet tab.", "success");
              }
            }}
          />
        </div>
      </div>
      <button
        type="button"
        className="st-btn sm danger"
        onClick={async () => {
          if (!window.confirm(`Delete ${image.filename} and its labels?`)) return;
          const idx = images.findIndex((i) => i.id === image.id);
          const ok = await run("Deleting image", () => api.deleteImage(pid, image.id));
          if (!ok) return;
          await onChanged();
          const rest = useStudio.getState().images;
          selectImage(rest.length ? rest[Math.min(idx, rest.length - 1)].id : null);
        }}
      >
        Delete image
      </button>
    </Section>
  );
}
