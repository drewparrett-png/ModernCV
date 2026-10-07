/**
 * Left-hand image strip: upload (button or drag-drop), synthetic pallets,
 * video frames, filters, and per-image badges.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { useStore } from "../store";
import * as api from "./api";
import { useStudio } from "./useStudio";
import { Segmented } from "./widgets";
import type { StudioImage } from "./types";

type Filter = "all" | "unlabeled" | "labeled" | "review" | "depth";

export function ImageList({
  selectedId,
  onSelect,
  compact = false,
  defaultFilter = "all",
}: {
  selectedId: string | null;
  onSelect: (id: string) => void;
  compact?: boolean;
  defaultFilter?: Filter;
}) {
  const pid = useStudio((s) => s.projectId);
  const images = useStudio((s) => s.images);
  const run = useStudio((s) => s.run);
  const refresh = useStudio((s) => s.refresh);
  const toast = useStudio((s) => s.toast);
  const videos = useStore((s) => s.videos);
  const [filter, setFilter] = useState<Filter>(defaultFilter);
  const [adding, setAdding] = useState(false);
  const [synthCount, setSynthCount] = useState(6);
  const [synthLabels, setSynthLabels] = useState(true);
  const [video, setVideo] = useState("");
  const [stride, setStride] = useState(30);
  const [maxFrames, setMaxFrames] = useState(20);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

  // Keep the selected image in view (←/→ navigation can move off-screen).
  useEffect(() => {
    if (!selectedId || !listRef.current) return;
    const el = listRef.current.querySelector<HTMLElement>(`[data-iid="${selectedId}"]`);
    el?.scrollIntoView({ block: "nearest" });
  }, [selectedId]);

  const shown = useMemo(() => {
    const f = (i: StudioImage) =>
      filter === "all" ||
      (filter === "unlabeled" && i.n_annotations === 0 && !i.negative) ||
      (filter === "labeled" && (i.n_annotations > 0 || i.negative)) ||
      (filter === "review" && i.n_suggestions > 0) ||
      (filter === "depth" && i.has_depth);
    return images.filter(f);
  }, [images, filter]);

  const upload = async (files: File[]) => {
    if (!pid || !files.length) return;
    const imgs = files.filter((f) => f.type.startsWith("image/") || /\.(jpe?g|png|webp|bmp|tiff?)$/i.test(f.name));
    if (!imgs.length) {
      toast("Drop image files (jpg, png, webp, bmp, tiff).", "error");
      return;
    }
    const out = await run(`Uploading ${imgs.length} image${imgs.length === 1 ? "" : "s"}`, () => api.uploadImages(pid, imgs));
    if (!out) return;
    await refresh();
    if (out.errors.length) toast(out.errors.join("\n"), "error");
    if (out.images[0]) onSelect(out.images[0].id);
  };

  const addSynthetic = async () => {
    if (!pid) return;
    const out = await run(`Rendering ${synthCount} synthetic pallet${synthCount === 1 ? "" : "s"}`, () =>
      api.addSynthetic(pid, { count: synthCount, with_labels: synthLabels }),
    );
    if (!out) return;
    await refresh();
    setAdding(false);
    toast(`Added ${out.images.length} synthetic pallets with exact depth${synthLabels ? " and ground-truth masks" : ""}.`, "success");
    if (out.images[0]) onSelect(out.images[0].id);
  };

  const addFrames = async () => {
    if (!pid || !video) return;
    const out = await run("Extracting frames", () => api.imagesFromVideo(pid, { video_path: video, stride, max_frames: maxFrames }));
    if (!out) return;
    await refresh();
    setAdding(false);
    toast(`Added ${out.images.length} frames.`, "success");
    if (out.images[0]) onSelect(out.images[0].id);
  };

  const counts = {
    all: images.length,
    unlabeled: images.filter((i) => i.n_annotations === 0 && !i.negative).length,
    labeled: images.filter((i) => i.n_annotations > 0 || i.negative).length,
    review: images.filter((i) => i.n_suggestions > 0).length,
    depth: images.filter((i) => i.has_depth).length,
  };

  return (
    <aside
      className={`st-images ${compact ? "compact" : ""} ${dragOver ? "drag" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        upload(Array.from(e.dataTransfer.files));
      }}
    >
      <header className="st-images-head">
        <h3>Images</h3>
        <div className="st-row">
          <button type="button" className="st-btn sm" onClick={() => fileRef.current?.click()} title="Upload images (or drag & drop)">
            Upload
          </button>
          <button type="button" className={`st-btn sm ${adding ? "on" : ""}`} onClick={() => setAdding(!adding)} title="More ways to add images">
            ＋
          </button>
        </div>
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          multiple
          hidden
          onChange={(e) => {
            upload(Array.from(e.target.files ?? []));
            e.target.value = "";
          }}
        />
      </header>

      {adding && (
        <div className="st-add-panel">
          <div className="st-add-block">
            <div className="st-add-title">Synthetic pallets</div>
            <p className="st-muted">Ray-traced cartons on a EUR pallet with exact depth + intrinsics — ground truth for testing heights.</p>
            <div className="st-row">
              <input className="st-input num" type="number" min={1} max={60} value={synthCount} onChange={(e) => setSynthCount(Number(e.target.value))} />
              <label className="st-toggle">
                <input type="checkbox" checked={synthLabels} onChange={(e) => setSynthLabels(e.target.checked)} />
                <span>GT masks</span>
              </label>
              <button type="button" className="st-btn sm primary" onClick={addSynthetic}>
                Render
              </button>
            </div>
          </div>
          <div className="st-add-block">
            <div className="st-add-title">Frames from a video</div>
            <select className="st-input" value={video} onChange={(e) => setVideo(e.target.value)}>
              <option value="">Choose a video in data/…</option>
              {videos.map((v) => (
                <option key={v} value={v}>
                  {v.replace(/^data\//, "")}
                </option>
              ))}
            </select>
            <div className="st-row">
              <label className="st-mini">
                every
                <input className="st-input num" type="number" min={1} value={stride} onChange={(e) => setStride(Number(e.target.value))} />
                frames
              </label>
              <label className="st-mini">
                max
                <input className="st-input num" type="number" min={1} max={500} value={maxFrames} onChange={(e) => setMaxFrames(Number(e.target.value))} />
              </label>
              <button type="button" className="st-btn sm primary" disabled={!video} onClick={addFrames}>
                Add
              </button>
            </div>
          </div>
        </div>
      )}

      {!compact && (
        <div className="st-images-filter">
          <Segmented
            size="sm"
            value={filter}
            onChange={setFilter}
            options={[
              { value: "all", label: `All ${counts.all}` },
              { value: "unlabeled", label: `To do ${counts.unlabeled}`, title: "No labels yet" },
              { value: "review", label: `Review ${counts.review}`, title: "Has suggestions to review" },
              { value: "labeled", label: `Done ${counts.labeled}` },
            ]}
          />
        </div>
      )}

      <ul className="st-thumbs" ref={listRef}>
        {shown.length === 0 && (
          <li className="st-empty">
            {images.length === 0 ? (
              <>
                Drop images here, click <b>Upload</b>, or <b>＋</b> to render synthetic pallets / grab video frames.
              </>
            ) : (
              "Nothing matches this filter."
            )}
          </li>
        )}
        {shown.map((img) => (
          <li
            key={img.id}
            data-iid={img.id}
            className={`st-thumb ${img.id === selectedId ? "on" : ""}`}
            onClick={() => onSelect(img.id)}
            title={`${img.filename} · ${img.width}×${img.height}`}
          >
            {pid && <img src={api.thumbUrl(pid, img.id)} loading="lazy" alt="" />}
            <div className="st-thumb-meta">
              <span className="st-thumb-name">{img.filename}</span>
              <span className="st-badges">
                {img.n_annotations > 0 && <span className="st-badge ok">{img.n_annotations}</span>}
                {img.negative && <span className="st-badge neg" title="negative (background) example">∅</span>}
                {img.n_suggestions > 0 && <span className="st-badge warn" title="suggestions to review">{img.n_suggestions}?</span>}
                {img.has_depth && <span className="st-badge depth" title="has a depth map">D</span>}
                {img.split !== "auto" && <span className="st-badge split">{img.split}</span>}
              </span>
            </div>
          </li>
        ))}
      </ul>
    </aside>
  );
}
