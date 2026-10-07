/**
 * Studio — YOLO26 image lab inside a project.
 *
 *   Label   prompt with boxes / brush strokes → SAM masks, YOLOE "find
 *           similar", text / prompt-free / model auto-labelling, review.
 *   Train   YOLO26 / YOLO11 detect / segment / OBB with live curves.
 *   Test    predict playground for every YOLO task + val / export /
 *           benchmark / video tracking.
 *   Pallet  Pal/DePal: carton masks + depth → heights, layers, pick order.
 */

import { useEffect } from "react";
import { useStore } from "../store";
import { LabelView } from "./LabelView";
import { PalletView } from "./PalletView";
import { TestView } from "./TestView";
import { TrainView } from "./TrainView";
import { useStudio, type StudioTab } from "./useStudio";
import "./studio.css";

const TABS: { id: StudioTab; label: string; blurb: string }[] = [
  { id: "label", label: "Label", blurb: "Prompt · segment · review" },
  { id: "train", label: "Train", blurb: "YOLO26 / YOLO11 · detect, seg, OBB" },
  { id: "test", label: "Test", blurb: "Predict · val · export · track" },
  { id: "pallet", label: "Pallet", blurb: "Pal/DePal heights & layers" },
];

export function Studio() {
  const projectId = useStore((s) => s.currentProjectId);
  const tab = useStudio((s) => s.tab);
  const setTab = useStudio((s) => s.setTab);
  const load = useStudio((s) => s.load);
  const images = useStudio((s) => s.images);
  const models = useStudio((s) => s.models);
  const busy = useStudio((s) => s.busy);
  const toasts = useStudio((s) => s.toasts);
  const dismiss = useStudio((s) => s.dismissToast);
  const refreshModels = useStudio((s) => s.refreshModels);

  useEffect(() => {
    if (projectId) load(projectId).catch((e) => console.error("studio load failed", e));
  }, [projectId, load]);

  // Poll while any training job is queued/running.
  const inFlight = models.some((m) => m.status === "queued" || m.status === "running");
  useEffect(() => {
    if (!inFlight) return;
    const t = window.setInterval(() => refreshModels().catch(() => {}), 3000);
    return () => window.clearInterval(t);
  }, [inFlight, refreshModels]);

  const counts: Record<StudioTab, string> = {
    label: images.length ? `${images.filter((i) => i.n_annotations > 0 || i.negative).length}/${images.length} labelled` : "",
    train: models.length ? `${models.length} model${models.length === 1 ? "" : "s"}${inFlight ? " · training" : ""}` : "",
    test: "",
    pallet: "",
  };

  return (
    <div className="studio">
      <nav className="st-tabs">
        {TABS.map((t) => (
          <button key={t.id} type="button" className={`st-tab ${tab === t.id ? "on" : ""}`} onClick={() => setTab(t.id)}>
            <span className="st-tab-label">{t.label}</span>
            <span className="st-tab-blurb">{counts[t.id] || t.blurb}</span>
          </button>
        ))}
      </nav>
      <div className="st-body">
        {tab === "label" && <LabelView />}
        {tab === "train" && <TrainView />}
        {tab === "test" && <TestView />}
        {tab === "pallet" && <PalletView />}
      </div>
      {busy && (
        <div className="st-busy">
          <span className="st-spinner" /> {busy}…
        </div>
      )}
      <div className="st-toasts">
        {toasts.map((t) => (
          <div key={t.id} className={`st-toast ${t.kind}`} onClick={() => dismiss(t.id)}>
            {t.text}
          </div>
        ))}
      </div>
    </div>
  );
}
