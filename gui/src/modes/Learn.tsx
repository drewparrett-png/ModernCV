/**
 * Learn mode — sidebar of Teacher runs (with the selected row expanded
 * inline) + main panel that hosts the New Teacher form.
 *
 * Earlier the selected Teacher's details lived below the form in the main
 * area, which forced you to scroll past the form to monitor a running
 * job. The sidebar is the right home for run-state because that's the
 * thing you keep coming back to. Click a row → it expands inline with
 * status, models, live progress, and the action buttons. The main panel
 * is now just "start another Teacher".
 */

import { useMemo, useState, type KeyboardEvent } from "react";
import { useStore } from "../store";
import type { RunDetail, RunProgress, Task } from "../types";
import { runOverlayUrl } from "../api";
import { ReviewStatusPill } from "../components/ReviewStatusPill";
import { VideoTreePicker } from "../components/VideoTreePicker";

const TASK_DESCRIPTIONS: Record<Task, string> = {
  detection:
    "Find things and draw bounding boxes around them. Faster, looser localization.",
  segmentation:
    "Find things and outline their pixel-level shape. Slower, more precise.",
};

export function Learn() {
  const form = useStore((s) => s.learnForm);
  const setField = useStore((s) => s.setLearnField);
  const videos = useStore((s) => s.videos);
  const error = useStore((s) => s.learnError);
  const teacherDetails = useStore((s) => s.teacherDetails);
  const selectedId = useStore((s) => s.selectedTeacherId);
  const select = useStore((s) => s.selectTeacher);
  const start = useStore((s) => s.startLearn);
  const del = useStore((s) => s.deleteTeacher);
  const setMode = useStore((s) => s.setMode);
  const startOptimize = useStore((s) => s.startOptimize);
  const openInspector = useStore((s) => s.openInspector);

  const teachers = useMemo(() => {
    const arr = Object.values(teacherDetails);
    const inFlight = (s: string) => s === "running" || s === "queued";
    arr.sort((a, b) => {
      const aIF = inFlight(a.manifest.status) ? 1 : 0;
      const bIF = inFlight(b.manifest.status) ? 1 : 0;
      if (aIF !== bIF) return bIF - aIF;
      return b.manifest.started_at.localeCompare(a.manifest.started_at);
    });
    return arr;
  }, [teacherDetails]);

  const inFlightCount = teachers.filter(
    (t) =>
      t.manifest.status === "running" || t.manifest.status === "queued",
  ).length;

  return (
    <div className="learn-mode-multi">
      <aside className="teachers-sidebar">
        <header className="sidebar-header">
          <h2>Teachers</h2>
          <span className="sidebar-count">{teachers.length}</span>
        </header>
        {teachers.length === 0 && (
          <div className="sidebar-empty">No Teachers yet — start one →</div>
        )}
        <ul className="teachers-list">
          {teachers.map((d) => {
            const isSelected = selectedId === d.manifest.id;
            return (
              <li
                key={d.manifest.id}
                className={`teacher-row ${isSelected ? "active expanded" : ""}`}
                onClick={() => select(d.manifest.id)}
              >
                <TeacherRowHeader
                  detail={d}
                  onDelete={() => del(d.manifest.id)}
                />
                {isSelected && (
                  <TeacherRowDetail
                    detail={d}
                    onInspect={() => openInspector(d.manifest.id)}
                    onStartStudent={async () => {
                      // Quick-start path: one teacher, no eval set. The user
                      // can switch to Optimize and add eval teachers later.
                      const id = await startOptimize({
                        train_teacher_ids: [d.manifest.id],
                        eval_teacher_ids: [],
                      });
                      if (id) setMode("optimize");
                    }}
                  />
                )}
              </li>
            );
          })}
        </ul>
        {inFlightCount > 0 && (
          <div className="sidebar-foot-hint">
            Teachers run sequentially — only one executes at a time so the
            per-frame timing numbers stay clean. Start as many as you want;
            they'll work through the queue in order.
          </div>
        )}
      </aside>

      <main className="learn-main">
        <section className="learn-form">
          <h3>New Teacher</h3>
          <p className="learn-blurb">
            Type what you're looking for. The Teacher pipeline finds it
            across every frame and saves the labels.
          </p>

          <label className="field">
            <span className="field-label">Task</span>
            <div className="task-toggle">
              {(["detection", "segmentation"] as Task[]).map((t) => (
                <button
                  key={t}
                  type="button"
                  className={`task-option ${form.task === t ? "active" : ""}`}
                  onClick={() => setField("task", t)}
                >
                  <div className="task-name">
                    {t === "detection" ? "Detection" : "Segmentation"}
                  </div>
                  <div className="task-desc">{TASK_DESCRIPTIONS[t]}</div>
                </button>
              ))}
            </div>
          </label>

          <label className="field">
            <span className="field-label">What are you looking for?</span>
            <PromptChips
              chips={form.prompts}
              onChange={(next) => setField("prompts", next)}
            />
            <span className="field-hint">
              Each chip is one class — type a phrase and press Enter (or
              comma) to add it. Detections will be labeled with the exact
              chip text. Use multiple chips to look for multiple things.
            </span>
          </label>

          <label className="field">
            <span className="field-label">Input video</span>
            <VideoTreePicker
              videos={videos}
              selected={form.videoPath}
              onSelect={(path) => setField("videoPath", path)}
            />
          </label>

          <label className="field field-inline">
            <span className="field-label">Max frames (optional cap)</span>
            <input
              type="number"
              value={form.maxFrames ?? ""}
              placeholder="all"
              min={1}
              onChange={(e) =>
                setField(
                  "maxFrames",
                  e.target.value === "" ? null : Number(e.target.value),
                )
              }
              className="frame-cap"
            />
            <span className="field-hint">
              Leave blank to process the whole clip.
            </span>
          </label>

          <details className="advanced-section">
            <summary>Advanced — detector thresholds</summary>
            <div className="advanced-body">
              <p className="advanced-blurb">
                Lower thresholds catch smaller / fainter objects but admit
                more false positives. For small-object prompts like
                <code> soccer ball</code> on wide stadium shots, try
                box ≈ 0.15. The default 0.30 works well for player-sized
                objects.
              </p>
              <label className="field field-inline">
                <span className="field-label">Box threshold</span>
                <input
                  type="number"
                  step={0.05}
                  min={0}
                  max={1}
                  value={form.boxThreshold}
                  onChange={(e) =>
                    setField("boxThreshold", Number(e.target.value))
                  }
                  className="frame-cap"
                />
                <span className="field-hint">
                  Score floor for a box to be kept (default 0.30).
                </span>
              </label>
              <label className="field field-inline">
                <span className="field-label">Text threshold</span>
                <input
                  type="number"
                  step={0.05}
                  min={0}
                  max={1}
                  value={form.textThreshold}
                  onChange={(e) =>
                    setField("textThreshold", Number(e.target.value))
                  }
                  className="frame-cap"
                />
                <span className="field-hint">
                  Score floor for matching the text prompt (default 0.25).
                </span>
              </label>
              <label className="field field-inline">
                <span className="field-label">Full resolution</span>
                <input
                  type="checkbox"
                  checked={form.fullResolution}
                  onChange={(e) =>
                    setField("fullResolution", e.target.checked)
                  }
                />
                <span className="field-hint">
                  Skip the GroundingDINO processor's resize (default
                  shrinks 1080p → 1333×750). Helps small objects like the
                  ball; ~2× slower per frame.
                </span>
              </label>
            </div>
          </details>

          <div className="run-row">
            <button type="button" className="run-button" onClick={() => start()}>
              Start Teacher
            </button>
            {error && <div className="learn-error">{error}</div>}
          </div>
        </section>
      </main>
    </div>
  );
}

function TeacherRowHeader({
  detail,
  onDelete,
}: {
  detail: RunDetail;
  onDelete: () => void;
}) {
  const { manifest, progress } = detail;
  return (
    <>
      <div className="teacher-row-header">
        <span className={`task-pill task-${manifest.task}`}>
          {manifest.task}
        </span>
        <span className="teacher-row-prompt">{manifest.prompt}</span>
        <button
          type="button"
          className="row-delete"
          title="Delete this Teacher"
          onClick={(e) => {
            e.stopPropagation();
            if (confirm(`Delete Teacher "${manifest.prompt}"?`)) onDelete();
          }}
        >
          ×
        </button>
      </div>
      <div className="teacher-row-meta">
        <span className={`status-pill status-${manifest.status}`}>
          {manifest.status}
        </span>
        {/* Review-status pill — only meaningful once the run is
            completed (a still-running Teacher has nothing to review).
            Hide it for non-terminal states to keep the sidebar quiet. */}
        {manifest.status === "completed" && (
          <ReviewStatusPill
            status={manifest.review_status}
            approvedAt={manifest.approved_at}
            compact
          />
        )}
        <span className="teacher-row-time mono">
          {manifest.started_at.replace("T", " ").replace("Z", "")}
        </span>
      </div>
      {(manifest.status === "running" || manifest.status === "queued") &&
        progress && <RowProgress progress={progress} />}
    </>
  );
}

function RowProgress({ progress }: { progress: RunProgress }) {
  const pct =
    progress.total_frames > 0
      ? Math.min(
          100,
          Math.round((progress.current_frame / progress.total_frames) * 100),
        )
      : 0;
  const indeterminate =
    progress.stage !== "running" || progress.total_frames === 0;
  return (
    <div className={`row-progress ${indeterminate ? "indeterminate" : ""}`}>
      <div
        className="row-progress-fill"
        style={{ width: indeterminate ? undefined : `${pct}%` }}
      />
      <span className="row-progress-label">
        {progress.stage.replace("_", " ")}
        {progress.total_frames > 0 && progress.stage === "running"
          ? ` · ${progress.current_frame}/${progress.total_frames}`
          : ""}
      </span>
    </div>
  );
}

/**
 * The expanded body of the active sidebar row. Models, progress message,
 * stats, action buttons. This used to live in a separate SelectedTeacher
 * card in the main scroll area; pulling it inline keeps the running state
 * visible without scrolling.
 */
function TeacherRowDetail({
  detail,
  onInspect,
  onStartStudent,
}: {
  detail: RunDetail;
  onInspect: () => void;
  onStartStudent: () => void;
}) {
  const { manifest, stats, progress } = detail;
  const stopPropagation = (e: React.MouseEvent) => e.stopPropagation();
  return (
    <div className="teacher-row-detail" onClick={stopPropagation}>
      <dl className="detail-grid">
        <dt>Run ID</dt>
        <dd className="mono">{manifest.id}</dd>
        <dt>Models</dt>
        <dd className="mono detail-models">
          {Object.entries(manifest.models).length === 0 ? (
            <span className="muted">(loading…)</span>
          ) : (
            Object.entries(manifest.models)
              .map(([k, v]) => `${k}=${v}`)
              .join("  ·  ")
          )}
        </dd>
        {(manifest.status === "running" || manifest.status === "queued") &&
          progress && (
            <>
              <dt>Stage</dt>
              <dd className="detail-stage">
                <span className={`stage-pill stage-${progress.stage}`}>
                  {progress.stage.replace("_", " ")}
                </span>
                <span className="detail-stage-msg">{progress.message}</span>
              </dd>
            </>
          )}
        {stats && stats.frames_processed > 0 && (
          <>
            <dt>Frames</dt>
            <dd>
              {stats.frames_processed} processed ·{" "}
              {stats.frames_with_detections} with detections ·{" "}
              {stats.n_detections_total} dets
            </dd>
            <dt>Time / frame</dt>
            <dd>
              avg {stats.avg_ms_per_frame.toFixed(1)} ms · p50{" "}
              {stats.p50_ms_per_frame.toFixed(1)} · p95{" "}
              {stats.p95_ms_per_frame.toFixed(1)}
            </dd>
          </>
        )}
        {manifest.error && (
          <>
            <dt>Error</dt>
            <dd className="error-text">{manifest.error}</dd>
          </>
        )}
      </dl>

      {manifest.status === "completed" &&
        stats &&
        stats.frames_with_detections === 0 && (
          <div className="result-hint">
            <strong>No detections were kept.</strong> Try lowering the
            <em> Box threshold</em> in Advanced (0.15–0.20 for small objects)
            or sharpening the prompt — e.g. <code>"white black soccer ball"</code>.
          </div>
        )}

      {manifest.status === "completed" && (
        <video
          key={manifest.id}
          src={runOverlayUrl(manifest.id)}
          controls
          className="overlay-video sidebar-video"
        />
      )}

      <div className="row-detail-actions">
        {manifest.status === "completed" && (
          <>
            <button
              type="button"
              className="secondary-button"
              onClick={onInspect}
            >
              Open Inspector
            </button>
            <button
              type="button"
              className="run-button"
              onClick={onStartStudent}
            >
              Start a Student →
            </button>
          </>
        )}
      </div>
    </div>
  );
}

/**
 * Multi-chip prompt input. Each chip is ONE user-intended class and
 * arrives at the GroundingDINO adapter atomically — no silent splitting
 * on whitespace, no confusion between "soccer" and "soccer ball".
 *
 * Interactions:
 *   - Enter or comma commits the current draft as a new chip
 *   - Backspace on an empty draft removes the last chip
 *   - Click the × on a chip to remove it
 *   - Pasting a string with commas splits into multiple chips
 *
 * Whitespace inside a chip is preserved on purpose — "soccer ball" and
 * "white black ball" are valid single phrases.
 */
function PromptChips({
  chips,
  onChange,
}: {
  chips: string[];
  onChange: (next: string[]) => void;
}) {
  const [draft, setDraft] = useState("");

  const commitDraft = (raw?: string) => {
    const text = (raw ?? draft).trim();
    if (!text) return;
    // Allow comma-separated paste in one go.
    const parts = text
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s.length > 0 && !chips.includes(s));
    if (parts.length === 0) {
      setDraft("");
      return;
    }
    onChange([...chips, ...parts]);
    setDraft("");
  };

  const removeAt = (idx: number) => {
    const next = chips.slice();
    next.splice(idx, 1);
    onChange(next);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      commitDraft();
    } else if (e.key === "Backspace" && draft === "" && chips.length > 0) {
      e.preventDefault();
      removeAt(chips.length - 1);
    }
  };

  return (
    <div className="prompt-chips">
      {chips.map((chip, i) => (
        <span key={`${chip}-${i}`} className="prompt-chip">
          <span className="prompt-chip-text">{chip}</span>
          <button
            type="button"
            className="prompt-chip-remove"
            onClick={() => removeAt(i)}
            title={`Remove "${chip}"`}
            aria-label={`Remove ${chip}`}
          >
            ×
          </button>
        </span>
      ))}
      <input
        type="text"
        className="prompt-chip-input"
        value={draft}
        placeholder={
          chips.length === 0
            ? "Type and hit Enter to add a search term"
            : "+ add another"
        }
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={onKeyDown}
        onBlur={() => commitDraft()}
      />
    </div>
  );
}
