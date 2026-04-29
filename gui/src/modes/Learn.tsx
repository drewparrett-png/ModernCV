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

import { useMemo } from "react";
import { useStore } from "../store";
import type { RunDetail, RunProgress } from "../types";
import { runOverlayUrl } from "../api";
import { VideoTreePicker } from "../components/VideoTreePicker";

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
  const project = useStore((s) => s.getCurrentProject());

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
            Pick a video. The Teacher will look for{" "}
            <strong>{project ? project.prompts.join(", ") : "…"}</strong>{" "}
            across every frame and save the labels.
          </p>

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
        {manifest.status === "completed" &&
          manifest.review_status !== "unreviewed" && (
            <span className={`curation-pill curation-${manifest.review_status}`}>
              {manifest.review_status === "approved"
                ? "Fully curated"
                : `${manifest.n_frames_reviewed}/${manifest.n_frames_total} reviewed`}
            </span>
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
        {manifest.status === "completed" &&
          manifest.review_status !== "unreviewed" && (
            <>
              <dt>Curation</dt>
              <dd>
                {manifest.review_status === "approved"
                  ? "Fully curated"
                  : `${manifest.n_frames_reviewed} / ${manifest.n_frames_total} frames reviewed`}
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
            <em> display threshold</em> in the inspector (down to 0.05),
            or sharpening the prompt — e.g.{" "}
            <code>"white black soccer ball"</code>.
          </div>
        )}

      {manifest.status === "completed" && (
        <video
          key={manifest.id}
          src={runOverlayUrl(manifest.project_id, manifest.id)}
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

