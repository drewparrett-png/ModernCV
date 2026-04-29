/**
 * StudentRunPanel — Phase 5 "Run" tab inside the Selected Student card.
 *
 * Lists past student-runs for the selected Student, lets the user start a
 * new one (against a video file or a Teacher dataset), and inlines an
 * overlay preview for completed runs. Polls while any run is active so
 * progress + status updates flow without manual refresh.
 *
 * The form lives at the top, run list below it. We do NOT mount a full
 * inspector here — clicking a completed run expands it inline with the
 * overlay video and headline stats. Teacher-style frame-by-frame review
 * isn't a Phase 5 goal; the predictions/per_frame.jsonl is on disk for
 * later stages.
 */

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useStore } from "../store";
import {
  deleteStudentRun,
  fetchStudentRunDetail,
  fetchStudentRuns,
  startStudentRun,
  studentRunOverlayUrl,
} from "../api";
import type {
  RunDetail,
  StudentDetail,
  StudentRunDetail,
  StudentRunInputKind,
  StudentRunManifest,
} from "../types";

const POLL_MS = 1500;

interface Props {
  student: StudentDetail;
  teachers: RunDetail[];
}

export function StudentRunPanel({ student, teachers }: Props) {
  const projectId = useStore((s) => s.currentProjectId);
  const videos = useStore((s) => s.videos);

  const [runs, setRuns] = useState<StudentRunManifest[]>([]);
  const [details, setDetails] = useState<Record<string, StudentRunDetail>>({});
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const studentId = student.manifest.id;

  const refresh = useCallback(async () => {
    if (!projectId || !studentId) return;
    try {
      const resp = await fetchStudentRuns(projectId, studentId);
      setRuns(resp.runs);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [projectId, studentId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Poll detail for the active run(s) so the GUI's status flips live.
  // We hydrate details for every visible run on first sight, then keep
  // polling only the ones still queued/running — completed/failed runs
  // are static.
  const activeIds = useMemo(
    () => runs.filter((r) => r.status === "queued" || r.status === "running").map((r) => r.id),
    [runs],
  );

  useEffect(() => {
    if (!projectId || !studentId) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const resp = await fetchStudentRuns(projectId, studentId);
        if (cancelled) return;
        setRuns(resp.runs);
        // Hydrate details for runs we don't have yet, plus any that
        // are still in flight.
        const want = new Set<string>(activeIds);
        for (const r of resp.runs) {
          if (!details[r.id]) want.add(r.id);
        }
        for (const id of want) {
          fetchStudentRunDetail(projectId, studentId, id)
            .then((d) => {
              if (!cancelled) {
                setDetails((prev) => ({ ...prev, [id]: d }));
              }
            })
            .catch(() => {
              /* missing detail is fine, will retry next tick */
            });
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
        }
      }
    };
    // Always tick once on mount; only continue polling if there's an
    // active run.
    void tick();
    if (activeIds.length === 0) return;
    const handle = setInterval(tick, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(handle);
    };
    // `details` deliberately excluded — we only want activeIds to drive
    // re-subscribing the polling loop.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, studentId, activeIds.join("|")]);

  const onStart = useCallback(
    async (kind: StudentRunInputKind, ref: string) => {
      if (!projectId || !studentId) return;
      setError(null);
      setSubmitting(true);
      try {
        const detail = await startStudentRun(projectId, studentId, {
          input_kind: kind,
          input_ref: ref,
        });
        setRuns((prev) => [detail.manifest, ...prev]);
        setDetails((prev) => ({ ...prev, [detail.manifest.id]: detail }));
        setExpandedId(detail.manifest.id);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setSubmitting(false);
      }
    },
    [projectId, studentId],
  );

  const onDelete = useCallback(
    async (runId: string) => {
      if (!projectId || !studentId) return;
      try {
        await deleteStudentRun(projectId, studentId, runId);
        setRuns((prev) => prev.filter((r) => r.id !== runId));
        setDetails((prev) => {
          const next = { ...prev };
          delete next[runId];
          return next;
        });
        if (expandedId === runId) setExpandedId(null);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    [projectId, studentId, expandedId],
  );

  if (!projectId) return null;

  const completedTeachers = teachers.filter(
    (t) => t.manifest.status === "completed",
  );

  return (
    <section className="student-run-panel">
      <header className="student-run-header">
        <h4>Run inference</h4>
        <p className="muted">
          Run this Student against a video file or one of the project's
          Teacher datasets. mAP is reported when input is a Teacher.
        </p>
      </header>

      <NewRunForm
        videos={videos}
        teachers={completedTeachers}
        submitting={submitting}
        onStart={onStart}
        disabled={student.manifest.status !== "completed"}
      />

      {error && <div className="student-run-error">{error}</div>}

      <div className="student-run-list">
        {runs.length === 0 && (
          <div className="muted">No runs yet — pick an input above.</div>
        )}
        {runs.map((m) => {
          const d = details[m.id];
          const expanded = expandedId === m.id;
          return (
            <div
              key={m.id}
              className={`student-run-row ${expanded ? "expanded" : ""}`}
            >
              <div
                className="student-run-row-summary"
                onClick={() => setExpandedId(expanded ? null : m.id)}
                role="button"
                tabIndex={0}
              >
                <span className={`status-pill status-${m.status}`}>
                  {m.status}
                </span>
                <span className="student-run-input mono">
                  {m.input_kind === "video" ? "video" : "teacher"} · {m.input_ref}
                </span>
                <span className="muted student-run-time mono">
                  {m.started_at}
                </span>
                <button
                  type="button"
                  className="row-delete"
                  title="Delete this run"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (confirm(`Delete run ${m.id}?`)) onDelete(m.id);
                  }}
                >
                  ×
                </button>
              </div>
              {expanded && d && <RunDetailBody detail={d} />}
            </div>
          );
        })}
      </div>
    </section>
  );
}

function NewRunForm({
  videos,
  teachers,
  submitting,
  onStart,
  disabled,
}: {
  videos: string[];
  teachers: RunDetail[];
  submitting: boolean;
  onStart: (kind: StudentRunInputKind, ref: string) => void;
  disabled: boolean;
}) {
  const [kind, setKind] = useState<StudentRunInputKind>("video");
  const [video, setVideo] = useState<string>("");
  const [teacherId, setTeacherId] = useState<string>("");

  // Auto-select first available option whenever kind flips. The video
  // and teacher lists may load asynchronously, so we re-pick when they
  // arrive.
  useEffect(() => {
    if (kind === "video" && !video && videos.length > 0) {
      setVideo(videos[0]);
    }
    if (kind === "teacher_dataset" && !teacherId && teachers.length > 0) {
      setTeacherId(teachers[0].manifest.id);
    }
  }, [kind, video, teacherId, videos, teachers]);

  const ref = kind === "video" ? video : teacherId;
  const canSubmit = !disabled && !submitting && !!ref;

  return (
    <form
      className="student-run-form"
      onSubmit={(e) => {
        e.preventDefault();
        if (canSubmit) onStart(kind, ref);
      }}
    >
      <div className="student-run-kind-row">
        <label className="student-run-radio">
          <input
            type="radio"
            name="kind"
            value="video"
            checked={kind === "video"}
            onChange={() => setKind("video")}
          />
          <span>Video file</span>
        </label>
        <label className="student-run-radio">
          <input
            type="radio"
            name="kind"
            value="teacher_dataset"
            checked={kind === "teacher_dataset"}
            onChange={() => setKind("teacher_dataset")}
          />
          <span>Teacher dataset (with mAP)</span>
        </label>
      </div>

      {kind === "video" ? (
        <select
          className="student-run-picker"
          value={video}
          onChange={(e) => setVideo(e.target.value)}
          disabled={videos.length === 0}
        >
          {videos.length === 0 && <option>No videos in data/</option>}
          {videos.map((v) => (
            <option key={v} value={v}>
              {v}
            </option>
          ))}
        </select>
      ) : (
        <select
          className="student-run-picker"
          value={teacherId}
          onChange={(e) => setTeacherId(e.target.value)}
          disabled={teachers.length === 0}
        >
          {teachers.length === 0 && <option>No completed Teachers</option>}
          {teachers.map((t) => (
            <option key={t.manifest.id} value={t.manifest.id}>
              {t.manifest.prompt} ({t.manifest.id})
            </option>
          ))}
        </select>
      )}

      <button
        type="submit"
        disabled={!canSubmit}
        className="student-run-submit"
      >
        {submitting ? "Starting…" : "Start run"}
      </button>
    </form>
  );
}

function RunDetailBody({ detail }: { detail: StudentRunDetail }) {
  const projectId = useStore((s) => s.currentProjectId);
  const { manifest, stats, progress } = detail;
  const overlayRef = useRef<HTMLVideoElement | null>(null);

  return (
    <div className="student-run-body">
      <div className="student-run-meta">
        <div>
          <span className="muted">Run id</span>{" "}
          <span className="mono">{manifest.id}</span>
        </div>
        {manifest.error && (
          <div className="student-run-error">Error: {manifest.error}</div>
        )}
        {progress && manifest.status === "running" && (
          <div className="student-run-progress muted">
            {progress.stage} · frame {progress.current_frame}
            {progress.total_frames > 0 && ` / ${progress.total_frames}`}
          </div>
        )}
        {stats && (
          <div className="student-run-stats">
            <div>
              <span className="muted">Frames</span>{" "}
              <span className="mono">{stats.n_frames}</span>
            </div>
            <div>
              <span className="muted">Detections</span>{" "}
              <span className="mono">{stats.n_detections}</span>
            </div>
            <div>
              <span className="muted">avg/p50/p95 ms</span>{" "}
              <span className="mono">
                {stats.avg_inference_ms.toFixed(1)} /{" "}
                {stats.p50_inference_ms.toFixed(1)} /{" "}
                {stats.p95_inference_ms.toFixed(1)}
              </span>
            </div>
            {stats.map50 !== null && (
              <div>
                <span className="muted">mAP @ 0.5</span>{" "}
                <span className="mono">{stats.map50.toFixed(3)}</span>
                {stats.map50_95 !== null && (
                  <span className="muted">
                    {" "}· {stats.map50_95.toFixed(3)} @ 0.5:0.95
                  </span>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {manifest.status === "completed" && projectId && (
        <video
          ref={overlayRef}
          className="student-run-video"
          src={studentRunOverlayUrl(projectId, manifest.student_id, manifest.id)}
          controls
          preload="metadata"
        />
      )}
    </div>
  );
}
