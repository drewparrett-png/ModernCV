/**
 * Optimize mode — sidebar of Student runs + main panel.
 *
 * Mirrors the Learn structure: left sidebar lists every Student on disk
 * with live progress; main panel hosts a "New Student" form (pick Train +
 * Eval teachers, tweak the toolchain) plus the "Selected" view.
 *
 * The trainer is stubbed this turn — clicking Start kicks off a daemon
 * that writes a real Student manifest, sleeps briefly to simulate work,
 * then marks failed with "trainer not yet wired". The plumbing through
 * to the trainer carries the new train + eval teacher lists, so the only
 * thing that changes when we wire Ultralytics is the body of
 * `pipeline.optimize._worker`.
 *
 * Why two columns
 * ---------------
 * Distillation accuracy is easy to fake: train on clip A, evaluate on
 * clip A, get high mAP, declare victory — the Student has just memorised
 * the training frames. Real generalisation needs *held-out* clips. The
 * left column picks Teachers that contribute to training; the right
 * picks Teachers reserved for evaluation. A Teacher can appear in both
 * (an "in-distribution sanity check") but the UI warns when this
 * happens because it inflates apparent transferability.
 */

import { useEffect, useMemo, useState } from "react";
import { useStore } from "../store";
import { implMeta } from "../blockMeta";
import type { OptimizeRequest, RunDetail, StudentDetail, Task } from "../types";

interface ToolchainStage {
  kind: "detect" | "segment" | "track";
  label: string;
  defaultImpl: string;
}

const TOOLCHAINS: Record<Task, ToolchainStage[]> = {
  detection: [
    { kind: "detect", label: "Student detector", defaultImpl: "yolov8n" },
    { kind: "track", label: "Tracker", defaultImpl: "bytetrack" },
  ],
  segmentation: [
    { kind: "detect", label: "Student detector", defaultImpl: "yolov8n" },
    { kind: "segment", label: "Student segmenter", defaultImpl: "fastsam" },
    { kind: "track", label: "Tracker", defaultImpl: "bytetrack" },
  ],
};

export function Optimize() {
  const teacherDetails = useStore((s) => s.teacherDetails);
  const studentDetails = useStore((s) => s.studentDetails);
  const selectedId = useStore((s) => s.selectedStudentId);
  const select = useStore((s) => s.selectStudent);
  const start = useStore((s) => s.startOptimize);
  const del = useStore((s) => s.deleteStudent);
  const error = useStore((s) => s.optimizeError);
  const blocks = useStore((s) => s.blocks);
  const openInspector = useStore((s) => s.openInspector);
  const loadStudents = useStore((s) => s.loadStudents);

  // Refresh on tab open — covers the case where another browser tab launched
  // a student.
  useEffect(() => {
    loadStudents().catch((e) => console.error("loadStudents failed", e));
  }, [loadStudents]);

  const completedTeachers = useMemo(
    () =>
      Object.values(teacherDetails)
        .filter((d) => d.manifest.status === "completed")
        .sort((a, b) => b.manifest.started_at.localeCompare(a.manifest.started_at)),
    [teacherDetails],
  );

  const students = useMemo(() => {
    const arr = Object.values(studentDetails);
    arr.sort((a, b) => {
      const aRunning = a.manifest.status === "running" ? 1 : 0;
      const bRunning = b.manifest.status === "running" ? 1 : 0;
      if (aRunning !== bRunning) return bRunning - aRunning;
      return b.manifest.started_at.localeCompare(a.manifest.started_at);
    });
    return arr;
  }, [studentDetails]);

  const selected = selectedId ? studentDetails[selectedId] : null;

  return (
    <div className="optimize-mode-multi">
      <aside className="students-sidebar">
        <header className="sidebar-header">
          <h2>Students</h2>
          <span className="sidebar-count">{students.length}</span>
        </header>
        {students.length === 0 && (
          <div className="sidebar-empty">No Students yet — start one →</div>
        )}
        <ul className="teachers-list">
          {students.map((d) => (
            <li
              key={d.manifest.id}
              className={`teacher-row ${selectedId === d.manifest.id ? "active" : ""}`}
              onClick={() => select(d.manifest.id)}
            >
              <StudentRow detail={d} onDelete={() => del(d.manifest.id)} />
            </li>
          ))}
        </ul>
      </aside>

      <main className="learn-main">
        <NewStudentForm
          teachers={completedTeachers}
          blocks={blocks}
          error={error}
          onStart={start}
        />
        {selected && (
          <SelectedStudent
            detail={selected}
            teacherDetails={teacherDetails}
            onInspectTeacher={(tid) => openInspector(tid)}
          />
        )}
      </main>
    </div>
  );
}

function StudentRow({
  detail,
  onDelete,
}: {
  detail: StudentDetail;
  onDelete: () => void;
}) {
  const { manifest, progress } = detail;
  const trainCount = manifest.train_teacher_ids.length;
  const evalCount = manifest.eval_teacher_ids.length;
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
          title="Delete this Student"
          onClick={(e) => {
            e.stopPropagation();
            if (confirm(`Delete Student "${manifest.prompt}"?`)) onDelete();
          }}
        >
          ×
        </button>
      </div>
      <div className="teacher-row-meta">
        <span className={`status-pill status-${manifest.status}`}>
          {manifest.status}
        </span>
        <span className="teacher-row-time mono">
          {trainCount} train{evalCount > 0 ? ` · ${evalCount} eval` : ""}
        </span>
      </div>
      {manifest.status === "running" && progress && (
        <div className="row-progress indeterminate">
          <div className="row-progress-fill" />
          <span className="row-progress-label">
            {progress.stage.replace("_", " ")}
          </span>
        </div>
      )}
    </>
  );
}

/**
 * Two-column Teacher picker.
 *
 * Each completed Teacher appears as one row spanning both columns. A row
 * carries two checkboxes: Train (left) and Eval (right). The user can:
 *   • Tick Train only        → contributes to the training set.
 *   • Tick Eval only         → held-out, used to score transferability.
 *   • Tick both              → in-distribution sanity check (warned).
 *   • Tick neither           → ignored.
 *
 * Auto-task gating: once any Train teacher is selected, Train rows for
 * teachers with a different task are disabled (the trainer can't mix
 * detection + segmentation in one Student). Eval rows are NOT disabled
 * the same way — a task-mismatched eval teacher is allowed; the trainer
 * surfaces it as a poor mAP rather than refusing.
 */
function NewStudentForm({
  teachers,
  blocks,
  error,
  onStart,
}: {
  teachers: RunDetail[];
  blocks: Record<string, string[]>;
  error: string | null;
  onStart: (req: OptimizeRequest) => Promise<string | null>;
}) {
  const [trainSet, setTrainSet] = useState<Set<string>>(new Set());
  const [evalSet, setEvalSet] = useState<Set<string>>(new Set());
  const [overrides, setOverrides] = useState<Record<string, string>>({});

  // The Student's task is fixed to the first Train teacher's task. The
  // Eval set can technically include any task (warning surfaces below);
  // the toolchain UI follows the Train task.
  const trainTeachers = useMemo(
    () => teachers.filter((t) => trainSet.has(t.manifest.id)),
    [teachers, trainSet],
  );
  const studentTask = trainTeachers[0]?.manifest.task ?? null;
  const stages = studentTask ? TOOLCHAINS[studentTask] : null;

  // Surface gotchas to the user before they hit Start.
  const overlapping = useMemo(
    () => [...trainSet].filter((id) => evalSet.has(id)),
    [trainSet, evalSet],
  );
  const taskMismatchedEval = useMemo(() => {
    if (!studentTask) return [] as string[];
    return teachers
      .filter(
        (t) =>
          evalSet.has(t.manifest.id) &&
          t.manifest.task !== studentTask,
      )
      .map((t) => t.manifest.id);
  }, [teachers, evalSet, studentTask]);

  const toggle = (
    setter: (s: Set<string>) => void,
    current: Set<string>,
    id: string,
  ) => {
    const next = new Set(current);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setter(next);
  };

  if (teachers.length === 0) {
    return (
      <section className="learn-form">
        <h3>New Student</h3>
        <p className="learn-blurb">
          No completed Teachers yet. Run something in <strong>Learn</strong>{" "}
          first — once a Teacher finishes, you can distill a Student from it
          here.
        </p>
      </section>
    );
  }

  return (
    <section className="learn-form">
      <h3>New Student</h3>
      <p className="learn-blurb">
        Pick which Teachers contribute their <strong>curated COCO labels</strong>{" "}
        to the Student's training set, and which are held out for{" "}
        <strong>transferability evaluation</strong>. Held-out Teachers test
        whether the Student generalises to clips it never saw — much more
        honest than reporting accuracy on the training data.
      </p>

      <div className="teacher-picker">
        <div className="teacher-picker-head">
          <span className="teacher-picker-col-label train">Train</span>
          <span className="teacher-picker-col-label eval">Eval (held-out)</span>
          <span className="teacher-picker-col-label teacher">Teacher</span>
        </div>
        <ul className="teacher-picker-list">
          {teachers.map((t) => {
            const id = t.manifest.id;
            const isTrain = trainSet.has(id);
            const isEval = evalSet.has(id);
            const trainDisabled =
              studentTask !== null && t.manifest.task !== studentTask && !isTrain;
            return (
              <li
                key={id}
                className={`teacher-picker-row ${isTrain ? "is-train" : ""} ${
                  isEval ? "is-eval" : ""
                }`}
              >
                <label
                  className={`teacher-picker-cell train ${
                    trainDisabled ? "disabled" : ""
                  }`}
                  title={
                    trainDisabled
                      ? `Different task (${t.manifest.task}) — can't mix tasks in one Student`
                      : "Use this Teacher's labels to train the Student"
                  }
                >
                  <input
                    type="checkbox"
                    checked={isTrain}
                    disabled={trainDisabled}
                    onChange={() => toggle(setTrainSet, trainSet, id)}
                  />
                </label>
                <label
                  className="teacher-picker-cell eval"
                  title="Hold this Teacher out — Student is scored against it but never trains on it"
                >
                  <input
                    type="checkbox"
                    checked={isEval}
                    onChange={() => toggle(setEvalSet, evalSet, id)}
                  />
                </label>
                <div className="teacher-picker-meta">
                  <div className="teacher-picker-row-top">
                    <span className={`task-pill task-${t.manifest.task}`}>
                      {t.manifest.task}
                    </span>
                    <span className="teacher-picker-prompt">
                      {t.manifest.prompt}
                    </span>
                  </div>
                  <div className="teacher-picker-row-sub mono">{id}</div>
                </div>
              </li>
            );
          })}
        </ul>
        <div className="teacher-picker-summary">
          <strong>{trainSet.size}</strong> train ·{" "}
          <strong>{evalSet.size}</strong> eval
          {overlapping.length > 0 && (
            <span className="teacher-picker-warn">
              ⚠ {overlapping.length} Teacher{overlapping.length === 1 ? "" : "s"}{" "}
              in both columns — accuracy on those won't reflect generalisation
            </span>
          )}
          {taskMismatchedEval.length > 0 && (
            <span className="teacher-picker-warn">
              ⚠ {taskMismatchedEval.length} eval Teacher
              {taskMismatchedEval.length === 1 ? "" : "s"} have a different
              task — mAP will be ~0
            </span>
          )}
        </div>
      </div>

      {stages && (
        <div className="toolchain-rows">
          {stages.map((stage) => {
            const available = blocks[stage.kind] ?? [];
            const current = overrides[stage.kind] ?? stage.defaultImpl;
            const meta = implMeta(stage.kind, current);
            return (
              <div key={stage.kind} className="toolchain-row">
                <div className="toolchain-stage">
                  <div className="toolchain-stage-label">{stage.label}</div>
                  <div className="toolchain-stage-kind">{stage.kind}</div>
                </div>
                <select
                  value={current}
                  onChange={(e) =>
                    setOverrides((o) => ({
                      ...o,
                      [stage.kind]: e.target.value,
                    }))
                  }
                >
                  {available.length === 0 && (
                    <option value={current}>{current}</option>
                  )}
                  {available.map((id) => (
                    <option key={id} value={id}>
                      {implMeta(stage.kind, id).label}
                    </option>
                  ))}
                </select>
                <div className="toolchain-impl-help">{meta.description}</div>
              </div>
            );
          })}
        </div>
      )}

      <div className="run-row">
        <button
          type="button"
          className="run-button"
          disabled={trainSet.size === 0}
          onClick={async () => {
            const id = await onStart({
              train_teacher_ids: [...trainSet],
              eval_teacher_ids: [...evalSet],
              detect_impl: overrides["detect"],
              segment_impl: overrides["segment"],
              track_impl: overrides["track"],
            });
            // On success, clear the form so the user gets a clean slate
            // for the next Student. Keep selections on failure so they
            // don't have to re-pick after fixing the error.
            if (id) {
              setTrainSet(new Set());
              setEvalSet(new Set());
              setOverrides({});
            }
          }}
        >
          Start Student
        </button>
        {trainSet.size === 0 && (
          <span className="learn-hint">Select at least one Train teacher</span>
        )}
        {error && <div className="learn-error">{error}</div>}
      </div>
    </section>
  );
}

function SelectedStudent({
  detail,
  teacherDetails,
  onInspectTeacher,
}: {
  detail: StudentDetail;
  teacherDetails: Record<string, RunDetail>;
  onInspectTeacher: (teacherId: string) => void;
}) {
  const { manifest, stats, progress } = detail;
  const trainTeachers = manifest.train_teacher_ids;
  const evalTeachers = manifest.eval_teacher_ids;
  return (
    <section className="learn-result">
      <h3>{manifest.prompt}</h3>
      <div className="result-card">
        <div className="result-row">
          <span className="result-key">Run ID</span>
          <span className="result-value mono">{manifest.id}</span>
        </div>
        <div className="result-row">
          <span className="result-key">Train Teachers</span>
          <span className="result-value">
            <TeacherChips
              ids={trainTeachers}
              teacherDetails={teacherDetails}
              onInspect={onInspectTeacher}
            />
          </span>
        </div>
        <div className="result-row">
          <span className="result-key">Eval Teachers</span>
          <span className="result-value">
            {evalTeachers.length > 0 ? (
              <TeacherChips
                ids={evalTeachers}
                teacherDetails={teacherDetails}
                onInspect={onInspectTeacher}
              />
            ) : (
              <span className="muted">— none (no transferability score)</span>
            )}
          </span>
        </div>
        <div className="result-row">
          <span className="result-key">Status</span>
          <span className={`result-value status-${manifest.status}`}>
            {manifest.status}
          </span>
        </div>
        <div className="result-row">
          <span className="result-key">Models</span>
          <span className="result-value mono">
            {Object.entries(manifest.models)
              .map(([k, v]) => `${k}=${v}`)
              .join("  ·  ")}
          </span>
        </div>
        {stats && stats.epochs > 0 && (
          <>
            <div className="result-row">
              <span className="result-key">Trained on</span>
              <span className="result-value">
                {stats.train_images} images · {stats.train_annotations}{" "}
                annotations · {stats.epochs} epochs ·{" "}
                {stats.train_seconds.toFixed(1)}s
              </span>
            </div>
            <div className="result-row">
              <span className="result-key">
                {stats.per_eval_teacher.length > 0
                  ? "Mean mAP @ 0.5"
                  : "mAP @ 0.5"}
              </span>
              <span className="result-value">
                {stats.map50.toFixed(3)}
                {stats.per_eval_teacher.length > 0 && (
                  <span className="muted">
                    {" "}
                    · {stats.map50_95.toFixed(3)} @ 0.5:0.95
                  </span>
                )}
              </span>
            </div>
            <div className="result-row">
              <span className="result-key">Inference latency</span>
              <span className="result-value">
                avg {stats.avg_inference_ms.toFixed(1)} ms · p95{" "}
                {stats.p95_inference_ms.toFixed(1)} ms
              </span>
            </div>
            <div className="result-row">
              <span className="result-key">Model size</span>
              <span className="result-value">
                {stats.model_size_mb.toFixed(1)} MB
              </span>
            </div>
          </>
        )}
        {manifest.error && (
          <div className="result-row">
            <span className="result-key">Error</span>
            <span className="result-value error">{manifest.error}</span>
          </div>
        )}
      </div>

      {stats && stats.per_eval_teacher.length > 0 && (
        <PerEvalTeacherTable
          rows={stats.per_eval_teacher}
          teacherDetails={teacherDetails}
          onInspect={onInspectTeacher}
        />
      )}

      {manifest.status === "running" && progress && (
        <div className="progress-card">
          <div className="progress-row">
            <span className={`stage-pill stage-${progress.stage}`}>
              {progress.stage.replace("_", " ")}
            </span>
            <span className="progress-message">{progress.message}</span>
          </div>
          <div className="progress-bar indeterminate">
            <div className="progress-bar-fill" />
          </div>
        </div>
      )}
    </section>
  );
}

/**
 * Transferability table — one row per held-out eval teacher.
 *
 * Each row shows the eval teacher (clickable → Inspector), how big its
 * eval set was, and the Student's mAP on it. mAP cells are colour-coded
 * so the "great on A, falls apart on B" pattern jumps out visually:
 *   • ≥ 0.7  green  (transferred well)
 *   • 0.4-0.7 amber (acceptable; may benefit from more training data)
 *   • < 0.4  red    (poor transfer; clip is out of distribution)
 *
 * The thresholds are obviously rough — for a learning project they're
 * "good enough to read at a glance"; tune later if it stops being useful.
 */
function PerEvalTeacherTable({
  rows,
  teacherDetails,
  onInspect,
}: {
  rows: import("../types").PerEvalTeacherStat[];
  teacherDetails: Record<string, RunDetail>;
  onInspect: (id: string) => void;
}) {
  const mapClass = (m: number): string => {
    if (m >= 0.7) return "map-good";
    if (m >= 0.4) return "map-okay";
    return "map-poor";
  };
  return (
    <div className="per-eval-table">
      <div className="per-eval-table-head">
        <span>Held-out Teacher</span>
        <span>Images</span>
        <span>mAP @ 0.5</span>
        <span>mAP @ 0.5:0.95</span>
      </div>
      {rows.map((r) => {
        const t = teacherDetails[r.teacher_id];
        const label = t?.manifest.prompt ?? r.teacher_id.replace("teacher_", "");
        return (
          <div key={r.teacher_id} className="per-eval-row">
            <button
              type="button"
              className="teacher-chip"
              title={`Inspect ${r.teacher_id}`}
              onClick={() => onInspect(r.teacher_id)}
            >
              {label}
            </button>
            <span className="per-eval-num">{r.n_images}</span>
            <span className={`per-eval-num ${mapClass(r.map50)}`}>
              {r.error ? "—" : r.map50.toFixed(3)}
            </span>
            <span className={`per-eval-num ${mapClass(r.map50_95)}`}>
              {r.error ? "—" : r.map50_95.toFixed(3)}
            </span>
            {r.error && (
              <div className="per-eval-error">⚠ {r.error}</div>
            )}
          </div>
        );
      })}
    </div>
  );
}

/** Renders a list of teacher IDs as clickable chips. Click → open in
 *  Inspector. Falls back to plain ID if the teacher isn't loaded
 *  client-side (e.g. user just opened the tab). */
function TeacherChips({
  ids,
  teacherDetails,
  onInspect,
}: {
  ids: string[];
  teacherDetails: Record<string, RunDetail>;
  onInspect: (id: string) => void;
}) {
  return (
    <div className="teacher-chip-row">
      {ids.map((id) => {
        const t = teacherDetails[id];
        const label = t?.manifest.prompt ?? id.replace("teacher_", "");
        return (
          <button
            key={id}
            type="button"
            className="teacher-chip"
            title={`Inspect ${id}`}
            onClick={() => onInspect(id)}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}
