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

import { useEffect, useMemo, useRef, useState } from "react";
import { useStore } from "../store";
import { implMeta } from "../blockMeta";
import { previewBuckets } from "../api";
import type {
  OptimizeRequest,
  PreviewBucketsResponse,
  RunDetail,
  StudentDetail,
  Task,
} from "../types";
import { Compare } from "./Compare";

// Confidence-band defaults — keep in lockstep with `OptimizeRequest`'s
// backend defaults (`server/schemas.py`) so the form's initial submission
// is a no-op against the backend's default behaviour.
const DEFAULT_T_HIGH = 0.35;
const DEFAULT_T_LOW = 0.15;
const PREVIEW_DEBOUNCE_MS = 250;

// Architecture default (Phase 1.4) — mirrors the only architecture that
// existed before the dispatcher refactor. The fetched `architectures`
// list from the store may be empty briefly on mount; the dropdown falls
// back to this single-entry list so the form is still usable.
const DEFAULT_ARCHITECTURE = "yolov8n";

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
  const optimizeTab = useStore((s) => s.optimizeTab);
  const setOptimizeTab = useStore((s) => s.setOptimizeTab);

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

  // Compare tab is gated on having ≥2 *completed* students. Showing it
  // earlier produces a useless empty Compare panel — better to lock the
  // tab and hint the user.
  const completedCount = useMemo(
    () => students.filter((d) => d.manifest.status === "completed").length,
    [students],
  );
  const compareLocked = completedCount < 2;

  const selected = selectedId ? studentDetails[selectedId] : null;

  return (
    <div className="optimize-tab-shell">
      <nav className="optimize-tabs">
        <button
          type="button"
          className={`optimize-tab ${optimizeTab === "new" ? "active" : ""}`}
          onClick={() => setOptimizeTab("new")}
        >
          New / Inspect
        </button>
        <button
          type="button"
          className={`optimize-tab ${optimizeTab === "compare" ? "active" : ""} ${
            compareLocked ? "locked" : ""
          }`}
          onClick={() => !compareLocked && setOptimizeTab("compare")}
          disabled={compareLocked}
          title={
            compareLocked
              ? "Compare needs at least 2 completed Students"
              : "Compare 2+ completed Students side-by-side"
          }
        >
          Compare
          {compareLocked && (
            <span className="optimize-tab-hint">— need ≥ 2 completed</span>
          )}
        </button>
      </nav>

      {optimizeTab === "new" && (
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
      )}

      {optimizeTab === "compare" && !compareLocked && (
        <Compare
          studentDetails={studentDetails}
          teacherDetails={teacherDetails}
          onInspectTeacher={(tid) => openInspector(tid)}
        />
      )}
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

  // Architecture selector (Phase 1.4). Default kept in lockstep with the
  // backend's `OptimizeRequest.architecture` default ("yolov8n") so the
  // form's initial submission produces the same Student as the pre-1.4
  // code path.
  const [architecture, setArchitecture] = useState<string>(DEFAULT_ARCHITECTURE);
  const architectures = useStore((s) => s.architectures);
  const projectId = useStore((s) => s.currentProjectId);

  // Confidence-band knobs (Phase 0.4). Local string state for the inputs
  // so the user can type a partial value (e.g. "0.") without React
  // immediately snapping it back to a number — we coerce on commit.
  const [tHigh, setTHigh] = useState<number>(DEFAULT_T_HIGH);
  const [tLow, setTLow] = useState<number>(DEFAULT_T_LOW);
  const [treatEmptyAsNegative, setTreatEmptyAsNegative] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  // Preview state — kept separate from the form values so a stale request
  // (slow network) doesn't overwrite a fresher one. We compare seq numbers
  // before committing.
  const [preview, setPreview] = useState<PreviewBucketsResponse | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const previewSeqRef = useRef(0);

  const thresholdsValid = tLow <= tHigh;

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

  // Debounced live preview. Re-fires whenever the train teacher set or the
  // threshold knobs change. We bail early when:
  //   • no Train teachers selected     → preview = null (empty state below)
  //   • thresholds are invalid (t_low > t_high) → don't waste a 422 round-trip
  // The seq guard prevents an in-flight slow response from clobbering a
  // fresher result, which matters once teacher COCOs get large.
  useEffect(() => {
    const trainIds = [...trainSet];
    if (trainIds.length === 0 || !projectId) {
      setPreview(null);
      setPreviewError(null);
      return;
    }
    if (!thresholdsValid) {
      // Don't ping the server when we know it'll 422; the form is going
      // to reject Start anyway.
      setPreviewError(null);
      return;
    }
    const seq = ++previewSeqRef.current;
    const handle = setTimeout(() => {
      previewBuckets(projectId, {
        teacher_ids: trainIds,
        t_high: tHigh,
        t_low: tLow,
        treat_empty_as_negative: treatEmptyAsNegative,
      })
        .then((res) => {
          if (seq !== previewSeqRef.current) return;  // a newer call superseded us
          setPreview(res);
          setPreviewError(null);
        })
        .catch((err: unknown) => {
          if (seq !== previewSeqRef.current) return;
          setPreviewError(
            err instanceof Error ? err.message : "preview failed",
          );
          setPreview(null);
        });
    }, PREVIEW_DEBOUNCE_MS);
    return () => clearTimeout(handle);
  }, [projectId, trainSet, tHigh, tLow, treatEmptyAsNegative, thresholdsValid]);

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

      <BucketPreviewLine
        preview={preview}
        previewError={previewError}
        trainTeacherCount={trainSet.size}
        thresholdsValid={thresholdsValid}
      />

      <AdvancedThresholdPanel
        open={advancedOpen}
        onToggle={() => setAdvancedOpen((v) => !v)}
        tHigh={tHigh}
        tLow={tLow}
        treatEmptyAsNegative={treatEmptyAsNegative}
        onTHighChange={setTHigh}
        onTLowChange={setTLow}
        onTreatEmptyAsNegativeChange={setTreatEmptyAsNegative}
      />

      {/* Architecture selector (Phase 1.4). Lives in its own toolchain-row
          block so it visually matches the detect/track rows below. The
          options come from `GET /students/architectures`; if that fetch
          hasn't resolved yet (or failed) we fall back to the single-entry
          [yolov8n] list so the form is usable in the worst case. */}
      <div className="toolchain-rows">
        <div className="toolchain-row">
          <div className="toolchain-stage">
            <div className="toolchain-stage-label">Architecture</div>
            <div className="toolchain-stage-kind">trainer</div>
          </div>
          <select
            value={architecture}
            onChange={(e) => setArchitecture(e.target.value)}
          >
            {(architectures.length > 0
              ? architectures
              : [DEFAULT_ARCHITECTURE]
            ).map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
          <div className="toolchain-impl-help">
            Backbone + training framework used to fit the Student.
          </div>
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
          disabled={trainSet.size === 0 || !thresholdsValid}
          onClick={async () => {
            const id = await onStart({
              train_teacher_ids: [...trainSet],
              eval_teacher_ids: [...evalSet],
              detect_impl: overrides["detect"],
              segment_impl: overrides["segment"],
              track_impl: overrides["track"],
              t_high: tHigh,
              t_low: tLow,
              treat_empty_as_negative: treatEmptyAsNegative,
              architecture,
            });
            // On success, clear the form so the user gets a clean slate
            // for the next Student. Keep selections on failure so they
            // don't have to re-pick after fixing the error. Thresholds
            // also reset to defaults so the next Student doesn't quietly
            // inherit a tweaked t_high.
            if (id) {
              setTrainSet(new Set());
              setEvalSet(new Set());
              setOverrides({});
              setTHigh(DEFAULT_T_HIGH);
              setTLow(DEFAULT_T_LOW);
              setTreatEmptyAsNegative(false);
              setAdvancedOpen(false);
              setArchitecture(DEFAULT_ARCHITECTURE);
            }
          }}
        >
          Start Student
        </button>
        {trainSet.size === 0 && (
          <span className="learn-hint">Select at least one Train teacher</span>
        )}
        {!thresholdsValid && (
          <span className="learn-hint learn-hint-warn">
            t_low must be ≤ t_high
          </span>
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
        {/* Architecture row (Phase 1.4). Old manifests predate this field;
            fall back to "yolov8n" — the only architecture that existed
            before the dispatcher refactor — so legacy runs still render a
            sensible value rather than "undefined". */}
        <div className="result-row">
          <span className="result-key">Architecture</span>
          <span className="result-value mono">
            {manifest.architecture ?? "yolov8n"}
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
            <FrameBucketsRow stats={stats} />
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

      {stats && stats.per_teacher_buckets.length > 0 && (
        <PerTrainTeacherBucketTable
          rows={stats.per_teacher_buckets}
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

/**
 * Frame buckets row on the Student detail card (Phase 0.5).
 *
 *   Frame buckets   432 positive · 87 uncertain (dropped) · 156 true negatives
 *                                                            (t_high=0.35, t_low=0.15)
 *
 * Old runs (pre-Phase-0) come back from the backend with all the bucket
 * counts defaulted to 0 and `per_teacher_buckets = []`. Render an em-dash
 * placeholder rather than a row of zeros — zeros would look like "we
 * trained on nothing" which is wrong.
 *
 * "Run was completed but had no positive/uncertain/negative frames" is
 * not a state Phase 0 can produce (the trainer would have failed at
 * `prepare_yolo_dataset`), so we treat all-zero as "legacy run, no
 * breakdown captured" rather than "ran but produced 0 frames".
 */
function FrameBucketsRow({
  stats,
}: {
  stats: import("../types").StudentStats;
}) {
  const hasBreakdown =
    stats.n_positive_frames > 0 ||
    stats.n_uncertain_dropped > 0 ||
    stats.n_true_negative_frames > 0 ||
    stats.per_teacher_buckets.length > 0;
  if (!hasBreakdown) {
    return (
      <div className="result-row">
        <span className="result-key">Frame buckets</span>
        <span className="result-value muted">— (legacy run, no breakdown)</span>
      </div>
    );
  }
  return (
    <div className="result-row">
      <span className="result-key">Frame buckets</span>
      <span className="result-value">
        {stats.n_positive_frames} positive ·{" "}
        {stats.n_uncertain_dropped} uncertain (dropped) ·{" "}
        {stats.n_true_negative_frames} true negatives{" "}
        <span className="muted">
          (t_high={stats.t_high.toFixed(2)}, t_low={stats.t_low.toFixed(2)}
          {stats.treat_empty_as_negative ? ", treat_empty_as_negative" : ""})
        </span>
      </span>
    </div>
  );
}

/**
 * Per-train-teacher bucket table (Phase 0.5).
 *
 * Mirrors `PerEvalTeacherTable`'s row shape so the eye doesn't have to
 * re-learn the layout. One row per Train teacher with that teacher's
 * three bucket counts — useful for spotting "Teacher A is dominating
 * the positive frames" or "Teacher B is contributing nothing but
 * uncertain ones (too noisy, raise its t_high?)".
 */
function PerTrainTeacherBucketTable({
  rows,
  teacherDetails,
  onInspect,
}: {
  rows: import("../types").PerTrainTeacherBucket[];
  teacherDetails: Record<string, RunDetail>;
  onInspect: (id: string) => void;
}) {
  return (
    <div className="per-eval-table per-train-bucket-table">
      <div className="per-eval-table-head per-train-bucket-head">
        <span>Train Teacher</span>
        <span>Positive</span>
        <span>Uncertain</span>
        <span>True neg.</span>
      </div>
      {rows.map((r) => {
        const t = teacherDetails[r.teacher_id];
        const label = t?.manifest.prompt ?? r.teacher_id.replace("teacher_", "");
        return (
          <div key={r.teacher_id} className="per-eval-row per-train-bucket-row">
            <button
              type="button"
              className="teacher-chip"
              title={`Inspect ${r.teacher_id}`}
              onClick={() => onInspect(r.teacher_id)}
            >
              {label}
            </button>
            <span className="per-eval-num">{r.positive}</span>
            <span className="per-eval-num">{r.uncertain}</span>
            <span className="per-eval-num">{r.true_negative}</span>
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

/**
 * One-line summary of what the trainer would extract right now.
 *
 *   Training data preview — 432 positive · 87 uncertain (excluded) · 156 true negatives · 12 classes from 3 teachers
 *
 * Empty state when no Train teacher is selected; error state when the
 * preview request fails (rare — only on backend bugs since validation
 * is mirrored client-side).
 */
function BucketPreviewLine({
  preview,
  previewError,
  trainTeacherCount,
  thresholdsValid,
}: {
  preview: PreviewBucketsResponse | null;
  previewError: string | null;
  trainTeacherCount: number;
  thresholdsValid: boolean;
}) {
  if (trainTeacherCount === 0) {
    return (
      <div className="bucket-preview bucket-preview-empty">
        <strong>Training data preview</strong> — pick at least one Train
        teacher to see what the trainer will extract.
      </div>
    );
  }
  if (!thresholdsValid) {
    return (
      <div className="bucket-preview bucket-preview-error">
        <strong>Training data preview</strong> — t_low must be ≤ t_high.
      </div>
    );
  }
  if (previewError) {
    return (
      <div className="bucket-preview bucket-preview-error">
        <strong>Training data preview</strong> — failed: {previewError}
      </div>
    );
  }
  if (!preview) {
    return (
      <div className="bucket-preview bucket-preview-loading">
        <strong>Training data preview</strong> — calculating…
      </div>
    );
  }
  const a = preview.aggregate;
  return (
    <div className="bucket-preview">
      <strong>Training data preview</strong> — {a.positive} positive ·{" "}
      {a.uncertain} uncertain (excluded) · {a.true_negative} true negatives ·{" "}
      {a.n_classes} {a.n_classes === 1 ? "class" : "classes"} from{" "}
      {trainTeacherCount} {trainTeacherCount === 1 ? "teacher" : "teachers"}
    </div>
  );
}

/**
 * Collapsible "Advanced" panel — surfaces the three confidence-band knobs.
 *
 * Default-collapsed because most users will run with the spec defaults
 * (t_high=0.35, t_low=0.15). Opening it reveals a vertical stack of
 * input rows; each one carries a one-sentence description sourced from
 * the spec (`docs/student-training.md` Phase 0.4) so the user doesn't
 * need a separate doc tab to know what they're doing.
 */
function AdvancedThresholdPanel({
  open,
  onToggle,
  tHigh,
  tLow,
  treatEmptyAsNegative,
  onTHighChange,
  onTLowChange,
  onTreatEmptyAsNegativeChange,
}: {
  open: boolean;
  onToggle: () => void;
  tHigh: number;
  tLow: number;
  treatEmptyAsNegative: boolean;
  onTHighChange: (v: number) => void;
  onTLowChange: (v: number) => void;
  onTreatEmptyAsNegativeChange: (v: boolean) => void;
}) {
  // Number inputs are tricky — we want to allow intermediate states like
  // "0." while typing without snapping to NaN. Strategy: hold a string in
  // local state, parse on commit, fall back to the previous value if the
  // user clears the field.
  return (
    <div className="advanced-panel">
      <button
        type="button"
        className="advanced-toggle"
        aria-expanded={open}
        onClick={onToggle}
      >
        <span className="advanced-toggle-caret">{open ? "▾" : "▸"}</span>
        Advanced
        {!open && (
          <span className="advanced-toggle-summary">
            t_high={tHigh.toFixed(2)} · t_low={tLow.toFixed(2)}
            {treatEmptyAsNegative ? " · treat_empty_as_negative" : ""}
          </span>
        )}
      </button>
      {open && (
        <div className="advanced-panel-body">
          <div className="advanced-row">
            <label className="advanced-row-label">
              <span className="advanced-row-name mono">t_high</span>
              <input
                type="number"
                step={0.05}
                min={0}
                max={1}
                value={tHigh}
                onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  if (!Number.isNaN(v)) onTHighChange(v);
                }}
              />
            </label>
            <span className="advanced-row-help">
              Detections at or above this confidence become labels.
            </span>
          </div>
          <div className="advanced-row">
            <label className="advanced-row-label">
              <span className="advanced-row-name mono">t_low</span>
              <input
                type="number"
                step={0.05}
                min={0}
                max={1}
                value={tLow}
                onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  if (!Number.isNaN(v)) onTLowChange(v);
                }}
                aria-invalid={tLow > tHigh}
              />
            </label>
            <span className="advanced-row-help">
              Frames with detections only between t_low and t_high are dropped
              (teacher was unsure).
            </span>
          </div>
          <div className="advanced-row">
            <label className="advanced-row-label">
              <input
                type="checkbox"
                checked={treatEmptyAsNegative}
                onChange={(e) => onTreatEmptyAsNegativeChange(e.target.checked)}
              />
              <span className="advanced-row-name mono">
                treat_empty_as_negative
              </span>
            </label>
            <span className="advanced-row-help">
              Treat every zero-detection frame as a true negative. Reproduces
              the old behaviour. Off by default.
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
