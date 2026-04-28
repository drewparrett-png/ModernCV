/**
 * Compare tab (Phase 3) — side-by-side analysis of completed Students.
 *
 * Lives inside Optimize as a sub-tab. The user picks ≥2 completed
 * Students from the multi-select panel; we render the comparison views
 * below.
 *
 *   3.4  comparability badges (mismatched imgsz/device/epochs)
 *   3.1  side-by-side details table, sortable, best-per-column highlight
 *   3.2  per-eval-teacher mAP matrix, with optional shared-only filter
 *   3.3  Pareto plot (mAP vs. p50 latency) on Recharts
 *
 * Selection lives in the Zustand store (`compareStudentIds`) so toggling
 * sub-tabs doesn't blow it away. Per spec, *not* persisted across reload.
 */

import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

import { useStore } from "../store";
import type { RunDetail, StudentDetail, StudentStats } from "../types";
import {
  bestPerColumn,
  comparabilityBadges,
  evalTeacherMatrix,
  mapColourClass,
  paretoPoints,
  sharedEvalTeachers,
  type CompareColumn,
  type ParetoPoint,
} from "../lib/compare";

// ---- Architecture palette (Pareto plot) ---------------------------------

// 6 distinguishable hues; categorical, not perceptually ordered. Picked to
// stay readable on the white canvas the rest of the GUI uses. Lookup is
// arch-name-stable: same architecture always gets the same colour
// regardless of which students are selected, which makes the Pareto plot
// easier to read across re-selections.
const ARCH_PALETTE = [
  "#2563eb", // blue
  "#dc2626", // red
  "#059669", // emerald
  "#d97706", // amber
  "#7c3aed", // violet
  "#0891b2", // cyan
];

function archColour(arch: string, allArchs: string[]): string {
  const idx = allArchs.indexOf(arch);
  if (idx < 0) return ARCH_PALETTE[0];
  return ARCH_PALETTE[idx % ARCH_PALETTE.length];
}

// ---- Column definitions for the details table --------------------------

interface TableColumn extends CompareColumn {
  label: string;
  /** How the cell renders the numeric value when present. */
  format: (stats: StudentStats) => string;
}

const TABLE_COLUMNS: TableColumn[] = [
  // Non-numeric columns are direction:none → never highlighted, never sortable.
  { key: "prompt", label: "Student", direction: "none", value: () => null, format: () => "" },
  { key: "architecture", label: "Arch", direction: "none", value: () => null, format: () => "" },
  {
    key: "train_buckets",
    label: "Train (pos / unc / neg)",
    direction: "none",
    value: () => null,
    format: (s) =>
      `${s.n_positive_frames} / ${s.n_uncertain_dropped} / ${s.n_true_negative_frames}`,
  },
  {
    key: "epochs",
    label: "Epochs",
    direction: "none",
    value: (s) => s.epochs,
    format: (s) => `${s.epochs}`,
  },
  {
    key: "train_seconds",
    label: "Train time",
    direction: "min",
    value: (s) => s.train_seconds,
    format: (s) => formatSeconds(s.train_seconds),
  },
  {
    key: "map50",
    label: "Mean mAP@0.5",
    direction: "max",
    value: (s) => s.map50,
    format: (s) => s.map50.toFixed(3),
  },
  {
    key: "map50_95",
    label: "mAP@0.5:0.95",
    direction: "max",
    value: (s) => s.map50_95,
    format: (s) => s.map50_95.toFixed(3),
  },
  {
    key: "p50_inference_ms",
    label: "Latency p50 (ms)",
    direction: "min",
    value: (s) => s.p50_inference_ms,
    format: (s) => s.p50_inference_ms.toFixed(1),
  },
  {
    key: "p95_inference_ms",
    label: "Latency p95 (ms)",
    direction: "min",
    value: (s) => s.p95_inference_ms,
    format: (s) => s.p95_inference_ms.toFixed(1),
  },
  {
    key: "model_size_mb",
    label: "Size (MB)",
    direction: "min",
    value: (s) => s.model_size_mb,
    format: (s) => s.model_size_mb.toFixed(1),
  },
];

function formatSeconds(s: number): string {
  if (s < 60) return `${s.toFixed(1)}s`;
  const mins = Math.floor(s / 60);
  const rem = s - mins * 60;
  return `${mins}m ${rem.toFixed(0)}s`;
}

// ---- Component ----------------------------------------------------------

export function Compare({
  studentDetails,
  teacherDetails,
  onInspectTeacher,
}: {
  studentDetails: Record<string, StudentDetail>;
  teacherDetails: Record<string, RunDetail>;
  onInspectTeacher: (id: string) => void;
}) {
  const compareIds = useStore((s) => s.compareStudentIds);
  const toggleStudent = useStore((s) => s.toggleCompareStudent);
  const clearStudents = useStore((s) => s.clearCompareStudents);

  // Eligible: completed Students with stats (epochs > 0). Failed runs and
  // running runs aren't useful for comparison — they have no numbers.
  const eligible = useMemo(
    () =>
      Object.values(studentDetails)
        .filter(
          (d) =>
            d.manifest.status === "completed" && d.stats && d.stats.epochs > 0,
        )
        .sort((a, b) =>
          b.manifest.started_at.localeCompare(a.manifest.started_at),
        ),
    [studentDetails],
  );

  const selected = useMemo(
    () =>
      compareIds
        .map((id) => studentDetails[id])
        .filter(
          (d): d is StudentDetail =>
            !!d &&
            d.manifest.status === "completed" &&
            !!d.stats &&
            d.stats.epochs > 0,
        ),
    [compareIds, studentDetails],
  );

  return (
    <div className="compare-shell">
      <CompareSelector
        eligible={eligible}
        selectedIds={new Set(compareIds)}
        onToggle={toggleStudent}
        onClear={clearStudents}
      />
      {selected.length < 2 ? (
        <div className="compare-empty">
          Pick at least 2 completed Students to compare.
        </div>
      ) : (
        <CompareBody
          selected={selected}
          teacherDetails={teacherDetails}
          onInspectTeacher={onInspectTeacher}
        />
      )}
    </div>
  );
}

// ---- Selector -----------------------------------------------------------

function CompareSelector({
  eligible,
  selectedIds,
  onToggle,
  onClear,
}: {
  eligible: StudentDetail[];
  selectedIds: Set<string>;
  onToggle: (id: string) => void;
  onClear: () => void;
}) {
  return (
    <section className="compare-selector">
      <header className="compare-selector-head">
        <h3>Pick Students</h3>
        <span className="compare-selector-count">
          {selectedIds.size} of {eligible.length} selected
        </span>
        {selectedIds.size > 0 && (
          <button
            type="button"
            className="compare-clear-btn"
            onClick={onClear}
          >
            Clear
          </button>
        )}
      </header>
      {eligible.length === 0 && (
        <div className="compare-selector-empty">
          No completed Students yet.
        </div>
      )}
      <ul className="compare-selector-list">
        {eligible.map((d) => {
          const id = d.manifest.id;
          const checked = selectedIds.has(id);
          return (
            <li
              key={id}
              className={`compare-selector-row ${checked ? "checked" : ""}`}
              onClick={() => onToggle(id)}
            >
              <input
                type="checkbox"
                checked={checked}
                onChange={() => onToggle(id)}
                onClick={(e) => e.stopPropagation()}
              />
              <span className={`task-pill task-${d.manifest.task}`}>
                {d.manifest.task}
              </span>
              <span className="compare-selector-prompt">
                {d.manifest.prompt}
              </span>
              <span className="compare-selector-arch mono">
                {d.manifest.architecture ?? "yolov8n"}
              </span>
              <span className="compare-selector-map mono">
                mAP {d.stats?.map50.toFixed(3) ?? "—"}
              </span>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

// ---- Body (badges + table + matrix + Pareto)

function CompareBody({
  selected,
  teacherDetails,
  onInspectTeacher,
}: {
  selected: StudentDetail[];
  teacherDetails: Record<string, RunDetail>;
  onInspectTeacher: (id: string) => void;
}) {
  const studentsForHelpers = useMemo(
    () =>
      selected.map((d) => ({
        id: d.manifest.id,
        stats: d.stats as StudentStats,
      })),
    [selected],
  );

  return (
    <div className="compare-body">
      <ComparabilityBadges students={studentsForHelpers} />
      <DetailsTable selected={selected} students={studentsForHelpers} />
      <EvalTeacherMatrixView
        students={studentsForHelpers}
        selected={selected}
        teacherDetails={teacherDetails}
        onInspectTeacher={onInspectTeacher}
      />
      <ParetoPlot selected={selected} />
    </div>
  );
}

// ---- 3.4 Comparability badges ------------------------------------------

function ComparabilityBadges({
  students,
}: {
  students: { id: string; stats: StudentStats }[];
}) {
  const badges = useMemo(() => comparabilityBadges(students), [students]);
  if (badges.length === 0) return null;
  return (
    <div className="compare-badges">
      {badges.map((b) => (
        <div key={b.field} className="compare-badge">
          <span className="compare-badge-icon">⚠</span>
          <span>{b.message}</span>
        </div>
      ))}
    </div>
  );
}

// ---- 3.1 Details table -------------------------------------------------

function DetailsTable({
  selected,
  students,
}: {
  selected: StudentDetail[];
  students: { id: string; stats: StudentStats }[];
}) {
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const winners = useMemo(
    () => bestPerColumn(students, TABLE_COLUMNS),
    [students],
  );

  const sortedDetails = useMemo(() => {
    if (!sortKey) return selected;
    const col = TABLE_COLUMNS.find((c) => c.key === sortKey);
    if (!col || col.direction === "none") return selected;
    const copy = [...selected];
    copy.sort((a, b) => {
      const va = col.value(a.stats as StudentStats);
      const vb = col.value(b.stats as StudentStats);
      // Nulls sort last.
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      return sortDir === "asc" ? va - vb : vb - va;
    });
    return copy;
  }, [selected, sortKey, sortDir]);

  const handleSortClick = (col: TableColumn) => {
    if (col.direction === "none") return;
    if (sortKey === col.key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(col.key);
      // Default sort matches the column's "best" direction so the first
      // click puts the winner at the top.
      setSortDir(col.direction === "max" ? "desc" : "asc");
    }
  };

  return (
    <section className="compare-section">
      <h3>Details</h3>
      <div className="compare-table-wrap">
        <table className="compare-table">
          <thead>
            <tr>
              {TABLE_COLUMNS.map((col) => {
                const isSorted = sortKey === col.key;
                const sortable = col.direction !== "none";
                return (
                  <th
                    key={col.key}
                    className={sortable ? "sortable" : ""}
                    onClick={() => handleSortClick(col)}
                    title={sortable ? "Click to sort" : undefined}
                  >
                    <span>{col.label}</span>
                    {sortable && (
                      <span className="compare-sort-indicator">
                        {isSorted ? (sortDir === "asc" ? " ▲" : " ▼") : " "}
                      </span>
                    )}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {sortedDetails.map((d) => {
              const stats = d.stats as StudentStats;
              const id = d.manifest.id;
              return (
                <tr key={id}>
                  {TABLE_COLUMNS.map((col) => {
                    const isWinner = winners.has(`${id}::${col.key}`);
                    let content: string;
                    if (col.key === "prompt") content = d.manifest.prompt;
                    else if (col.key === "architecture")
                      content = d.manifest.architecture ?? "yolov8n";
                    else content = col.format(stats);
                    return (
                      <td
                        key={col.key}
                        className={`${
                          col.direction !== "none" ? "compare-td-num" : ""
                        } ${isWinner ? "compare-best" : ""}`}
                      >
                        {content}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

// ---- 3.2 Per-eval-teacher matrix ---------------------------------------

function EvalTeacherMatrixView({
  students,
  selected,
  teacherDetails,
  onInspectTeacher,
}: {
  students: { id: string; stats: StudentStats }[];
  selected: StudentDetail[];
  teacherDetails: Record<string, RunDetail>;
  onInspectTeacher: (id: string) => void;
}) {
  const [sharedOnly, setSharedOnly] = useState(false);
  const fullMatrix = useMemo(() => evalTeacherMatrix(students), [students]);
  const matrix = sharedOnly ? sharedEvalTeachers(fullMatrix) : fullMatrix;

  // Map student id → display label (prompt) for the header.
  const studentLabel = (id: string): string => {
    const d = selected.find((x) => x.manifest.id === id);
    return d?.manifest.prompt ?? id;
  };

  if (fullMatrix.teacherIds.length === 0) {
    return (
      <section className="compare-section">
        <h3>Per-eval-teacher mAP@0.5</h3>
        <div className="compare-empty-inline">
          None of the selected Students have eval teachers.
        </div>
      </section>
    );
  }

  return (
    <section className="compare-section">
      <header className="compare-section-head">
        <h3>Per-eval-teacher mAP@0.5</h3>
        <label className="compare-toggle">
          <input
            type="checkbox"
            checked={sharedOnly}
            onChange={(e) => setSharedOnly(e.target.checked)}
          />
          Filter to shared eval teachers
        </label>
      </header>
      <div className="compare-table-wrap">
        <table className="compare-table compare-matrix-table">
          <thead>
            <tr>
              <th>Eval teacher</th>
              {matrix.studentIds.map((sid) => (
                <th key={sid}>{studentLabel(sid)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.teacherIds.map((tid) => {
              const tDet = teacherDetails[tid];
              const tLabel = tDet?.manifest.prompt ?? tid.replace("teacher_", "");
              return (
                <tr key={tid}>
                  <td>
                    <button
                      type="button"
                      className="teacher-chip"
                      title={`Inspect ${tid}`}
                      onClick={() => onInspectTeacher(tid)}
                    >
                      {tLabel}
                    </button>
                  </td>
                  {matrix.studentIds.map((sid) => {
                    const cell = matrix.cells[tid]?.[sid];
                    if (!cell || cell.map50 == null) {
                      return (
                        <td
                          key={sid}
                          className="compare-cell-missing"
                          title="This Student didn't include this teacher in its eval set — no comparable score."
                        >
                          —
                        </td>
                      );
                    }
                    return (
                      <td
                        key={sid}
                        className={`compare-td-num ${mapColourClass(cell.map50)}`}
                      >
                        {cell.map50.toFixed(3)}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
        {sharedOnly && matrix.teacherIds.length === 0 && (
          <div className="compare-empty-inline">
            No teachers in common across the selected Students.
          </div>
        )}
      </div>
    </section>
  );
}

// ---- 3.3 Pareto plot ---------------------------------------------------

function ParetoPlot({ selected }: { selected: StudentDetail[] }) {
  const points = useMemo(
    () => paretoPoints(selected.map((d) => ({ detail: d }))),
    [selected],
  );

  // Group points by architecture so each gets its own <Scatter> layer with
  // a distinct colour and a legend entry.
  const grouped = useMemo(() => {
    const m = new Map<string, ParetoPoint[]>();
    for (const p of points) {
      const arr = m.get(p.architecture) ?? [];
      arr.push(p);
      m.set(p.architecture, arr);
    }
    return m;
  }, [points]);

  const allArchs = useMemo(() => [...grouped.keys()].sort(), [grouped]);

  if (points.length === 0) return null;

  return (
    <section className="compare-section">
      <h3>Pareto: mAP@0.5 vs. p50 latency</h3>
      <div className="compare-pareto-blurb">
        Top-left is the desirable region: high mAP, low latency. Dot size
        scales with model size (cosmetic).
      </div>
      <div className="compare-pareto-wrap">
        <ResponsiveContainer width="100%" height={360}>
          <ScatterChart margin={{ top: 16, right: 24, bottom: 48, left: 56 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              type="number"
              dataKey="p50_inference_ms"
              name="p50 latency (ms)"
              label={{
                value: "p50 latency (ms) — lower is better",
                position: "insideBottom",
                offset: -28,
                fill: "#374151",
              }}
              stroke="#6b7280"
            />
            <YAxis
              type="number"
              dataKey="map50"
              name="mAP@0.5"
              domain={[0, 1]}
              label={{
                value: "mAP@0.5 — higher is better",
                angle: -90,
                position: "insideLeft",
                offset: -4,
                fill: "#374151",
              }}
              stroke="#6b7280"
            />
            <ZAxis
              type="number"
              dataKey="model_size_mb"
              range={[60, 240]}
              name="model size (MB)"
            />
            <Tooltip
              cursor={{ strokeDasharray: "3 3" }}
              content={<ParetoTooltip />}
            />
            <Legend />
            {allArchs.map((arch) => (
              <Scatter
                key={arch}
                name={arch}
                data={grouped.get(arch) ?? []}
                fill={archColour(arch, allArchs)}
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

interface RechartsTooltipProps {
  active?: boolean;
  payload?: { payload: ParetoPoint }[];
}

function ParetoTooltip({ active, payload }: RechartsTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  const p = payload[0].payload;
  return (
    <div className="compare-pareto-tooltip">
      <div className="compare-pareto-tooltip-header">{p.prompt}</div>
      <div className="compare-pareto-tooltip-arch mono">{p.architecture}</div>
      <table>
        <tbody>
          <tr>
            <td>mAP@0.5</td>
            <td className="mono">{p.map50.toFixed(3)}</td>
          </tr>
          <tr>
            <td>mAP@0.5:0.95</td>
            <td className="mono">{p.map50_95.toFixed(3)}</td>
          </tr>
          <tr>
            <td>Latency p50/p95</td>
            <td className="mono">
              {p.p50_inference_ms.toFixed(1)} / {p.p95_inference_ms.toFixed(1)} ms
            </td>
          </tr>
          <tr>
            <td>Size</td>
            <td className="mono">{p.model_size_mb.toFixed(1)} MB</td>
          </tr>
          <tr>
            <td>Train</td>
            <td className="mono">
              {p.train_images} imgs · {formatSeconds(p.train_seconds)}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
