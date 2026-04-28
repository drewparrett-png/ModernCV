/**
 * Compare tab (Phase 3) — side-by-side analysis of completed Students.
 *
 * Lives inside Optimize as a sub-tab. The user picks ≥2 completed
 * Students from the multi-select panel; we render the comparison views
 * below.
 *
 *   3.1  side-by-side details table, sortable, best-per-column highlight
 *   3.2  per-eval-teacher mAP matrix (next commit)
 *   3.3  Pareto plot (next commit)
 *   3.4  comparability badges (next commit)
 *
 * Selection lives in the Zustand store (`compareStudentIds`) so toggling
 * sub-tabs doesn't blow it away. Per spec, *not* persisted across reload.
 */

import { useMemo, useState } from "react";

import { useStore } from "../store";
import type { RunDetail, StudentDetail, StudentStats } from "../types";
import { bestPerColumn, type CompareColumn } from "../lib/compare";

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
        <CompareBody selected={selected} />
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

// ---- Body (3.1 details table — more sections appended in subsequent commits)

function CompareBody({ selected }: { selected: StudentDetail[] }) {
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
      <DetailsTable selected={selected} students={studentsForHelpers} />
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
