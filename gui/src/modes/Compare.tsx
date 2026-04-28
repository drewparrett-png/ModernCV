/**
 * Compare tab (Phase 3) — side-by-side analysis of completed Students.
 *
 * Lives inside Optimize as a sub-tab. Selection lives in the Zustand
 * store (`compareStudentIds`) so toggling sub-tabs doesn't blow it away.
 *
 * Empty scaffolding for Phase 3.0 — the four sections (3.1 details
 * table, 3.2 eval-teacher matrix, 3.3 Pareto plot, 3.4 comparability
 * badges) are filled in by subsequent commits.
 */

import { useMemo } from "react";
import { useStore } from "../store";
import type { RunDetail, StudentDetail } from "../types";

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

  // Eligible: completed Students with stats. Failed / running runs aren't
  // useful for comparison — they have no numbers.
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

  const selectedIds = new Set(compareIds);
  const selectedCount = compareIds.filter((id) =>
    eligible.some((d) => d.manifest.id === id),
  ).length;

  return (
    <div className="compare-shell">
      <section className="compare-selector">
        <header className="compare-selector-head">
          <h3>Pick Students</h3>
          <span className="compare-selector-count">
            {selectedCount} of {eligible.length} selected
          </span>
          {selectedCount > 0 && (
            <button
              type="button"
              className="compare-clear-btn"
              onClick={clearStudents}
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
                onClick={() => toggleStudent(id)}
              >
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => toggleStudent(id)}
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
      {selectedCount < 2 ? (
        <div className="compare-empty">
          Pick at least 2 completed Students to compare.
        </div>
      ) : (
        <div className="compare-empty">
          Compare views (table, matrix, Pareto, badges) coming up next.
        </div>
      )}
    </div>
  );
}
