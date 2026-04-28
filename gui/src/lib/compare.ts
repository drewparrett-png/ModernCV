/**
 * Pure helpers for the Compare tab (Phase 3).
 *
 * UI-free so they're easy to reason about and (eventually) easy to test.
 * The Compare component composes these and adds the React/Recharts shell.
 *
 * No JS test runner is set up in this repo yet — these helpers stay pure
 * so a runner can be added later without rewriting them. Manual smoke-
 * testing happens through the Compare tab in the GUI.
 */

import type { StudentStats } from "../types";

// ---- 3.1 best-per-column highlight ---------------------------------------

/** Direction "max" → bigger is better; "min" → smaller is better. */
export type ColumnDirection = "max" | "min" | "none";

export interface CompareColumn {
  key: string;
  direction: ColumnDirection;
  /** Pulls the numeric value out of a StudentStats. Returns null when the
   *  student doesn't have that column populated (legacy run, missing
   *  stats). Null cells are skipped when picking the best — they can never
   *  be the winner. */
  value: (stats: StudentStats) => number | null;
}

/**
 * For each column, find the student id(s) whose value is the best in that
 * column. Ties are all marked as "best" — the user gets to see two
 * tied-winner cells highlighted, which honestly reflects the data.
 *
 * Returns a flat Set of `${studentId}::${columnKey}` strings; the
 * component checks membership when rendering each cell.
 */
export function bestPerColumn(
  students: { id: string; stats: StudentStats }[],
  columns: CompareColumn[],
): Set<string> {
  const winners = new Set<string>();
  if (students.length === 0) return winners;

  for (const col of columns) {
    if (col.direction === "none") continue;

    // Collect (id, value) pairs, dropping nulls.
    const valued: { id: string; v: number }[] = [];
    for (const s of students) {
      const v = col.value(s.stats);
      if (v != null && Number.isFinite(v)) {
        valued.push({ id: s.id, v });
      }
    }
    if (valued.length === 0) continue;

    let bestVal = valued[0].v;
    for (const { v } of valued) {
      if (col.direction === "max" && v > bestVal) bestVal = v;
      if (col.direction === "min" && v < bestVal) bestVal = v;
    }
    for (const { id, v } of valued) {
      if (v === bestVal) winners.add(`${id}::${col.key}`);
    }
  }

  return winners;
}
