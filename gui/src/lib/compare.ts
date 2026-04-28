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

// ---- 3.2 per-eval-teacher matrix ----------------------------------------

export interface EvalTeacherMatrixCell {
  /** null = student didn't include this teacher in its eval set. */
  map50: number | null;
}

export interface EvalTeacherMatrix {
  /** Eval teacher ids in the order they should be rendered as rows. */
  teacherIds: string[];
  /** Student ids in the order they should be rendered as columns. */
  studentIds: string[];
  /** cells[teacherId][studentId] → cell. Missing entries = null map50. */
  cells: Record<string, Record<string, EvalTeacherMatrixCell>>;
}

/**
 * Build the per-eval-teacher matrix from a list of selected students.
 *
 * Rows: union of `eval_teacher_ids` across the selection (one row per
 * unique teacher id any student evaluated against).
 *
 * Cells: the student's `map50` for that teacher, or null if the student
 * didn't include that teacher in its eval set. Errored eval entries
 * (`r.error` set) are also rendered as null — the score isn't trustworthy.
 *
 * Rows are sorted by frequency (most students first), tie-broken
 * alphabetically. The frequency sort puts the "most-comparable" rows at
 * the top; users can scan the gaps below.
 */
export function evalTeacherMatrix(
  students: { id: string; stats: StudentStats }[],
): EvalTeacherMatrix {
  const studentIds = students.map((s) => s.id);

  // Count how many students evaluated against each teacher (non-error).
  const teacherCounts = new Map<string, number>();
  const cells: Record<string, Record<string, EvalTeacherMatrixCell>> = {};

  for (const s of students) {
    for (const r of s.stats.per_eval_teacher) {
      if (!cells[r.teacher_id]) cells[r.teacher_id] = {};
      const map50 = r.error ? null : r.map50;
      cells[r.teacher_id][s.id] = { map50 };
      if (map50 != null) {
        teacherCounts.set(
          r.teacher_id,
          (teacherCounts.get(r.teacher_id) ?? 0) + 1,
        );
      }
    }
  }

  const teacherIds = [...teacherCounts.keys()].sort((a, b) => {
    const ca = teacherCounts.get(a) ?? 0;
    const cb = teacherCounts.get(b) ?? 0;
    if (ca !== cb) return cb - ca;
    return a.localeCompare(b);
  });

  return { teacherIds, studentIds, cells };
}

/**
 * Filter to teachers that every selected student evaluated against (no
 * missing cells). Useful when the user wants apples-to-apples and is
 * willing to lose rows where coverage is incomplete.
 */
export function sharedEvalTeachers(
  matrix: EvalTeacherMatrix,
): EvalTeacherMatrix {
  const filtered = matrix.teacherIds.filter((tid) => {
    const row = matrix.cells[tid] ?? {};
    return matrix.studentIds.every((sid) => {
      const cell = row[sid];
      return cell != null && cell.map50 != null;
    });
  });
  return { ...matrix, teacherIds: filtered };
}

/** Shared map@0.5 colour bucket — keeps the matrix in lockstep with
 *  PerEvalTeacherTable's existing thresholds (≥0.7 good, ≥0.4 okay,
 *  else poor). */
export function mapColourClass(m: number | null): string {
  if (m == null) return "";
  if (m >= 0.7) return "map-good";
  if (m >= 0.4) return "map-okay";
  return "map-poor";
}
