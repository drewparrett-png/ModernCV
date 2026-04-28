/**
 * Top-level mode shell — the front door of the app.
 *
 * Two tabs: Learn (the simple wizard) and Optimize (distill from a Teacher
 * run). Optimize is disabled until at least one completed Teacher run
 * exists on disk.
 */

import { useStore } from "../store";
import type { Mode } from "../types";
import { Learn } from "../modes/Learn";
import { Optimize } from "../modes/Optimize";
import { RunInspector } from "./RunInspector";

interface TabDef {
  mode: Mode;
  label: string;
  blurb: string;
}

const TABS: TabDef[] = [
  { mode: "learn", label: "Learn", blurb: "Teach the system from scratch" },
  { mode: "optimize", label: "Optimize", blurb: "Distill a fast student" },
];

export function ModeShell() {
  const mode = useStore((s) => s.mode);
  const setMode = useStore((s) => s.setMode);
  const teacherDetails = useStore((s) => s.teacherDetails);
  const inspectingRunId = useStore((s) => s.inspectingRunId);

  const hasCompletedRun = Object.values(teacherDetails).some(
    (d) => d.manifest.status === "completed",
  );

  return (
    <div className="mode-shell">
      <nav className="mode-tabs">
        {TABS.map((t) => {
          const locked = t.mode === "optimize" && !hasCompletedRun;
          return (
            <button
              key={t.mode}
              type="button"
              className={`mode-tab ${mode === t.mode ? "active" : ""} ${
                locked ? "locked" : ""
              }`}
              onClick={() => !locked && setMode(t.mode)}
              disabled={locked}
              title={
                locked
                  ? "Run something in Learn first — Optimize unlocks once a Teacher run completes."
                  : t.blurb
              }
            >
              <span className="mode-tab-label">{t.label}</span>
              <span className="mode-tab-blurb">
                {locked ? "🔒 needs a Learn run first" : t.blurb}
              </span>
            </button>
          );
        })}
      </nav>

      <div className="mode-body">
        {mode === "learn" && <Learn />}
        {mode === "optimize" && <Optimize />}
      </div>

      {inspectingRunId && <RunInspector />}
    </div>
  );
}
