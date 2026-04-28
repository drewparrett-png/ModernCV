import { useEffect } from "react";
import { ModeShell } from "./components/ModeShell";
import { startStudentPoll, startTeacherPoll, useStore } from "./store";

export default function App() {
  const loadBlocks = useStore((s) => s.loadBlocks);
  const loadVideos = useStore((s) => s.loadVideos);
  const loadTeachers = useStore((s) => s.loadTeachers);
  const loadStudents = useStore((s) => s.loadStudents);
  const loadArchitectures = useStore((s) => s.loadArchitectures);

  useEffect(() => {
    // Initial catalog + run-list pulls. After they settle, re-arm pollers
    // for any run still flagged "running" — covers the case where the user
    // closes the browser, the server keeps grinding, and they come back.
    Promise.all([
      loadBlocks(),
      loadVideos(),
      loadTeachers(),
      loadStudents(),
      loadArchitectures(),
    ])
      .then(() => {
        const { teacherDetails, studentDetails } = useStore.getState();
        for (const [id, d] of Object.entries(teacherDetails)) {
          // Re-arm polling for anything still in-flight — including
          // queued runs (they'll start running when their turn comes up).
          if (
            d.manifest.status === "running" ||
            d.manifest.status === "queued"
          ) {
            startTeacherPoll(id);
          }
        }
        for (const [id, d] of Object.entries(studentDetails)) {
          if (d.manifest.status === "running") startStudentPoll(id);
        }
      })
      .catch((err) => console.error("initial load failed", err));
  }, [loadBlocks, loadVideos, loadTeachers, loadStudents, loadArchitectures]);

  return (
    <div className="app">
      <header className="app-header">
        <h1>ModernCV</h1>
      </header>
      <ModeShell />
    </div>
  );
}
