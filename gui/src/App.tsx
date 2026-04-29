import { useEffect } from "react";
import { ModeShell } from "./components/ModeShell";
import { ProjectHeader } from "./components/ProjectHeader";
import { Projects } from "./modes/Projects";
import { startStudentPoll, startTeacherPoll, useStore } from "./store";

export default function App() {
  const loadBlocks = useStore((s) => s.loadBlocks);
  const loadVideos = useStore((s) => s.loadVideos);
  const loadProjects = useStore((s) => s.loadProjects);
  const loadTeachers = useStore((s) => s.loadTeachers);
  const loadStudents = useStore((s) => s.loadStudents);
  const loadArchitectures = useStore((s) => s.loadArchitectures);
  const currentProjectId = useStore((s) => s.currentProjectId);

  // Initial catalog + project list. Project-scoped data (teachers, students,
  // architectures) is loaded by `setCurrentProject` when the user picks one.
  useEffect(() => {
    Promise.all([loadBlocks(), loadVideos(), loadProjects()]).catch((err) =>
      console.error("initial load failed", err),
    );
  }, [loadBlocks, loadVideos, loadProjects]);

  // Re-arm pollers when a project becomes active and its lists have loaded.
  useEffect(() => {
    if (!currentProjectId) return;
    let cancelled = false;
    Promise.all([loadTeachers(), loadStudents(), loadArchitectures()])
      .then(() => {
        if (cancelled) return;
        const { teacherDetails, studentDetails } = useStore.getState();
        for (const [id, d] of Object.entries(teacherDetails)) {
          if (
            d.manifest.status === "running" ||
            d.manifest.status === "queued"
          ) {
            startTeacherPoll(currentProjectId, id);
          }
        }
        for (const [id, d] of Object.entries(studentDetails)) {
          if (
            d.manifest.status === "running" ||
            d.manifest.status === "queued"
          ) {
            startStudentPoll(currentProjectId, id);
          }
        }
      })
      .catch((err) => console.error("project load failed", err));
    return () => {
      cancelled = true;
    };
  }, [currentProjectId, loadTeachers, loadStudents, loadArchitectures]);

  return (
    <div className="app">
      <header className="app-header">
        <h1>ModernCV</h1>
        <ProjectHeader />
      </header>
      {currentProjectId ? <ModeShell /> : <Projects />}
    </div>
  );
}
