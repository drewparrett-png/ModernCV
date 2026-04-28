/**
 * Persistent header strip showing the active project (if any).
 *
 * When inside a project: name + task chip + prompts as chips + "Switch
 * project" button. When at the picker: nothing.
 */

import { useStore } from "../store";

export function ProjectHeader() {
  const currentProjectId = useStore((s) => s.currentProjectId);
  const project = useStore((s) => s.getCurrentProject());
  const setCurrentProject = useStore((s) => s.setCurrentProject);

  if (!currentProjectId) return null;
  if (!project) {
    // Project list hasn't loaded yet but a project id is set (e.g. fast
    // navigation). Show a minimal placeholder so the header doesn't reflow.
    return (
      <div className="project-header">
        <span className="project-header-name">Loading project…</span>
      </div>
    );
  }

  return (
    <div className="project-header">
      <span className="project-header-name">{project.name}</span>
      <span className={`project-header-task task-${project.task}`}>
        {project.task}
      </span>
      <span className="project-header-prompts">
        {project.prompts.map((p) => (
          <span key={p} className="project-header-prompt-chip">
            {p}
          </span>
        ))}
      </span>
      <button
        type="button"
        className="project-header-switch"
        onClick={() => setCurrentProject(null)}
      >
        Switch project
      </button>
    </div>
  );
}
