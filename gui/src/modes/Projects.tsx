/**
 * Projects landing page — picker for existing projects + "Create" form.
 *
 * Phase 1: this is the front door. The mode shell only mounts once a
 * project is selected, so creating or picking one is the path into Learn /
 * Optimize.
 */

import { useState } from "react";
import { useStore } from "../store";
import type { ProjectSummary, Task } from "../types";

export function Projects() {
  const projects = useStore((s) => s.projects);
  const setCurrentProject = useStore((s) => s.setCurrentProject);
  const projectError = useStore((s) => s.projectError);

  const [showCreate, setShowCreate] = useState(false);

  return (
    <div className="projects-page">
      <div className="projects-header-row">
        <h2>Projects</h2>
        <button
          type="button"
          className="projects-create-btn"
          onClick={() => setShowCreate(true)}
        >
          + Create project
        </button>
      </div>

      {projectError && <p className="projects-error">{projectError}</p>}

      {projects.length === 0 && !showCreate && (
        <div className="projects-empty">
          <p>No projects yet.</p>
          <p>
            A project pins a task ("detection" or "segmentation") and what
            you're looking for ("soccer ball", "player", …) so every Teacher
            and Student you train under it is comparable.
          </p>
          <button
            type="button"
            className="projects-create-btn"
            onClick={() => setShowCreate(true)}
          >
            Create your first project
          </button>
        </div>
      )}

      {showCreate && <CreateProjectForm onDone={() => setShowCreate(false)} />}

      {projects.length > 0 && (
        <ul className="projects-list">
          {projects.map((p) => (
            <li key={p.id} className="projects-card">
              <ProjectCard project={p} onOpen={() => setCurrentProject(p.id)} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function ProjectCard({
  project,
  onOpen,
}: {
  project: ProjectSummary;
  onOpen: () => void;
}) {
  const deleteProject = useStore((s) => s.deleteProject);
  const renameProject = useStore((s) => s.renameProject);
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(project.name);

  return (
    <div className="project-card">
      <div className="project-card-head">
        {editing ? (
          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (name.trim() && name !== project.name) {
                void renameProject(project.id, name.trim());
              }
              setEditing(false);
            }}
            className="project-card-rename"
          >
            <input
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              onBlur={() => setEditing(false)}
            />
          </form>
        ) : (
          <h3 className="project-card-name" onDoubleClick={() => setEditing(true)}>
            {project.name}
          </h3>
        )}
        <span className={`project-card-task task-${project.task}`}>
          {project.task}
        </span>
      </div>

      <div className="project-card-prompts">
        {project.prompts.map((p) => (
          <span key={p} className="project-card-prompt-chip">
            {p}
          </span>
        ))}
      </div>

      <div className="project-card-counters">
        <Counter label="Teachers" value={project.n_teacher_datasets} />
        <Counter label="Reviewed" value={project.n_human_reviewed_datasets} />
        <Counter label="Students" value={project.n_students} />
        {project.n_running > 0 && (
          <span className="project-card-running" title="Runs currently in progress">
            ● {project.n_running} running
          </span>
        )}
      </div>

      <div className="project-card-actions">
        <button type="button" className="project-card-open" onClick={onOpen}>
          Open
        </button>
        <button
          type="button"
          className="project-card-delete"
          onClick={() => {
            if (
              window.confirm(
                `Delete project "${project.name}" and all its runs? This cannot be undone.`,
              )
            ) {
              void deleteProject(project.id);
            }
          }}
        >
          Delete
        </button>
      </div>
    </div>
  );
}

function Counter({ label, value }: { label: string; value: number }) {
  return (
    <span className="project-card-counter">
      <span className="project-card-counter-value">{value}</span>
      <span className="project-card-counter-label">{label}</span>
    </span>
  );
}

function CreateProjectForm({ onDone }: { onDone: () => void }) {
  const createProject = useStore((s) => s.createProject);
  const setCurrentProject = useStore((s) => s.setCurrentProject);

  const [name, setName] = useState("");
  const [task, setTask] = useState<Task>("detection");
  const [promptInput, setPromptInput] = useState("");
  const [prompts, setPrompts] = useState<string[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const addPrompt = () => {
    const t = promptInput.trim();
    if (!t) return;
    if (prompts.includes(t)) {
      setPromptInput("");
      return;
    }
    setPrompts([...prompts, t]);
    setPromptInput("");
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErr(null);
    if (!name.trim()) {
      setErr("Name is required.");
      return;
    }
    if (prompts.length === 0) {
      setErr("Add at least one prompt — that's what your Teachers will look for.");
      return;
    }
    setSubmitting(true);
    const project = await createProject({
      name: name.trim(),
      task,
      prompts,
    });
    setSubmitting(false);
    if (project) {
      setCurrentProject(project.id);
      onDone();
    } else {
      setErr("Could not create project — see project error above.");
    }
  };

  return (
    <form className="create-project-form" onSubmit={submit}>
      <h3>New project</h3>

      <label>
        <span>Name</span>
        <input
          autoFocus
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. SoccerNet ball tracking"
        />
      </label>

      <label>
        <span>Task</span>
        <select value={task} onChange={(e) => setTask(e.target.value as Task)}>
          <option value="detection">Detection</option>
          <option value="segmentation">Segmentation</option>
        </select>
      </label>

      <label>
        <span>What are you looking for?</span>
        <div className="prompt-input-row">
          <input
            value={promptInput}
            onChange={(e) => setPromptInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === ",") {
                e.preventDefault();
                addPrompt();
              }
            }}
            placeholder='Type a phrase and press Enter (e.g. "soccer ball")'
          />
          <button type="button" onClick={addPrompt}>
            Add
          </button>
        </div>
        <div className="prompt-chip-row">
          {prompts.map((p) => (
            <span key={p} className="prompt-chip">
              {p}
              <button
                type="button"
                onClick={() => setPrompts(prompts.filter((x) => x !== p))}
                aria-label={`Remove ${p}`}
              >
                ×
              </button>
            </span>
          ))}
        </div>
      </label>

      <p className="create-project-help">
        Task and prompts <strong>cannot be changed</strong> later — they
        define what this project is. Only the name is editable.
      </p>

      {err && <p className="create-project-error">{err}</p>}

      <div className="create-project-actions">
        <button type="button" onClick={onDone}>
          Cancel
        </button>
        <button type="submit" disabled={submitting}>
          {submitting ? "Creating…" : "Create project"}
        </button>
      </div>
    </form>
  );
}
