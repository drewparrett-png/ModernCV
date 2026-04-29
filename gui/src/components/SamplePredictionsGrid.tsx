import { useEffect, useState } from "react";
import { fetchStudentSamples, studentSampleUrl } from "../api";
import type { RunDetail } from "../types";

/**
 * Thumbnail grid of side-by-side prediction comparison images for a
 * trained Student's eval set — green = Teacher ground truth, red =
 * Student prediction. Click a thumbnail for a full-size lightbox.
 *
 * Renders nothing when the backend has no samples (still training,
 * pre-Phase-7 Student, or eval failed before sample render).
 */
export function SamplePredictionsGrid({
  projectId,
  studentId,
  teacherDetails,
  status,
}: {
  projectId: string;
  studentId: string;
  teacherDetails: Record<string, RunDetail>;
  status: string;
}) {
  const [samples, setSamples] = useState<Record<string, string[]>>({});
  const [lightbox, setLightbox] = useState<
    { teacherId: string; name: string } | null
  >(null);

  useEffect(() => {
    let cancelled = false;
    fetchStudentSamples(projectId, studentId)
      .then((s) => {
        if (!cancelled) setSamples(s);
      })
      .catch(() => {
        // 404 or transient — silently treat as empty.
      });
    return () => {
      cancelled = true;
    };
  }, [projectId, studentId, status]);

  const teacherIds = Object.keys(samples);
  if (teacherIds.length === 0) return null;

  return (
    <div className="sample-grid-wrap">
      <div className="sample-grid-legend">
        <span>
          <span className="sample-legend-swatch sample-legend-gt" />
          Teacher (ground truth)
        </span>
        <span>
          <span className="sample-legend-swatch sample-legend-pred" />
          Student prediction
        </span>
      </div>
      {teacherIds.map((tid) => {
        const t = teacherDetails[tid];
        const label = t?.manifest.prompt ?? tid.replace("teacher_", "");
        return (
          <div key={tid} className="sample-grid-section">
            <div className="sample-grid-title">{label}</div>
            <div className="sample-grid">
              {samples[tid].map((name) => (
                <button
                  type="button"
                  key={name}
                  className="sample-thumb"
                  onClick={() => setLightbox({ teacherId: tid, name })}
                  title={name}
                >
                  <img
                    src={studentSampleUrl(projectId, studentId, tid, name)}
                    alt={name}
                    loading="lazy"
                  />
                </button>
              ))}
            </div>
          </div>
        );
      })}
      {lightbox && (
        <div
          className="sample-lightbox"
          onClick={() => setLightbox(null)}
          role="dialog"
        >
          <img
            src={studentSampleUrl(
              projectId,
              studentId,
              lightbox.teacherId,
              lightbox.name,
            )}
            alt={lightbox.name}
          />
          <button
            type="button"
            className="sample-lightbox-close"
            aria-label="Close"
            onClick={(e) => {
              e.stopPropagation();
              setLightbox(null);
            }}
          >
            ×
          </button>
        </div>
      )}
    </div>
  );
}
