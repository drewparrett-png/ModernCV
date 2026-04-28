/**
 * ReviewStatusPill — three-state human-review indicator for a Teacher
 * dataset. Phase 1 of `docs/evaluate.md` §1.1.
 *
 * The same component renders inline next to a Teacher run's name in:
 *   - Learn sidebar rows (compact: icon only)
 *   - Optimize train/eval teacher selectors (compact: icon only)
 *   - Run Inspector title bar (full: icon + label)
 *   - Evaluate dataset header (Phase 2)
 *
 * Visual language:
 *   unreviewed  ○  gray  — "No human review yet."
 *   reviewed    ◐  amber — "Partially reviewed — some detections rejected."
 *   approved    ✓  green — "Approved as ground truth on {date}."
 *
 * The state is derived server-side (RunManifestModel.review_status) from
 * `approved_at` plus rejection-file presence; this component is pure
 * presentation. Approval/un-approval actions live in RunInspector — we
 * deliberately keep the un-approve gesture out of casual surfaces.
 */

import type { ReviewStatus } from "../types";

interface Props {
  status: ReviewStatus;
  /** ISO 8601 timestamp from `manifest.approved_at`. Only consumed when
   *  `status === "approved"` — used to render the date in the user's
   *  locale inside the tooltip. */
  approvedAt?: string | null;
  /** Compact = icon only, no text label. Use in dense lists (sidebars,
   *  selector rows). Default false renders icon + short label. */
  compact?: boolean;
}

const LABELS: Record<ReviewStatus, string> = {
  unreviewed: "Unreviewed",
  reviewed: "Reviewed",
  approved: "Approved",
};

const ICONS: Record<ReviewStatus, string> = {
  // Plain unicode keeps this dependency-free and printable in tests.
  // CSS handles colour and shape via `.review-pill-{status}` classes so
  // a future redesign can swap glyphs for SVGs in one place.
  unreviewed: "○",
  reviewed: "◐",
  approved: "✓",
};

function tooltipFor(status: ReviewStatus, approvedAt?: string | null): string {
  if (status === "approved") {
    if (approvedAt) {
      // User-locale date so "Approved on 2026-04-28" reads naturally for
      // an EU vs US user without us having to pick a format.
      const formatted = new Date(approvedAt).toLocaleString();
      return `Approved as ground truth on ${formatted}.`;
    }
    // Defensive: server should always send approved_at when status is
    // "approved", but if it doesn't we still render a useful tooltip.
    return "Approved as ground truth.";
  }
  if (status === "reviewed") {
    return "Partially reviewed — some detections rejected.";
  }
  return "No human review yet.";
}

export function ReviewStatusPill({
  status,
  approvedAt,
  compact = false,
}: Props) {
  const cls = `review-pill review-pill-${status}${compact ? " compact" : ""}`;
  return (
    <span
      className={cls}
      title={tooltipFor(status, approvedAt)}
      role="img"
      aria-label={`Review status: ${LABELS[status].toLowerCase()}`}
    >
      <span className="review-pill-icon" aria-hidden>
        {ICONS[status]}
      </span>
      {!compact && status !== "unreviewed" && (
        <span className="review-pill-label">{LABELS[status]}</span>
      )}
    </span>
  );
}
