import { useEffect, useRef, useState } from "react";

/**
 * Tiny "(?)" popover used to attach plain-English explanations to
 * jargon-heavy stat labels (mAP, frame buckets, threshold names).
 *
 * Click to toggle; click outside or press Escape to dismiss. No
 * dependencies — just a button + an absolutely-positioned div.
 *
 * Usage:
 *   <InfoTip title="mAP" body="0-1 score for how well..." />
 */
export function InfoTip({
  title,
  body,
  align = "right",
}: {
  title: string;
  body: string;
  /** Which edge of the trigger the popover anchors to. */
  align?: "left" | "right";
}) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!open) return;
    function onDocClick(e: MouseEvent) {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <span className="info-tip-wrap" ref={wrapRef}>
      <button
        type="button"
        className="info-tip-trigger"
        aria-label={`What is ${title}?`}
        aria-expanded={open}
        onClick={(e) => {
          e.stopPropagation();
          setOpen((v) => !v);
        }}
      >
        ?
      </button>
      {open && (
        <span
          className={`info-tip-popover info-tip-${align}`}
          role="tooltip"
          onClick={(e) => e.stopPropagation()}
        >
          <span className="info-tip-title">{title}</span>
          <span className="info-tip-body">{body}</span>
        </span>
      )}
    </span>
  );
}
