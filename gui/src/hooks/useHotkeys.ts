import { useEffect } from "react";

type HotkeyHandler = (e: KeyboardEvent) => void;

/**
 * Register a keydown listener while `enabled` is true.
 *
 * Attached to `window`, but matched against the active keys map; the
 * caller is responsible for not registering a hotkey that conflicts with
 * the user typing into a focused input. We bail out when the active
 * element is an editable field as a defensive default.
 *
 * Keys are matched against `event.key` exactly (case-insensitive).
 * Modifier keys are ignored — pass `Shift+A` only if you want to fire on
 * shifted A specifically (the matched key would be "A" then).
 */
export function useHotkeys(
  bindings: Record<string, HotkeyHandler>,
  enabled = true,
): void {
  useEffect(() => {
    if (!enabled) return;
    const lowered: Record<string, HotkeyHandler> = {};
    for (const [k, fn] of Object.entries(bindings)) {
      lowered[k.toLowerCase()] = fn;
    }
    const onKeyDown = (e: KeyboardEvent): void => {
      const target = e.target as HTMLElement | null;
      if (target) {
        const tag = target.tagName;
        if (
          tag === "INPUT" ||
          tag === "TEXTAREA" ||
          tag === "SELECT" ||
          target.isContentEditable
        ) {
          return;
        }
      }
      const handler = lowered[e.key.toLowerCase()];
      if (!handler) return;
      e.preventDefault();
      handler(e);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
    // The `bindings` object is fresh every render; depending on it would
    // re-register the listener every render. The caller is expected to
    // wrap stable handlers in `useCallback` and pass them via the same
    // keys; we depend on the stringified key list so a key change
    // re-registers.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, Object.keys(bindings).join("|"), ...Object.values(bindings)]);
}
