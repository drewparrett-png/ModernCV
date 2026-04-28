/**
 * Stable class → color mapping shared by the Teachers card, the Inspector
 * overlays, and the per-frame chart. Same class always paints the same
 * color across UI surfaces.
 *
 * Approach: deterministic hash → palette index. The palette is hand-picked
 * for contrast on a sports-pitch green background and against each other.
 * Rejected detections still render red elsewhere — these colors are for
 * *kept* detections only.
 */
const PALETTE = [
  "#22c55e", // green-500   (legacy default — keep first so existing screenshots match)
  "#38bdf8", // sky-400
  "#fbbf24", // amber-400
  "#f472b6", // pink-400
  "#a78bfa", // violet-400
  "#f87171", // red-400 (used here for a class, distinct from rejection red elsewhere)
  "#34d399", // emerald-400
  "#fb923c", // orange-400
];

function hashString(s: string): number {
  // FNV-1a 32-bit. We only need stable bucketing — collisions are fine
  // (worst case: two classes share a color in a run with >8 classes).
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}

export function colorForClass(className: string): string {
  return PALETTE[hashString(className) % PALETTE.length];
}
