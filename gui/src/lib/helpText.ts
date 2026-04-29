/**
 * Plain-English explanations for the jargon that shows up on the
 * Optimize mode UI — frame buckets, mAP, the confidence-band thresholds.
 *
 * Both the Advanced settings panel and the Student detail card pull
 * from here so the same term has the same explanation everywhere.
 * If you tweak wording, you only have to do it in one place.
 */

export const HELP = {
  export_threshold: {
    title: "export_threshold",
    body: "Detections at or above this confidence become labels the Student trains on. Lower it to be more permissive (more positives, more noise); raise it to be stricter (fewer positives, cleaner labels).",
  },
  t_low: {
    title: "t_low",
    body: "Lower edge of the confidence dead-zone. Frames whose top detection sits between t_low and export_threshold are dropped entirely — the Teacher saw something but wasn't sure, and including it as either a positive or a negative would bias the Student.",
  },
  treat_empty_as_negative: {
    title: "treat_empty_as_negative",
    body: "Reclassifies every zero-detection frame as a true negative — including the dead-zone frames the Teacher was uncertain about. Reproduces the pre-Phase-0 behaviour. Off by default.",
  },
  frame_buckets: {
    title: "Frame buckets",
    body: "How each training frame is classified by the Teacher's top detection confidence. Positive = used to teach the Student labels. Skipped = teacher uncertain, frame dropped. True negative = empty frame, used to teach the Student to leave it alone.",
  },
  positives: {
    title: "Used as positives",
    body: "Frames where the Teacher was confident enough (≥ export_threshold) for its detections to be taught to the Student as ground truth.",
  },
  uncertain_skipped: {
    title: "Skipped — uncertain",
    body: "Frames whose top detection sat in the t_low → export_threshold dead-zone. Including them would bias the Student because the Teacher itself wasn't sure.",
  },
  true_negatives: {
    title: "Used as negatives",
    body: "Empty frames the Student learns to leave alone. Confirmed empty = no detections at all OR top score below t_low.",
  },
  map: {
    title: "mAP — mean Average Precision",
    body: "0–1 score for how well the Student's detections match the Teacher's labels on a held-out clip the Student never trained on. Higher is better. mAP @ 0.5 is the headline number (50% IoU); mAP @ 0.5:0.95 averages across stricter overlap thresholds and is always lower.",
  },
  generalization: {
    title: "Generalization",
    body: "How well the Student does on Teachers it was never trained on. This is the real signal — high in-distribution mAP just means the Student memorised its training set.",
  },
  inference_latency: {
    title: "Inference latency",
    body: "Wall-clock time per frame on this machine, measured after a warmup run. avg = mean across the timing batch; p95 = the worst 5% of frames are slower than this.",
  },
} as const;

/**
 * mAP color thresholds. Object detection rules of thumb — anything
 * under 0.30 is essentially noise, 0.30–0.60 is "the model has learned
 * something but it's not production-ready", ≥ 0.60 is solidly useful.
 * Tune later if these stop matching how you read the numbers.
 */
export const MAP_THRESHOLDS = {
  poor: 0.3,
  okay: 0.6,
};

export function mapTier(m: number): "poor" | "okay" | "good" {
  if (m >= MAP_THRESHOLDS.okay) return "good";
  if (m >= MAP_THRESHOLDS.poor) return "okay";
  return "poor";
}
