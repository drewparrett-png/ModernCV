/**
 * Display metadata for blocks and their impls.
 *
 * The `BlockKind` enum values ("detect", "reid", …) are *stable identifiers*
 * — they appear in the JSON graph spec, the Python registry, and the API.
 * They're not user-facing labels. This module is the single source of truth
 * for what each kind/impl is *called* in the UI and what it *means* in plain
 * English.
 *
 * If you add a new BlockKind in `types.ts`, add an entry here too. Same for
 * impls: when a backend impl shows up in the registry that isn't listed here,
 * we fall through to a passable default (the raw id) — see `implMeta`.
 */

import type { BlockKind } from "./types";

export interface ImplInfo {
  /** Identifier matching the backend impl name (yolov8n, sam2-tiny, …). */
  id: string;
  /** Short, human-friendly label for dropdowns. */
  label: string;
  /** One-liner: what it is, when you'd pick it. Shown under the impl picker. */
  description: string;
}

export interface BlockKindMeta {
  /** Friendly name shown in the palette and on canvas nodes. */
  label: string;
  /** What this stage of the pipeline does, in plain English. Shown as tooltip. */
  description: string;
  /** Per-impl notes. Keys match backend impl ids. */
  impls: Record<string, ImplInfo>;
}

export const BLOCK_META: Record<BlockKind, BlockKindMeta> = {
  input: {
    label: "Video Source",
    description:
      "Reads frames from a video file. The first stage of every pipeline.",
    impls: {
      opencv: {
        id: "opencv",
        label: "OpenCV",
        description: "cv2.VideoCapture — fastest, works for most clips.",
      },
      ffmpeg: {
        id: "ffmpeg",
        label: "FFmpeg",
        description: "Use when OpenCV chokes on the codec.",
      },
    },
  },

  detect: {
    label: "Object Detection",
    description:
      "Draws bounding boxes around objects in each frame — players, ball, etc.",
    impls: {
      yolov8n: {
        id: "yolov8n",
        label: "YOLOv8 Nano",
        description: "Smallest YOLOv8. Fastest; great first run on M-series.",
      },
      yolov8s: {
        id: "yolov8s",
        label: "YOLOv8 Small",
        description: "A bit more accurate than Nano, still real-time on MPS.",
      },
      yolov11n: {
        id: "yolov11n",
        label: "YOLOv11 Nano",
        description:
          "Newer YOLO generation — similar speed, generally better accuracy.",
      },
      rtdetr: {
        id: "rtdetr",
        label: "RT-DETR",
        description:
          "Transformer-based detector. Stronger on small objects (the ball).",
      },
      groundingdino: {
        id: "groundingdino",
        label: "GroundingDINO (text prompt)",
        description:
          "Type a phrase like \"soccer ball\" — no training needed. Slower, very flexible.",
      },
    },
  },

  segment: {
    label: "Segmentation",
    description:
      "Pixel-level masks for each detection — outlines the shape, not just a box.",
    impls: {
      "sam2-tiny": {
        id: "sam2-tiny",
        label: "SAM2 Tiny",
        description:
          "Meta SAM2 with video-native mask propagation. Recommended.",
      },
      "sam2-small": {
        id: "sam2-small",
        label: "SAM2 Small",
        description: "Bigger SAM2 — sharper masks if you have the MPS budget.",
      },
      mobilesam: {
        id: "mobilesam",
        label: "MobileSAM",
        description:
          "Light single-frame fallback. No temporal propagation across frames.",
      },
      fastsam: {
        id: "fastsam",
        label: "FastSAM",
        description:
          "YOLO-based segmenter — very fast, less accurate than SAM2.",
      },
    },
  },

  reid: {
    label: "Appearance Embedding",
    description:
      "Computes a feature vector — a \"fingerprint\" — for each detected crop. Doesn't assign IDs itself; the Tracker downstream uses these vectors to recognize the same object across occlusions and crossovers, not just by box overlap. The DINO impls are general-purpose vision models; we use their features as the appearance signal.",
    impls: {
      "dinov3-vits16": {
        id: "dinov3-vits16",
        label: "DINOv3 ViT-S/16",
        description: "Small DINOv3 backbone. Fast on M-series; recommended.",
      },
      "dinov3-vitb16": {
        id: "dinov3-vitb16",
        label: "DINOv3 ViT-B/16",
        description: "Base DINOv3 — slower, stronger features.",
      },
      "dinov2-vits14": {
        id: "dinov2-vits14",
        label: "DINOv2 ViT-S/14",
        description: "Older DINOv2. Battle-tested fallback.",
      },
      osnet: {
        id: "osnet",
        label: "OSNet",
        description:
          "Classical Re-ID baseline. Useful for comparing against learned features.",
      },
    },
  },

  track: {
    label: "Tracker",
    description:
      "Maintains stable IDs over time. Matches each new detection to existing tracks using box overlap, motion prediction, and — if a Re-ID block is upstream — appearance similarity. Without appearance signals, IDs tend to swap whenever objects cross paths.",
    impls: {
      bytetrack: {
        id: "bytetrack",
        label: "ByteTrack",
        description:
          "Fast, IoU-based. Uses Re-ID embeddings if a Re-ID block is upstream.",
      },
      botsort: {
        id: "botsort",
        label: "BoT-SORT",
        description: "IoU + appearance — robust on people.",
      },
      ocsort: {
        id: "ocsort",
        label: "OC-SORT",
        description: "Observation-centric — good through occlusions.",
      },
      "sam2-mask": {
        id: "sam2-mask",
        label: "SAM2 Mask Propagation",
        description:
          "Use SAM2's own temporal propagation — no separate tracker needed.",
      },
    },
  },

  stats: {
    label: "Metrics",
    description:
      "Aggregates per-track stats: positions, distance covered, possession, team assignment.",
    impls: {
      basic: {
        id: "basic",
        label: "Basic (pixels)",
        description: "Pixel-space positions and distances. Fine for visualization.",
      },
      calibrated: {
        id: "calibrated",
        label: "Calibrated (meters)",
        description:
          "Real-world distance and speed. Needs a Calibrate block upstream.",
      },
      "team-cluster": {
        id: "team-cluster",
        label: "Team Clustering",
        description:
          "KMeans on Re-ID embeddings — assigns Team A / Team B / referee.",
      },
    },
  },

  output: {
    label: "Output",
    description: "Where the annotated video and labels are written.",
    impls: {
      "overlay-mp4": {
        id: "overlay-mp4",
        label: "Overlay MP4",
        description: "Render boxes/masks/IDs onto frames; write a video file.",
      },
      "json-tracks": {
        id: "json-tracks",
        label: "Tracks JSON",
        description: "Dump tracks as JSON for downstream analysis.",
      },
      preview: {
        id: "preview",
        label: "Live Preview",
        description: "Stream frames to the GUI via websocket.",
      },
    },
  },
};

/** Display info for a block kind. */
export function blockMeta(kind: BlockKind): BlockKindMeta {
  return BLOCK_META[kind];
}

/**
 * Display info for an impl. Falls back to `{label: id, description: ""}` if
 * the backend advertises an impl we don't have metadata for — keeps the GUI
 * working when the registry is ahead of this file.
 */
export function implMeta(kind: BlockKind, impl: string): ImplInfo {
  const meta = BLOCK_META[kind];
  return meta?.impls[impl] ?? { id: impl, label: impl, description: "" };
}
