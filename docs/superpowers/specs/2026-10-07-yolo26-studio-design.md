# YOLO26 Studio — Design

**Date:** 2026-10-07
**Goal:** Test and train Ultralytics YOLO26 models inside ModernCV, prompt
them by drawing boxes or painting brush strokes, expose YOLO26's modes
(train / val / predict / export / track / benchmark) and tasks (detect,
segment, pose, OBB, classify, open-vocabulary YOLOE-26), and answer the
Pal/DePal question: *can we segment boxes on a pallet and recover their
relative heights?*

## Shape

A third mode tab inside a project — **Studio** — next to Learn/Optimize.
The existing modes are video-centric (Teacher runs → Student distillation);
Studio is image-centric, like Roboflow: one image dataset per project, many
trained models.

```
Studio
├── Label    image list · prompt canvas · suggestions review   (Roboflow "Annotate")
├── Train    YOLO26 detect/segment/OBB, sizes n–x, live curves  (Roboflow "Train")
├── Test     predict playground, val, export, benchmark, track  (Roboflow "Deploy")
└── Pallet   segmentation + depth → heights, layers, pick order (Pal/DePal)
```

## Prompting

The prompt canvas holds a *draft* of prompts for the current image:

| Prompt | Gesture | Used by |
|---|---|---|
| Box | drag with Box tool | SAM (box prompt), YOLOE visual prompt (rect mask), plain box label |
| Positive stroke | paint with Brush tool | SAM (points sampled along stroke), YOLOE visual prompt (mask), plain mask label |
| Negative stroke | paint with Erase tool / ⌥-drag | SAM negative points, subtracts from mask prompts |

Draft actions:

* **Segment (SAM 2.1)** — whole draft → one object mask (Roboflow "smart
  polygon"). Optional *live* mode re-runs on every prompt change; Enter
  accepts. Image features are cached per image so re-prompting is fast.
* **Find similar (YOLOE-26 visual prompt)** — positive prompts grouped by
  class → visual-prompt embeddings (VPE) → detect every similar object in
  this image, all images, or only unlabeled ones. Existing annotations can
  be used as extra examples; VPEs from several images are averaged.
* **Add as label** — boxes become box annotations; strokes become a mask
  annotation directly (manual painting).

Also: **text prompts** (YOLOE-26 + MobileCLIP2), **prompt-free** YOLOE-26
(built-in vocabulary), and **model assist** (any trained/pretrained model).
Everything except SAM lands as *suggestions* that are reviewed (accept /
reject / accept ≥ conf) before becoming annotations.

The Ultralytics high-level API only accepts box visual prompts and has a
shape bug with 2-D mask prompts, so `pipeline/studio/engines.py` drives the
VPE path directly with a predictor subclass that letterboxes masks itself.

## Storage

```
runs/projects/{pid}/studio/
  classes.json                [{id, name, color}]
  images.json                 index: id, file, size, split, source, depth, intrinsics, counts
  images/{iid}.{ext}          original bytes
  depth/{iid}.png             optional 16-bit depth (mm)
  annotations/{iid}.json      [{id, class_id, bbox, polygon|null, source, score}]
  suggestions/{iid}.json      pending predictions awaiting review
  models/{mid}/               model.json, progress.json, dataset/ snapshot,
                              ultralytics/ run dir, best.pt, exports/
  tracks/{tid}/               tracking jobs: manifest, overlay.mp4, tracks.jsonl
```

## Training

A dataset *snapshot* (hard-linked images + YOLO labels for the chosen task)
is written when a job is queued, so later edits don't change what the model
saw. Jobs run sequentially in a **subprocess** (`python -m
pipeline.studio.train_job`) — isolates MPS crashes, enables cancel — and
write `progress.json` each epoch; curves come from Ultralytics'
`results.csv`. OBB labels are derived from polygons via `minAreaRect`,
which is exactly the gripper yaw a depal robot needs.

## Pal/DePal analysis

1. Instances: annotations, suggestions, or a model run on the image.
2. Depth: uploaded depth map (RGB-D camera, 16-bit PNG in mm or `.npy` in
   metres) **or** monocular metric depth (Depth Anything V2 Metric-Indoor).
3. Back-project with intrinsics (stored with the image, or from an assumed
   horizontal FOV).
4. Reference plane by RANSAC: auto (background pixels = pallet deck/floor),
   a painted region, or fallback to the box tops' common normal.
5. Per box: top-face points (highest band of the mask) → height above plane,
   top-face dimensions (PCA/minAreaRect in the plane), yaw, tilt.
6. Layers by gap clustering of heights; pick order = top layer first.

Monocular heights are approximate and labelled as such; RGB-D depth gives
real heights. A synthetic pallet renderer (ray-cast boxes with exact depth
and GT masks) validates the geometry end to end.

## Findings from testing (2026-10-07)

Measured on two CC-licensed real photos (a cobot palletizer cell, a carton
stack in a corridor) and synthetic renders:

* **Visual prompts, not text, for cartons.** Text prompts ("cardboard box",
  "carton", "box", "package") scored stacked cartons < 0.3 in every YOLOE
  variant tried (26 s/m/l, 11 s/l). Box visual prompts: YOLOE-11s found
  11–14 cartons per real photo at > 0.25 while YOLOE-26 s/m/l stayed
  < 0.25. Find similar defaults to YOLOE-11 (YOLOE-26 selectable).
* **Visual prompts are scene-specific.** Carton embeddings from two
  different photos had cosine similarity 0.45; a prompt from one photo
  scored ≤ 0.26 on the other. Within one scene type (synthetic renders)
  similarity was 0.83–0.95 and averaging 4 reference images found 7/7
  cartons on an unseen scene. So "find similar" / "find more like my
  labels" is a labelling accelerator for one camera setup; new scenes need
  a trained model.
* **SAM granularity matters for strokes.** A stroke on one carton in a
  stack made SAM's single-mask output pick the whole stack. Strokes now use
  SAM's three hypotheses; "auto" = smallest one containing the whole stroke
  (exactly one carton on the palletizer photo); Fine/Medium/Coarse override.
* **Hidden tops.** In oblique views lower cartons show only side faces.
  The first version measured those faces' upper edge as the "top"
  (errors up to 13 cm). Tops are now found from per-pixel surface normals
  (upward-facing pixels only); masks with none are flagged `top_hidden`
  and reported as lower bounds. Result on 13 synthetic scenes with exact
  depth: every visible top within 0.1 mm, layers correct.
* **Monocular depth**: Depth Anything V2 metric-indoor Base + known camera
  height → ~17 cm mean height error, rank correlation ~0.9. Small model is
  worse (Spearman 0.72). Fine for "which is higher", not for picking.
* **Wall vs floor.** Without gravity a wall facing the camera and a floor
  seen from above are indistinguishable by normal; on a side-view photo the
  auto plane picked the back wall. The analysis now warns when the plane
  never reaches the bottom of the frame, and when most cartons show only
  side faces (a side view).
* **YOLO26 head toggle.** The first end-to-end forward fuses the
  one-to-many head away, so NMS mode needs a separate model instance;
  `max_det` / `agnostic_nms` only reach the e2e head when a predictor is
  built. Both handled in `infer.configure_head` / `engines.load_yolo`.
* **End to end on held-out scenes.** YOLO26n-seg trained 40 epochs on 30
  labelled images (10.7 min on an M4; box mAP50 0.51, mask mAP50 0.50),
  then Pallet with sensor depth on 3 unseen synthetic pallets: every
  detected carton's height within 0.2 mm of truth, layers and pick order
  correct. Recall is the bottleneck at this data size (one scene needed
  conf 0.1); more images / epochs is the lever, not geometry.
* **Deploy latency** (yolo26n-seg, 640 px, M4): ONNX Runtime CPU p50 28 ms,
  PyTorch CPU 32 ms, PyTorch MPS 39 ms, TorchScript CPU 40 ms. Benchmarks
  warm up once per input shape (MPS compiles kernels per shape).
* **Dev reloads vs training.** `uvicorn --reload` restarts only the API
  process; trainers keep running and are re-adopted by PID on startup
  (queued jobs are re-queued).

## Out of scope (for now)

Pose-keypoint and classification *training* (no keypoint / image-tag
annotation tools yet — pretrained pose/cls models are available in Test),
SAM 3 (gated weights), multi-polygon instances, vertex editing.
