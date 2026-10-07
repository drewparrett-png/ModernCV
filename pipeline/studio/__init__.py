"""YOLO26 Studio — image datasets, prompting, training and Pal/DePal analysis.

Modules
-------
  store      on-disk dataset: classes, images, annotations, suggestions
  engines    cached model handles (YOLO26, YOLOE-26, SAM 2.1, depth) + the
             inference lock that serialises MPS access
  prompting  draft prompts (boxes / brush strokes) → SAM masks or YOLOE
             visual-prompt detections
  infer      generic predict → JSON for every YOLO26 task
  export     annotations → YOLO-format dataset (detect / segment / OBB)
  training   sequential training queue (subprocess per job)
  pallet     depth + segmentation → per-carton heights, layers, pick order
  synth      synthetic pallet scenes with exact depth for testing
  track      video tracking jobs (ByteTrack / BoT-SORT)
"""
