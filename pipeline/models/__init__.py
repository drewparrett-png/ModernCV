"""Model adapters.

Each file is a thin wrapper around a third-party library so the rest of the
pipeline doesn't have to know about ultralytics, sam2, etc. directly. Adapters
expose a small, uniform interface (Adapter / SourceAdapter) and register
themselves with the registry via @register at import time.

Importing any submodule here triggers its registrations as a side effect, so
this __init__ pulls in every wired adapter module. Add a new line as new
adapters land.
"""

# Import order doesn't matter — each module's @register decorators populate
# the ADAPTERS dict at import time.
from . import video  # noqa: F401 — Input/opencv, Output/overlay-mp4
from . import grounding_dino  # noqa: F401 — Detect/groundingdino
from . import bytetrack  # noqa: F401 — Track/bytetrack
