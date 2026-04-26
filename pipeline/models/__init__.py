"""Model adapters.

Each file is a thin wrapper around a third-party library so the rest of the
pipeline doesn't have to know about ultralytics, sam2, etc. directly. Adapters
expose a small, uniform interface that the corresponding Block calls into.

Lazy imports recommended — we don't want to pay a torch import cost just to
list available implementations from the GUI.
"""
