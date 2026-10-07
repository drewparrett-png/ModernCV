"""Child-process body for one Studio training job.

    python -m pipeline.studio.train_job <models/{mid} dir>

Reads `model.json`, trains with Ultralytics, writes `progress.json` after
every epoch, then a final validation pass for per-class metrics, and
records the outcome in `model.json`. The parent (`training._run_job`) only
waits on the process; on cancel it marks the manifest and sends SIGTERM,
which we turn into an immediate exit.
"""

from __future__ import annotations

import os
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path


def main(mdir: Path) -> int:
    from pipeline.studio import engines
    from pipeline.studio.store import now_iso
    from pipeline.studio.training import _write_json, metrics_to_json, read_manifest, write_manifest

    signal.signal(signal.SIGTERM, lambda *_: os._exit(130))

    m = read_manifest(mdir)
    cfg = m["config"]
    task, size, base = m["task"], m["size"], m["base"]
    progress_path = mdir / "progress.json"

    def prog(**kw: object) -> None:
        try:
            _write_json(progress_path, {**kw, "updated_at": now_iso()})
        except Exception as e:  # never let progress reporting kill training
            print(f"progress write failed: {e}", flush=True)

    t0 = time.time()
    try:
        from ultralytics import YOLO

        prog(stage="starting", epoch=0, epochs=cfg["epochs"], message="loading model and dataset")
        family = str(m.get("family") or "26")
        if base == "pretrained":
            weights = str(engines.ensure_weights(engines.yolo_name(family, task, size)))
        elif base == "scratch":
            weights = f"yolo{family}{size}{engines.TASK_SUFFIX[task]}.yaml"
        else:
            weights = str(mdir.parent / base[len("model:"):] / "best.pt")
        model = YOLO(weights, task=task)

        def on_train_start(trainer: object) -> None:
            prog(stage="training", epoch=0, epochs=int(trainer.epochs), message="first epoch running")

        def on_fit_epoch_end(trainer: object) -> None:
            ep, n = int(trainer.epoch) + 1, int(trainer.epochs)
            metrics = {k: round(float(v), 5) for k, v in (trainer.metrics or {}).items()}
            loss = {}
            if getattr(trainer, "tloss", None) is not None:
                try:
                    loss = {k: round(float(v), 5) for k, v in trainer.label_loss_items(trainer.tloss, prefix="train").items()}
                except Exception:
                    loss = {}
            elapsed = time.time() - t0
            prog(stage="training", epoch=ep, epochs=n, metrics=metrics, loss=loss,
                 elapsed_s=round(elapsed, 1), eta_s=round(elapsed / ep * (n - ep), 1))

        model.add_callback("on_train_start", on_train_start)
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        kw: dict = {
            "data": str(mdir / "dataset" / "data.yaml"),
            "epochs": int(cfg["epochs"]),
            "imgsz": int(cfg["imgsz"]),
            "batch": int(cfg["batch"]),
            "patience": int(cfg["patience"]),
            "optimizer": cfg.get("optimizer") or "auto",
            "cos_lr": bool(cfg.get("cos_lr")),
            "seed": int(cfg.get("seed") or 0),
            "workers": int(cfg.get("workers", 2)),
            "cache": cfg.get("cache") or False,
            "device": engines.device(),
            "project": str(mdir / "ultralytics"),
            "name": "train",
            "exist_ok": True,
            "plots": True,
            "verbose": True,
        }
        if cfg.get("lr0"):
            kw["lr0"] = float(cfg["lr0"])
        if cfg.get("freeze"):
            kw["freeze"] = int(cfg["freeze"])
        if base == "scratch":
            kw["pretrained"] = False
        kw.update(cfg.get("augment") or {})
        model.train(**kw)

        wdir = mdir / "ultralytics" / "train" / "weights"
        best = wdir / "best.pt" if (wdir / "best.pt").exists() else wdir / "last.pt"
        if not best.exists():
            raise RuntimeError(f"Ultralytics produced no weights under {wdir}")
        shutil.copy2(best, mdir / "best.pt")

        prog(stage="validating", epoch=int(cfg["epochs"]), epochs=int(cfg["epochs"]), message="final validation")
        final = YOLO(str(mdir / "best.pt")).val(
            # plots=True so Ultralytics fills the confusion matrix.
            data=kw["data"], split="val", device=engines.device(), plots=True, verbose=False,
            imgsz=kw["imgsz"], project=str(mdir / "val_runs"), name="final", exist_ok=True,
        )
        mj = metrics_to_json(final, m["dataset"]["names"], task)

        m = read_manifest(mdir)
        if m["status"] != "running":
            return 1
        m.update(
            status="completed",
            finished_at=now_iso(),
            train_seconds=round(time.time() - t0, 1),
            metrics=mj["summary"],
            per_class=mj["per_class"],
            speed=mj["speed"],
            confusion=mj["confusion"],
            weights_bytes=(mdir / "best.pt").stat().st_size,
        )
        write_manifest(mdir, m)
        prog(stage="done", epoch=int(cfg["epochs"]), epochs=int(cfg["epochs"]), elapsed_s=round(time.time() - t0, 1))
        return 0
    except Exception as e:
        traceback.print_exc()
        m = read_manifest(mdir)
        if m["status"] == "running":
            m.update(status="failed", finished_at=now_iso(), error=f"{type(e).__name__}: {e}")
            write_manifest(mdir, m)
        prog(stage="failed", epoch=0, epochs=int(cfg["epochs"]), message=str(e)[:500])
        return 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
