from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union

from ultralytics import YOLO
from ultralytics.data import base as yolo_base
from ultralytics.data import dataset as yolo_dataset


class SequentialPool:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def __enter__(self) -> "SequentialPool":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        return None

    def imap(self, func, iterable):
        for item in iterable:
            yield func(item)


def parse_batch(value: str) -> Union[int, float]:
    if value == "-1":
        return -1
    if "." in value:
        return float(value)
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO model for helmet/person detection."
    )
    parser.add_argument("--model", default="yolo26x.pt", help="Base YOLO model or checkpoint.")
    parser.add_argument(
        "--data",
        default="datasets/hiyoung_ppe_local.yaml",
        help="Dataset yaml path.",
    )
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=960, help="Image size.")
    parser.add_argument(
        "--batch",
        type=parse_batch,
        default=0.7,
        help="Batch size. Use -1 for auto or a fraction like 0.70 for target GPU utilization.",
    )
    parser.add_argument("--device", default="0", help="Device id, e.g. 0, cpu.")
    parser.add_argument(
        "--patience", type=int, default=25, help="Early stopping patience in epochs."
    )
    parser.add_argument(
        "--project",
        default="runs",
        help="Project directory for training outputs.",
    )
    parser.add_argument(
        "--name",
        default="helmet_person_yolo26x_960_rtx3080",
        help="Run name under the project directory.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument(
        "--cache",
        default=False,
        help="Ultralytics cache mode: False, True, ram, or disk.",
    )
    parser.add_argument(
        "--optimizer",
        default="auto",
        help="Optimizer name. Use auto unless you have a reason to override it.",
    )
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate.")
    parser.add_argument("--lrf", type=float, default=None, help="Final learning rate factor.")
    parser.add_argument("--momentum", type=float, default=None, help="Optimizer momentum.")
    parser.add_argument("--weight-decay", type=float, default=0.00027, help="Weight decay.")
    parser.add_argument("--warmup-epochs", type=float, default=3.0, help="Warmup epochs.")
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=10,
        help="Disable mosaic augmentation for the last N epochs.",
    )
    parser.add_argument("--mosaic", type=float, default=0.9, help="Mosaic augmentation ratio.")
    parser.add_argument("--mixup", type=float, default=0.2, help="MixUp augmentation ratio.")
    parser.add_argument(
        "--copy-paste",
        type=float,
        default=0.2,
        help="Copy-paste augmentation ratio.",
    )
    parser.add_argument("--degrees", type=float, default=0.0, help="Rotation augmentation.")
    parser.add_argument("--translate", type=float, default=0.2, help="Translation augmentation.")
    parser.add_argument("--scale", type=float, default=0.85, help="Scale augmentation.")
    parser.add_argument("--fliplr", type=float, default=0.3, help="Horizontal flip ratio.")
    parser.add_argument("--flipud", type=float, default=0.0, help="Vertical flip ratio.")
    parser.add_argument("--shear", type=float, default=0.0, help="Shear augmentation.")
    parser.add_argument("--perspective", type=float, default=0.0, help="Perspective augmentation.")
    parser.add_argument("--hsv-h", type=float, default=0.013, help="HSV hue augmentation.")
    parser.add_argument("--hsv-s", type=float, default=0.35, help="HSV saturation augmentation.")
    parser.add_argument("--hsv-v", type=float, default=0.2, help="HSV value augmentation.")
    parser.add_argument("--bgr", type=float, default=0.0, help="BGR channel swap ratio.")
    parser.add_argument("--erasing", type=float, default=0.1, help="Random erasing ratio.")
    parser.add_argument("--box", type=float, default=9.83, help="Box loss gain.")
    parser.add_argument("--cls", type=float, default=0.65, help="Classification loss gain.")
    parser.add_argument("--cls-pw", type=float, default=0.25, help="Class weighting power.")
    parser.add_argument("--dfl", type=float, default=0.96, help="DFL loss gain.")
    parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="Freeze first N layers. Use 10 for conservative fine-tuning on small data.",
    )
    parser.add_argument("--cos-lr", action="store_true", help="Use cosine learning rate decay.")
    parser.add_argument("--resume", action="store_true", help="Resume the latest matching run.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing an existing run name.")
    parser.add_argument(
        "--skip-engine-export",
        action="store_true",
        help="Skip TensorRT .engine export after training.",
    )
    parser.add_argument(
        "--engine-batch",
        type=int,
        default=1,
        help="Inference batch size to bake into the TensorRT engine export.",
    )
    parser.add_argument(
        "--engine-workspace",
        type=float,
        default=4.0,
        help="TensorRT workspace size in GiB for export.",
    )
    parser.add_argument(
        "--engine-dynamic",
        action="store_true",
        help="Export a dynamic-shape TensorRT engine.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Windows environments can deny multiprocessing pipe creation during label caching.
    # Fall back to a sequential pool so training can proceed reliably.
    yolo_dataset.ThreadPool = SequentialPool
    yolo_base.ThreadPool = SequentialPool
    model = YOLO(args.model)
    train_args = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        workers=args.workers,
        cache=args.cache,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        close_mosaic=args.close_mosaic,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        fliplr=args.fliplr,
        flipud=args.flipud,
        shear=args.shear,
        perspective=args.perspective,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        bgr=args.bgr,
        erasing=args.erasing,
        box=args.box,
        cls=args.cls,
        cls_pw=args.cls_pw,
        dfl=args.dfl,
        freeze=args.freeze,
        cos_lr=args.cos_lr,
        resume=args.resume,
        exist_ok=args.exist_ok,
        amp=True,
        plots=True,
        val=True,
        deterministic=True,
        seed=0,
    )
    if args.lr0 is not None:
        train_args["lr0"] = args.lr0
    if args.lrf is not None:
        train_args["lrf"] = args.lrf
    if args.momentum is not None:
        train_args["momentum"] = args.momentum
    model.train(**train_args)

    if args.skip_engine_export:
        return

    save_dir = Path(model.trainer.save_dir).resolve()
    best_path = save_dir / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"best.pt not found after training: {best_path}")

    print(f"Exporting TensorRT engine from: {best_path}")
    export_model = YOLO(str(best_path))
    engine_path = export_model.export(
        format="engine",
        imgsz=args.imgsz,
        batch=args.engine_batch,
        device=args.device,
        half=True,
        workspace=args.engine_workspace,
        dynamic=args.engine_dynamic,
    )
    print(f"Exported TensorRT engine to: {engine_path}")


if __name__ == "__main__":
    main()
