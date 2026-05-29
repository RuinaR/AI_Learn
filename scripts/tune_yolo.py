from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run focused hyperparameter tuning for yolo26x helmet/person training."
    )
    parser.add_argument("--model", default="yolo26x.pt", help="Base YOLO model or checkpoint.")
    parser.add_argument(
        "--data",
        default="datasets/hiyoung_ppe_local.yaml",
        help="Dataset yaml path.",
    )
    parser.add_argument("--imgsz", type=int, default=960, help="Image size.")
    parser.add_argument("--epochs", type=int, default=40, help="Epochs per tuning trial.")
    parser.add_argument("--iterations", type=int, default=30, help="Number of tuning trials.")
    parser.add_argument("--device", default="0", help="Device id, e.g. 0, cpu.")
    parser.add_argument("--batch", type=float, default=0.6, help="GPU utilization fraction.")
    parser.add_argument("--project", default="runs", help="Project directory.")
    parser.add_argument(
        "--name",
        default="tune_helmet_person_yolo26x_960_rtx3080",
        help="Run name under the project directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    search_space = {
        "lr0": (3e-4, 2e-3),
        "lrf": (0.5, 0.95),
        "momentum": (0.9, 0.98),
        "weight_decay": (1e-4, 8e-4),
        "warmup_epochs": (1.0, 4.0),
        "box": (7.0, 12.0),
        "cls": (0.4, 1.2),
        "cls_pw": (0.15, 0.7),
        "dfl": (0.8, 1.4),
        "hsv_h": (0.008, 0.03),
        "hsv_s": (0.2, 0.5),
        "hsv_v": (0.15, 0.35),
        "translate": (0.05, 0.3),
        "scale": (0.6, 0.95),
        "fliplr": (0.1, 0.5),
        "mosaic": (0.6, 1.0),
        "mixup": (0.0, 0.3),
        "copy_paste": (0.0, 0.3),
        "close_mosaic": (5.0, 15.0),
    }
    model.tune(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        iterations=args.iterations,
        device=args.device,
        batch=args.batch,
        optimizer="AdamW",
        project=args.project,
        name=args.name,
        space=search_space,
        plots=False,
        save=True,
        val=True,
    )


if __name__ == "__main__":
    main()
