from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO model for helmet/person detection.")
    parser.add_argument("--model", default="yolo11m.pt", help="Base YOLO model or checkpoint.")
    parser.add_argument(
        "--data",
        default="/content/drive/MyDrive/AI_Learn/datasets/hiyoung_ppe.yaml",
        help="Dataset yaml path.",
    )
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=960, help="Image size.")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size. Use -1 for auto.")
    parser.add_argument("--device", default="0", help="Device id, e.g. 0, cpu.")
    parser.add_argument(
        "--patience", type=int, default=30, help="Early stopping patience in epochs."
    )
    parser.add_argument(
        "--project",
        default="/content/drive/MyDrive/AI_Learn/runs",
        help="Project directory for training outputs.",
    )
    parser.add_argument(
        "--name",
        default="helmet_person_yolo11m_960",
        help="Run name under the project directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()

