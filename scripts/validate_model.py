from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


REQUIRED_CLASSES = {"helmet", "person"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a trained YOLO model and run prediction on an image or video."
    )
    parser.add_argument("--weights", required=True, help="Path to trained .pt weights file.")
    parser.add_argument(
        "--source",
        required=True,
        help="Path to an image, directory, video file, or camera stream source.",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Prediction confidence threshold.")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save prediction outputs to the default Ultralytics runs directory.",
    )
    return parser.parse_args()


def validate_class_names(model: YOLO) -> bool:
    names = model.names
    print(f"model.names = {names}")

    actual_names = set(names.values()) if isinstance(names, dict) else set(names)
    missing = REQUIRED_CLASSES - actual_names
    if missing:
        print(f"Missing required classes: {sorted(missing)}")
        return False

    print("Required classes found: helmet, person")
    return True


def main() -> int:
    args = parse_args()
    weights_path = Path(args.weights).resolve()

    if not weights_path.exists():
        print(f"Weights file not found: {weights_path}")
        return 1

    model = YOLO(str(weights_path))
    valid = validate_class_names(model)

    results = model.predict(source=args.source, conf=args.conf, save=args.save)
    print(f"Prediction completed. Results objects: {len(results)}")

    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

