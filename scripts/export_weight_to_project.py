from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


REQUIRED_CLASSES = {"helmet", "person"}
DEFAULT_TARGET_NAME = "hiyoung_helmet_person_yolo11m.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a trained YOLO weight file into an external project weights directory."
    )
    parser.add_argument("--source", required=True, help="Path to the trained best.pt file.")
    parser.add_argument(
        "--target-dir",
        required=True,
        help="Target weights directory in the external project.",
    )
    parser.add_argument(
        "--target-name",
        default=DEFAULT_TARGET_NAME,
        help="Target filename to use in the external project.",
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
    source_path = Path(args.source).resolve()
    target_dir = Path(args.target_dir).resolve()
    target_path = target_dir / args.target_name

    if not source_path.exists():
        print(f"Source weight file not found: {source_path}")
        return 1

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    print(f"Copied weights to: {target_path}")

    model = YOLO(str(target_path))
    valid = validate_class_names(model)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

