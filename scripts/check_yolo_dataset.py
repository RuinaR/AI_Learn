from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate YOLO detection dataset structure and label files."
    )
    parser.add_argument(
        "--yaml",
        default="datasets/hiyoung_ppe.yaml",
        help="Path to the YOLO dataset yaml file.",
    )
    return parser.parse_args()


def load_dataset_config(yaml_path: Path) -> dict:
    with yaml_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_dataset_root(yaml_path: Path, config: dict) -> Path:
    base_path = Path(config["path"])
    if base_path.is_absolute():
        return base_path
    return (yaml_path.parent / base_path).resolve()


def validate_required_dirs(dataset_root: Path) -> list[Path]:
    required = [
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
    ]

    missing = [path for path in required if not path.exists()]
    for path in required:
        status = "OK" if path.exists() else "MISSING"
        print(f"[{status}] {path}")
    return missing


def count_files(directory: Path, patterns: tuple[str, ...]) -> int:
    total = 0
    for pattern in patterns:
        total += len(list(directory.glob(pattern)))
    return total


def print_counts(dataset_root: Path) -> None:
    image_patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    label_pattern = ("*.txt",)
    sections = ("train", "val", "test")

    for section in sections:
        image_dir = dataset_root / "images" / section
        label_dir = dataset_root / "labels" / section
        image_count = count_files(image_dir, image_patterns) if image_dir.exists() else 0
        label_count = count_files(label_dir, label_pattern) if label_dir.exists() else 0
        print(f"{section}: images={image_count}, labels={label_count}")


def validate_label_file(label_path: Path) -> list[str]:
    errors: list[str] = []
    with label_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                errors.append(
                    f"{label_path} line {line_number}: expected 5 values, got {len(parts)}"
                )
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                errors.append(f"{label_path} line {line_number}: invalid class id '{parts[0]}'")
                continue

            if class_id not in (0, 1):
                errors.append(
                    f"{label_path} line {line_number}: class id {class_id} is outside 0~1"
                )

            try:
                bbox = [float(value) for value in parts[1:]]
            except ValueError:
                errors.append(f"{label_path} line {line_number}: bbox values must be floats")
                continue

            for index, value in enumerate(bbox, start=1):
                if not 0.0 <= value <= 1.0:
                    field_name = ("x_center", "y_center", "width", "height")[index - 1]
                    errors.append(
                        f"{label_path} line {line_number}: {field_name}={value} is outside 0~1"
                    )

    return errors


def validate_labels(dataset_root: Path) -> list[str]:
    errors: list[str] = []
    for split in ("train", "val", "test"):
        label_dir = dataset_root / "labels" / split
        if not label_dir.exists():
            continue
        for label_path in sorted(label_dir.glob("*.txt")):
            errors.extend(validate_label_file(label_path))
    return errors


def main() -> int:
    args = parse_args()
    yaml_path = Path(args.yaml).resolve()

    if not yaml_path.exists():
        print(f"Dataset yaml not found: {yaml_path}")
        return 1

    config = load_dataset_config(yaml_path)
    dataset_root = resolve_dataset_root(yaml_path, config)

    print(f"Dataset yaml: {yaml_path}")
    print(f"Dataset root: {dataset_root}")

    missing_dirs = validate_required_dirs(dataset_root)
    print_counts(dataset_root)

    errors = validate_labels(dataset_root)

    if missing_dirs:
        print("\nMissing required directories:")
        for path in missing_dirs:
            print(f"- {path}")

    if errors:
        print("\nLabel validation errors:")
        for error in errors:
            print(f"- {error}")

    if missing_dirs or errors:
        print("\nDataset validation failed.")
        return 1

    print("\nDataset validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

