from __future__ import annotations

import argparse
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


TARGET_CLASS_IDS = {
    "helmet": 0,
    "person": 1,
    "vest": 2,
}

SOURCE_TO_TARGET = {
    "person": "person",
    "helmet": "helmet",
    "safetyvest": "vest",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class ConversionStats:
    split_images: Counter = field(default_factory=Counter)
    class_labels: Counter = field(default_factory=Counter)
    excluded_labels: int = 0
    skipped_empty_images: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SH17 dataset into the hiyoung PPE YOLO dataset format."
    )
    parser.add_argument("--src-root", default="raw_datasets/sh17", help="SH17 source root.")
    parser.add_argument(
        "--dst-root",
        default="datasets/hiyoung_ppe",
        help="Destination root for the converted YOLO dataset.",
    )
    parser.add_argument(
        "--train-list",
        default="raw_datasets/sh17/train_files.txt",
        help="Text file listing train images.",
    )
    parser.add_argument(
        "--val-list",
        default="raw_datasets/sh17/val_files.txt",
        help="Text file listing validation images.",
    )
    parser.add_argument(
        "--test-list",
        default="raw_datasets/sh17/test_files.txt",
        help="Optional text file listing test images.",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep images even when no target labels remain after filtering.",
    )
    return parser.parse_args()


def normalize_name(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def load_source_class_mapping(classes_path: Path) -> dict[int, int]:
    if not classes_path.exists():
        raise FileNotFoundError(f"classes.txt not found: {classes_path}")

    source_to_target_id: dict[int, int] = {}
    with classes_path.open("r", encoding="utf-8") as file:
        for source_id, raw_line in enumerate(file):
            class_name = raw_line.strip()
            if not class_name:
                continue

            normalized = normalize_name(class_name)
            target_name = SOURCE_TO_TARGET.get(normalized)
            if target_name is None:
                continue

            source_to_target_id[source_id] = TARGET_CLASS_IDS[target_name]

    required = {"helmet", "person", "vest"}
    found = {name for source_id, target_id in source_to_target_id.items() for name, cid in TARGET_CLASS_IDS.items() if cid == target_id}
    missing = required - found
    if missing:
        raise ValueError(
            f"Missing required SH17 classes in {classes_path}: {sorted(missing)}"
        )

    return source_to_target_id


def read_file_list(list_path: Path) -> list[str]:
    if not list_path.exists():
        return []
    with list_path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def resolve_image_path(src_root: Path, entry: str) -> Path:
    candidate = Path(entry)
    candidates = []

    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.append(src_root / candidate)
        candidates.append(src_root / "images" / candidate)

    if candidate.suffix:
        for item in candidates:
            if item.exists():
                return item.resolve()
    else:
        expanded: list[Path] = []
        for base in candidates:
            for extension in IMAGE_EXTENSIONS:
                expanded.append(base.with_suffix(extension))
        for item in expanded:
            if item.exists():
                return item.resolve()

    raise FileNotFoundError(f"Image not found for list entry: {entry}")


def infer_label_path(src_root: Path, image_path: Path) -> Path:
    sibling = image_path.with_suffix(".txt")
    if sibling.exists():
        return sibling

    image_parts = list(image_path.parts)
    for index, part in enumerate(image_parts):
        if part.lower() == "images":
            label_parts = image_parts.copy()
            label_parts[index] = "labels"
            label_path = Path(*label_parts).with_suffix(".txt")
            if label_path.exists():
                return label_path

    relative = image_path.relative_to(src_root)
    fallback = (src_root / "labels" / relative).with_suffix(".txt")
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Label not found for image: {image_path}")


def derive_output_relative_path(src_root: Path, image_path: Path) -> Path:
    try:
        relative = image_path.relative_to(src_root / "images")
    except ValueError:
        try:
            relative = image_path.relative_to(src_root)
        except ValueError:
            relative = Path(image_path.name)
    return relative


def convert_label_file(
    label_path: Path, source_to_target_id: dict[int, int], stats: ConversionStats
) -> list[str]:
    converted_lines: list[str] = []

    with label_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"Invalid YOLO label format at {label_path} line {line_number}: '{line}'"
                )

            source_id = int(parts[0])
            target_id = source_to_target_id.get(source_id)
            if target_id is None:
                stats.excluded_labels += 1
                continue

            converted_lines.append(" ".join([str(target_id), *parts[1:]]))
            class_name = next(name for name, cid in TARGET_CLASS_IDS.items() if cid == target_id)
            stats.class_labels[class_name] += 1

    return converted_lines


def ensure_split_dirs(dst_root: Path, split: str) -> tuple[Path, Path]:
    image_dir = dst_root / "images" / split
    label_dir = dst_root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    return image_dir, label_dir


def write_split(
    split: str,
    entries: list[str],
    src_root: Path,
    dst_root: Path,
    source_to_target_id: dict[int, int],
    keep_empty: bool,
    stats: ConversionStats,
) -> None:
    if not entries:
        return

    image_root, label_root = ensure_split_dirs(dst_root, split)

    for entry in entries:
        image_path = resolve_image_path(src_root, entry)
        label_path = infer_label_path(src_root, image_path)
        converted_lines = convert_label_file(label_path, source_to_target_id, stats)

        if not converted_lines and not keep_empty:
            stats.skipped_empty_images += 1
            continue

        relative_output = derive_output_relative_path(src_root, image_path)
        dst_image_path = image_root / relative_output
        dst_label_path = (label_root / relative_output).with_suffix(".txt")

        dst_image_path.parent.mkdir(parents=True, exist_ok=True)
        dst_label_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(image_path, dst_image_path)
        dst_label_path.write_text("\n".join(converted_lines) + ("\n" if converted_lines else ""), encoding="utf-8")
        stats.split_images[split] += 1


def print_summary(stats: ConversionStats) -> None:
    print("\nConversion summary")
    print(f"- train images: {stats.split_images['train']}")
    print(f"- val images: {stats.split_images['val']}")
    print(f"- test images: {stats.split_images['test']}")
    print(f"- helmet labels: {stats.class_labels['helmet']}")
    print(f"- person labels: {stats.class_labels['person']}")
    print(f"- vest labels: {stats.class_labels['vest']}")
    print(f"- excluded labels: {stats.excluded_labels}")
    print(f"- skipped empty images: {stats.skipped_empty_images}")

    print("\nNext commands")
    print("python scripts/check_yolo_dataset.py")
    print(
        "python scripts/train_yolo.py --model yolo11s.pt --epochs 10 --imgsz 640 --data datasets/hiyoung_ppe_local.yaml --name test_helmet_person_vest_yolo11s"
    )


def main() -> int:
    args = parse_args()
    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()

    classes_path = src_root / "classes.txt"
    source_to_target_id = load_source_class_mapping(classes_path)
    stats = ConversionStats()

    print(f"Source root: {src_root}")
    print(f"Destination root: {dst_root}")
    print(f"Class id remap: source ids -> target ids {source_to_target_id}")
    print("Target classes: helmet=0, person=1, vest=2")

    split_to_entries = {
        "train": read_file_list(Path(args.train_list).resolve()),
        "val": read_file_list(Path(args.val_list).resolve()),
    }

    test_entries = read_file_list(Path(args.test_list).resolve())
    if test_entries:
        split_to_entries["test"] = test_entries

    for split, entries in split_to_entries.items():
        print(f"Processing {split}: {len(entries)} entries")
        write_split(
            split=split,
            entries=entries,
            src_root=src_root,
            dst_root=dst_root,
            source_to_target_id=source_to_target_id,
            keep_empty=args.keep_empty,
            stats=stats,
        )

    print_summary(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

