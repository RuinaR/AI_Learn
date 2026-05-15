from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
BOX_COLORS = {
    0: "#e4572e",
    1: "#17bebb",
    2: "#ffc914",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a few YOLO-labeled samples to verify class names before training."
    )
    parser.add_argument(
        "--yaml",
        default="datasets/hiyoung_ppe_local.yaml",
        help="Path to the YOLO dataset yaml file.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=("train", "val", "test"),
        help="Dataset split to visualize.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=6,
        help="Number of labeled samples to render.",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="Only render images whose label file contains this class id.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/hiyoung_ppe_previews",
        help="Directory where preview images will be written.",
    )
    return parser.parse_args()


def load_dataset_config(yaml_path: Path) -> dict:
    with yaml_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_class_names(config: dict) -> list[str]:
    names = config.get("names", {})
    if isinstance(names, dict):
        return [names[key] for key in sorted(names)]
    if isinstance(names, list):
        return names
    raise ValueError("Dataset yaml must define names as a list or dict.")


def resolve_dataset_root(yaml_path: Path, config: dict) -> Path:
    base_path = Path(config["path"])
    if base_path.is_absolute():
        return base_path
    return (yaml_path.parent / base_path).resolve()


def find_image_for_label(image_dir: Path, label_dir: Path, label_path: Path) -> Path | None:
    relative = label_path.relative_to(label_dir).with_suffix("")
    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{relative}{extension}"
        if candidate.exists():
            return candidate
    return None


def yolo_to_xyxy(width: int, height: int, parts: list[str]) -> tuple[float, float, float, float]:
    x_center = float(parts[1]) * width
    y_center = float(parts[2]) * height
    box_width = float(parts[3]) * width
    box_height = float(parts[4]) * height
    x1 = x_center - box_width / 2
    y1 = y_center - box_height / 2
    x2 = x_center + box_width / 2
    y2 = y_center + box_height / 2
    return x1, y1, x2, y2


def draw_label(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    box: tuple[float, float, float, float],
    class_name: str,
    color: str,
) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

    text_bbox = draw.textbbox((x1, y1), class_name, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_top = max(0, y1 - text_height - 6)
    text_bottom = text_top + text_height + 4
    draw.rectangle((x1, text_top, x1 + text_width + 8, text_bottom), fill=color)
    draw.text((x1 + 4, text_top + 2), class_name, fill="black", font=font)


def collect_label_paths(label_dir: Path) -> list[Path]:
    return sorted(path for path in label_dir.rglob("*.txt") if path.is_file())


def parse_label_rows(label_path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            continue
        rows.append(parts)
    return rows


def main() -> int:
    args = parse_args()
    yaml_path = Path(args.yaml).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not yaml_path.exists():
        print(f"Dataset yaml not found: {yaml_path}")
        return 1

    config = load_dataset_config(yaml_path)
    class_names = get_class_names(config)
    dataset_root = resolve_dataset_root(yaml_path, config)
    image_dir = dataset_root / "images" / args.split
    label_dir = dataset_root / "labels" / args.split

    if not image_dir.exists() or not label_dir.exists():
        print(f"Split directories not found for '{args.split}': {image_dir} / {label_dir}")
        return 1

    label_paths = [path for path in collect_label_paths(label_dir) if path.read_text(encoding="utf-8").strip()]
    if not label_paths:
        print(f"No labeled samples found in {label_dir}")
        return 1

    font = ImageFont.load_default()
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    for label_path in label_paths:
        label_rows = parse_label_rows(label_path)
        if not label_rows:
            continue

        if args.class_id is not None and not any(int(parts[0]) == args.class_id for parts in label_rows):
            continue

        image_path = find_image_for_label(image_dir, label_dir, label_path)
        if image_path is None:
            continue

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size

        has_box = False
        for parts in label_rows:
            class_id = int(parts[0])
            class_name = class_names[class_id]
            color = BOX_COLORS.get(class_id, "#ffffff")
            draw_label(draw, font, yolo_to_xyxy(width, height, parts), class_name, color)
            has_box = True

        if not has_box:
            continue

        preview_name = f"{args.split}_{rendered:02d}_{image_path.stem}.jpg"
        image.save(output_dir / preview_name, quality=95)
        print(f"Wrote {output_dir / preview_name}")
        rendered += 1

        if rendered >= args.count:
            break

    if rendered == 0:
        print("No preview images were rendered.")
        return 1

    print(f"Rendered {rendered} preview images to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
