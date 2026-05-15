from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
TARGET_COLOR = "#ff3b30"
OTHER_COLOR = "#c7c7c7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize SH17 source label ids so the real class meaning can be inspected by eye."
    )
    parser.add_argument(
        "--src-root",
        default="raw_datasets/sh17",
        help="SH17 dataset root containing images/ and labels/.",
    )
    parser.add_argument(
        "--output-dir",
        default="raw_datasets/sh17_source_id_previews",
        help="Directory where source-id preview images will be written.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=3,
        help="Maximum number of preview images to save for each source class id.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=17,
        help="Number of source class ids to inspect, starting from 0.",
    )
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def yolo_to_xyxy(width: int, height: int, parts: list[str]) -> tuple[float, float, float, float]:
    x_center = float(parts[1]) * width
    y_center = float(parts[2]) * height
    box_width = float(parts[3]) * width
    box_height = float(parts[4]) * height
    return (
        x_center - box_width / 2,
        y_center - box_height / 2,
        x_center + box_width / 2,
        y_center + box_height / 2,
    )


def find_image_path(images_dir: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def draw_header(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    source_id: int,
) -> None:
    text = f"SOURCE ID {source_id:02d}"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    padding = 12
    draw.rounded_rectangle(
        (12, 12, 12 + text_width + padding * 2, 12 + text_height + padding * 2),
        radius=10,
        fill=TARGET_COLOR,
    )
    draw.text((12 + padding, 12 + padding), text, fill="white", font=font)


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    color: str,
    width: int,
) -> None:
    draw.rectangle(box, outline=color, width=width)


def parse_label_file(label_path: Path) -> list[list[str]]:
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


def collect_samples(
    labels_dir: Path,
    images_dir: Path,
    max_per_class: int,
    num_classes: int,
) -> dict[int, list[tuple[Path, Path]]]:
    samples: dict[int, list[tuple[Path, Path]]] = defaultdict(list)

    for label_path in sorted(labels_dir.glob("*.txt")):
        rows = parse_label_file(label_path)
        if not rows:
            continue

        present_ids = sorted({int(parts[0]) for parts in rows if parts[0].isdigit()})
        if not present_ids:
            continue

        image_path = find_image_path(images_dir, label_path.stem)
        if image_path is None:
            continue

        for source_id in present_ids:
            if source_id < 0 or source_id >= num_classes:
                continue
            if len(samples[source_id]) >= max_per_class:
                continue
            samples[source_id].append((image_path, label_path))

        if all(len(samples[source_id]) >= max_per_class for source_id in range(num_classes)):
            break

    return samples


def render_preview(
    image_path: Path,
    label_path: Path,
    source_id: int,
    destination: Path,
    header_font: ImageFont.ImageFont,
) -> None:
    rows = parse_label_file(label_path)
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for parts in rows:
        current_id = int(parts[0])
        box = yolo_to_xyxy(width, height, parts)
        if current_id == source_id:
            draw_box(draw, box, TARGET_COLOR, width=5)
        else:
            draw_box(draw, box, OTHER_COLOR, width=2)

    draw_header(draw, header_font, source_id)
    image.save(destination, quality=95)


def main() -> int:
    args = parse_args()
    src_root = Path(args.src_root).resolve()
    images_dir = src_root / "images"
    labels_dir = src_root / "labels"
    output_dir = Path(args.output_dir).resolve()

    if not images_dir.exists() or not labels_dir.exists():
        print(f"Missing SH17 directories: {images_dir} / {labels_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    header_font = load_font(size=34)
    samples = collect_samples(labels_dir, images_dir, args.max_per_class, args.num_classes)

    total_written = 0
    for source_id in range(args.num_classes):
        class_samples = samples.get(source_id, [])
        if not class_samples:
            print(f"source_id {source_id:02d}: no samples found")
            continue

        for sample_index, (image_path, label_path) in enumerate(class_samples, start=1):
            destination = output_dir / f"source_id_{source_id:02d}_sample_{sample_index:02d}.jpg"
            render_preview(image_path, label_path, source_id, destination, header_font)
            print(f"Wrote {destination}")
            total_written += 1

    print(f"Rendered {total_written} previews to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
