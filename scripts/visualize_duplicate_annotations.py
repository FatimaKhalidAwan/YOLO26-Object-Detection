
from pathlib import Path
import csv
import math

import yaml
from PIL import Image, ImageDraw

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset"
CSV_FILE = PROJECT_DIR / "duplicate_audit.csv"
OUTPUT_DIR = PROJECT_DIR / "duplicate_review"
CONFIG_FILE = PROJECT_DIR / "data.yaml"


def load_class_names():
    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    names = config["names"]

    if isinstance(names, dict):
        classes = [
            names.get(i, names.get(str(i)))
            for i in range(int(config["nc"]))
        ]
    else:
        classes = names

    if len(classes) != int(config["nc"]) or any(name is None for name in classes):
        raise ValueError("Class names do not match data.yaml.")

    return classes


def resolve_image_path(path_value, split):
    image_path = Path(path_value)

    if image_path.is_file():
        return image_path

    fallback = DATASET_DIR / "images" / split / image_path.name

    if fallback.is_file():
        return fallback

    raise FileNotFoundError(f"Image not found: {image_path.name} ({split})")


def draw_annotations(image_path, label_path, title, classes):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    if not label_path.is_file():
        raise FileNotFoundError(f"Label not found: {label_path}")

    with open(label_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.split()

            if not parts:
                continue

            if len(parts) != 5:
                print(f"Skipping malformed annotation: {label_path.name}, line {line_number}")
                continue

            try:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                print(f"Skipping invalid annotation: {label_path.name}, line {line_number}")
                continue

            values = (x, y, w, h)

            if not 0 <= class_id < len(classes) or not all(math.isfinite(v) for v in values):
                continue

            if not (
                0 <= x <= 1 and 0 <= y <= 1
                and 0 < w <= 1 and 0 < h <= 1
                and x - w / 2 >= 0 and y - h / 2 >= 0
                and x + w / 2 <= 1 and y + h / 2 <= 1
            ):
                continue

            x1 = int((x - w / 2) * width)
            y1 = int((y - h / 2) * height)
            x2 = int((x + w / 2) * width)
            y2 = int((y + h / 2) * height)

            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            draw.text((x1, max(0, y1 - 18)), classes[class_id], fill="red")

    return image, title


def create_comparison(row, classes):
    split_1 = row["split_1"]
    split_2 = row["split_2"]

    image_1 = resolve_image_path(row["image_1"], split_1)
    image_2 = resolve_image_path(row["image_2"], split_2)

    label_1 = DATASET_DIR / "labels" / split_1 / f"{image_1.stem}.txt"
    label_2 = DATASET_DIR / "labels" / split_2 / f"{image_2.stem}.txt"

    left, _ = draw_annotations(image_1, label_1, split_1, classes)
    right, _ = draw_annotations(image_2, label_2, split_2, classes)

    header_height = 40
    canvas = Image.new(
        "RGB",
        (left.width + right.width, max(left.height, right.height) + header_height),
        "white"
    )

    canvas.paste(left, (0, header_height))
    canvas.paste(right, (left.width, header_height))

    draw = ImageDraw.Draw(canvas)
    draw.text((10, 12), f"LEFT: {split_1} - {image_1.name}", fill="black")
    draw.text(
        (left.width + 10, 12),
        f"RIGHT: {split_2} - {image_2.name}",
        fill="black"
    )

    output_name = (
        f"{split_1}_{image_1.stem}_vs_{split_2}_{image_2.stem}.png"
    )

    output_path = OUTPUT_DIR / output_name
    canvas.save(output_path)

    return output_path


def main():
    if not CSV_FILE.is_file():
        raise FileNotFoundError(f"Duplicate audit CSV not found: {CSV_FILE}")

    classes = load_class_names()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CSV_FILE, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        required_columns = {
            "image_1", "image_2", "split_1", "split_2", "match_type"
        }

        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError("Duplicate audit CSV is missing required columns.")

        rows = list(reader)

    exact_rows = [
        row for row in rows
        if (row.get("match_type") or "").startswith("Exact")
    ]

    print("Exact duplicate pairs:", len(exact_rows))

    saved = 0
    errors = 0

    for row in exact_rows:
        try:
            output_path = create_comparison(row, classes)
            print("Saved:", output_path)
            saved += 1
        except Exception as error:
            print(f"Could not process pair: {error}")
            errors += 1

    print("\nVISUAL COMPARISON COMPLETED")
    print("Images saved:", saved)
    print("Pairs skipped due to errors:", errors)
    print("Output folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
