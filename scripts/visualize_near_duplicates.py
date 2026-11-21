
from pathlib import Path
import csv
import math

import yaml
from PIL import Image, ImageDraw

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset"
CSV_FILE = PROJECT_DIR / "duplicate_audit.csv"
OUTPUT_DIR = PROJECT_DIR / "duplicate_review" / "near_duplicate_candidates"
CONFIG_FILE = PROJECT_DIR / "data.yaml"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_SIZE = (720, 520)


def load_class_names():
    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    number_of_classes = int(config["nc"])
    names = config["names"]

    if isinstance(names, dict):
        classes = [
            names.get(i, names.get(str(i)))
            for i in range(number_of_classes)
        ]
    else:
        classes = names

    if len(classes) != number_of_classes or any(name is None for name in classes):
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


def draw_annotations(image, label_path, classes):
    draw = ImageDraw.Draw(image)
    width, height = image.size

    if not label_path.is_file():
        print(f"Missing label: {label_path.name}")
        return

    with open(label_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.split()

            if len(parts) != 5:
                continue

            try:
                class_id = int(parts[0])
                x, y, box_width, box_height = map(float, parts[1:])
            except ValueError:
                continue

            values = (x, y, box_width, box_height)

            if not 0 <= class_id < len(classes):
                continue

            if not all(math.isfinite(value) for value in values):
                continue

            if not (
                0 <= x <= 1 and 0 <= y <= 1
                and 0 < box_width <= 1 and 0 < box_height <= 1
                and x - box_width / 2 >= 0
                and y - box_height / 2 >= 0
                and x + box_width / 2 <= 1
                and y + box_height / 2 <= 1
            ):
                continue

            x1 = (x - box_width / 2) * width
            y1 = (y - box_height / 2) * height
            x2 = (x + box_width / 2) * width
            y2 = (y + box_height / 2) * height

            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            draw.text((x1, max(0, y1 - 16)), classes[class_id], fill="red")


def prepare_image(image_path, split, classes):
    with Image.open(image_path) as source:
        image = source.convert("RGB")

    label_path = (
        DATASET_DIR / "labels" / split / f"{image_path.stem}.txt"
    )

    draw_annotations(image, label_path, classes)
    image.thumbnail(IMAGE_SIZE)

    return image


def create_comparison(row, index, classes):
    split_1 = row["split_1"]
    split_2 = row["split_2"]

    image_1_path = resolve_image_path(row["image_1"], split_1)
    image_2_path = resolve_image_path(row["image_2"], split_2)

    image_1 = prepare_image(image_1_path, split_1, classes)
    image_2 = prepare_image(image_2_path, split_2, classes)

    gap = 20
    header_height = 75

    canvas_width = image_1.width + image_2.width + gap
    canvas_height = max(image_1.height, image_2.height) + header_height

    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text(
        (10, 10),
        f"{image_1_path.name} ({split_1})",
        fill="black"
    )
    draw.text(
        (image_1.width + gap, 10),
        f"{image_2_path.name} ({split_2})",
        fill="black"
    )
    draw.text(
        (10, 40),
        f"pHash distance: {row['phash_distance']}",
        fill="blue"
    )

    canvas.paste(image_1, (0, header_height))
    canvas.paste(image_2, (image_1.width + gap, header_height))

    output_path = OUTPUT_DIR / f"near_duplicate_{index}.png"
    canvas.save(output_path)

    return output_path


def main():
    if not CSV_FILE.is_file():
        raise FileNotFoundError(f"Duplicate audit CSV not found: {CSV_FILE}")

    classes = load_class_names()

    with open(CSV_FILE, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        required_columns = {
            "image_1", "image_2", "split_1", "split_2",
            "match_type", "phash_distance"
        }

        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError("Duplicate audit CSV is missing required columns.")

        rows = [
            row for row in reader
            if "near" in (row.get("match_type") or "").lower()
        ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("VISUALIZING NEAR-DUPLICATE CANDIDATES")
    print("Candidate pairs:", len(rows))

    saved = 0
    errors = 0

    for index, row in enumerate(rows, start=1):
        try:
            output_path = create_comparison(row, index, classes)
            print("Saved:", output_path)
            saved += 1
        except Exception as error:
            print(f"Could not process pair {index}: {error}")
            errors += 1

    print("\nVISUALIZATION COMPLETED")
    print("Images saved:", saved)
    print("Pairs skipped due to errors:", errors)
    print("Output folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
