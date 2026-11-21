
from collections import Counter
from math import isfinite
from pathlib import Path
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
YAML_FILE = PROJECT_DIR / "data.yaml"
TOLERANCE = 1e-6
SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_config():
    if not YAML_FILE.is_file():
        raise FileNotFoundError(f"Configuration not found: {YAML_FILE}")

    config = yaml.safe_load(YAML_FILE.read_text(encoding="utf-8"))

    if not isinstance(config, dict):
        raise ValueError("Invalid data.yaml configuration")

    num_classes = config.get("nc")
    names = config.get("names")

    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("Invalid number of classes in data.yaml")

    if isinstance(names, list):
        class_names = names
    elif isinstance(names, dict):
        try:
            class_names = [names[i] for i in range(num_classes)]
        except KeyError as error:
            raise ValueError("Class IDs in names must be consecutive from 0") from error
    else:
        raise ValueError("Invalid class names in data.yaml")

    if len(class_names) != num_classes:
        raise ValueError("Class count does not match nc in data.yaml")

    dataset_path = Path(config.get("path", ""))
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_DIR / dataset_path

    return dataset_path, num_classes, class_names


def validate_annotation(parts, num_classes):
    if len(parts) != 5:
        return None, "Invalid annotation format"

    try:
        class_id = int(parts[0])
        xc, yc, width, height = map(float, parts[1:])
    except ValueError:
        return None, "Invalid numeric value"

    if not all(isfinite(value) for value in (xc, yc, width, height)):
        return None, "Non-finite coordinate"

    if not 0 <= class_id < num_classes:
        return None, "Invalid class ID"

    if not (
        0 <= xc <= 1 and 0 <= yc <= 1
        and 0 < width <= 1 and 0 < height <= 1
    ):
        return None, "Invalid center or box size"

    left = xc - width / 2
    right = xc + width / 2
    top = yc - height / 2
    bottom = yc + height / 2

    if not (
        left >= -TOLERANCE and right <= 1 + TOLERANCE
        and top >= -TOLERANCE and bottom <= 1 + TOLERANCE
    ):
        return None, "Bounding box exceeds image boundaries"

    return class_id, None


def verify_split(dataset_dir, split, num_classes, class_names):
    image_dir = dataset_dir / "images" / split
    label_dir = dataset_dir / "labels" / split

    errors = []
    counts = Counter()
    objects = 0
    empty_labels = 0

    if not image_dir.is_dir():
        return {
            "images": 0, "labels": 0, "objects": 0,
            "empty": 0, "counts": counts,
            "errors": [f"Missing image directory: {image_dir}"]
        }

    if not label_dir.is_dir():
        return {
            "images": 0, "labels": 0, "objects": 0,
            "empty": 0, "counts": counts,
            "errors": [f"Missing label directory: {label_dir}"]
        }

    images = [
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    labels = list(label_dir.glob("*.txt"))

    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}

    if len(image_stems) != len(images):
        errors.append("Duplicate image filename stems detected")

    for stem in sorted(image_stems - label_stems):
        errors.append(f"Missing label: {stem}")

    for stem in sorted(label_stems - image_stems):
        errors.append(f"Missing image: {stem}")

    for label_path in labels:
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()

            if not any(line.strip() for line in lines):
                empty_labels += 1
                continue

            for line_number, line in enumerate(lines, 1):
                if not line.strip():
                    continue

                class_id, error = validate_annotation(
                    line.split(), num_classes
                )

                if error:
                    errors.append(
                        f"{label_path.name}, line {line_number}: {error}"
                    )
                    continue

                counts[class_id] += 1
                objects += 1

        except (OSError, UnicodeError) as error:
            errors.append(f"Cannot read {label_path.name}: {error}")

    print(f"\n{split.upper()} DATASET")
    print("Images:", len(images))
    print("Labels:", len(labels))
    print("Empty labels:", empty_labels)
    print("Valid objects:", objects)
    print("Errors:", len(errors))

    print("\nClass distribution:")
    for class_id, name in enumerate(class_names):
        print(f"{class_id}: {name}: {counts[class_id]}")

    if errors:
        print("\nFirst 20 errors:")
        for error in errors[:20]:
            print("-", error)

    return {
        "images": len(images),
        "labels": len(labels),
        "objects": objects,
        "empty": empty_labels,
        "counts": counts,
        "errors": errors
    }


def main():
    dataset_dir, num_classes, class_names = load_config()

    print("YOLO DATASET VERIFICATION")
    print("Dataset:", dataset_dir)
    print("Classes:", num_classes)

    results = {}
    for split in SPLITS:
        results[split] = verify_split(
            dataset_dir, split, num_classes, class_names
        )

    total_images = sum(item["images"] for item in results.values())
    total_labels = sum(item["labels"] for item in results.values())
    total_objects = sum(item["objects"] for item in results.values())
    total_errors = sum(len(item["errors"]) for item in results.values())

    print("\nFINAL VERIFICATION SUMMARY")
    print("Total images:", total_images)
    print("Total label files:", total_labels)
    print("Total valid objects:", total_objects)
    print("Total errors:", total_errors)

    if total_errors == 0:
        print("\nSUCCESS: Dataset verification passed.")
    else:
        print("\nFAILED: Dataset issues were detected.")

    return total_errors == 0


if __name__ == "__main__":
    if not main():
        raise SystemExit(1)
