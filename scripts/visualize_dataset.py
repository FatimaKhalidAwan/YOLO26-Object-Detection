
from pathlib import Path
from collections import Counter
import csv
import math
import random

import cv2
import matplotlib.pyplot as plt
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "dataset_audit"

SPLITS = ("train", "val", "test")
NUMBER_OF_IMAGES = 12
RANDOM_SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_config():
    config_path = PROJECT_DIR / "data.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    number_of_classes = int(config["nc"])
    names = config["names"]

    if isinstance(names, dict):
        classes = [names.get(i, names.get(str(i))) for i in range(number_of_classes)]
    else:
        classes = names

    if len(classes) != number_of_classes or any(name is None for name in classes):
        raise ValueError("Class names do not match the configured class count.")

    return number_of_classes, classes


def count_classes(number_of_classes):
    counts = Counter()

    for split in SPLITS:
        label_dir = DATASET_DIR / "labels" / split

        if not label_dir.is_dir():
            raise FileNotFoundError(f"Missing label directory: {label_dir}")

        label_files = sorted(label_dir.glob("*.txt"))
        print(f"Processing {split}: {len(label_files)} label files")

        for label_path in label_files:
            with open(label_path, "r", encoding="utf-8") as file:
                for line in file:
                    parts = line.split()

                    if len(parts) != 5:
                        continue

                    try:
                        class_id = int(parts[0])
                        values = [float(value) for value in parts[1:]]
                    except ValueError:
                        continue

                    if not all(math.isfinite(value) for value in values):
                        continue

                    if 0 <= class_id < number_of_classes:
                        counts[class_id] += 1

    return counts


def save_class_distribution(classes, counts):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    class_ids = range(len(classes))
    values = [counts[class_id] for class_id in class_ids]

    fig, ax = plt.subplots(figsize=(12, 9))
    bars = ax.barh(classes, values)

    ax.set_title("Object Distribution Across the Entire Dataset")
    ax.set_xlabel("Number of Annotated Objects")
    ax.set_ylabel("Object Classes")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for bar, count in zip(bars, values):
        ax.text(
            bar.get_width() + 8,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center"
        )

    fig.tight_layout()
    chart_path = OUTPUT_DIR / "class_distribution.png"
    fig.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    csv_path = OUTPUT_DIR / "class_distribution.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Class ID", "Class Name", "Object Count"])

        for class_id, class_name in enumerate(classes):
            writer.writerow([class_id, class_name, counts[class_id]])

    return chart_path, csv_path


def visualize_image(image_path, label_dir, classes):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read image: {image_path.name}")
        return None

    label_path = label_dir / f"{image_path.stem}.txt"

    if not label_path.is_file():
        print(f"Missing label: {image_path.name}")
        return None

    height, width = image.shape[:2]

    with open(label_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.split()

            if len(parts) != 5:
                continue

            try:
                class_id = int(parts[0])
                x_center, y_center, box_width, box_height = map(float, parts[1:])
            except ValueError:
                continue

            values = (x_center, y_center, box_width, box_height)

            if not 0 <= class_id < len(classes):
                continue

            if not all(math.isfinite(value) for value in values):
                continue

            if not (
                0 <= x_center <= 1
                and 0 <= y_center <= 1
                and 0 < box_width <= 1
                and 0 < box_height <= 1
                and x_center - box_width / 2 >= 0
                and y_center - box_height / 2 >= 0
                and x_center + box_width / 2 <= 1
                and y_center + box_height / 2 <= 1
            ):
                continue

            x1 = int((x_center - box_width / 2) * width)
            y1 = int((y_center - box_height / 2) * height)
            x2 = int((x_center + box_width / 2) * width)
            y2 = int((y_center + box_height / 2) * height)

            color = (
                (37 * class_id + 80) % 256,
                (97 * class_id + 150) % 256,
                (157 * class_id + 200) % 256
            )

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                classes[class_id],
                (x1, max(y1 - 7, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_annotated_samples(classes):
    image_dir = DATASET_DIR / "images" / "train"
    label_dir = DATASET_DIR / "labels" / "train"

    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError("Training image or label directory is missing.")

    images = sorted(
        image_path
        for image_path in image_dir.iterdir()
        if image_path.is_file()
        and image_path.suffix.lower() in IMAGE_EXTENSIONS
        and (label_dir / f"{image_path.stem}.txt").is_file()
    )

    if not images:
        raise FileNotFoundError("No training images with matching labels found.")

    rng = random.Random(RANDOM_SEED)
    samples = rng.sample(images, min(NUMBER_OF_IMAGES, len(images)))

    columns = 4
    rows = (len(samples) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(20, 5 * rows))
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    valid_samples = 0

    for ax, image_path in zip(axes, samples):
        image = visualize_image(image_path, label_dir, classes)

        if image is None:
            ax.axis("off")
            continue

        ax.imshow(image)
        ax.set_title(image_path.name, fontsize=10)
        ax.axis("off")
        valid_samples += 1

    for ax in axes[len(samples):]:
        ax.axis("off")

    fig.suptitle("Annotated Training Image Samples", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    samples_path = OUTPUT_DIR / "annotated_samples.png"
    fig.savefig(samples_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return samples_path, valid_samples


def main():
    number_of_classes, classes = load_config()

    print("=" * 55)
    print("YOLO DATASET VISUAL AUDIT")
    print("=" * 55)
    print("Number of classes:", number_of_classes)
    print("Classes:", classes)

    counts = count_classes(number_of_classes)
    chart_path, csv_path = save_class_distribution(classes, counts)
    samples_path, valid_samples = save_annotated_samples(classes)

    print("\n" + "=" * 55)
    print("VISUAL AUDIT COMPLETED")
    print("=" * 55)
    print("Total annotated objects:", sum(counts.values()))
    print("Number of classes:", number_of_classes)
    print("Valid sample images:", valid_samples)

    print("\nGenerated files:")
    print("1.", chart_path)
    print("2.", samples_path)
    print("3.", csv_path)

    print("\nClass distribution:")
    for class_id, class_name in enumerate(classes):
        print(f"{class_name}: {counts[class_id]}")


if __name__ == "__main__":
    main()
