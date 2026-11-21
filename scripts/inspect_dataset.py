
import xml.etree.ElementTree as ET
from collections import Counter
from math import isfinite
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset" / "original"

CLASSES = {
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def inspect_dataset():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

    files = list(DATASET_DIR.iterdir())

    image_groups = {}
    xml_groups = {}

    for path in files:
        if not path.is_file():
            continue

        if path.suffix.lower() in IMAGE_EXTENSIONS:
            image_groups.setdefault(path.stem, []).append(path)
        elif path.suffix.lower() == ".xml":
            xml_groups.setdefault(path.stem, []).append(path)

    duplicate_images = [
        stem for stem, paths in image_groups.items() if len(paths) > 1
    ]
    duplicate_xml = [
        stem for stem, paths in xml_groups.items() if len(paths) > 1
    ]

    images = {stem: paths[0] for stem, paths in image_groups.items()}
    xml_files = {stem: paths[0] for stem, paths in xml_groups.items()}

    missing_xml = sorted(set(images) - set(xml_files))
    missing_images = sorted(set(xml_files) - set(images))

    corrupted_xml = []
    invalid_boxes = []
    unknown_classes = []
    class_counts = Counter()

    for stem, xml_path in xml_files.items():
        try:
            root = ET.parse(xml_path).getroot()

            if root.tag != "annotation":
                raise ValueError("Invalid XML root element")

            size = root.find("size")
            if size is None:
                raise ValueError("Missing image dimensions")

            width = float(size.findtext("width", "nan"))
            height = float(size.findtext("height", "nan"))

            if not (
                isfinite(width) and isfinite(height)
                and width > 0 and height > 0
            ):
                raise ValueError("Invalid image dimensions")

            for index, obj in enumerate(root.findall("object"), 1):
                class_name = (obj.findtext("name") or "").strip()

                if not class_name:
                    corrupted_xml.append((stem, f"Object {index}: missing class"))
                    continue

                class_counts[class_name] += 1

                if class_name not in CLASSES:
                    unknown_classes.append((stem, class_name))

                bbox = obj.find("bndbox")
                if bbox is None:
                    invalid_boxes.append((stem, class_name, "Missing bounding box"))
                    continue

                try:
                    xmin = float(bbox.findtext("xmin", "nan"))
                    ymin = float(bbox.findtext("ymin", "nan"))
                    xmax = float(bbox.findtext("xmax", "nan"))
                    ymax = float(bbox.findtext("ymax", "nan"))

                    valid = all(
                        isfinite(value)
                        for value in (xmin, ymin, xmax, ymax)
                    ) and (
                        1 <= xmin <= xmax <= width
                        and 1 <= ymin <= ymax <= height
                    )

                    if not valid:
                        invalid_boxes.append((stem, class_name, "Invalid coordinates"))

                except (ValueError, TypeError):
                    invalid_boxes.append((stem, class_name, "Non-numeric coordinates"))

        except (ET.ParseError, OSError, ValueError, TypeError) as error:
            corrupted_xml.append((stem, str(error)))

    print("\nDATASET INSPECTION REPORT")
    print("=" * 40)
    print("Dataset location:", DATASET_DIR)
    print("Total images:", len(images))
    print("Total XML files:", len(xml_files))
    print("Missing XML files:", len(missing_xml))
    print("Missing images:", len(missing_images))
    print("Duplicate image stems:", len(duplicate_images))
    print("Duplicate XML stems:", len(duplicate_xml))
    print("Corrupted XML files:", len(corrupted_xml))
    print("Invalid bounding boxes:", len(invalid_boxes))
    print("Unknown class annotations:", len(unknown_classes))

    print("\nOBJECT CLASS DISTRIBUTION")
    for name, count in class_counts.most_common():
        print(f"{name}: {count}")

    print("\nTotal classes:", len(class_counts))
    print("Total annotated objects:", sum(class_counts.values()))

    problems = {
        "Images without XML": missing_xml,
        "XML without images": missing_images,
        "Duplicate image stems": duplicate_images,
        "Duplicate XML stems": duplicate_xml,
        "Corrupted XML": corrupted_xml,
        "Invalid bounding boxes": invalid_boxes,
        "Unknown classes": unknown_classes
    }

    for label, items in problems.items():
        if items:
            print(f"\n{label} (first 20):")
            for item in items[:20]:
                print(item)

    print("\nInspection completed.")


if __name__ == "__main__":
    inspect_dataset()
