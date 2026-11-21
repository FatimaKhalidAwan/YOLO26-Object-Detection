
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "dataset" / "original"
DATASET_DIR = PROJECT_DIR / "dataset"
IMAGE_DIR = DATASET_DIR / "images"
LABEL_DIR = DATASET_DIR / "labels"

RANDOM_SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def convert_xml_to_yolo(xml_path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")

    if size is None:
        raise ValueError(f"Missing image dimensions: {xml_path.name}")

    width = float(size.findtext("width"))
    height = float(size.findtext("height"))

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {xml_path.name}")

    annotations = []

    for obj in root.findall("object"):
        class_name = obj.findtext("name")
        bbox = obj.find("bndbox")

        if class_name not in CLASS_TO_ID:
            raise ValueError(f"Unknown class '{class_name}': {xml_path.name}")

        if bbox is None:
            raise ValueError(f"Missing bounding box: {xml_path.name}")

        xmin = float(bbox.findtext("xmin"))
        ymin = float(bbox.findtext("ymin"))
        xmax = float(bbox.findtext("xmax"))
        ymax = float(bbox.findtext("ymax"))

        if not (1 <= xmin <= xmax <= width and 1 <= ymin <= ymax <= height):
            raise ValueError(f"Invalid bounding box in {xml_path.name}")

        x_min, y_min = xmin - 1, ymin - 1
        x_max, y_max = xmax, ymax

        box_width = round((x_max - x_min) / width, 8)
        box_height = round((y_max - y_min) / height, 8)
        x_center = ((x_min + x_max) / 2) / width
        y_center = ((y_min + y_max) / 2) / height

        x_center = max(box_width / 2, min(x_center, 1 - box_width / 2))
        y_center = max(box_height / 2, min(y_center, 1 - box_height / 2))

        class_id = CLASS_TO_ID[class_name]
        annotations.append(
            f"{class_id} {x_center:.8f} {y_center:.8f} "
            f"{box_width:.8f} {box_height:.8f}"
        )

    return annotations


def check_output_folders():
    for split in ["train", "val", "test"]:
        for folder in [IMAGE_DIR / split, LABEL_DIR / split]:
            if folder.exists() and any(folder.iterdir()):
                raise FileExistsError(
                    f"{folder} is not empty. Remove previous output before converting."
                )


def collect_images():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Original dataset folder not found: {SOURCE_DIR}")

    images = sorted(
        path for path in SOURCE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise FileNotFoundError(f"No images found in {SOURCE_DIR}")

    stems = [image.stem for image in images]

    if len(stems) != len(set(stems)):
        raise ValueError("Duplicate image filenames detected.")

    xml_files = list(SOURCE_DIR.glob("*.xml"))
    xml_stems = [xml.stem for xml in xml_files]

    if len(xml_stems) != len(set(xml_stems)):
        raise ValueError("Duplicate XML annotation filenames detected.")

    image_stems = set(stems)
    xml_stem_set = set(xml_stems)

    missing_xml = image_stems - xml_stem_set
    unmatched_xml = xml_stem_set - image_stems

    if missing_xml:
        raise ValueError(f"Missing XML annotation for: {sorted(missing_xml)[0]}")

    if unmatched_xml:
        raise ValueError(f"XML without matching image: {sorted(unmatched_xml)[0]}")

    return images


def split_dataset(images):
    random.seed(RANDOM_SEED)
    images = images.copy()
    random.shuffle(images)

    total = len(images)
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    return {
        "train": images[:train_count],
        "val": images[train_count:train_count + val_count],
        "test": images[train_count + val_count:]
    }


def create_dataset_folders():
    for split in ["train", "val", "test"]:
        (IMAGE_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABEL_DIR / split).mkdir(parents=True, exist_ok=True)


def process_split(split_name, image_list):
    print(f"\nProcessing {split_name}...")
    object_counter = Counter()

    for image_path in image_list:
        xml_path = SOURCE_DIR / f"{image_path.stem}.xml"
        annotations = convert_xml_to_yolo(xml_path)

        shutil.copy2(image_path, IMAGE_DIR / split_name / image_path.name)

        label_path = LABEL_DIR / split_name / f"{image_path.stem}.txt"
        label_path.write_text(
            "\n".join(annotations) + ("\n" if annotations else ""),
            encoding="utf-8"
        )

        for annotation in annotations:
            class_id = int(annotation.split()[0])
            object_counter[CLASSES[class_id]] += 1

    print(f"Images: {len(image_list)}")
    print(f"Objects: {sum(object_counter.values())}")
    return object_counter


def create_data_yaml():
    yaml_path = PROJECT_DIR / "data.yaml"
    content = [
        "path: dataset",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(CLASSES)}",
        "",
        "names:"
    ]

    content.extend(
        f"  {index}: {name}" for index, name in enumerate(CLASSES)
    )

    yaml_path.write_text("\n".join(content) + "\n", encoding="utf-8")
    print(f"\nYOLO configuration saved: {yaml_path}")


def main():
    print("PASCAL VOC TO YOLO DATASET CONVERSION")
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {DATASET_DIR}")

    check_output_folders()
    images = collect_images()
    splits = split_dataset(images)

    print("\nDataset split:")
    for name, image_list in splits.items():
        print(f"{name}: {len(image_list)}")

    create_dataset_folders()

    total_objects = Counter()

    for split_name, image_list in splits.items():
        total_objects.update(process_split(split_name, image_list))

    create_data_yaml()

    print("\nDATASET CONVERSION COMPLETED")
    print(f"Total images: {len(images)}")
    print(f"Total objects: {sum(total_objects.values())}")

    print("\nObject distribution:")
    for class_name in CLASSES:
        print(f"{class_name:15s}: {total_objects[class_name]}")

    print("\nNext step: Run verify_yolo_dataset.py")


if __name__ == "__main__":
    main()
