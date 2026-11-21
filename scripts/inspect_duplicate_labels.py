
import csv
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
LABEL_DIR = PROJECT_DIR / "dataset" / "labels"
CSV_FILE = PROJECT_DIR / "duplicate_audit.csv"


def read_annotations(label_path):
    annotations = []

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()

        if len(parts) != 5:
            raise ValueError(f"Invalid annotation format: {label_path.name}")

        class_id = int(parts[0])
        values = tuple(float(value) for value in parts[1:])

        annotations.append((class_id, values))

    return annotations


def compare_annotations():
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"Duplicate audit CSV not found: {CSV_FILE}")

    with open(CSV_FILE, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    exact_pairs = [
        row for row in rows
        if row.get("match_type", "").startswith("Exact")
    ]

    print("COMPARING EXACT DUPLICATE ANNOTATIONS")
    print("Exact duplicate pairs:", len(exact_pairs))

    identical = 0
    different = 0
    missing = 0
    invalid = 0

    for row in exact_pairs:
        split_1 = row.get("split_1", "")
        split_2 = row.get("split_2", "")

        if split_1 not in {"train", "val", "test"}:
            print("\nInvalid first split:", split_1)
            invalid += 1
            continue

        if split_2 not in {"train", "val", "test"}:
            print("\nInvalid second split:", split_2)
            invalid += 1
            continue

        image_1 = Path(row.get("image_1", ""))
        image_2 = Path(row.get("image_2", ""))

        label_1 = LABEL_DIR / split_1 / f"{image_1.stem}.txt"
        label_2 = LABEL_DIR / split_2 / f"{image_2.stem}.txt"

        print(f"\n{image_1.name} ({split_1}) vs {image_2.name} ({split_2})")

        if not label_1.is_file() or not label_2.is_file():
            print("RESULT: Annotation file missing")
            missing += 1
            continue

        try:
            annotations_1 = read_annotations(label_1)
            annotations_2 = read_annotations(label_2)
        except (ValueError, OSError) as error:
            print("RESULT: Invalid annotation:", error)
            invalid += 1
            continue

        if annotations_1 == annotations_2:
            print("RESULT: Annotations are identical")
            identical += 1
        else:
            print("RESULT: Annotations are different")
            different += 1

    print("\nANNOTATION COMPARISON SUMMARY")
    print("Identical:", identical)
    print("Different:", different)
    print("Missing:", missing)
    print("Invalid:", invalid)
    print("Comparison completed.")


if __name__ == "__main__":
    compare_annotations()
