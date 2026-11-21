
from pathlib import Path
from PIL import Image
import hashlib
import imagehash
import csv

PROJECT_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_DIR / "dataset" / "images"
SPLITS = ["train", "val", "test"]
PHASH_THRESHOLD = 6
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def calculate_sha256(image_path):
    sha256 = hashlib.sha256()
    with open(image_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def calculate_phash(image_path):
    with Image.open(image_path) as image:
        return int(str(imagehash.phash(image.convert("RGB"))), 16)


def save_csv(file_path, fieldnames, records):
    with open(file_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main():
    image_records = {split: [] for split in SPLITS}
    errors = []

    print("Starting dataset duplicate audit")

    for split in SPLITS:
        folder = IMAGE_DIR / split

        if not folder.exists():
            raise FileNotFoundError(f"Dataset folder not found: {folder}")

        image_paths = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"\nProcessing {split}: {len(image_paths)} images")

        for image_path in image_paths:
            try:
                image_records[split].append({
                    "path": image_path,
                    "sha": calculate_sha256(image_path),
                    "phash": calculate_phash(image_path)
                })
            except Exception as error:
                errors.append({
                    "path": str(image_path),
                    "error": str(error)
                })

        print(f"Successfully processed: {len(image_records[split])}")

    all_images = [
        record
        for split in SPLITS
        for record in image_records[split]
    ]

    print("\nChecking exact duplicates...")

    sha_groups = {}
    for record in all_images:
        sha_groups.setdefault(record["sha"], []).append(record)

    matches = []

    for group in sha_groups.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                first, second = group[i], group[j]

                if first["path"].parent.name == second["path"].parent.name:
                    continue

                matches.append({
                    "image_1": str(first["path"]),
                    "image_2": str(second["path"]),
                    "split_1": first["path"].parent.name,
                    "split_2": second["path"].parent.name,
                    "match_type": "Exact duplicate (SHA-256)",
                    "phash_distance": 0
                })

    print("Checking near duplicates using pHash...")

    split_pairs = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test")
    ]

    for split_1, split_2 in split_pairs:
        print(f"Comparing {split_1} vs {split_2}...")

        for first in image_records[split_1]:
            for second in image_records[split_2]:
                if first["sha"] == second["sha"]:
                    continue

                distance = (first["phash"] ^ second["phash"]).bit_count()

                if distance <= PHASH_THRESHOLD:
                    matches.append({
                        "image_1": str(first["path"]),
                        "image_2": str(second["path"]),
                        "split_1": split_1,
                        "split_2": split_2,
                        "match_type": "Near duplicate (pHash)",
                        "phash_distance": distance
                    })

    fields = [
        "image_1", "image_2", "split_1",
        "split_2", "match_type", "phash_distance"
    ]

    output_file = PROJECT_DIR / "duplicate_audit.csv"
    error_file = PROJECT_DIR / "duplicate_audit_errors.csv"

    save_csv(output_file, fields, matches)
    save_csv(error_file, ["path", "error"], errors)

    exact_count = sum(
        item["match_type"].startswith("Exact") for item in matches
    )
    near_count = sum(
        item["match_type"].startswith("Near") for item in matches
    )

    print("\nDuplicate audit summary")
    print("Total readable images:", len(all_images))
    print("Exact duplicate pairs:", exact_count)
    print("Near duplicate candidates:", near_count)
    print("Unreadable images:", len(errors))
    print("Results saved to:", output_file)
    print("Image errors saved to:", error_file)
    print("Audit finished")


if __name__ == "__main__":
    main()
