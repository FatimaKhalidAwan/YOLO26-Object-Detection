
from pathlib import Path
from collections import Counter
from math import isfinite

PROJECT_DIR = Path(__file__).resolve().parents[1]
LABEL_DIR = PROJECT_DIR / "dataset" / "labels"
SPLITS = ["train", "val", "test"]
NUM_CLASSES = 20
TOLERANCE = 1e-6


def diagnose_split(split):
    folder = LABEL_DIR / split

    if not folder.exists():
        raise FileNotFoundError(f"Label folder not found: {folder}")

    stats = Counter()
    examples = []

    for file in sorted(folder.glob("*.txt")):
        for line_no, line in enumerate(
            file.read_text(encoding="utf-8").splitlines(), 1
        ):
            parts = line.split()

            if len(parts) != 5:
                stats["invalid_format"] += 1
                continue

            try:
                class_id = int(parts[0])
                xc, yc, width, height = map(float, parts[1:])
            except ValueError:
                stats["invalid_number"] += 1
                continue

            if not all(isfinite(value) for value in (xc, yc, width, height)):
                stats["non_finite_value"] += 1
                continue

            if not 0 <= class_id < NUM_CLASSES:
                stats["invalid_class"] += 1

            if not (0 <= xc <= 1 and 0 <= yc <= 1):
                stats["invalid_center"] += 1

            if not (0 < width <= 1 and 0 < height <= 1):
                stats["invalid_size"] += 1

            if not (
                0 <= xc <= 1 and 0 <= yc <= 1
                and 0 < width <= 1 and 0 < height <= 1
            ):
                continue

            left = xc - width / 2
            right = xc + width / 2
            top = yc - height / 2
            bottom = yc + height / 2

            excess = max(0, -left, right - 1, -top, bottom - 1)

            if excess > 0:
                category = (
                    "tiny_boundary_rounding"
                    if excess <= TOLERANCE
                    else "large_boundary_violation"
                )
                stats[category] += 1

                if len(examples) < 5:
                    examples.append((file.name, line_no, excess))

    print(f"\n{split.upper()}")
    print("Annotation files:", len(list(folder.glob("*.txt"))))

    for key, value in sorted(stats.items()):
        print(f"{key}: {value}")

    print("Examples:", examples)


def main():
    for split in SPLITS:
        diagnose_split(split)


if __name__ == "__main__":
    main()
