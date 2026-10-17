from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(r"C:\QASense-AI\datasets\processed\visionqa_atomblock")

CLASS_NAMES = {
    0: "button",
    1: "link",
    2: "input",
    3: "image",
    4: "icon",
    5: "checkbox",
    6: "radio",
    7: "select",
    8: "block-nav",
    9: "block-sidebar",
    10: "block-footer",
    11: "block-form",
    12: "block-table",
    13: "block",
}

all_ok = True

for split in ["train", "val", "test"]:
    label_dir = ROOT / split / "labels"

    class_counts = Counter()

    invalid_lines = []
    invalid_classes = []
    invalid_boxes = []
    empty_labels = []

    total_boxes = 0

    for label_file in label_dir.glob("*.txt"):
        text = label_file.read_text(encoding="utf-8").strip()

        if not text:
            empty_labels.append(label_file.name)
            continue

        for line_num, line in enumerate(text.splitlines(), start=1):
            parts = line.strip().split()

            if len(parts) != 5:
                invalid_lines.append(
                    (label_file.name, line_num, line)
                )
                continue

            try:
                class_id = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                invalid_lines.append(
                    (label_file.name, line_num, line)
                )
                continue

            if class_id not in CLASS_NAMES:
                invalid_classes.append(
                    (label_file.name, line_num, class_id)
                )
                continue

            if not (
                0 <= x <= 1
                and 0 <= y <= 1
                and 0 < w <= 1
                and 0 < h <= 1
            ):
                invalid_boxes.append(
                    (label_file.name, line_num, x, y, w, h)
                )
                continue

            class_counts[class_id] += 1
            total_boxes += 1

    print(f"\n{'=' * 60}")
    print(f"SPLIT: {split.upper()}")
    print(f"{'=' * 60}")

    print(f"Total bounding boxes: {total_boxes}")
    print(f"Empty label files: {len(empty_labels)}")
    print(f"Invalid lines: {len(invalid_lines)}")
    print(f"Invalid class IDs: {len(invalid_classes)}")
    print(f"Invalid bounding boxes: {len(invalid_boxes)}")

    print("\nClass distribution:")

    for class_id in range(14):
        count = class_counts[class_id]
        percentage = (
            (count / total_boxes * 100)
            if total_boxes
            else 0
        )

        print(
            f"{class_id:2d} "
            f"{CLASS_NAMES[class_id]:15s} "
            f"{count:8d} "
            f"{percentage:6.2f}%"
        )

    if (
        invalid_lines
        or invalid_classes
        or invalid_boxes
    ):
        all_ok = False

        print("\nFirst few problems:")

        for item in invalid_lines[:5]:
            print("Invalid line:", item)

        for item in invalid_classes[:5]:
            print("Invalid class:", item)

        for item in invalid_boxes[:5]:
            print("Invalid box:", item)


print("\n" + "=" * 60)

if all_ok:
    print("AUDIT RESULT: PASS")
else:
    print("AUDIT RESULT: FAIL - review invalid annotations")

print("=" * 60)