from pathlib import Path
import os

SRC = Path(r"C:\QASense-AI\datasets\atomblock-webui\yolo-dataset")
DST = Path(r"C:\QASense-AI\datasets\processed\visionqa_atomblock")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

for split in ["train", "val", "test"]:
    src_images = SRC / split / "images"
    src_labels = SRC / split / "labels"

    dst_images = DST / split / "images"
    dst_labels = DST / split / "labels"

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    images = [
        p for p in src_images.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    copied = 0
    skipped = 0

    for i, image in enumerate(images, start=1):
        label = src_labels / f"{image.stem}.txt"

        if not label.exists():
            skipped += 1
            continue

        dst_img = dst_images / image.name
        dst_lbl = dst_labels / label.name

        if not dst_img.exists():
            os.link(image, dst_img)

        if not dst_lbl.exists():
            os.link(label, dst_lbl)

        copied += 1

        if i % 500 == 0:
            print(f"{split}: processed {i}/{len(images)}")

    print(
        f"{split}: matched={copied}, "
        f"missing_labels={skipped}"
    )

print("\nPreparation complete.")