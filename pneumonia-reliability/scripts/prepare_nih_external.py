"""Organize NIH ChestX-ray14 images for pneumreliab external evaluation.

Usage:
  python scripts/prepare_nih_external.py --images PATH_TO_EXTRACTED_IMAGES \
      --csv PATH_TO_Data_Entry_2017_v2020.csv --out data/external

Images labeled with Pneumonia (including co-occurring findings) go into
PNEUMONIA. Images labeled exactly No Finding go into NORMAL. Other images are
excluded because this project evaluates a binary pneumonia task.
"""
import argparse
import csv
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path,
                        help="Root folder containing extracted NIH image archives")
    parser.add_argument("--csv", required=True, type=Path,
                        help="NIH Data_Entry_2017_v2020.csv file")
    parser.add_argument("--out", type=Path, default=Path("data/external"),
                        help="Output folder (default: data/external)")
    args = parser.parse_args()

    if not args.images.is_dir():
        parser.error(f"Image folder does not exist: {args.images}")
    if not args.csv.is_file():
        parser.error(f"Label CSV does not exist: {args.csv}")

    # Index once because NIH stores images in multiple archive subfolders.
    images = {}
    for path in args.images.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.setdefault(path.name.lower(), path)

    counts = {"NORMAL": 0, "PNEUMONIA": 0}
    missing = 0
    with args.csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"Image Index", "Finding Labels"}.issubset(reader.fieldnames):
            parser.error("CSV must contain 'Image Index' and 'Finding Labels' columns")

        for row in reader:
            name = row["Image Index"].strip()
            labels = {label.strip() for label in row["Finding Labels"].split("|")}
            if "Pneumonia" in labels:
                cls = "PNEUMONIA"
            elif labels == {"No Finding"}:
                cls = "NORMAL"
            else:
                continue

            source = images.get(name.lower())
            if source is None:
                missing += 1
                continue

            destination_dir = args.out / cls
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / source.name
            if not destination.exists():
                shutil.copy2(source, destination)
            counts[cls] += 1

    print(f"Prepared NIH external set at: {args.out.resolve()}")
    print(f"NORMAL: {counts['NORMAL']}")
    print(f"PNEUMONIA: {counts['PNEUMONIA']}")
    if missing:
        print(f"CSV images not found in extracted image folder: {missing}")
    if not all(counts.values()):
        raise SystemExit("Could not prepare both classes. Check the CSV and extracted image paths.")


if __name__ == "__main__":
    main()
