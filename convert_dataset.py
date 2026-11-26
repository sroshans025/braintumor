#!/usr/bin/env python3
"""
Build the YOLO classification dataset from the raw images.

Input:  dataset/train/{class}/ and dataset/test/{class}/
Output: data/train/{class}/, data/val/{class}/, data/test/{class}/

- Exact duplicate images (same file bytes) are kept only once, so the same scan
  can never land in both train and val.
- Train/val split is stratified per class and seeded, so it is reproducible.
- The output folder is rebuilt from scratch each run (no stale or synthetic files).
- Any image that also appears in test is dropped from train/val (leakage guard).
"""

import argparse
import hashlib
import random
import shutil
from pathlib import Path

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def unique_images(folder: Path, seen: set) -> tuple[list[Path], int]:
    """Images in folder whose bytes have not been seen yet; returns (kept, dropped)."""
    kept, dropped = [], 0
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        h = md5(p)
        if h in seen:
            dropped += 1
            continue
        seen.add(h)
        kept.append(p)
    return kept, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="./dataset", help="Raw dataset with train/ and test/")
    ap.add_argument("--out_dir", default="./data", help="Output YOLO dataset")
    ap.add_argument("--val_split", type=float, default=0.15, help="Validation share of train")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw, out = Path(args.raw_dir), Path(args.out_dir)
    if out.exists():
        shutil.rmtree(out)
    rng = random.Random(args.seed)

    # Test first, so its hashes can be excluded from train/val.
    test_seen: set = set()
    counts = {}
    for cls in CLASSES:
        imgs, dropped = unique_images(raw / "test" / cls, test_seen)
        dest = out / "test" / cls
        dest.mkdir(parents=True)
        for p in imgs:
            shutil.copy2(p, dest / p.name)
        counts[("test", cls)] = len(imgs)
        print(f"test  {cls:11s} kept {len(imgs):4d}  dropped duplicates {dropped}")

    train_seen = set(test_seen)
    for cls in CLASSES:
        imgs, dropped = unique_images(raw / "train" / cls, train_seen)
        rng.shuffle(imgs)
        n_val = round(len(imgs) * args.val_split)
        for split, chunk in (("val", imgs[:n_val]), ("train", imgs[n_val:])):
            dest = out / split / cls
            dest.mkdir(parents=True)
            for p in chunk:
                shutil.copy2(p, dest / p.name)
            counts[(split, cls)] = len(chunk)
        print(f"train {cls:11s} kept {len(imgs):4d}  dropped duplicates/test-overlap {dropped}")

    print("\nSplit sizes")
    for split in ("train", "val", "test"):
        row = "  ".join(f"{c}={counts[(split, c)]}" for c in CLASSES)
        print(f"  {split:5s} {row}  total={sum(counts[(split, c)] for c in CLASSES)}")


if __name__ == "__main__":
    main()
