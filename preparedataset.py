import argparse
import json
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def copy_or_preprocess(files, dest_dir: Path, target_size: int, preprocess: bool):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for image_path in files:
        dst_path = dest_dir / f"{image_path.stem}.jpg"
        if preprocess:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img = ImageOps.autocontrast(img)
                img = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
                img.save(dst_path, format="JPEG", quality=95, optimize=True)
        else:
            shutil.copy2(image_path, dst_path)


def build_dataset_stats(out_dir: Path):
    rows = []
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            cls_dir = out_dir / split / cls
            if not cls_dir.exists():
                continue
            for image_path in list_images(cls_dir):
                with Image.open(image_path) as img:
                    width, height = img.size
                rows.append({"split": split, "class": cls, "image": image_path.name, "width": width, "height": height})

    if not rows:
        return None

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(out_dir / "reports" / "dataset_stats.csv", index=False)

    summary = {
        "class_counts": stats_df.groupby(["class", "split"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "mean_size": stats_df.groupby("class")[["width", "height"]].mean().round(1).to_dict(orient="index"),
    }
    (out_dir / "reports" / "dataset_summary.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts = stats_df.groupby(["class", "split"]).size().unstack(fill_value=0)
    counts.plot(kind="bar", ax=axes[0], rot=0)
    axes[0].set_title("Class counts by split")
    axes[0].set_ylabel("Images")

    size_summary = stats_df.groupby("class")[["width", "height"]].mean()
    size_summary.plot(kind="bar", ax=axes[1], rot=0)
    axes[1].set_title("Average image size")
    axes[1].set_ylabel("Pixels")

    fig.tight_layout()
    fig.savefig(out_dir / "reports" / "dataset_overview.png", dpi=200)
    plt.close(fig)
    return stats_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, required=True,
                     help="Path containing Training/ and Testing/ folders")
    ap.add_argument("--out_dir", type=str, default="./data",
                     help="Output path for YOLO-formatted dataset")
    ap.add_argument("--val_split", type=float, default=0.1,
                     help="Fraction of Training data held out for validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--imgsz", type=int, default=256,
                     help="Resize images to this size before training")
    ap.add_argument("--preprocess", action="store_true",
                     help="Standardize contrast/size before training")
    args = ap.parse_args()

    random.seed(args.seed)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_src = raw_dir / "Training"
    test_src = raw_dir / "Testing"

    assert train_src.exists(), f"Not found: {train_src}"
    assert test_src.exists(), f"Not found: {test_src}"

    for cls in CLASSES:
        src_cls_dir = train_src / cls
        assert src_cls_dir.exists(), f"Missing class folder: {src_cls_dir}"
        images = list_images(src_cls_dir)
        random.shuffle(images)

        if len(images) <= 1:
            n_val = 0
        else:
            n_val = int(len(images) * args.val_split)
            if n_val < 1:
                n_val = 1
            if n_val >= len(images):
                n_val = max(1, len(images) // 2)

        val_imgs = images[:n_val]
        train_imgs = images[n_val:]

        copy_or_preprocess(train_imgs, out_dir / "train" / cls, args.imgsz, args.preprocess)
        copy_or_preprocess(val_imgs, out_dir / "val" / cls, args.imgsz, args.preprocess)

        test_cls_dir = test_src / cls
        test_imgs = list_images(test_cls_dir)
        copy_or_preprocess(test_imgs, out_dir / "test" / cls, args.imgsz, args.preprocess)

        print(f"[{cls:12s}] train={len(train_imgs):4d}  val={len(val_imgs):4d}  test={len(test_imgs):4d}")

    build_dataset_stats(out_dir)
    print(f"\nDone. YOLO-ready dataset written to: {out_dir.resolve()}")
    print(f"Dataset reports saved to: {reports_dir.resolve()}")


if __name__ == "__main__":
    main()
