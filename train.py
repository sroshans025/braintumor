import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO


def plot_training_metrics(run_dir: Path, output_path: Path):
    results_csv = next(run_dir.rglob("results.csv"), None)
    if results_csv is None:
        print("No training results.csv was found; skipping curve plot")
        return

    df = pd.read_csv(results_csv)
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    loss_cols = [col for col in df.columns if "loss" in col.lower()]
    score_cols = [col for col in df.columns if "accuracy" in col.lower() or "top1" in col.lower() or "top5" in col.lower()]

    if loss_cols:
        for col in loss_cols:
            axes[0].plot(df.index + 1, df[col], label=col)
        axes[0].set_title("Training / Validation Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend(loc="best")

    if score_cols:
        for col in score_cols:
            axes[1].plot(df.index + 1, df[col], label=col)
        axes[1].set_title("Accuracy Metrics")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].legend(loc="best")

    if not loss_cols and not score_cols:
        axes[0].text(0.5, 0.5, "No metric columns found", ha="center", va="center")
        axes[1].axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="./data",
                     help="Path to dataset root containing train/ and val/")
    ap.add_argument("--model", type=str, default="yolo11m-cls.pt",
                     help="Base model / weights to start from")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--patience", type=int, default=15,
                     help="Early-stopping patience (epochs with no improvement)")
    ap.add_argument("--project", type=str, default="runs_classify")
    ap.add_argument("--name", type=str, default="brain_tumor_yolo11")
    ap.add_argument("--device", type=str, default="0",
                     help="'0' for first GPU, 'cpu' for CPU, '0,1' for multi-GPU")
    args = ap.parse_args()

    data_root = Path(args.data)
    has_val = (data_root / "val").exists() and any((data_root / "val").iterdir())
    if has_val:
        val_flag = True
    else:
        val_flag = False
        print("No validation images found; training will proceed without validation.")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
        cache=True,
        workers=max(2, min(8, os.cpu_count() or 2)),
        amp=True,
        hsv_h=0.0,          # MRI is greyscale: hue/saturation jitter adds nothing
        hsv_s=0.0,
        hsv_v=0.3,
        degrees=15,
        translate=0.15,
        scale=0.25,
        shear=0.1,
        flipud=0.0,         # scans are acquired upright; vertical flips create unrealistic anatomy
        fliplr=0.5,
        erasing=0.3,
        optimizer="AdamW",  # "auto" would silently ignore lr0
        lr0=0.0005,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        momentum=0.937,
        weight_decay=0.0005,
        label_smoothing=0.05,
        dropout=0.1,
        val=val_flag,
        plots=True,
        save=True,
        seed=42,
    )

    run_dir = getattr(model.trainer, "save_dir", None)
    if run_dir is None or not Path(run_dir).exists():
        run_dir = Path(args.project) / args.name
        if not run_dir.exists():
            run_dir = Path("runs") / "classify" / Path(args.project).name / args.name

    run_dir = Path(run_dir)
    plot_training_metrics(run_dir, run_dir / "training_metrics.png")

    print("\nTraining complete.")
    print(f"Best weights saved at: {run_dir / 'weights' / 'best.pt'}")
    print(f"Training curves saved at: {run_dir / 'training_metrics.png'}")


if __name__ == "__main__":
    main()
