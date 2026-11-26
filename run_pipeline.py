import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Run the full YOLO brain tumor classification workflow")
    ap.add_argument("--raw_dir", type=str, required=True, help="Path containing Training/ and Testing/ folders")
    ap.add_argument("--out_dir", type=str, default="./data")
    ap.add_argument("--model", type=str, default="yolo11m-cls.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--project", type=str, default="runs_classify")
    ap.add_argument("--name", type=str, default="brain_tumor_yolo11")
    ap.add_argument("--preprocess", action="store_true")
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parent

    prepare_cmd = [
        sys.executable,
        str(base_dir / "preparedataset.py"),
        "--raw_dir", args.raw_dir,
        "--out_dir", args.out_dir,
        "--val_split", str(args.val_split),
        "--imgsz", str(args.imgsz),
    ]
    if args.preprocess:
        prepare_cmd.append("--preprocess")

    train_cmd = [
        sys.executable,
        str(base_dir / "train.py"),
        "--data", args.out_dir,
        "--model", args.model,
        "--epochs", str(args.epochs),
        "--imgsz", str(args.imgsz),
        "--batch", str(args.batch),
        "--device", args.device,
        "--project", args.project,
        "--name", args.name,
    ]

    evaluate_cmd = [
        sys.executable,
        str(base_dir / "evaluate.py"),
        "--weights", str(Path(args.project) / args.name / "weights" / "best.pt"),
        "--data", str(Path(args.out_dir) / "test"),
        "--out_dir", "./eval_results",
    ]

    for cmd in [prepare_cmd, train_cmd, evaluate_cmd]:
        print("\nRunning:", " ".join(cmd))
        result = subprocess.run(cmd, cwd=base_dir)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
