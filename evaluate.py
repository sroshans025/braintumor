import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from ultralytics import YOLO

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to best.pt from training")
    ap.add_argument("--data", required=True, help="test/ folder with one subfolder per class")
    ap.add_argument("--imgsz", type=int, default=320, help="Must match the training image size")
    ap.add_argument("--out_dir", default="./eval_results")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    # The model's own class order is the source of truth for prediction indices.
    names = [model.names[i] for i in range(len(model.names))]
    assert names == CLASSES, f"Model classes {names} differ from {CLASSES}"

    rows = []
    for cls_idx, cls_name in enumerate(CLASSES):
        paths = sorted(p for p in (Path(args.data) / cls_name).iterdir() if p.suffix.lower() in IMG_EXTS)
        for r in model.predict(source=paths, imgsz=args.imgsz, stream=True, verbose=False):
            probs = r.probs.data.cpu().numpy()
            rows.append({
                "image": Path(r.path).name,
                "true": cls_name,
                "pred": CLASSES[int(probs.argmax())],
                "confidence": float(probs.max()),
                **{f"p_{c}": float(probs[i]) for i, c in enumerate(CLASSES)},
            })
        print(f"[{cls_name:11s}] {len(paths)} test images evaluated")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "predictions.csv", index=False)
    y_true, y_pred = df["true"], df["pred"]

    report_txt = classification_report(y_true, y_pred, labels=CLASSES, digits=4)
    report = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True)
    (out_dir / "classification_report.txt").write_text(report_txt)
    pd.DataFrame(report).T.to_csv(out_dir / "classification_report.csv")

    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    summary = {
        "test_images": int(len(df)),
        "correct": int((y_true == y_pred).sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro"),
        "macro_recall": recall_score(y_true, y_pred, average="macro"),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "per_class": {c: {k: report[c][k] for k in ("precision", "recall", "f1-score", "support")} for c in CLASSES},
        "confusion_matrix": {"labels": CLASSES, "matrix": cm.tolist()},
        "imgsz": args.imgsz,
        "weights": str(args.weights),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "evaluation_summary.txt").write_text(
        f"Test images:     {summary['test_images']}\n"
        f"Correct:         {summary['correct']}\n"
        f"Accuracy:        {summary['accuracy']*100:.2f}%\n"
        f"Macro precision: {summary['macro_precision']*100:.2f}%\n"
        f"Macro recall:    {summary['macro_recall']*100:.2f}%\n"
        f"Macro F1:        {summary['macro_f1']*100:.2f}%\n"
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(4), CLASSES)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Test confusion matrix (accuracy {summary['accuracy']*100:.2f}%)")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)

    print("\n" + report_txt)
    print((out_dir / "evaluation_summary.txt").read_text())
    print(f"Results written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
