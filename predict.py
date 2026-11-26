import argparse
from pathlib import Path
from typing import Any
from ultralytics import YOLO
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
def extract_prediction(result: Any) -> tuple[list[float], int, float]:
    probs_obj = getattr(result, "probs", None)
    if probs_obj is None:
        return [], 0, 0.0
    probs_data = getattr(probs_obj, "data", None)
    if probs_data is None:
        return [], 0, 0.0
    probs = list(probs_data)
    top1 = int(getattr(probs_obj, "top1", 0))
    conf = float(getattr(probs_obj, "top1conf", 0.0))
    return probs, top1, conf
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--source", type=str, required=True,
                     help="Path to an image file or a folder of images")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--save", action="store_true",
                     help="Save annotated output images")
    args = ap.parse_args()

    model = YOLO(args.weights)
    results = model.predict(source=args.source, imgsz=args.imgsz, save=args.save, verbose=False)

    print("\n" + "="*70)
    print("BRAIN TUMOR MRI CLASSIFICATION RESULTS")
    print("="*70)

    for r in results:
        probs, top1, conf = extract_prediction(r)
        if not probs:
            continue

        name = Path(getattr(r, "path", "unknown")).name
        predicted_class = CLASSES[top1]

        print(f"\nImage: {name}")
        print("-" * 70)
        print(f"Predicted Class: {predicted_class.upper()}")
        print(f"Confidence:      {conf*100:.2f}%")
        print("-" * 70)
        print(f"{'Class':<15} {'Probability':>15} {'Confidence':>15}")
        print("-" * 70)

        sorted_probs = sorted(zip(CLASSES, probs), key=lambda x: -x[1])
        for idx, (cls_name, p) in enumerate(sorted_probs, 1):
            print(f"{cls_name:<15} {p:>14.2%} {p*100:>14.2f}%")

        print("-" * 70)
        print(f"\nTop 3 Predictions:")
        for idx, (cls_name, p) in enumerate(sorted_probs[:3], 1):
            print(f"  {idx}. {cls_name:<12s} - {p*100:>6.2f}% confidence")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
