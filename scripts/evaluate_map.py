import argparse
import os
from collections import defaultdict

import numpy as np


def parse_yolo_label(file_path: str) -> list[tuple[str, float, float, float, float]]:
    boxes = []
    if not os.path.exists(file_path):
        return boxes

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = str(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            boxes.append((class_id, x_min, y_min, x_max, y_max))
    return boxes


def calculate_iou(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    return 0.0 if union_area == 0 else intersection_area / union_area


def calculate_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for idx in range(len(precision) - 2, -1, -1):
        precision[idx] = max(precision[idx], precision[idx + 1])

    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    return float(np.sum((recall[indices] - recall[indices - 1]) * precision[indices]))


def evaluate_detections(pred_dir: str, gt_dir: str, iou_threshold: float = 0.5) -> dict:
    pred_files = [name for name in os.listdir(pred_dir) if name.endswith(".txt")]
    gt_files = [name for name in os.listdir(gt_dir) if name.endswith(".txt")]

    pred_boxes = defaultdict(list)
    gt_boxes = defaultdict(list)

    for file_name in gt_files:
        image_id = os.path.splitext(file_name)[0]
        for class_id, x_min, y_min, x_max, y_max in parse_yolo_label(os.path.join(gt_dir, file_name)):
            gt_boxes[class_id].append(
                {
                    "image_id": image_id,
                    "bbox": (x_min, y_min, x_max, y_max),
                    "used": False,
                }
            )

    for file_name in pred_files:
        image_id = os.path.splitext(file_name)[0]
        with open(os.path.join(pred_dir, file_name), "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue

                class_id = str(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                confidence = float(parts[5])
                pred_boxes[class_id].append(
                    {
                        "image_id": image_id,
                        "bbox": (
                            x_center - width / 2,
                            y_center - height / 2,
                            x_center + width / 2,
                            y_center + height / 2,
                        ),
                        "confidence": confidence,
                    }
                )

    for class_id in pred_boxes:
        pred_boxes[class_id].sort(key=lambda item: item["confidence"], reverse=True)

    aps = {}
    all_classes = set(gt_boxes.keys())
    pred_classes = set(pred_boxes.keys())

    for class_id in all_classes:
        class_preds = pred_boxes.get(class_id, [])
        class_gts = gt_boxes.get(class_id, [])
        if not class_preds:
            aps[class_id] = 0.0
            continue

        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        for gt in class_gts:
            gt["used"] = False

        for idx, pred in enumerate(class_preds):
            image_gts = [gt for gt in class_gts if gt["image_id"] == pred["image_id"]]
            best_iou = 0.0
            best_gt = None

            for gt in image_gts:
                iou = calculate_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

            if best_gt is not None and best_iou >= iou_threshold and not best_gt["used"]:
                tp[idx] = 1
                best_gt["used"] = True
            else:
                fp[idx] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall = cum_tp / max(len(class_gts), 1)
        precision = cum_tp / (cum_tp + cum_fp + 1e-10)
        aps[class_id] = calculate_ap(recall, precision)

    mean_ap = float(np.mean([aps[class_id] for class_id in all_classes])) if all_classes else 0.0
    return {"aps": aps, "map": mean_ap, "all_classes": all_classes, "pred_classes": pred_classes}


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate mAP for YOLO txt detections.")
    parser.add_argument("--pred-dir", required=True, help="Prediction txt directory")
    parser.add_argument("--gt-dir", required=True, help="Ground-truth txt directory")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not os.path.isdir(args.pred_dir):
        raise FileNotFoundError(f"Prediction directory not found: {args.pred_dir}")
    if not os.path.isdir(args.gt_dir):
        raise FileNotFoundError(f"Ground-truth directory not found: {args.gt_dir}")
    if not 0 < args.iou_threshold < 1:
        raise ValueError("IoU threshold must be between 0 and 1")

    results = evaluate_detections(args.pred_dir, args.gt_dir, args.iou_threshold)
    print("=" * 50)
    print("mAP Evaluation Results")
    print("=" * 50)
    for class_id in sorted(results["aps"].keys()):
        status = "found" if class_id in results["pred_classes"] else "missing"
        print(f"Class {class_id}: AP = {results['aps'][class_id]:.4f} ({status})")
    print("=" * 50)
    print(f"mAP@{args.iou_threshold}: {results['map']:.4f}")
    print(f"Total classes: {len(results['all_classes'])}")


if __name__ == "__main__":
    main()
