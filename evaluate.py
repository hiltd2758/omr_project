"""
evaluate.py
Đánh giá định lượng hệ thống OMR: Accuracy, Precision/Recall/F1, IoU.
Không sửa logic pipeline gốc trong modules/ — chỉ gọi lại và so sánh kết quả.

Cách chạy:
    python evaluate.py --image input/sample1.jpg --gt input/sample1_gt.json
    python evaluate.py --batch input/            # chạy toàn bộ ảnh có file *_gt.json kèm theo
"""

import os
import sys
import json
import argparse

import cv2
import numpy as np

MODULES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules")
sys.path.insert(0, MODULES_DIR)

from preprocessing import preprocess_pipeline
from detection import find_corner_markers
from perspective import warp_perspective
from segmentation import segment_all
from recognition import recognize_full_sheet

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Chạy pipeline nhận dạng (tái sử dụng nguyên logic gốc)
# ---------------------------------------------------------------------------
def run_pipeline(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    pre = preprocess_pipeline(image)
    corners, _ = find_corner_markers(pre["binary"])
    warped, _ = warp_perspective(image, corners)
    segments = segment_all(warped)
    result = recognize_full_sheet(segments)
    return result, corners


# ---------------------------------------------------------------------------
# 2. ACCURACY — tỉ lệ nhận dạng đúng từng phần
# ---------------------------------------------------------------------------
def accuracy_phan1(pred: dict, gt: dict) -> dict:
    total, correct = 0, 0
    for cau, dap_gt in gt.items():
        total += 1
        dap_pred = str(pred.get(int(cau), "")).strip().upper()
        if dap_pred == str(dap_gt).strip().upper():
            correct += 1
    return {"total": total, "correct": correct,
            "accuracy": round(correct / total, 4) if total else 0.0}


def accuracy_phan3(pred: dict, gt: dict) -> dict:
    total, correct = 0, 0
    for cau, dap_gt in gt.items():
        total += 1
        dap_pred = str(pred.get(int(cau), "")).strip()
        if dap_pred == str(dap_gt).strip():
            correct += 1
    return {"total": total, "correct": correct,
            "accuracy": round(correct / total, 4) if total else 0.0}


# ---------------------------------------------------------------------------
# 3. PRECISION / RECALL / F1 — coi mỗi lựa chọn là một lớp phân loại
#    Phần I: các lớp = {A, B, C, D}
#    Phần II: mỗi ý là bài toán nhị phân {D, S}
# ---------------------------------------------------------------------------
def precision_recall_f1(y_true: list, y_pred: list, labels: list) -> dict:
    metrics = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        metrics[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    # macro-average
    macro_p = round(np.mean([m["precision"] for m in metrics.values()]), 4)
    macro_r = round(np.mean([m["recall"] for m in metrics.values()]), 4)
    macro_f1 = round(np.mean([m["f1"] for m in metrics.values()]), 4)
    metrics["macro_avg"] = {"precision": macro_p, "recall": macro_r, "f1": macro_f1}
    return metrics


def eval_phan1_prf(pred: dict, gt: dict) -> dict:
    y_true, y_pred = [], []
    for cau, dap_gt in gt.items():
        y_true.append(str(dap_gt).strip().upper())
        y_pred.append(str(pred.get(int(cau), "")).strip().upper())
    return precision_recall_f1(y_true, y_pred, labels=["A", "B", "C", "D"])


def eval_phan2_prf(pred: dict, gt: dict) -> dict:
    y_true, y_pred = [], []
    for cau, dap_gt_list in gt.items():
        dap_pred_list = pred.get(int(cau), ["", "", "", ""])
        for i in range(4):
            gt_val = str(dap_gt_list[i]).strip().upper() if i < len(dap_gt_list) else ""
            pred_val = str(dap_pred_list[i]).strip().upper() if i < len(dap_pred_list) else ""
            y_true.append(gt_val)
            y_pred.append(pred_val)
    return precision_recall_f1(y_true, y_pred, labels=["D", "S"])


# ---------------------------------------------------------------------------
# 4. IoU — đánh giá vùng phát hiện marker / segmentation so với ground truth
#    box = [x_min, y_min, x_max, y_max]
# ---------------------------------------------------------------------------
def compute_iou(box_pred: list, box_gt: list) -> float:
    x1 = max(box_pred[0], box_gt[0])
    y1 = max(box_pred[1], box_gt[1])
    x2 = min(box_pred[2], box_gt[2])
    y2 = min(box_pred[3], box_gt[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area_pred = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])
    area_gt = (box_gt[2] - box_gt[0]) * (box_gt[3] - box_gt[1])
    union_area = area_pred + area_gt - inter_area

    return round(inter_area / union_area, 4) if union_area > 0 else 0.0


def corners_to_bbox(corners: dict) -> list:
    """Chuyển 4 điểm marker góc (dict {tl,tr,bl,br}) thành bounding box [x_min,y_min,x_max,y_max]."""
    xs = [p[0] for p in corners.values()]
    ys = [p[1] for p in corners.values()]
    return [min(xs), min(ys), max(xs), max(ys)]


# ---------------------------------------------------------------------------
# 5. TỔNG HỢP — chạy đánh giá cho 1 ảnh
# ---------------------------------------------------------------------------
def evaluate_one(image_path: str, gt_path: str) -> dict:
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    pred, corners = run_pipeline(image_path)

    report = {"image": os.path.basename(image_path)}

    # --- Accuracy ---
    if "phan1" in gt:
        report["accuracy_phan1"] = accuracy_phan1(pred.get("phan1", {}), gt["phan1"])
    if "phan3" in gt:
        report["accuracy_phan3"] = accuracy_phan3(pred.get("phan3", {}), gt["phan3"])

    # SBD / Mã đề: so khớp chuỗi
    if "sbd" in gt:
        report["sbd_correct"] = str(pred.get("sbd", "")).strip() == str(gt["sbd"]).strip()
    if "made" in gt:
        report["made_correct"] = str(pred.get("made", "")).strip() == str(gt["made"]).strip()

    # --- Precision / Recall / F1 ---
    if "phan1" in gt:
        report["prf_phan1"] = eval_phan1_prf(pred.get("phan1", {}), gt["phan1"])
    if "phan2" in gt:
        report["prf_phan2"] = eval_phan2_prf(pred.get("phan2", {}), gt["phan2"])

    # --- IoU (vùng marker) ---
    if "marker_box_gt" in gt:
        box_pred = corners_to_bbox(corners)
        report["iou_marker"] = compute_iou(box_pred, gt["marker_box_gt"])

    return report


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Đánh giá định lượng hệ thống OMR")
    parser.add_argument("--image", type=str, help="Đường dẫn ảnh cần đánh giá")
    parser.add_argument("--gt", type=str, help="Đường dẫn file ground truth JSON")
    parser.add_argument("--batch", type=str, help="Thư mục chứa nhiều ảnh + file *_gt.json tương ứng")
    args = parser.parse_args()

    all_reports = []

    if args.batch:
        for fname in sorted(os.listdir(args.batch)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                base = os.path.splitext(fname)[0]
                gt_file = os.path.join(args.batch, f"{base}_gt.json")
                if os.path.exists(gt_file):
                    img_path = os.path.join(args.batch, fname)
                    print(f"Đang đánh giá: {fname} ...")
                    try:
                        r = evaluate_one(img_path, gt_file)
                        all_reports.append(r)
                        print(json.dumps(r, ensure_ascii=False, indent=2))
                    except Exception as e:
                        print(f"  Lỗi: {e}")
                else:
                    print(f"Bỏ qua {fname}: không có file ground truth ({gt_file})")

    elif args.image and args.gt:
        r = evaluate_one(args.image, args.gt)
        all_reports.append(r)
        print(json.dumps(r, ensure_ascii=False, indent=2))

    else:
        parser.error("Cần chỉ định --image + --gt, hoặc --batch <thư_mục>")

    # Lưu kết quả
    if all_reports:
        out_name = "batch_metrics.json" if args.batch else \
            f"{os.path.splitext(os.path.basename(args.image))[0]}_metrics.json"
        out_path = os.path.join(OUT_DIR, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_reports, f, ensure_ascii=False, indent=2)
        print(f"\nĐã lưu kết quả đánh giá tại: {out_path}")


if __name__ == "__main__":
    main()