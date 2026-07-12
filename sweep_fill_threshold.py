"""
sweep_fill_threshold.py (v2 - khớp đúng recognize_answers_mcq thật)
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
from recognition import recognize_answers_mcq

OUT_DIR = "sweep_results/fill_threshold"
THRESHOLD_VALUES = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
SKIP_LABELS = {"M", "."}


def run_phan1(image, fill_threshold: float) -> dict:
    """Chạy pipeline đến bước nhận dạng Phần I. Trả về {câu(str): đáp án}."""
    pre = preprocess_pipeline(image)
    corners, _ = find_corner_markers(pre["binary"])
    warped, _ = warp_perspective(image, corners)
    segments = segment_all(warped)

    all_answers = {}
    for b in range(1, 5):
        key = f"phan1_block{b}"
        if key not in segments:
            continue
        crop = segments[key]
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

        # dùng đúng binarize của pipeline chính, không tự chế
        crop_pre = preprocess_pipeline(crop)
        binary_crop = crop_pre["binary"]

        results = recognize_answers_mcq(
            gray_crop, binary_crop,
            n_rows=10, n_cols=4,
            choice_labels=["A", "B", "C", "D"],
            fill_threshold=fill_threshold,
        )

        offset = (b - 1) * 10
        for i, ans in enumerate(results):
            cau = str(offset + i + 1)
            if ans is None:
                all_answers[cau] = "."
            elif ans == "MULTI":
                all_answers[cau] = "M"
            else:
                all_answers[cau] = ans

    return all_answers


def compare(pred: dict, gt: dict) -> dict:
    """So sánh dự đoán với GT, loại câu GT = M/. ra khỏi thống kê chính."""
    total, correct = 0, 0
    false_negative, false_positive_multi, other_wrong = 0, 0, 0
    skipped = 0

    for cau, gt_val in gt.items():
        gt_val = str(gt_val).strip().upper()
        if gt_val in SKIP_LABELS:
            skipped += 1
            continue

        pred_val = str(pred.get(cau, ".")).strip().upper()
        total += 1

        if pred_val == gt_val:
            correct += 1
        elif pred_val == ".":
            false_negative += 1       # bỏ sót ô tô nhạt
        elif pred_val == "M":
            false_positive_multi += 1  # hệ thống nhận nhầm thành multi (do 2 vết mờ gần bằng nhau)
        else:
            other_wrong += 1          # nhận sai đáp án khác

    return {
        "total": total, "correct": correct, "skipped": skipped,
        "wrong": total - correct,
        "false_negative": false_negative,
        "false_positive_multi": false_positive_multi,
        "other_wrong": other_wrong,
        "accuracy": round(correct / total, 4) if total else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    gt_files = [f for f in sorted(os.listdir(args.dir)) if f.endswith("_gt.json")]
    if not gt_files:
        print("Không tìm thấy file *_gt.json trong thư mục.")
        return

    summary = {t: {"total": 0, "correct": 0, "false_negative": 0,
                    "false_positive_multi": 0, "other_wrong": 0} for t in THRESHOLD_VALUES}

    for gt_fname in gt_files:
        base = gt_fname.replace("_gt.json", "")
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            p = os.path.join(args.dir, base + ext)
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            print(f"Bỏ qua {gt_fname}: không tìm thấy ảnh gốc")
            continue

        with open(os.path.join(args.dir, gt_fname), "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        gt_phan1 = gt_data.get("phan1", {})
        if not gt_phan1:
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue

        print(f"\n--- {base} ---")
        for t in THRESHOLD_VALUES:
            try:
                pred = run_phan1(image, fill_threshold=t)
            except Exception as e:
                print(f"  threshold={t}: lỗi pipeline ({e})")
                continue
            r = compare(pred, gt_phan1)
            summary[t]["total"] += r["total"]
            summary[t]["correct"] += r["correct"]
            summary[t]["false_negative"] += r["false_negative"]
            summary[t]["false_positive_multi"] += r["false_positive_multi"]
            summary[t]["other_wrong"] += r["other_wrong"]
            print(f"  threshold={t:.2f} | đúng {r['correct']}/{r['total']} "
                  f"| bỏ sót={r['false_negative']} | multi giả={r['false_positive_multi']} "
                  f"| sai khác={r['other_wrong']} | skip(M/.)={r['skipped']}")

    print("\n=== TỔNG HỢP TOÀN BỘ ẢNH ===")
    results_out = []
    for t, s in summary.items():
        if s["total"] == 0:
            continue
        acc = s["correct"] / s["total"] * 100
        err_rate = 100 - acc
        print(f"threshold={t:.2f} | tổng={s['total']} | đúng={s['correct']} ({acc:.2f}%) "
              f"| sai={s['total']-s['correct']} ({err_rate:.2f}%) "
              f"| bỏ sót={s['false_negative']} | multi giả={s['false_positive_multi']} "
              f"| sai khác={s['other_wrong']}")
        results_out.append({"threshold": t, **s, "accuracy": round(acc, 2), "error_rate": round(err_rate, 2)})

    out_path = os.path.join(OUT_DIR, "fill_threshold_sweep.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_out, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu: {out_path}")


if __name__ == "__main__":
    main()