"""
sweep_hough_param2.py
Khảo sát param2 của detect_circles() trên vùng Phần I (kỳ vọng 10 hàng x 4 cột).
Đo: số hàng detect đúng đủ 4 cột / tổng 10 hàng kỳ vọng.
"""

import os
import sys
import argparse
import cv2

MODULES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules")
sys.path.insert(0, MODULES_DIR)

from preprocessing import preprocess_pipeline
from detection import find_corner_markers
from perspective import warp_perspective
from segmentation import segment_all
from recognition import detect_circles, cluster_to_grid

PARAM2_VALUES = [10, 12, 15, 18, 20, 25]
EXPECTED_ROWS = 10
EXPECTED_COLS = 4


def eval_param2(gray_crop, param2: int) -> tuple:
    """Trả về (số hàng đúng đủ 4 cột, tổng số hàng detect được)."""
    circles = detect_circles(gray_crop, param2=param2)
    grid_rows = cluster_to_grid(circles)
    if len(grid_rows) > EXPECTED_ROWS:
        grid_rows = grid_rows[-EXPECTED_ROWS:]
    correct = sum(1 for row in grid_rows if len(row) == EXPECTED_COLS)
    return correct, len(grid_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    args = parser.parse_args()

    files = [f for f in sorted(os.listdir(args.dir))
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    results = {v: {"correct_rows": 0, "total_expected": 0, "n_blocks": 0} for v in PARAM2_VALUES}

    for fname in files:
        path = os.path.join(args.dir, fname)
        image = cv2.imread(path)
        if image is None:
            continue
        try:
            pre = preprocess_pipeline(image)
            corners, _ = find_corner_markers(pre["binary"])
            warped, _ = warp_perspective(image, corners)
            segments = segment_all(warped)
        except Exception as e:
            print(f"{fname}: lỗi pipeline ({e}), bỏ qua")
            continue

        for b in range(1, 5):
            key = f"phan1_block{b}"
            if key not in segments:
                continue
            crop = segments[key]
            crop_pre = preprocess_pipeline(crop)
            gray = crop_pre["gray"]

            for param2 in PARAM2_VALUES:
                correct, n_rows = eval_param2(gray, param2)
                results[param2]["correct_rows"] += correct
                results[param2]["total_expected"] += EXPECTED_ROWS
                results[param2]["n_blocks"] += 1
                print(f"{fname} | {key} | param2={param2:2d} | {correct}/{EXPECTED_ROWS} hàng đúng "
                      f"(detect tổng {n_rows} hàng)")

    print("\n=== KẾT QUẢ TỔNG HỢP ===")
    for param2, r in results.items():
        rate = r["correct_rows"] / r["total_expected"] * 100 if r["total_expected"] else 0
        print(f"param2={param2:2d}: {r['correct_rows']}/{r['total_expected']} hàng đúng "
              f"({rate:.2f}%) trên {r['n_blocks']} block")


if __name__ == "__main__":
    main()