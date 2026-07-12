"""
sweep_marker_area.py (v2 - khớp đúng detection.py thật)
"""

import os
import sys
import argparse
import cv2

MODULES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules")
sys.path.insert(0, MODULES_DIR)

from preprocessing import preprocess_pipeline
from detection import find_marker_candidates

MIN_AREA_VALUES = [100, 200, 300, 400, 500]


def pick_corners(candidates, h_img, w_img):
    """Tái hiện đúng logic chọn 4 góc trong find_corner_markers(), nhưng
    nhận candidates đã tính sẵn theo từng min_area thay vì hardcode 300."""
    if len(candidates) < 4:
        return None
    corners_ref = {
        "top_left": (0, 0), "top_right": (w_img, 0),
        "bottom_left": (0, h_img), "bottom_right": (w_img, h_img),
    }
    result = {}
    for name, (rx, ry) in corners_ref.items():
        best = min(candidates, key=lambda c: (c["center"][0] - rx) ** 2 + (c["center"][1] - ry) ** 2)
        result[name] = best["center"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    args = parser.parse_args()

    files = [f for f in sorted(os.listdir(args.dir))
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    results = {v: {"pass": 0, "total": 0, "candidates_sum": 0} for v in MIN_AREA_VALUES}

    for fname in files:
        path = os.path.join(args.dir, fname)
        image = cv2.imread(path)
        if image is None:
            continue
        pre = preprocess_pipeline(image)
        h_img, w_img = pre["binary"].shape

        for min_area in MIN_AREA_VALUES:
            candidates = find_marker_candidates(pre["binary"], min_area=min_area, max_area=5000)
            corners = pick_corners(candidates, h_img, w_img)
            ok = corners is not None

            results[min_area]["total"] += 1
            results[min_area]["candidates_sum"] += len(candidates)
            if ok:
                results[min_area]["pass"] += 1

            print(f"{fname:25s} | min_area={min_area:4d} | candidates={len(candidates):3d} | {'OK' if ok else 'FAIL'}")

    print("\n=== KẾT QUẢ TỔNG HỢP ===")
    for min_area, r in results.items():
        rate = r["pass"] / r["total"] * 100 if r["total"] else 0
        avg_cand = r["candidates_sum"] / r["total"] if r["total"] else 0
        print(f"min_area={min_area:4d}: {r['pass']}/{r['total']} OK ({rate:.2f}%) | candidates TB={avg_cand:.1f}")


if __name__ == "__main__":
    main()