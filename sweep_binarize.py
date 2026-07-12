"""
sweep_binarize.py (v3)
Khảo sát Otsu vs Adaptive Threshold bằng chỉ số THẬT phản ánh chất lượng:
- flood_ratio: tỉ lệ diện tích bị quét thành 1 khối trắng liền lớn nhất
  (chỉ số càng cao = càng mất nhiều cấu trúc thật, do ngưỡng toàn cục
  hiểu nhầm vùng tối/bóng đổ là nội dung)
- marker_ok: có phát hiện đủ 4 marker góc hay không
- black_pixel_ratio: tỉ lệ pixel đen trên toàn ảnh (dùng để phát hiện
  bất thường — quá cao/quá thấp so với ảnh sáng đều là dấu hiệu lỗi)
"""

import os
import sys
import argparse
import cv2
import numpy as np

MODULES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules")
sys.path.insert(0, MODULES_DIR)

from preprocessing import to_grayscale, denoise, equalize_lighting, binarize
from detection import find_marker_candidates, find_corner_markers

OUT_DIR = "sweep_results/binarize"


def largest_white_blob_ratio(binary: np.ndarray) -> float:
    """
    Tính tỉ lệ diện tích của khối trắng liền lớn nhất so với toàn ảnh.
    Giá trị cao bất thường (ví dụ > 0.15-0.2) cho thấy có một vùng lớn
    bị quét trắng đồng nhất — dấu hiệu ngưỡng toàn cục "nuốt" cả vùng
    tối do bóng đổ, không phải nội dung ô tròn/nét mực thật.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return 0.0
    # bỏ qua nhãn 0 (nền), tìm component trắng lớn nhất
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_area = areas.max()
    total_area = binary.shape[0] * binary.shape[1]
    return round(largest_area / total_area, 4)


def black_pixel_ratio(binary: np.ndarray) -> float:
    return round(np.sum(binary == 0) / binary.size, 4)


def process(image, method: str):
    gray = to_grayscale(image)
    denoised = denoise(gray, method="gaussian")
    equalized = equalize_lighting(denoised)
    binary = binarize(equalized, method=method)

    candidates = find_marker_candidates(binary, min_area=300, max_area=5000)
    try:
        corners, _ = find_corner_markers(binary)
        ok = True
    except ValueError:
        corners, ok = None, False

    flood_ratio = largest_white_blob_ratio(binary)
    black_ratio = black_pixel_ratio(binary)

    return {
        "binary": binary,
        "n_candidates": len(candidates),
        "marker_ok": ok,
        "flood_ratio": flood_ratio,
        "black_ratio": black_ratio,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    files = [f for f in sorted(os.listdir(args.dir))
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    rows = []
    for fname in files:
        path = os.path.join(args.dir, fname)
        image = cv2.imread(path)
        if image is None:
            continue
        base = os.path.splitext(fname)[0]

        r_otsu = process(image, "otsu")
        r_adap = process(image, "adaptive")

        # lưu ảnh so sánh
        h, w = r_otsu["binary"].shape
        combined = np.zeros((h, w * 2 + 20), dtype=np.uint8)
        combined[:, :w] = r_otsu["binary"]
        combined[:, w + 20:] = r_adap["binary"]
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
        cv2.putText(combined_bgr, "OTSU", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(combined_bgr, "ADAPTIVE", (w + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(OUT_DIR, f"{base}_compare.jpg"), combined_bgr)

        rows.append({"file": fname, "otsu": r_otsu, "adaptive": r_adap})

        print(f"{fname:25s} | OTSU     marker_ok={r_otsu['marker_ok']!s:5} "
              f"flood_ratio={r_otsu['flood_ratio']:.4f} black_ratio={r_otsu['black_ratio']:.4f}")
        print(f"{'':25s} | ADAPTIVE marker_ok={r_adap['marker_ok']!s:5} "
              f"flood_ratio={r_adap['flood_ratio']:.4f} black_ratio={r_adap['black_ratio']:.4f}")

    print(f"\nĐã lưu ảnh so sánh vào: {OUT_DIR}/")
    print("=== TỔNG HỢP ===")
    n = len(rows)
    otsu_marker_ok = sum(1 for r in rows if r["otsu"]["marker_ok"])
    adap_marker_ok = sum(1 for r in rows if r["adaptive"]["marker_ok"])
    otsu_flood_avg = np.mean([r["otsu"]["flood_ratio"] for r in rows])
    adap_flood_avg = np.mean([r["adaptive"]["flood_ratio"] for r in rows])

    print(f"Otsu:     marker_ok {otsu_marker_ok}/{n} | flood_ratio trung bình = {otsu_flood_avg:.4f}")
    print(f"Adaptive: marker_ok {adap_marker_ok}/{n} | flood_ratio trung bình = {adap_flood_avg:.4f}")

    # cảnh báo ảnh có flood_ratio bất thường cao (nghi ngờ mất nội dung)
    print("\n=== CẢNH BÁO FLOOD (khả nghi mất cấu trúc do bóng đổ) ===")
    for r in rows:
        for method in ["otsu", "adaptive"]:
            if r[method]["flood_ratio"] > 0.10:
                print(f"{r['file']:25s} | {method:10s} | flood_ratio={r[method]['flood_ratio']:.4f} "
                      f"⚠ nghi ngờ mất cấu trúc vùng lớn")


if __name__ == "__main__":
    main()