"""
Module: Detection
Áp dụng kiến thức: Phân đoạn ảnh theo ngưỡng, Contour Detection, Đối sánh đặc trưng hình học
- Tìm các marker hình vuông đen ở 4 góc phiếu để làm điểm neo cho Perspective Transform
- Dùng contour + lọc theo diện tích, tỉ lệ cạnh (aspect ratio), độ vuông (extent) để loại nhiễu
"""

import cv2
import numpy as np


def find_marker_candidates(binary: np.ndarray, min_area=300, max_area=5000):
    """
    Tìm các contour hình vuông nhỏ (marker) trên ảnh nhị phân.
    Lọc theo: diện tích, tỉ lệ cạnh gần 1 (vuông), độ đặc (solidity).
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        if not (0.7 <= aspect_ratio <= 1.3):
            continue

        rect_area = w * h
        extent = area / float(rect_area)
        if extent < 0.6:  # phải đặc (là hình vuông tô đen), không phải viền rỗng
            continue

        cx, cy = x + w / 2, y + h / 2
        candidates.append({"center": (cx, cy), "bbox": (x, y, w, h), "area": area})

    return candidates


def find_corner_markers(binary: np.ndarray):
    """
    Từ các candidate marker, chọn ra 4 marker ở 4 góc ngoài cùng
    (trên-trái, trên-phải, dưới-trái, dưới-phải) dựa trên vị trí cực trị.
    """
    h_img, w_img = binary.shape
    candidates = find_marker_candidates(binary)

    if len(candidates) < 4:
        raise ValueError(f"Chỉ tìm thấy {len(candidates)} marker, cần ít nhất 4. "
                          f"Kiểm tra lại ảnh hoặc điều chỉnh ngưỡng.")

    # Tính khoảng cách tới 4 góc ảnh, chọn candidate gần nhất mỗi góc
    corners_ref = {
        "top_left": (0, 0),
        "top_right": (w_img, 0),
        "bottom_left": (0, h_img),
        "bottom_right": (w_img, h_img),
    }

    result = {}
    for name, (rx, ry) in corners_ref.items():
        best = min(candidates, key=lambda c: (c["center"][0] - rx) ** 2 + (c["center"][1] - ry) ** 2)
        result[name] = best["center"]

    return result, candidates


def draw_debug_markers(image: np.ndarray, corners: dict, candidates: list) -> np.ndarray:
    """Vẽ tất cả candidate (xanh) và 4 marker góc đã chọn (đỏ, đánh số) để debug."""
    debug_img = image.copy()
    if len(debug_img.shape) == 2:
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)

    for c in candidates:
        x, y, w, h = c["bbox"]
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 1)

    for i, (name, (cx, cy)) in enumerate(corners.items()):
        cv2.circle(debug_img, (int(cx), int(cy)), 10, (0, 0, 255), 2)
        cv2.putText(debug_img, name, (int(cx) + 12, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return debug_img


if __name__ == "__main__":
    import os
    from preprocessing import preprocess_pipeline

    img = cv2.imread("../input/sample1.jpg")
    if img is None:
        img = cv2.imread("input/sample1.jpg")

    pre = preprocess_pipeline(img)
    binary = pre["binary"]

    corners, candidates = find_corner_markers(binary)
    print("4 góc marker phát hiện được:", corners)
    print(f"Tổng số candidate marker tìm thấy: {len(candidates)}")

    debug_img = draw_debug_markers(img, corners, candidates)
    os.makedirs("output/debug", exist_ok=True)
    cv2.imwrite("output/debug/02_markers.jpg", debug_img)
    print("Đã lưu output/debug/02_markers.jpg")