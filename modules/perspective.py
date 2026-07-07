"""
Module: Perspective Correction
Áp dụng kiến thức: Biến đổi hình học (Geometric Transform) - Perspective Transform
- Dùng 4 điểm marker góc làm điểm nguồn (src) 
- Ánh xạ sang 4 điểm đích (dst) tạo thành hình chữ nhật chuẩn
- Dùng ma trận biến đổi phối cảnh 3x3 (Homography) để warp toàn bộ ảnh
- Khắc phục trường hợp ảnh chụp bị nghiêng, lệch góc
"""

import cv2
import numpy as np

# Kích thước ảnh chuẩn sau khi warp (cố định để các module sau dùng tọa độ tuyệt đối)
STANDARD_WIDTH = 1100
STANDARD_HEIGHT = 1500


def order_points(corners: dict) -> np.ndarray:
    """
    Sắp xếp 4 điểm góc theo thứ tự: top_left, top_right, bottom_right, bottom_left
    (thứ tự chuẩn mà cv2.getPerspectiveTransform yêu cầu)
    """
    pts = np.array([
        corners["top_left"],
        corners["top_right"],
        corners["bottom_right"],
        corners["bottom_left"],
    ], dtype="float32")
    return pts


def warp_perspective(image: np.ndarray, corners: dict,
                      out_w: int = STANDARD_WIDTH, out_h: int = STANDARD_HEIGHT,
                      margin: int = 50) -> np.ndarray:
    """
    margin: khoảng đệm (pixel) chừa quanh 4 marker để không cắt mất nội dung
    nằm ngoài marker (do marker lùi vào trong so với mép giấy thật).
    """
    src_pts = order_points(corners)

    dst_pts = np.array([
        [margin, margin],
        [out_w - margin, margin],
        [out_w - margin, out_h - margin],
        [margin, out_h - margin],
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (out_w, out_h))
    return warped, M

if __name__ == "__main__":
    import os
    from preprocessing import preprocess_pipeline
    from detection import find_corner_markers

    img = cv2.imread("../input/sample1.jpg")
    if img is None:
        img = cv2.imread("input/sample1.jpg")

    pre = preprocess_pipeline(img)
    binary = pre["binary"]

    corners, _ = find_corner_markers(binary)
    warped, M = warp_perspective(img, corners)

    os.makedirs("output/debug", exist_ok=True)
    cv2.imwrite("output/debug/03_warped.jpg", warped)
    print("Đã lưu output/debug/03_warped.jpg — kích thước:", warped.shape)