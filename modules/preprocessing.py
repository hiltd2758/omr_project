"""
Module: Preprocessing
Áp dụng kiến thức: Toán tử điểm ảnh, Lọc tuyến tính/phi tuyến, Nhị phân hóa
- Chuyển ảnh xám: giảm chiều dữ liệu, chuẩn hóa đầu vào
- Khử nhiễu: Gaussian Blur (lọc tuyến tính) / Median Blur (lọc phi tuyến, tốt cho nhiễu muối tiêu)
- Cân bằng sáng: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Nhị phân hóa: Adaptive Threshold / Otsu (toán tử điểm ảnh dựa ngưỡng)
"""

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Chuyển ảnh màu sang ảnh xám."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def denoise(gray: np.ndarray, method: str = "gaussian") -> np.ndarray:
    """
    Khử nhiễu ảnh.
    - gaussian: lọc tuyến tính, làm mượt nhiễu Gauss
    - median: lọc phi tuyến, tốt cho nhiễu muối tiêu (salt-pepper) từ ảnh chụp điện thoại
    """
    if method == "gaussian":
        return cv2.GaussianBlur(gray, (5, 5), 0)
    elif method == "median":
        return cv2.medianBlur(gray, 5)
    else:
        return gray


def equalize_lighting(gray: np.ndarray) -> np.ndarray:
    """
    Cân bằng sáng cục bộ bằng CLAHE.
    Xử lý trường hợp ảnh chụp bị bóng đổ / ánh sáng không đều.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def binarize(gray: np.ndarray, method: str = "adaptive") -> np.ndarray:
    """
    Nhị phân hóa ảnh (toán tử điểm ảnh dựa ngưỡng).
    - adaptive: ngưỡng thích nghi theo vùng cục bộ -> chống chịu ánh sáng không đều
    - otsu: ngưỡng toàn cục tự động theo histogram -> nhanh, phù hợp ảnh sáng đều
    Kết quả: nền trắng (0), nét/ô tô đen thành giá trị 255 (đảo ngược để thuận tiện đếm pixel)
    """
    if method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=25,
            C=10
        )
    else:  # otsu
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    return binary


def preprocess_pipeline(image: np.ndarray, debug_dir: str = None) -> dict:
    """
    Chạy toàn bộ pipeline tiền xử lý, trả về dict chứa ảnh trung gian
    để phục vụ debug và các module sau sử dụng.
    """
    gray = to_grayscale(image)
    denoised = denoise(gray, method="gaussian")
    equalized = equalize_lighting(denoised)
    binary = binarize(equalized, method="adaptive")

    results = {
        "gray": gray,
        "denoised": denoised,
        "equalized": equalized,
        "binary": binary,
    }

    if debug_dir:
        import os
        os.makedirs(debug_dir, exist_ok=True)
        for name, img in results.items():
            cv2.imwrite(os.path.join(debug_dir, f"01_{name}.jpg"), img)

    return results


if __name__ == "__main__":
    # Test nhanh module độc lập
    img = cv2.imread("input/sample1.jpg")
    if img is None:
        print("Không tìm thấy ảnh input/sample1.jpg")
    else:
        out = preprocess_pipeline(img, debug_dir="output/debug")
        print("Đã xử lý xong, xem kết quả trong output/debug/")