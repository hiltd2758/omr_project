"""
Module: Segmentation (v2 - tự động, dùng Contour Detection)
Áp dụng kiến thức: Phân đoạn ảnh bằng Contour Detection + phân loại theo hình dạng/kích thước
- Mỗi vùng câu hỏi được thiết kế có viền đen bao quanh (rectangle)
- Tìm tất cả contour dạng hình chữ nhật lớn, sắp xếp theo vị trí (trên->dưới, trái->phải)
  để gán đúng tên block, thay vì đoán tọa độ tay theo %.
"""

import cv2
import numpy as np


def find_block_contours(binary: np.ndarray, min_area=8000):
    """
    Tìm các contour hình chữ nhật lớn (khung block câu hỏi).
    binary: ảnh nhị phân (đã đảo, nét/viền = 255)
    """
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / float(h)
        # Lọc bỏ contour quá dẹt (đường kẻ ngang/dọc dài) hoặc quá vuông nhỏ (marker)
        if aspect < 0.15 or aspect > 6:
            continue
        boxes.append((x, y, w, h))

    return boxes


def dedupe_nested_boxes(boxes, iou_thresh=0.85):
    """Loại các box lồng nhau gần trùng (contour ngoài + trong của cùng 1 khung)."""
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for b in boxes:
        x1, y1, w1, h1 = b
        is_dup = False
        for k in kept:
            x2, y2, w2, h2 = k
            # box b nằm gần như trọn trong box k đã giữ
            ix1, iy1 = max(x1, x2), max(y1, y2)
            ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            if inter / float(w1 * h1) > iou_thresh:
                is_dup = True
                break
        if not is_dup:
            kept.append(b)
    return kept


def draw_debug_boxes(image, boxes):
    debug = image.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(debug, str(i), (x + 3, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return debug

def split_by_vertical_lines(binary_crop: np.ndarray, n_parts: int = 6):
    """
    Tách 1 khối lớn thành n_parts cột bằng cách tìm đường kẻ dọc
    qua projection profile (đếm pixel đen theo từng cột x).
    Áp dụng kiến thức: Projection Histogram (thị giác máy tính truyền thống)
    """
    col_sum = np.sum(binary_crop, axis=0)  # tổng pixel đen mỗi cột x
    h, w = binary_crop.shape

    # Tìm các cột có mật độ đen cao (đường kẻ dọc) -> là điểm chia
    threshold = 0.5 * h * 255
    line_cols = np.where(col_sum > threshold)[0]

    # Gom các cột liền kề thành 1 đường kẻ (lấy điểm giữa)
    lines = []
    if len(line_cols) > 0:
        group = [line_cols[0]]
        for c in line_cols[1:]:
            if c - group[-1] <= 5:
                group.append(c)
            else:
                lines.append(int(np.mean(group)))
                group = [c]
        lines.append(int(np.mean(group)))

    # Thêm biên trái/phải nếu thiếu
    boundaries = [0] + lines + [w]
    boundaries = sorted(set(boundaries))

    # Nếu số đoạn không đúng n_parts, chia đều theo tỉ lệ (fallback an toàn)
    if len(boundaries) - 1 != n_parts:
        boundaries = [int(i * w / n_parts) for i in range(n_parts + 1)]

    parts = []
    for i in range(n_parts):
        x1, x2 = boundaries[i], boundaries[i + 1]
        parts.append((x1, 0, x2 - x1, h))
    return parts


def classify_and_label_boxes(boxes, img_shape):
    """
    Gán tên cho từng box dựa trên vị trí (x, y) và kích thước tương đối.
    Sắp xếp theo hàng (y) rồi theo cột (x) trong mỗi hàng.
    """
    h_img, w_img = img_shape[:2]

    # Lọc bỏ box quá to (gần bằng cả ảnh) hoặc quá nhỏ
    boxes = [b for b in boxes if b[2] * b[3] < 0.9 * w_img * h_img]

    # Phân theo dải y (row band) để nhóm: SBD/made | Phan1 | Phan2 | Phan3
    rows = {"sbd_made": [], "phan1": [], "phan2": [], "phan3": []}
    for b in boxes:
        x, y, w, h = b
        y_ratio = y / h_img
        x_ratio = x / w_img
        if y_ratio < 0.32:
            if x_ratio > 0.65:          # chỉ lấy box bên phải (SBD/Mã đề)
                rows["sbd_made"].append(b)
            # bỏ qua box bên trái (khung họ tên cán bộ coi thi)
        elif y_ratio < 0.55:
            rows["phan1"].append(b)
        elif y_ratio < 0.68:
            rows["phan2"].append(b)
        else:
            rows["phan3"].append(b)

    labeled = {}

    # sbd/made: sắp theo x, box trái = sbd, phải = made
    sbd_made = sorted(rows["sbd_made"], key=lambda b: b[0])
    if len(sbd_made) >= 2:
        labeled["sbd"] = sbd_made[0]
        labeled["made"] = sbd_made[-1]

    # phan1, phan2: sắp theo x, gán block1..4
    for i, b in enumerate(sorted(rows["phan1"], key=lambda b: b[0])):
        labeled[f"phan1_block{i+1}"] = b
    for i, b in enumerate(sorted(rows["phan2"], key=lambda b: b[0])):
        labeled[f"phan2_block{i+1}"] = b

    # phan3: nếu chỉ có 1 box lớn -> tách 6 cột; nếu đã có 6 box -> dùng luôn
    p3 = sorted(rows["phan3"], key=lambda b: b[0])
    if len(p3) == 1:
        x, y, w, h = p3[0]
        sub_parts = split_by_vertical_lines(
            np.zeros((h, w), dtype=np.uint8), n_parts=6  # placeholder, xem lưu ý dưới
        )
        for i, (sx, sy, sw, sh) in enumerate(sub_parts):
            labeled[f"phan3_block{i+1}"] = (x + sx, y + sy, sw, sh)
    else:
        for i, b in enumerate(p3):
            labeled[f"phan3_block{i+1}"] = b

    return labeled

def segment_all(warped: np.ndarray, debug_dir: str = None) -> dict:
    """
    Pipeline đầy đủ: tìm block bằng contour -> gán tên -> trả về dict crop ảnh.
    Thay thế cho bản segment_all cũ (dùng tỉ lệ % cố định).
    """
    from preprocessing import preprocess_pipeline

    warped_pre = preprocess_pipeline(warped)
    binary = warped_pre["binary"]

    boxes = find_block_contours(binary, min_area=8000)
    boxes = dedupe_nested_boxes(boxes)

    h_img, w_img = warped.shape[:2]

    # Tách riêng Phần III nếu bị gộp thành 1 khối lớn
    boxes_filtered = [b for b in boxes if b[2] * b[3] < 0.9 * w_img * h_img]
    rows_phan3 = [b for b in boxes_filtered if (b[1] / h_img) >= 0.68]

    sub_parts = None
    if len(rows_phan3) == 1:
        x, y, w, h = rows_phan3[0]
        sub_binary = binary[y:y + h, x:x + w]
        sub_parts = split_by_vertical_lines(sub_binary, n_parts=6)
        px, py = x, y

    labeled = classify_and_label_boxes(boxes, warped.shape)

    if sub_parts:
        for i, (sx, sy, sw, sh) in enumerate(sub_parts):
            labeled[f"phan3_block{i+1}"] = (px + sx, py + sy, sw, sh)

    segments = {}
    debug_img = warped.copy()
    for name, (x, y, w, h) in labeled.items():
        crop = warped[y:y + h, x:x + w]
        segments[name] = crop
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(debug_img, name, (x + 3, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    if debug_dir:
        import os
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(f"{debug_dir}/06_labeled_overlay.jpg", debug_img)
        for name, crop in segments.items():
            cv2.imwrite(f"{debug_dir}/06_{name}.jpg", crop)

    return segments
if __name__ == "__main__":
    import os
    from preprocessing import preprocess_pipeline
    from detection import find_corner_markers
    from perspective import warp_perspective

    img = cv2.imread("../input/sample1.jpg")
    if img is None:
        img = cv2.imread("input/sample1.jpg")

    pre = preprocess_pipeline(img)
    corners, _ = find_corner_markers(pre["binary"])
    warped, _ = warp_perspective(img, corners)

    warped_pre = preprocess_pipeline(warped)
    binary = warped_pre["binary"]

    boxes = find_block_contours(binary, min_area=8000)
    boxes = dedupe_nested_boxes(boxes)

    # Gán tên, nhưng patch riêng phan3 dùng binary thật
    h_img, w_img = warped.shape[:2]
    boxes_filtered = [b for b in boxes if b[2] * b[3] < 0.9 * w_img * h_img]

    rows_phan3 = [b for b in boxes_filtered if (b[1] / h_img) >= 0.68]
    if len(rows_phan3) == 1:
        x, y, w, h = rows_phan3[0]
        sub_binary = binary[y:y+h, x:x+w]
        sub_parts = split_by_vertical_lines(sub_binary, n_parts=6)

    labeled = classify_and_label_boxes(boxes, warped.shape)
    # override phan3 bằng kết quả tách thật nếu có
    if len(rows_phan3) == 1:
        for i, (sx, sy, sw, sh) in enumerate(sub_parts):
            labeled[f"phan3_block{i+1}"] = (x + sx, y + sy, sw, sh)

    print("Các vùng đã gán tên:", list(labeled.keys()))

    debug = warped.copy()
    os.makedirs("output/debug", exist_ok=True)
    for name, (x, y, w, h) in labeled.items():
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(debug, name, (x + 3, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imwrite(f"output/debug/06_{name}.jpg", warped[y:y+h, x:x+w])

    cv2.imwrite("output/debug/06_labeled_overlay.jpg", debug)
    print("Xem output/debug/06_labeled_overlay.jpg")