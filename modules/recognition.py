"""
Module: Recognition
Áp dụng kiến thức: Hough Circle Transform, Connected Components, 
Phân đoạn theo lưới (Grid-based segmentation), toán tử điểm ảnh (đếm pixel)
- Phát hiện tâm các ô tròn bằng Hough Circle
- Sắp xếp thành lưới hàng x cột theo tọa độ
- Tính tỉ lệ lấp đầy (fill ratio) từng ô để xác định ô nào được tô đen
"""

import cv2
import numpy as np


def detect_circles(gray_crop, min_dist=15, param2=15, min_r=6, max_r=14):
    """
    Phát hiện tâm và bán kính các ô tròn bằng Hough Circle Transform.
    """
    blurred = cv2.GaussianBlur(gray_crop, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist,
        param1=50, param2=param2, minRadius=min_r, maxRadius=max_r
    )
    if circles is None:
        return []
    circles = np.round(circles[0, :]).astype(int)
    return [(x, y, r) for x, y, r in circles]

def detect_circles_small(gray_crop):
    raw = detect_circles(gray_crop, min_dist=12, param2=14, min_r=5, max_r=11)
    return dedupe_circles(raw, min_dist=10)
def dedupe_circles(circles, min_dist=10):
    """
    Loại bỏ các circle trùng lặp (tâm quá gần nhau) - lỗi thường gặp khi
    Hough Circle với param2 thấp phát hiện nhiều vòng cho cùng 1 ô thực.
    Giữ lại circle có bán kính lớn nhất trong mỗi cụm trùng.
    """
    if not circles:
        return []
    circles = sorted(circles, key=lambda c: -c[2])  # ưu tiên bán kính lớn trước
    kept = []
    for c in circles:
        x, y, r = c
        is_dup = False
        for k in kept:
            kx, ky, kr = k
            if (x - kx) ** 2 + (y - ky) ** 2 < min_dist ** 2:
                is_dup = True
                break
        if not is_dup:
            kept.append(c)
    return kept
def cluster_to_grid(circles, row_tol=10):
    """
    Gom các tâm ô tròn thành lưới hàng/cột dựa trên tọa độ y (hàng) và x (cột).
    Tự động gộp các hàng bị tách nhầm quá gần nhau (do dedupe circle chưa đủ).
    """
    if not circles:
        return []
    circles = sorted(circles, key=lambda c: c[1])

    rows = []
    current_row = [circles[0]]
    for c in circles[1:]:
        if abs(c[1] - current_row[-1][1]) <= row_tol:
            current_row.append(c)
        else:
            rows.append(sorted(current_row, key=lambda p: p[0]))
            current_row = [c]
    rows.append(sorted(current_row, key=lambda p: p[0]))

    # ---- Gộp thêm các hàng có y trung bình quá gần nhau (dư hàng do nhiễu) ----
    if len(rows) > 1:
        row_ys = [np.mean([p[1] for p in r]) for r in rows]
        gaps = [row_ys[i+1] - row_ys[i] for i in range(len(row_ys)-1)]
        median_gap = np.median(gaps) if gaps else 999

        merged = [rows[0]]
        for i in range(1, len(rows)):
            gap = row_ys[i] - row_ys[i-1]
            if gap < median_gap * 0.5:  # hàng quá gần hàng trước -> gộp
                merged[-1] = sorted(merged[-1] + rows[i], key=lambda p: p[0])
                row_ys[i] = np.mean([p[1] for p in merged[-1]])
            else:
                merged.append(rows[i])
        rows = merged

    return rows


def fill_ratio(binary_crop: np.ndarray, x, y, r):
    """
    Tính tỉ lệ pixel đen (đã tô) trong vùng tròn tâm (x,y) bán kính r.
    Áp dụng: toán tử điểm ảnh - đếm số pixel foreground / tổng pixel vùng.
    """
    h, w = binary_crop.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (x, y), max(r - 2, 3), 255, -1)  # thu nhỏ nhẹ để tránh viền ô
    region = cv2.bitwise_and(binary_crop, mask)
    filled = cv2.countNonZero(region)
    total = cv2.countNonZero(mask)
    return filled / float(total) if total > 0 else 0


def recognize_answers_mcq(gray_crop, binary_crop, n_rows, n_cols, choice_labels,
                           fill_threshold=0.35, relative_ratio=None):
    """
    Nhận dạng đáp án trắc nghiệm A/B/C/D (dùng cho Phần I).
    Cải tiến v2: Dùng relative thresholding để giảm false positive do vết chì/bóng đổ.
    """
    RELATIVE_RATIO = relative_ratio if relative_ratio is not None else 0.65

    circles = detect_circles(gray_crop)
    grid_rows = cluster_to_grid(circles)

    if len(grid_rows) > n_rows:
        grid_rows = grid_rows[-n_rows:]
    elif len(grid_rows) < n_rows:
        print(f"[Cảnh báo] Phát hiện {len(grid_rows)} hàng, kỳ vọng {n_rows}. "
              f"Có thể cần chỉnh param2/min_dist.")

    results = []
    for row in grid_rows:
        if len(row) != n_cols:
            results.append(None)
            continue

        ratios = [fill_ratio(binary_crop, x, y, r) for (x, y, r) in row]
        print("MCQ:", ratios)
        max_ratio = max(ratios)

        if max_ratio < fill_threshold:
            results.append(None)
            continue

        relative_cutoff = RELATIVE_RATIO * max_ratio
        above = [i for i, r in enumerate(ratios)
                 if r >= fill_threshold and r >= relative_cutoff]

        if len(above) > 1:
            results.append("MULTI")
        elif len(above) == 1:
            results.append(choice_labels[above[0]])
        else:
            results.append(None)

    return results


def _detect_dung_sai_cascade(gray, n_rows=4, n_cols=2, row_tol=12):
    """
    Thử nhiều bộ param Hough theo thứ tự, chọn bộ cho nhiều hàng đúng nhất.
    Với những hàng vẫn sai (len != n_cols), dùng tọa độ x từ các hàng đúng để suy ra.
    """
    PARAM_SETS = [
        dict(min_dist=20, param2=18, min_r=5, max_r=13, dedup_dist=14),
        dict(min_dist=18, param2=15, min_r=5, max_r=13, dedup_dist=12),
        dict(min_dist=22, param2=20, min_r=5, max_r=12, dedup_dist=16),
        dict(min_dist=15, param2=12, min_r=5, max_r=14, dedup_dist=10),
        dict(min_dist=20, param2=25, min_r=5, max_r=13, dedup_dist=14),
    ]

    best_rows = None
    best_score = -1

    for params in PARAM_SETS:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1,
            minDist=params["min_dist"], param1=50, param2=params["param2"],
            minRadius=params["min_r"], maxRadius=params["max_r"])
        if circles is None:
            continue
        raw = [(int(round(x)), int(round(y)), int(round(r))) for x, y, r in circles[0]]
        raw = dedupe_circles(raw, min_dist=params["dedup_dist"])
        grid_rows = cluster_to_grid(raw, row_tol=row_tol)
        if len(grid_rows) > n_rows:
            grid_rows = grid_rows[-n_rows:]
        score = sum(1 for r in grid_rows if len(r) == n_cols)
        if score > best_score:
            best_score = score
            best_rows = grid_rows

    if best_rows is None:
        return []

    # Suy ra tọa độ x chuẩn từ các hàng detect đúng
    good_rows = [r for r in best_rows if len(r) == n_cols]
    if good_rows:
        col_xs = [
            int(np.mean([r[ci][0] for r in good_rows]))
            for ci in range(n_cols)
        ]
        avg_r = int(np.mean([c[2] for r in good_rows for c in r]))
    else:
        col_xs = None
        avg_r = 8

    # Với hàng sai, reconstruct bằng tọa độ x chuẩn + y trung bình của hàng đó
    fixed_rows = []
    for row in best_rows:
        if len(row) == n_cols:
            fixed_rows.append(row)
        elif col_xs is not None and len(row) > 0:
            avg_y = int(np.mean([c[1] for c in row]))
            fixed_rows.append([(cx, avg_y, avg_r) for cx in col_xs])
        else:
            fixed_rows.append(row)  # vẫn sai, để caller xử lý

    return fixed_rows


def recognize_answers_dung_sai(gray_crop, binary_crop, fill_threshold=0.18):
    """
    Nhận dạng Phần II (Đúng/Sai) — 4 hàng x 2 cột.
    - Cascade param Hough để tối đa hàng detect đúng.
    - Reconstruct hàng sai bằng tọa độ x từ hàng đúng.
    - Winner-takes-all với margin check thay vì MULTI.
    """
    N_ROWS, N_COLS = 4, 2
    LABELS = ["D", "S"]
    MARGIN_RATIO = 0.20  # winner phải hơn ô kia ít nhất 20%

    h, w = binary_crop.shape
    cell_h = h / N_ROWS
    cell_w = w / N_COLS

    grid_rows = _detect_dung_sai_cascade(gray_crop, n_rows=N_ROWS, n_cols=N_COLS)
    if len(grid_rows) > N_ROWS:
        grid_rows = grid_rows[-N_ROWS:]

    results = []
    for ri in range(N_ROWS):
        row = grid_rows[ri] if ri < len(grid_rows) else []

        if len(row) == N_COLS:
            ratios = [fill_ratio(binary_crop, x, y, r) for (x, y, r) in row]
            print("DungSai:", ratios)
        else:
            # Fallback grid nếu vẫn không detect được
            ratios = []
            for ci in range(N_COLS):
                y1 = int(ri * cell_h); y2 = int((ri + 1) * cell_h)
                x1 = int(ci * cell_w); x2 = int((ci + 1) * cell_w)
                cell = binary_crop[y1:y2, x1:x2]
                ch, cw = cell.shape
                cy, cx = ch // 2, cw // 2
                rad = int(min(ch, cw) * 0.35)
                mask = np.zeros((ch, cw), dtype=np.uint8)
                cv2.circle(mask, (cx, cy), rad, 255, -1)
                region = cv2.bitwise_and(cell, mask)
                total = cv2.countNonZero(mask)
                ratios.append(cv2.countNonZero(region) / float(total) if total > 0 else 0)

        max_r = max(ratios)
        if max_r < fill_threshold:
            results.append(None)
            continue

        winner = ratios.index(max_r)
        other_max = max(r for i, r in enumerate(ratios) if i != winner)

        if other_max >= fill_threshold and (max_r - other_max) < MARGIN_RATIO * max_r:
            results.append("MULTI")
        else:
            results.append(LABELS[winner])

    return results
def recognize_full_sheet(segments: dict, fill_threshold=0.35) -> dict:
    """
    Nhận dạng toàn bộ phiếu từ dict segments (do segmentation.py trả về).
    """
    from preprocessing import preprocess_pipeline

    final = {}

    # ---- SBD & Mã đề ----
    for key, n_digits in [("sbd", 6), ("made", 3)]:
        crop = segments[key]
        pre = preprocess_pipeline(crop)
        final[key] = snap_grid_digits(pre["gray"], pre["binary"], n_digits)

    # ---- Phần I: 4 block x 10 câu x 4 đáp án A/B/C/D ----
    phan1_all = []
    for b in range(1, 5):
        crop = segments[f"phan1_block{b}"]
        pre = preprocess_pipeline(crop)
        ans = recognize_answers_mcq(pre["gray"], pre["binary"],
                                     n_rows=10, n_cols=4,
                                     choice_labels=["A", "B", "C", "D"],
                                     fill_threshold=fill_threshold)
        phan1_all.extend(ans)
    final["phan1"] = {i + 1: a for i, a in enumerate(phan1_all)}

    # ---- Phần II: 4 block, mỗi block 2 câu x 4 ý x Đúng/Sai ----
    # Thêm morphological opening + CLAHE mạnh hơn cho Phần II (nhỏ, tối).
    phan2_all = {}
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clahe_strong = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    for b in range(1, 5):
        crop = segments[f"phan2_block{b}"]
        gray_p2 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        gray_p2 = cv2.GaussianBlur(gray_p2, (3, 3), 0)
        gray_p2 = clahe_strong.apply(gray_p2)
        binary_p2 = cv2.adaptiveThreshold(
            gray_p2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=15, C=6
        )
        binary_cleaned = cv2.morphologyEx(binary_p2, cv2.MORPH_OPEN, kernel_open)
        pre = {"gray": gray_p2, "binary": binary_p2}
        h, w = pre["gray"].shape
        half = w // 2
        for side, (x1, x2) in enumerate([(0, half), (half, w)]):
            sub_gray = pre["gray"][:, x1:x2]
            sub_bin  = binary_cleaned[:, x1:x2]
            ans = recognize_answers_dung_sai(sub_gray, sub_bin,
                                               fill_threshold=0.18)
            cau_index = (b - 1) * 2 + side + 1
            phan2_all[cau_index] = ans
    final["phan2"] = phan2_all

    # ---- Phần III: 6 block, số 4 chữ số, lưới 0-9 ----
    phan3_all = {}
    for b in range(1, 7):
        crop = segments[f"phan3_block{b}"]
        pre = preprocess_pipeline(crop)
        phan3_all[b] = snap_grid_digits(pre["gray"], pre["binary"], n_digits=4)
    final["phan3"] = phan3_all

    return final

def snap_grid_digits(gray_crop, binary_crop, n_digits, fill_threshold=0.33):
    """
    Đọc lưới số 0-9: dùng cluster_to_grid để nhóm đúng 10 hàng (đã xử lý merge nhiễu),
    sau đó ghép cột theo tọa độ x thực tế (không chuẩn hóa min-max toàn cục vì dễ lệch
    khi có nhiễu ở rìa).
    """
    circles = detect_circles_small(gray_crop)
    grid_rows = cluster_to_grid(circles)

    # Tách riêng 2 hàng đặc biệt (dấu trừ "-", dấu phẩy ",") nằm phía trên 10 hàng số 0-9
    special_rows = []
    if len(grid_rows) > 10:
        special_rows = grid_rows[:len(grid_rows) - 10]
        grid_rows = grid_rows[-10:]  # giữ 10 hàng cuối = hàng số 0-9
    if len(grid_rows) < 10:
        return "?" * n_digits

    # Xác định cột chuẩn bằng cách gom tọa độ x của tất cả circle trong 10 hàng
    all_circles = [c for row in grid_rows for c in row]
    all_circles_sorted = sorted(all_circles, key=lambda c: c[0])

    col_centers = []
    col_tol = 12
    for c in all_circles_sorted:
        x, y, r = c
        placed = False
        for col in col_centers:
            if abs(x - col["x"]) <= col_tol:
                col["xs"].append(x)
                col["x"] = sum(col["xs"]) / len(col["xs"])
                placed = True
                break
        if not placed:
            col_centers.append({"x": x, "xs": [x]})

    col_centers = sorted(col_centers, key=lambda c: c["x"])
    # Nếu phát hiện dư/thiếu cột so với n_digits, cắt hoặc bỏ qua cột lệch nhất
    col_centers = col_centers[:n_digits] if len(col_centers) >= n_digits else col_centers
    col_xs = [c["x"] for c in col_centers]

    if len(col_xs) < n_digits:
        return "?" * n_digits

    best_ratio = [0.0] * n_digits
    digits = [None] * n_digits
    for digit_val, row in enumerate(grid_rows):
        for (x, y, r) in row:
            col_idx = min(range(len(col_xs)), key=lambda i: abs(col_xs[i] - x))
            ratio = fill_ratio(binary_crop, x, y, r)
            if ratio >= fill_threshold and ratio > best_ratio[col_idx]:
                best_ratio[col_idx] = ratio
                digits[col_idx] = str(digit_val)

    return "".join(d if d else "?" for d in digits)
def read_digit_grid_fixed(binary_crop, n_digits, n_rows=10, fill_threshold=0.35,
                           top_skip_ratio=0.18):
    """
    Đọc lưới số 0-9 bằng cách CHIA Ô CỐ ĐỊNH thay vì dò Hough Circle.
    Áp dụng: Phân đoạn ảnh dạng lưới đều (grid-based segmentation) - 
    phù hợp khi ảnh đã chuẩn hóa phối cảnh nên lưới ô tròn cách đều nhau.

    top_skip_ratio: bỏ qua phần trăm chiều cao phía trên (chứa ô nhập số, dấu -, dấu ,)
                    trước khi bắt đầu lưới 0-9.
    """
    h, w = binary_crop.shape
    y_start = int(h * top_skip_ratio)
    grid_h = h - y_start
    cell_h = grid_h / n_rows
    cell_w = w / n_digits

    digits = [None] * n_digits
    best_ratio = [0.0] * n_digits

    for row in range(n_rows):
        y1 = int(y_start + row * cell_h)
        y2 = int(y_start + (row + 1) * cell_h)
        for col in range(n_digits):
            x1 = int(col * cell_w)
            x2 = int((col + 1) * cell_w)

            cell = binary_crop[y1:y2, x1:x2]
            ch, cw = cell.shape
            if ch == 0 or cw == 0:
                continue

            # Chỉ lấy vùng tròn ở giữa ô (tránh viền ô/đường kẻ lân cận)
            cy, cx = ch // 2, cw // 2
            radius = int(min(ch, cw) * 0.35)
            mask = np.zeros((ch, cw), dtype=np.uint8)
            cv2.circle(mask, (cx, cy), radius, 255, -1)
            region = cv2.bitwise_and(cell, mask)

            filled = cv2.countNonZero(region)
            total = cv2.countNonZero(mask)
            ratio = filled / float(total) if total > 0 else 0

            if ratio >= fill_threshold and ratio > best_ratio[col]:
                best_ratio[col] = ratio
                digits[col] = str(row)

    return "".join(d if d else "?" for d in digits)
if __name__ == "__main__":
    from preprocessing import preprocess_pipeline

    # Test nhanh trên 1 crop Phần I block 1 (đã lưu ở bước segmentation)
    crop = cv2.imread("output/debug/06_phan1_block1.jpg")
    if crop is None:
        print("Chưa có file crop, chạy segmentation.py trước.")
    else:
        pre = preprocess_pipeline(crop)
        answers = recognize_answers_mcq(
            pre["gray"], pre["binary"],
            n_rows=10, n_cols=4, choice_labels=["A", "B", "C", "D"]
        )
        for i, ans in enumerate(answers, start=1):
            print(f"Câu {i}: {ans}")