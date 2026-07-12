"""
make_charts.py
Vẽ biểu đồ cho Chương 4 từ số liệu đã thu thập (hardcode trực tiếp,
không cần đọc lại file JSON — copy đúng số bạn đã có).
"""

import matplotlib.pyplot as plt
import os

OUT_DIR = "report_charts"
os.makedirs(OUT_DIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Đã lưu: {path}")


# ---- 4.4.1: Otsu vs Adaptive (flood_ratio) ----
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["Otsu", "Adaptive"], [0.1936, 0.0157], color=["#e74c3c", "#2ecc71"])
ax.set_ylabel("Flood ratio trung bình")
ax.set_title("4.4.1 Otsu vs Adaptive Threshold")
for i, v in enumerate([0.1936, 0.0157]):
    ax.text(i, v + 0.003, f"{v:.4f}", ha="center")
save(fig, "4_4_1_otsu_vs_adaptive.png")

# ---- 4.4.2: Fill Threshold sweep ----
thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
accuracy = [91.75, 91.75, 93.50, 93.00, 93.50, 91.50, 87.25]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(thresholds, accuracy, marker="o", color="#3498db")
ax.axvline(0.30, color="#2ecc71", linestyle="--", label="Chọn: 0.30")
ax.set_xlabel("Fill Threshold")
ax.set_ylabel("Accuracy (%)")
ax.set_title("4.4.2 Khảo sát Fill Threshold")
ax.legend()
save(fig, "4_4_2_fill_threshold.png")

# ---- 4.4.3: min_area sweep ----
min_areas = [100, 200, 300, 400, 500]
pass_rate = [100.0, 100.0, 100.0, 100.0, 23.53]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(min_areas, pass_rate, marker="o", color="#9b59b6")
ax.axvline(300, color="#2ecc71", linestyle="--", label="Đang dùng: 300")
ax.set_xlabel("min_area")
ax.set_ylabel("Tỉ lệ phát hiện đủ marker (%)")
ax.set_title("4.4.3 Khảo sát min_area Marker")
ax.legend()
save(fig, "4_4_3_min_area.png")

# ---- 4.4.4: param2 sweep ----
param2_vals = [10, 12, 15, 18, 20, 25]
row_acc = [54.85, 85.44, 98.09, 99.85, 99.41, 74.12]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(param2_vals, row_acc, marker="o", color="#e67e22")
ax.axvline(15, color="#2ecc71", linestyle="--", label="Đang dùng: 15")
ax.set_xlabel("param2 (Hough Circle)")
ax.set_ylabel("Tỉ lệ hàng đúng (%)")
ax.set_title("4.4.4 Khảo sát param2 Hough Circle")
ax.legend()
save(fig, "4_4_4_param2.png")

# ---- 4.5: Accuracy từng ảnh ----
images = [f"s{i:02d}" for i in range(1, 11)]
acc_per_image = [100, 100, 97.5, 97.5, 95, 77.5, 87.5, 85, 97.5, 92.5]
fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#e74c3c" if a < 85 else "#3498db" for a in acc_per_image]
ax.bar(images, acc_per_image, color=colors)
ax.axhline(93.0, color="#2ecc71", linestyle="--", label="TB = 93.0%")
ax.set_ylabel("Accuracy (%)")
ax.set_title("4.5 Accuracy theo từng ảnh (Phần I)")
ax.legend()
save(fig, "4_5_accuracy_per_image.png")

print(f"\nToàn bộ biểu đồ đã lưu trong thư mục: {OUT_DIR}/")