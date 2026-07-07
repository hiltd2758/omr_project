"""Debug Phần II v6. Chạy: python debug_phan2.py input/student01.jpg"""
import os, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "modules"))
from detection import find_corner_markers
from perspective import warp_perspective
from segmentation import segment_all
import preprocessing as pp
from recognition import recognize_answers_dung_sai

def preprocess_phan2(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=15, C=6)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return gray, cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

def main():
    if len(sys.argv) < 2:
        print("Dùng: python debug_phan2.py <ảnh>"); sys.exit(1)
    image = cv2.imread(sys.argv[1])
    pre = pp.preprocess_pipeline(image)
    corners, _ = find_corner_markers(pre["binary"])
    warped, _ = warp_perspective(image, corners)
    segments = segment_all(warped)

    print("=" * 40)
    for b in range(1, 5):
        crop = segments[f"phan2_block{b}"]
        gray, binary = preprocess_phan2(crop)
        h, w = gray.shape
        half = w // 2
        for side, (x1, x2) in enumerate([(0, half), (half, w)]):
            cau = (b-1)*2 + side + 1
            ans = recognize_answers_dung_sai(gray[:, x1:x2], binary[:, x1:x2], fill_threshold=0.18)
            print(f"Câu {cau}: {', '.join(str(a) if a else '(bỏ trống)' for a in ans)}")
    print("=" * 40)

if __name__ == "__main__":
    main()