"""
make_ground_truth.py (v2)
Hỗ trợ tạo ground-truth JSON cho Phần I, có xử lý câu multi-mark và bỏ trống.

Ký tự hợp lệ khi nhập:
    A B C D   -> đáp án rõ ràng, tô đúng 1 ô
    .         -> bỏ trống, không tô ô nào
    M         -> multi-mark, thí sinh tô từ 2 ô trở lên (không tính vào accuracy)
"""

import os
import json
import argparse


VALID_CHARS = set("ABCD.M")


def collect_answers(n_questions: int = 40) -> dict:
    print(f"\nNhập đáp án Phần I ({n_questions} câu). Mỗi dòng 10 câu, không dấu cách.")
    print("Ký tự hợp lệ: A B C D (đáp án rõ) | . (bỏ trống) | M (multi-mark, tô ≥2 ô)\n")

    answers = {}
    q = 1
    while q <= n_questions:
        remaining = min(10, n_questions - q + 1)
        line = input(f"Câu {q}-{q + remaining - 1}: ").strip().upper()
        line = line.replace(" ", "")

        if len(line) != remaining:
            print(f"  ⚠ Cần đúng {remaining} ký tự, bạn nhập {len(line)}. Nhập lại dòng này.")
            continue

        invalid = [c for c in line if c not in VALID_CHARS]
        if invalid:
            print(f"  ⚠ Ký tự không hợp lệ: {invalid}. Chỉ dùng A/B/C/D/./M. Nhập lại dòng này.")
            continue

        for i, ch in enumerate(line):
            answers[str(q + i)] = ch
        q += remaining

    return answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--n", type=int, default=40)
    args = parser.parse_args()

    base = os.path.splitext(os.path.basename(args.image))[0]
    out_dir = os.path.dirname(args.image) or "."
    out_path = os.path.join(out_dir, f"{base}_gt.json")

    print(f"=== Tạo ground-truth cho: {args.image} ===")
    phan1 = collect_answers(args.n)
    data = {"phan1": phan1}

    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing["phan1"] = phan1
        data = existing

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    n_multi = sum(1 for v in phan1.values() if v == "M")
    n_blank = sum(1 for v in phan1.values() if v == ".")
    print(f"\n✔ Đã lưu: {out_path}")
    print(f"  Số câu multi-mark: {n_multi} | Số câu bỏ trống: {n_blank}")


if __name__ == "__main__":
    main()