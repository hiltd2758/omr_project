"""
Module: Chấm điểm (Grading)
Chỉ so sánh kết quả nhận dạng (từ recognition.py) với đáp án chuẩn do người dùng
nhập, KHÔNG can thiệp vào thuật toán xử lý ảnh.

Thang điểm mặc định theo định dạng đề thi THPT có 3 phần:
- Phần I (trắc nghiệm A/B/C/D): mỗi câu đúng = 0.25 điểm
- Phần II (Đúng/Sai, 4 ý/câu): số ý đúng -> 1 ý=0.1, 2 ý=0.25, 3 ý=0.5, 4 ý=1.0
- Phần III (điền số): mỗi câu đúng = 0.25 điểm
"""

PHAN2_PARTIAL = {0: 0.0, 1: 0.1, 2: 0.25, 3: 0.5, 4: 1.0}


def grade_phan1(student: dict, key: dict, diem_per_cau: float = 0.25):
    chi_tiet, diem = {}, 0.0
    for cau, dap_an in key.items():
        tra_loi = student.get(cau)
        dung = (tra_loi == dap_an)
        if dung:
            diem += diem_per_cau
        chi_tiet[cau] = {"dap_an": dap_an, "tra_loi": tra_loi, "dung": dung}
    return diem, chi_tiet


def _chuan_hoa(ky_tu):
    ky_tu = str(ky_tu).strip().upper()
    return "D" if ky_tu in ("D", "Đ") else ky_tu

def grade_phan2(student: dict, key: dict):
    chi_tiet, diem = {}, 0.0
    for cau, dap_an in key.items():
        dap_an_chuan = [_chuan_hoa(x) for x in dap_an]
        tra_loi = [_chuan_hoa(x) for x in student.get(cau, [])]
        so_y_dung = sum(
            1 for i, y in enumerate(dap_an_chuan)
            if i < len(tra_loi) and tra_loi[i] == y
        )
        diem_cau = PHAN2_PARTIAL.get(so_y_dung, 0.0)
        diem += diem_cau
        chi_tiet[cau] = {"dap_an": dap_an, "tra_loi": student.get(cau, []),
                          "so_y_dung": so_y_dung, "diem": diem_cau}
    return diem, chi_tiet   

def grade_phan3(student: dict, key: dict, diem_per_cau: float = 0.25):
    chi_tiet, diem = {}, 0.0
    for cau, dap_an in key.items():
        tra_loi = student.get(cau)
        dung = (str(tra_loi) == str(dap_an))
        if dung:
            diem += diem_per_cau
        chi_tiet[cau] = {"dap_an": dap_an, "tra_loi": tra_loi, "dung": dung}
    return diem, chi_tiet


def grade_submission(result: dict, answer_key: dict) -> dict:
    """
    result: kết quả trả về từ recognition.recognize_full_sheet()
    answer_key: {"phan1": {...}, "phan2": {...}, "phan3": {...}}
    """
    d1, ct1 = grade_phan1(result.get("phan1", {}), answer_key.get("phan1", {}))
    d2, ct2 = grade_phan2(result.get("phan2", {}), answer_key.get("phan2", {}))
    d3, ct3 = grade_phan3(result.get("phan3", {}), answer_key.get("phan3", {}))
    tong_diem = round((d1 / 10.0) * 2.5 + (d2 / 8.0) * 4.0 + (d3 / 1.5) * 3.5, 2)
    return {
        "sbd": result.get("sbd", "?"),
        "made": result.get("made", "?"),
        "diem_phan1": round(d1, 2),
        "diem_phan2": round(d2, 2),
        "diem_phan3": round(d3, 2),
        "tong_diem": tong_diem,
        "chi_tiet": {"phan1": ct1, "phan2": ct2, "phan3": ct3},
    }