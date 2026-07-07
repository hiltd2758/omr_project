"""
Module: Chấm điểm (Grading)
Chỉ so sánh kết quả nhận dạng (từ recognition.py) với đáp án chuẩn do người dùng
nhập, KHÔNG can thiệp vào thuật toán xử lý ảnh.

Thang điểm theo đề thi THPT 3 phần (thang 10):
- Phần I (trắc nghiệm A/B/C/D): 40 câu, tổng 4 điểm, mỗi câu đúng = 0.1 điểm
- Phần II (Đúng/Sai, 4 ý/câu): 8 câu, tổng 4 điểm (raw max = 8, quy đổi chia 2)
  Đúng 1 ý = 0.1 điểm, 2 ý = 0.25 điểm, 3 ý = 0.5 điểm, 4 ý = 1.0 điểm
- Phần III (điền số): 6 câu, tổng 2 điểm, mỗi câu đúng = 0.333... điểm
"""

PHAN2_PARTIAL = {0: 0.0, 1: 0.1, 2: 0.25, 3: 0.5, 4: 1.0}


def grade_phan1(student: dict, key: dict, diem_per_cau: float = 0.1):
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

def grade_phan3(student: dict, key: dict, diem_per_cau: float = 2.0/6.0):
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
    
    Phần II có raw score tối đa 8 điểm, quy đổi về thang 10 bằng cách chia 2.
    """
    d1, ct1 = grade_phan1(result.get("phan1", {}), answer_key.get("phan1", {}))
    d2, ct2 = grade_phan2(result.get("phan2", {}), answer_key.get("phan2", {}))
    d3, ct3 = grade_phan3(result.get("phan3", {}), answer_key.get("phan3", {}))
    
    d2_quy_doi = d2 / 2  # Quy đổi Phần II từ thang 8 về thang 4
    
    tong_diem = round(d1 + d2_quy_doi + d3, 2)
    return {
        "sbd": result.get("sbd", "?"),
        "made": result.get("made", "?"),
        "diem_phan1": round(d1, 2),
        "diem_phan2": round(d2_quy_doi, 2),
        "diem_phan2_raw": round(d2, 2),
        "diem_phan3": round(d3, 2),
        "tong_diem": tong_diem,
        "chi_tiet": {"phan1": ct1, "phan2": ct2, "phan3": ct3},
    }