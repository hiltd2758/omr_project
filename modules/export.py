"""
Module: Xuất kết quả (Export)
Xuất bảng điểm tổng hợp ra Excel (.xlsx) bằng pandas/openpyxl.
"""

import io
import json
import pandas as pd


def history_to_dataframe(history: list) -> pd.DataFrame:
    """history: list các dict trả về từ grading.grade_submission (đã gộp thêm 'ten_file')."""
    rows = []
    for item in history:
        rows.append({
            "Tên file": item.get("ten_file", ""),
            "SBD": item.get("sbd", ""),
            "Mã đề": item.get("made", ""),
            "Điểm Phần I": item.get("diem_phan1", 0),
            "Điểm Phần II": item.get("diem_phan2", 0),
            "Điểm Phần III": item.get("diem_phan3", 0),
            "Tổng điểm": item.get("tong_diem", 0),
        })
    return pd.DataFrame(rows)


def export_excel_bytes(history: list) -> bytes:
    df = history_to_dataframe(history)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="KetQua")
    buffer.seek(0)
    return buffer.read()


def parse_answer_key_file(uploaded_file) -> dict:
    """
    Đọc file đáp án chuẩn từ Excel (.xlsx) hoặc JSON (.json).

    Định dạng Excel — 3 cột: Phan (1/2/3), Cau (số), DapAn (text).
      Phan 1: DapAn là "A"/"B"/"C"/"D"
      Phan 2: DapAn là 4 ký tự Đ/S, vd "DSDS"
      Phan 3: DapAn là chuỗi số, vd "1234"

    Định dạng JSON:
      {"phan1": {"1": "A", "2": "B"},
       "phan2": {"1": "DSDS"},
       "phan3": {"1": "1234"}}
    """
    name = uploaded_file.name.lower()
    key = {"phan1": {}, "phan2": {}, "phan3": {}}

    if name.endswith(".json"):
        data = json.load(uploaded_file)
        for phan in ("phan1", "phan2", "phan3"):
            for cau, dap in data.get(phan, {}).items():
                cau = int(cau)
                if phan == "phan2":
                    key[phan][cau] = list(str(dap).upper())
                elif phan == "phan1":
                    key[phan][cau] = str(dap).upper()
                else:
                    key[phan][cau] = str(dap)

    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
        df.columns = [str(c).strip().lower() for c in df.columns]
        for _, row in df.iterrows():
            phan = str(row["phan"]).strip()
            cau = int(row["cau"])
            dap = str(row["dapan"]).strip()
            if phan == "1":
                key["phan1"][cau] = dap.upper()
            elif phan == "2":
                key["phan2"][cau] = list(dap.upper())
            elif phan == "3":
                key["phan3"][cau] = dap

    else:
        raise ValueError("Chỉ hỗ trợ file .xlsx hoặc .json")

    return key