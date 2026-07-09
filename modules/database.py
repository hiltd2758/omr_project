"""
Module: Lưu trữ kết quả chấm bài bằng SQLite (thay cho session_state để dữ liệu
không bị mất khi reload trang).
"""

import os
import json
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "omr_results.db")


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ket_qua (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ten_file TEXT,
            sbd TEXT,
            made TEXT,
            diem_phan1 REAL,
            diem_phan2 REAL,
            diem_phan2_raw REAL,
            diem_phan3 REAL,
            tong_diem REAL,
            chi_tiet TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def insert_result(graded: dict):
    conn = get_connection()
    conn.execute(
        """INSERT INTO ket_qua
           (ten_file, sbd, made, diem_phan1, diem_phan2, diem_phan2_raw, diem_phan3, tong_diem, chi_tiet)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            graded.get("ten_file"),
            graded.get("sbd"),
            graded.get("made"),
            graded.get("diem_phan1"),
            graded.get("diem_phan2"),
            graded.get("diem_phan2_raw"),
            graded.get("diem_phan3"),
            graded.get("tong_diem"),
            json.dumps(graded.get("chi_tiet", {}), ensure_ascii=False),
        ),
    )
    conn.commit()
    conn.close()


def get_all_results() -> list:
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM ket_qua ORDER BY id").fetchall()
    conn.close()

    results = []
    for row in rows:
        item = dict(row)
        item["chi_tiet"] = json.loads(item["chi_tiet"]) if item["chi_tiet"] else {}
        results.append(item)
    return results


def delete_all():
    conn = get_connection()
    conn.execute("DELETE FROM ket_qua")
    conn.commit()
    conn.close()