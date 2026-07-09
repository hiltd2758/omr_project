"""
Giao diện Web Demo hệ thống OMR (Streamlit)
- CHỈ ghép nối (orchestration) các hàm đã có sẵn trong modules/, KHÔNG chỉnh sửa
  thuật toán / logic xử lý ảnh gốc.
- 3 trang: Dashboard, Chấm bài, Kết quả.
- Luồng xử lý: preprocessing -> detection (marker) -> perspective (warp)
  -> segmentation (tách vùng) -> recognition (nhận dạng SBD/Mã đề/Đáp án)
  -> grading (chấm điểm theo đáp án chuẩn, tùy chọn) -> export (xuất Excel)
- Lịch sử kết quả được lưu bền vững vào SQLite (modules/database.py) thay vì
  session_state, nên không bị mất khi reload trang hoặc restart app.
"""

import os
import sys
import json

import cv2
import numpy as np
import pandas as pd
import streamlit as st

MODULES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules")
sys.path.insert(0, MODULES_DIR)

from preprocessing import preprocess_pipeline
from detection import find_corner_markers, draw_debug_markers
from perspective import warp_perspective
from segmentation import segment_all
from recognition import recognize_full_sheet
from grading import grade_submission
from export import export_excel_bytes, history_to_dataframe, parse_answer_key_file
from database import init_db, insert_result, get_all_results, delete_all

st.set_page_config(page_title="Hệ thống OMR", layout="wide", page_icon="📝")

init_db()  # Đảm bảo bảng SQLite đã tồn tại trước khi dùng

if "answer_key" not in st.session_state:
    st.session_state.answer_key = None      # đáp án chuẩn (tùy chọn, vẫn giữ trong session)


# ---------------------------------------------------------------------------
# Hàm xử lý dùng chung — gọi đúng pipeline gốc, không thay đổi logic
# ---------------------------------------------------------------------------
def process_image(image: np.ndarray):
    pre = preprocess_pipeline(image)
    corners, candidates = find_corner_markers(pre["binary"])
    marker_debug = draw_debug_markers(image, corners, candidates)
    warped, _ = warp_perspective(image, corners)
    segments = segment_all(warped)
    result = recognize_full_sheet(segments)
    return {
        "marker_debug": marker_debug,
        "warped": warped,
        "segments": segments,
        "result": result,
        # Các bước tiền xử lý
        "gray": pre["gray"],
        "denoised": pre["denoised"],
        "equalized": pre["equalized"],
        "binary": pre["binary"],
    }


def parse_answer_key(text_phan1, text_phan2, text_phan3):
    """Phân tích đáp án chuẩn do người dùng nhập tay dạng text đơn giản."""
    key = {"phan1": {}, "phan2": {}, "phan3": {}}
    for pair in text_phan1.split(","):
        if ":" in pair:
            cau, dap = pair.split(":")
            key["phan1"][int(cau.strip())] = dap.strip().upper()
    for pair in text_phan2.split(","):
        if ":" in pair:
            cau, dap = pair.split(":")
            key["phan2"][int(cau.strip())] = list(dap.strip().upper())
    for pair in text_phan3.split(","):
        if ":" in pair:
            cau, dap = pair.split(":")
            key["phan3"][int(cau.strip())] = dap.strip()
    return key


def answer_key_to_dataframe(key: dict) -> pd.DataFrame:
    """Chuyển đáp án chuẩn thành DataFrame để hiển thị dạng bảng."""
    rows = []
    for cau, dap in sorted(key.get("phan1", {}).items()):
        rows.append({"Phần": "I", "Câu": cau, "Đáp án": dap})
    for cau, dap_list in sorted(key.get("phan2", {}).items()):
        dap_str = "".join(dap_list)
        rows.append({"Phần": "II", "Câu": cau, "Đáp án": dap_str})
    for cau, dap in sorted(key.get("phan3", {}).items()):
        rows.append({"Phần": "III", "Câu": cau, "Đáp án": dap})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sidebar điều hướng
# ---------------------------------------------------------------------------
st.sidebar.title("Hệ thống OMR")
page = st.sidebar.radio("Điều hướng", ["Dashboard", "Chấm bài", "Kết quả"])
st.sidebar.markdown("---")
st.sidebar.caption("Demo trực quan hóa pipeline OMR. Không thay đổi thuật toán backend.")


# ---------------------------------------------------------------------------
# TRANG 1: DASHBOARD
# ---------------------------------------------------------------------------
if page == "Dashboard":
    st.title("Dashboard tổng quan")

    history = get_all_results()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số bài đã chấm", len(history))
    if history:
        diem_list = [h["tong_diem"] for h in history]
        c2.metric("Điểm trung bình", round(sum(diem_list) / len(diem_list), 2))
        c3.metric("Điểm cao nhất", max(diem_list))
        c4.metric("Điểm thấp nhất", min(diem_list))

        st.subheader("Phân bố điểm")
        df = history_to_dataframe(history)
        st.bar_chart(df.set_index("SBD")["Tổng điểm"])

        st.subheader("5 bài chấm gần nhất")
        st.dataframe(df.tail(5), use_container_width=True, hide_index=True)
    else:
        c2.metric("Điểm trung bình", "—")
        c3.metric("Điểm cao nhất", "—")
        c4.metric("Điểm thấp nhất", "—")
        st.info("Chưa có bài nào được chấm. Vào trang **Chấm bài** để bắt đầu.")

    st.markdown("---")
    st.subheader("Đáp án chuẩn hiện tại")
    if st.session_state.answer_key:
        st.dataframe(
            answer_key_to_dataframe(st.session_state.answer_key),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("Chưa thiết lập đáp án chuẩn — hệ thống vẫn nhận dạng được bài "
                    "nhưng chưa thể tính điểm. Thiết lập ở trang **Chấm bài**.")


# ---------------------------------------------------------------------------
# TRANG 2: CHẤM BÀI
# ---------------------------------------------------------------------------
elif page == "Chấm bài":
    st.title("Chấm bài trắc nghiệm")

    with st.expander("Thiết lập đáp án chuẩn (tùy chọn — để tính điểm)"):
        tab_file, tab_json, tab_manual = st.tabs(["Tải file (Excel/JSON)", "Dán JSON", "Nhập tay"])

        with tab_file:
            st.caption(
                "Excel (.xlsx): 3 cột **Phan, Cau, DapAn** — Phần I: A/B/C/D, "
                "Phần II: 4 ký tự Đ/S (vd DSDS), Phần III: chuỗi số.  \n"
                "JSON: `{\"phan1\":{\"1\":\"A\"}, \"phan2\":{\"1\":\"DSDS\"}, \"phan3\":{\"1\":\"1234\"}}`"
            )
            key_file = st.file_uploader("Chọn file đáp án chuẩn", type=["xlsx", "json"], key="key_file")
            if key_file is not None:
                try:
                    st.session_state.answer_key = parse_answer_key_file(key_file)
                    st.success("Đã nạp đáp án chuẩn từ file.")
                    st.subheader("Xem trước đáp án đã nạp")
                    st.dataframe(
                        answer_key_to_dataframe(st.session_state.answer_key),
                        use_container_width=True,
                        hide_index=True
                    )
                except Exception as e:
                    st.error(f"Lỗi đọc file đáp án: {e}")

        with tab_json:
            st.caption(
                "Dán trực tiếp JSON đáp án chuẩn, ví dụ:\n\n"
                '`{"phan1":{"1":"A","2":"B"}, "phan2":{"1":"DSDS"}, "phan3":{"1":"1234"}}`'
            )
            json_text = st.text_area("Nội dung JSON", height=150, key="json_text")
            if st.button("Lưu đáp án chuẩn (JSON)"):
                try:
                    data = json.loads(json_text)
                    key = {"phan1": {}, "phan2": {}, "phan3": {}}
                    for phan in ("phan1", "phan2", "phan3"):
                        for cau, dap in data.get(phan, {}).items():
                            cau = int(cau)
                            if phan == "phan2":
                                key[phan][cau] = list(str(dap).upper())
                            elif phan == "phan1":
                                key[phan][cau] = str(dap).upper()
                            else:
                                key[phan][cau] = str(dap)
                    st.session_state.answer_key = key
                    st.success("Đã lưu đáp án chuẩn từ JSON.")
                except Exception as e:
                    st.error(f"JSON không hợp lệ: {e}")

        with tab_manual:
            st.caption("Định dạng: `câu:đáp_án, câu:đáp_án, ...`. "
                       "Phần II mỗi câu 4 ý Đúng/Sai, ví dụ `1:DSDS`.")
            col1, col2, col3 = st.columns(3)
            t1 = col1.text_area("Phần I (A/B/C/D)", placeholder="1:A, 2:B, 3:C, ...", height=100)
            t2 = col2.text_area("Phần II (Đ/S x4 ý)", placeholder="1:DSDS, 2:SSDD, ...", height=100)
            t3 = col3.text_area("Phần III (điền số)", placeholder="1:1234, 2:0567, ...", height=100)
            if st.button("Lưu đáp án chuẩn (nhập tay)"):
                st.session_state.answer_key = parse_answer_key(t1, t2, t3)
                st.success("Đã lưu đáp án chuẩn.")

    st.markdown("---")
    uploaded_file = st.file_uploader("Tải lên ảnh phiếu trả lời trắc nghiệm",
                                      type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Không thể đọc ảnh. Vui lòng thử ảnh khác.")
            st.stop()

        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", width=350)

        if st.button("Thực hiện chấm bài", type="primary"):
            status = st.empty()
            try:
                status.info("Đang tiền xử lý ảnh...")
                status.info("Đang phát hiện marker 4 góc và chỉnh phối cảnh...")
                out = process_image(image)
                status.info("Đang phân đoạn và nhận dạng đáp án...")
                result = out["result"]
                status.success("Chấm bài thành công!")

                st.subheader("Quy trình xử lý")

                st.markdown("**1. Tiền xử lý ảnh**")
                p1, p2, p3, p4 = st.columns(4)
                with p1:
                    st.image(cv2.cvtColor(out["gray"], cv2.COLOR_BGR2RGB),
                             caption="1.1 Ảnh xám (Grayscale)")
                with p2:
                    st.image(cv2.cvtColor(out["denoised"], cv2.COLOR_GRAY2BGR) if len(out["denoised"].shape) == 2 else out["denoised"],
                             caption="1.2 Khử nhiễu (Gaussian Blur)")
                with p3:
                    st.image(cv2.cvtColor(out["equalized"], cv2.COLOR_GRAY2BGR) if len(out["equalized"].shape) == 2 else out["equalized"],
                             caption="1.3 Cân bằng sáng (CLAHE)")
                with p4:
                    st.image(cv2.cvtColor(out["binary"], cv2.COLOR_GRAY2BGR) if len(out["binary"].shape) == 2 else out["binary"],
                             caption="1.4 Nhị phân hóa (Adaptive Threshold)")

                st.markdown("**2. Phát hiện & Chỉnh phối cảnh**")
                c1, c2 = st.columns(2)
                with c1:
                    st.image(cv2.cvtColor(out["marker_debug"], cv2.COLOR_BGR2RGB),
                             caption="2.1 Phát hiện marker 4 góc")
                with c2:
                    st.image(cv2.cvtColor(out["warped"], cv2.COLOR_BGR2RGB),
                             caption="2.2 Ảnh sau chỉnh phối cảnh (Warp)")

                st.markdown("**3. Phân đoạn vùng (Segmentation)**")

                # Nhóm các vùng theo loại
                sbd_crop = out["segments"].get("sbd")
                made_crop = out["segments"].get("made")
                phan1_crops = {k: v for k, v in out["segments"].items() if k.startswith("phan1")}
                phan2_crops = {k: v for k, v in out["segments"].items() if k.startswith("phan2")}
                phan3_crops = {k: v for k, v in out["segments"].items() if k.startswith("phan3")}

                # Hiển thị SBD và Mã đề
                if sbd_crop is not None or made_crop is not None:
                    sbd_made_cols = st.columns(2)
                    if sbd_crop is not None:
                        with sbd_made_cols[0]:
                            st.image(cv2.cvtColor(sbd_crop, cv2.COLOR_BGR2RGB),
                                     caption="Số báo danh (SBD)", width=250)
                    if made_crop is not None:
                        with sbd_made_cols[1]:
                            st.image(cv2.cvtColor(made_crop, cv2.COLOR_BGR2RGB),
                                     caption="Mã đề", width=250)

                # Hiển thị Phần I - 40 câu trắc nghiệm
                if phan1_crops:
                    st.markdown("**Phần I — 40 câu trắc nghiệm**")
                    phan1_list = list(phan1_crops.items())
                    for i in range(0, len(phan1_list), 5):
                        row_crops = phan1_list[i:i+5]
                        cols = st.columns(5)
                        for j, (name, crop) in enumerate(row_crops):
                            with cols[j]:
                                st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                                         caption=name, width=120)

                # Hiển thị Phần II - 10 câu Đúng/Sai
                if phan2_crops:
                    st.markdown("**Phần II — 10 câu Đúng/Sai**")
                    phan2_list = list(phan2_crops.items())
                    for i in range(0, len(phan2_list), 5):
                        row_crops = phan2_list[i:i+5]
                        cols = st.columns(5)
                        for j, (name, crop) in enumerate(row_crops):
                            with cols[j]:
                                st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                                         caption=name, width=120)

                # Hiển thị Phần III - 6 câu điền số
                if phan3_crops:
                    st.markdown("**Phần III — 6 câu điền số**")
                    phan3_list = list(phan3_crops.items())
                    cols = st.columns(6)
                    for j, (name, crop) in enumerate(phan3_list):
                        with cols[j]:
                            st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                                     caption=name, width=150)

                st.markdown("**4. Nhận dạng (Recognition)**")

                # Hiển thị kết quả nhận dạng
                res_col1, res_col2 = st.columns(2)
                res_col1.metric("Số báo danh (SBD)", result.get("sbd", "?"))
                res_col2.metric("Mã đề", result.get("made", "?"))

                # Phần I - Trắc nghiệm
                st.markdown("**Phần I — Trắc nghiệm A/B/C/D**")
                phan1_data = [{"Câu": k, "Đáp án": v if v else "(bỏ trống)"}
                              for k, v in sorted(result.get("phan1", {}).items())]
                if phan1_data:
                    st.dataframe(phan1_data, use_container_width=True, hide_index=True)

                # Phần II - Đúng/Sai
                st.markdown("**Phần II — Đúng/Sai**")
                phan2_data = [{"Câu": k, "Ý 1": ans[0] if len(ans) > 0 else "",
                               "Ý 2": ans[1] if len(ans) > 1 else "",
                               "Ý 3": ans[2] if len(ans) > 2 else "",
                               "Ý 4": ans[3] if len(ans) > 3 else ""}
                              for k, ans in sorted(result.get("phan2", {}).items())]
                if phan2_data:
                    st.dataframe(phan2_data, use_container_width=True, hide_index=True)

                # Phần III - Điền số
                st.markdown("**Phần III — Điền số**")
                phan3_data = [{"Câu": k, "Đáp số": v} for k, v in sorted(result.get("phan3", {}).items())]
                if phan3_data:
                    st.dataframe(phan3_data, use_container_width=True, hide_index=True)

                # ---- Chấm điểm nếu đã có đáp án chuẩn ----
                if st.session_state.answer_key:
                    graded = grade_submission(result, st.session_state.answer_key)
                    graded["ten_file"] = uploaded_file.name
                    st.write("DEBUG - Chi tiết Phần I:", graded["chi_tiet"]["phan1"])
                    st.write("DEBUG - Chi tiết Phần II:", graded["chi_tiet"]["phan2"], "(Raw:", graded.get("diem_phan2_raw", 0), "→ Quy đổi:", graded["diem_phan2"], ")")
                    st.write("DEBUG - Chi tiết Phần III:", graded["chi_tiet"]["phan3"])
                    st.subheader("Điểm số")
                    g1, g2, g3, g4 = st.columns(4)
                    g1.metric("Phần I", graded["diem_phan1"])
                    g2.metric("Phần II", graded["diem_phan2"])
                    g3.metric("Phần III", graded["diem_phan3"])
                    g4.metric("Tổng điểm", graded["tong_diem"])
                    insert_result(graded)
                    st.success("Đã lưu kết quả vào cơ sở dữ liệu (xem tại trang Kết quả).")
                else:
                    st.info("Chưa có đáp án chuẩn nên chỉ hiển thị kết quả nhận dạng, "
                            "chưa tính điểm.")

            except ValueError as e:
                status.error(f"Lỗi xử lý ảnh: {e}")
            except Exception as e:
                status.error(f"Lỗi không xác định: {e}")
    else:
        st.info("Vui lòng tải lên một ảnh phiếu trả lời để bắt đầu.")


# ---------------------------------------------------------------------------
# TRANG 3: KẾT QUẢ
# ---------------------------------------------------------------------------
elif page == "Kết quả":
    st.title("Bảng kết quả tổng hợp")

    history = get_all_results()
    if not history:
        st.info("Chưa có bài nào được chấm. Vào trang **Chấm bài** để bắt đầu.")
    else:
        df = history_to_dataframe(history)
        st.dataframe(df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            excel_bytes = export_excel_bytes(history)
            st.download_button(
                "Tải bảng điểm (Excel)",
                data=excel_bytes,
                file_name="ket_qua_omr.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with col2:
            if st.button("Xóa toàn bộ lịch sử"):
                delete_all()
                st.rerun()

        st.markdown("---")
        st.subheader("Xem chi tiết một bài")
        ten_files = [h["ten_file"] for h in history]
        chon = st.selectbox("Chọn bài", ten_files)
        item = next(h for h in history if h["ten_file"] == chon)
        st.json(item["chi_tiet"])