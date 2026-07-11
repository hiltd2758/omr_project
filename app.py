"""
Giao diện Web Demo hệ thống OMR (Streamlit)
- CHỈ ghép nối (orchestration) các hàm đã có sẵn trong modules/, KHÔNG chỉnh sửa
  thuật toán / logic xử lý ảnh gốc.
- 3 trang: Dashboard, Chấm bài, Kết quả.
- Luồng xử lý: preprocessing -> detection (marker) -> perspective (warp)
  -> segmentation (tách vùng) -> recognition (nhận dạng SBD/Mã đề/Đáp án)
  -> XÁC NHẬN / CHỈNH SỬA đáp án (nếu hệ thống nhận diện sai)
  -> grading (chấm điểm theo đáp án chuẩn, tùy chọn) -> export (xuất Excel)
- Lịch sử kết quả được lưu bền vững vào SQLite (modules/database.py) thay vì
  session_state, nên không bị mất khi reload trang hoặc restart app.
"""

import os
import sys
import json
import copy

import cv2
import numpy as np
import pandas as pd
import streamlit as st

MODULES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules")
sys.path.insert(0, MODULES_DIR)

from modules.preprocessing import preprocess_pipeline
from modules.detection import find_corner_markers, draw_debug_markers
from modules.perspective import warp_perspective
from modules.segmentation import segment_all
from modules.recognition import recognize_full_sheet
from modules.grading import grade_submission
from modules.export import export_excel_bytes, history_to_dataframe, parse_answer_key_file
from modules.database import init_db, insert_result, get_all_results, delete_all
from modules.preprocessing import preprocess_pipeline, binarize
st.set_page_config(page_title="Hệ thống OMR", layout="wide", page_icon="📝")

init_db()  # Đảm bảo bảng SQLite đã tồn tại trước khi dùng

if "answer_key" not in st.session_state:
    st.session_state.answer_key = None      # đáp án chuẩn (tùy chọn, vẫn giữ trong session)
if "pending_out" not in st.session_state:
    st.session_state.pending_out = None       # kết quả xử lý ảnh (out) đang chờ xác nhận
if "pending_result" not in st.session_state:
    st.session_state.pending_result = None    # bản kết quả nhận dạng CÓ THỂ CHỈNH SỬA
if "pending_file_name" not in st.session_state:
    st.session_state.pending_file_name = None
if "confirmed" not in st.session_state:
    st.session_state.confirmed = False


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


def reset_pending():
    st.session_state.pending_out = None
    st.session_state.pending_result = None
    st.session_state.pending_file_name = None
    st.session_state.confirmed = False


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

        # Nếu người dùng đổi file mới thì reset trạng thái xác nhận cũ
        if st.session_state.pending_file_name != uploaded_file.name:
            reset_pending()

        file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Không thể đọc ảnh. Vui lòng thử ảnh khác.")
            st.stop()

        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", width=350)

        # --------------------------------------------------------------
        # BƯỚC 1: CHẠY PIPELINE NHẬN DẠNG (chỉ chạy khi chưa có kết quả
        # chờ xác nhận cho đúng file này)
        # --------------------------------------------------------------
        if st.session_state.pending_out is None:
            if st.button("Thực hiện nhận dạng", type="primary"):
                status = st.empty()
                try:
                    status.info("Đang tiền xử lý ảnh...")
                    status.info("Đang phát hiện marker 4 góc và chỉnh phối cảnh...")
                    out = process_image(image)
                    status.info("Đang phân đoạn và nhận dạng đáp án...")
                    status.success("Nhận dạng xong! Vui lòng kiểm tra và xác nhận đáp án bên dưới.")

                    st.session_state.pending_out = out
                    # Bản sao có thể chỉnh sửa, tách biệt với kết quả gốc do máy nhận dạng
                    st.session_state.pending_result = copy.deepcopy(out["result"])
                    st.session_state.pending_file_name = uploaded_file.name
                    st.session_state.confirmed = False
                    st.rerun()

                except ValueError as e:
                    status.error(f"Lỗi xử lý ảnh: {e}")
                except Exception as e:
                    status.error(f"Lỗi không xác định: {e}")
            else:
                st.info("Nhấn **Thực hiện nhận dạng** để bắt đầu.")

        # --------------------------------------------------------------
        # BƯỚC 2: HIỂN THỊ CÁC BƯỚC XỬ LÝ + FORM XÁC NHẬN / CHỈNH SỬA
        # --------------------------------------------------------------
        if st.session_state.pending_out is not None:
            out = st.session_state.pending_out
            edit_result = st.session_state.pending_result

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

            st.markdown("**So sánh nhị phân hóa: Adaptive Threshold vs Otsu**")
            binary_adaptive = binarize(out["equalized"], method="adaptive")
            binary_otsu = binarize(out["equalized"], method="otsu")

            cmp1, cmp2 = st.columns(2)
            with cmp1:
                st.image(binary_adaptive, caption="Adaptive Threshold (đang dùng trong pipeline)")
                ok1, buf1 = cv2.imencode(".png", binary_adaptive)
                st.download_button(
                    "Tải ảnh Adaptive Threshold",
                    data=buf1.tobytes(),
                    file_name=f"{uploaded_file.name}_adaptive.png",
                    mime="image/png",
                    key="dl_adaptive",
                )
            with cmp2:
                st.image(binary_otsu, caption="Otsu (tham khảo)")
                ok2, buf2 = cv2.imencode(".png", binary_otsu)
                st.download_button(
                    "Tải ảnh Otsu",
                    data=buf2.tobytes(),
                    file_name=f"{uploaded_file.name}_otsu.png",
                    mime="image/png",
                    key="dl_otsu",
                )

            st.caption(
                f"Số pixel trắng (vùng tô/nét) — Adaptive: {int(np.sum(binary_adaptive == 255))} | "
                f"Otsu: {int(np.sum(binary_otsu == 255))}"
            )
            st.markdown("**2. Phát hiện & Chỉnh phối cảnh**")
            c1, c2 = st.columns(2)
            with c1:
                st.image(cv2.cvtColor(out["marker_debug"], cv2.COLOR_BGR2RGB),
                         caption="2.1 Phát hiện marker 4 góc")
            with c2:
                st.image(cv2.cvtColor(out["warped"], cv2.COLOR_BGR2RGB),
                         caption="2.2 Ảnh sau chỉnh phối cảnh (Warp)")

            cnts_binary, _ = cv2.findContours(binary_adaptive, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            st.caption(f"Số contour tìm được trên ảnh nhị phân: {len(cnts_binary)}")
            st.markdown("**3. Phân đoạn vùng (Segmentation)**")

            sbd_crop = out["segments"].get("sbd")
            made_crop = out["segments"].get("made")
            phan1_crops = {k: v for k, v in out["segments"].items() if k.startswith("phan1")}
            phan2_crops = {k: v for k, v in out["segments"].items() if k.startswith("phan2")}
            phan3_crops = {k: v for k, v in out["segments"].items() if k.startswith("phan3")}

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

            if phan3_crops:
                st.markdown("**Phần III — 6 câu điền số**")
                phan3_list = list(phan3_crops.items())
                cols = st.columns(6)
                for j, (name, crop) in enumerate(phan3_list):
                    with cols[j]:
                        st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                                 caption=name, width=150)

            # ------------------------------------------------------------
            # BƯỚC XÁC NHẬN / CHỈNH SỬA ĐÁP ÁN NHẬN DẠNG
            # ------------------------------------------------------------
            st.markdown("---")
            st.markdown("**4. Nhận dạng — Xác nhận hoặc chỉnh sửa nếu chưa chính xác**")
            st.caption("Hệ thống nhận dạng bằng thuật toán xử lý ảnh nên có thể sai lệch. "
                       "Vui lòng kiểm tra và sửa lại các ô bên dưới trước khi xác nhận.")

            res_col1, res_col2 = st.columns(2)
            edit_result["sbd"] = res_col1.text_input(
                "Số báo danh (SBD)", value=str(edit_result.get("sbd", "")), key="edit_sbd")
            edit_result["made"] = res_col2.text_input(
                "Mã đề", value=str(edit_result.get("made", "")), key="edit_made")

            st.markdown("**Phần I — Trắc nghiệm A/B/C/D**")
            phan1_items = sorted(edit_result.get("phan1", {}).items())
            for i in range(0, len(phan1_items), 5):
                cols = st.columns(5)
                for j, (cau, dap) in enumerate(phan1_items[i:i+5]):
                    with cols[j]:
                        raw_p1 = st.text_input(
                            f"Câu {cau}", value=(dap or ""), key=f"edit_p1_{cau}",
                            max_chars=1)
                        edit_result["phan1"][cau] = (raw_p1 or "").strip().upper()

            st.markdown("**Phần II — Đúng/Sai (Đ/S x4 ý)**")
            for cau, ans in sorted(edit_result.get("phan2", {}).items()):
                cols = st.columns(5)
                cols[0].markdown(f"Câu {cau}")
                new_ans = []
                for i in range(4):
                    val = ans[i] if i < len(ans) else ""
                    raw_p2 = cols[i + 1].text_input(
                        f"Ý {i+1} (câu {cau})", value=val, key=f"edit_p2_{cau}_{i}",
                        max_chars=1, label_visibility="collapsed")
                    new_ans.append((raw_p2 or "").strip().upper())
                edit_result["phan2"][cau] = new_ans

            st.markdown("**Phần III — Điền số**")
            phan3_items = sorted(edit_result.get("phan3", {}).items())
            cols = st.columns(6)
            for j, (cau, dap) in enumerate(phan3_items):
                with cols[j % 6]:
                    edit_result["phan3"][cau] = st.text_input(
                        f"Câu {cau}", value=str(dap or ""), key=f"edit_p3_{cau}")

            st.session_state.pending_result = edit_result

            action_col1, action_col2 = st.columns(2)
            confirm_clicked = action_col1.button(
                "Xác nhận đáp án và chấm điểm", type="primary")
            redo_clicked = action_col2.button("Làm lại (nhận dạng lại từ đầu)")

            if redo_clicked:
                reset_pending()
                st.rerun()

            if confirm_clicked:
                final_result = st.session_state.pending_result

                if st.session_state.answer_key:
                    graded = grade_submission(final_result, st.session_state.answer_key)
                    graded["ten_file"] = uploaded_file.name
                    st.subheader("Điểm số")
                    g1, g2, g3, g4 = st.columns(4)
                    g1.metric("Phần I", graded["diem_phan1"])
                    g2.metric("Phần II", graded["diem_phan2"])
                    g3.metric("Phần III", graded["diem_phan3"])
                    g4.metric("Tổng điểm", graded["tong_diem"])
                    insert_result(graded)
                    st.success("Đã xác nhận đáp án và lưu kết quả vào cơ sở dữ liệu "
                               "(xem tại trang Kết quả).")
                else:
                    st.info("Chưa có đáp án chuẩn nên chỉ lưu kết quả nhận dạng đã xác nhận, "
                            "chưa tính điểm. Thiết lập đáp án chuẩn ở phần trên để chấm điểm.")

                reset_pending()

    else:
        reset_pending()
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