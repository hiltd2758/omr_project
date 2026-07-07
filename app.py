"""
Giao diện Web Demo hệ thống OMR (Streamlit)
- CHỈ ghép nối (orchestration) các hàm đã có sẵn trong modules/, KHÔNG chỉnh sửa
  thuật toán / logic xử lý ảnh gốc.
- Luồng xử lý: preprocessing -> detection (marker) -> perspective (warp)
  -> segmentation (tách vùng) -> recognition (nhận dạng SBD/Mã đề/Đáp án)
"""

import os
import sys
import cv2
import numpy as np
import streamlit as st

MODULES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules")
sys.path.insert(0, MODULES_DIR)

from preprocessing import preprocess_pipeline
from detection import find_corner_markers, draw_debug_markers
from perspective import warp_perspective
from segmentation import segment_all
from recognition import recognize_full_sheet

st.set_page_config(page_title="Demo hệ thống OMR", layout="wide")
st.title("Hệ thống chấm bài trắc nghiệm tự động (OMR)")
st.caption("Giao diện demo — chỉ trực quan hóa pipeline xử lý ảnh, không thay đổi thuật toán backend.")

uploaded_file = st.file_uploader("Tải lên ảnh phiếu trả lời trắc nghiệm", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Không thể đọc ảnh. Vui lòng thử ảnh khác.")
        st.stop()

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", width=350)

    if st.button("Thực hiện chấm bài"):
        status = st.empty()
        try:
            status.info("Đang tiền xử lý ảnh...")
            pre = preprocess_pipeline(image)

            status.info("Đang phát hiện marker 4 góc...")
            corners, candidates = find_corner_markers(pre["binary"])
            marker_debug = draw_debug_markers(image, corners, candidates)

            status.info("Đang chỉnh phối cảnh (perspective transform)...")
            warped, _ = warp_perspective(image, corners)

            status.info("Đang phân đoạn các vùng trên phiếu...")
            segments = segment_all(warped)

            status.info("Đang nhận dạng đáp án...")
            result = recognize_full_sheet(segments)

            status.success("Chấm bài thành công!")

            st.subheader("Quy trình xử lý")
            c1, c2 = st.columns(2)
            with c1:
                st.image(cv2.cvtColor(marker_debug, cv2.COLOR_BGR2RGB), caption="Phát hiện marker 4 góc")
            with c2:
                st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), caption="Ảnh sau khi chỉnh phối cảnh")

            st.subheader("Kết quả nhận dạng")
            r1, r2 = st.columns(2)
            r1.metric("Số báo danh (SBD)", result.get("sbd", "?"))
            r2.metric("Mã đề", result.get("made", "?"))

            st.markdown("**Phần I — Trắc nghiệm A/B/C/D**")
            st.dataframe(
                [{"Câu": k, "Đáp án": v if v else "(bỏ trống)"} for k, v in result.get("phan1", {}).items()],
                use_container_width=True, hide_index=True,
            )

            st.markdown("**Phần II — Đúng/Sai**")
            for cau, ans in result.get("phan2", {}).items():
                st.write(f"Câu {cau}: " + ", ".join(a if a else "(bỏ trống)" for a in ans))

            st.markdown("**Phần III — Điền số**")
            st.dataframe(
                [{"Câu": k, "Đáp số": v} for k, v in result.get("phan3", {}).items()],
                use_container_width=True, hide_index=True,
            )

            with st.expander("Xem chi tiết các vùng đã tách (debug)"):
                cols = st.columns(4)
                for i, (name, crop) in enumerate(segments.items()):
                    with cols[i % 4]:
                        st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption=name)

        except ValueError as e:
            status.error(f"Lỗi xử lý ảnh: {e}")
        except Exception as e:
            status.error(f"Lỗi không xác định: {e}")
else:
    st.info("Vui lòng tải lên một ảnh phiếu trả lời để bắt đầu.")