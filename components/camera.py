"""
DigitVision AI - Camera Capture Component
Enables real-time camera snapshot capture, quality validation,
noise suppression, CNN inference, explainability, and speech feedback.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image

from predict import predict_digit
from components.upload import render_prediction_results


def render_camera_page():
    """Renders the live camera capture and recognition module."""
    st.markdown("""
    <div class="glass-card">
        <h2 style="color: #f8fafc; font-weight: 800; margin-bottom: 6px;">📷 Live Camera Digit Recognition</h2>
        <p style="color: #94a3b8; font-size: 14.5px; margin-bottom: 0;">
            Hold up a sheet of paper with a handwritten digit (0–9) clearly drawn in dark pen or marker. 
            Allow camera permissions when prompted, snap a frame, and let the CNN model detect the digit in real-time.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_cam, col_guide = st.columns([1.4, 1])

    with col_guide:
        st.markdown("""
        <div class="glass-card" style="background: rgba(17, 24, 39, 0.5);">
            <h4 style="color: #a5b4fc; margin-top: 0;">💡 Tips for Highest Accuracy</h4>
            <ul style="color: #cbd5e1; font-size: 13.5px; line-height: 1.7; padding-left: 18px;">
                <li>Draw the digit large enough on plain white paper.</li>
                <li>Ensure uniform lighting without strong shadows.</li>
                <li>Hold the digit steady in the center of the frame.</li>
                <li>Avoid having multiple digits in the same frame.</li>
                <li>Use a bold pen or marker for best stroke contrast.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_cam:
        camera_img = st.camera_input("Point camera at handwritten digit and capture", key="live_camera_feed")

    if camera_img is not None:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            detect_clicked = st.button("🔍 Detect Digit From Snapshot", key="btn_detect_camera", use_container_width=True)

        if detect_clicked or "last_camera_res" in st.session_state:
            if detect_clicked:
                with st.spinner("Processing camera snapshot & running CNN inference..."):
                    res = predict_digit(camera_img, input_source="Camera")
                    st.session_state["last_camera_res"] = res
            else:
                res = st.session_state["last_camera_res"]

            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
            render_prediction_results(res, key_prefix="cam")

