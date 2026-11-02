"""
DigitVision AI - Interactive Drawing Canvas Component
Enables real-time freehand digit drawing using mouse or stylus,
brush width configuration, MNIST normalization, Grad-CAM, and voice readout.
"""

import streamlit as st
import numpy as np

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

from predict import predict_digit
from components.upload import render_prediction_results


def render_drawing_page():
    """Renders the interactive drawing canvas recognition interface."""
    st.markdown("""
    <div class="glass-card">
        <h2 style="color: #f8fafc; font-weight: 800; margin-bottom: 6px;">✍️ Draw Your Own Digit</h2>
        <p style="color: #94a3b8; font-size: 14.5px; margin-bottom: 0;">
            Draw any digit from 0 to 9 on the interactive black canvas below using your mouse, trackpad, or touchscreen. 
            Adjust brush thickness if needed, and hit <b>Predict Handwritten Digit</b> to trigger the CNN inference.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not CANVAS_AVAILABLE:
        st.error("The 'streamlit-drawable-canvas' package is still completing installation. Please refresh in a moment.")
        return

    col_canvas, col_ctrl = st.columns([1.2, 1])

    with col_ctrl:
        st.markdown("""
        <div class="glass-card" style="background: rgba(17, 24, 39, 0.5);">
            <h4 style="color: #a5b4fc; margin-top: 0;">🎨 Canvas Controls</h4>
        """, unsafe_allow_html=True)

        stroke_width = st.slider(
            "Stroke Thickness (px)",
            min_value=10,
            max_value=35,
            value=18,
            step=1,
            help="Simulates pen or marker thickness. 16-22 px recommended."
        )

        st.caption("White ink on black background (matches native MNIST polarity).")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_canvas:
        st.markdown("<div style='display: flex; justify-content: center; margin-bottom: 12px;'>", unsafe_allow_html=True)
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="handwriting_canvas",
            display_toolbar=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Prediction Action
    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        predict_draw_btn = st.button("🚀 Recognize Drawn Digit", key="btn_predict_drawing", use_container_width=True)

    if predict_draw_btn or "last_canvas_res" in st.session_state:
        if predict_draw_btn:
            if canvas_result.image_data is None or np.sum(canvas_result.image_data[:, :, :3]) == 0:
                st.warning("⚠️ The canvas is empty! Please draw a digit (0–9) before clicking predict.")
                return

            with st.spinner("Analyzing drawn digit strokes with CNN..."):
                res = predict_digit(canvas_result.image_data, input_source="Canvas")
                st.session_state["last_canvas_res"] = res
        else:
            res = st.session_state["last_canvas_res"]

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        render_prediction_results(res, key_prefix="canvas")

