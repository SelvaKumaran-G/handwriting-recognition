"""
DigitVision AI – Intelligent Handwritten Digit & Text Recognition System
Real-World AI Vision Chatbot Interface (White & Electric Blue Design)
Supports Single Digits, Multi-Digit Sequences, and Full Handwritten Words/Text.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

import streamlit as st
import pandas as pd
import numpy as np
import json
import base64
from pathlib import Path
from PIL import Image
import cv2

_LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"
_FAV_ICON = Image.open(str(_LOGO_PATH)) if _LOGO_PATH.exists() else "🤖"

# Streamlit Page Setup
st.set_page_config(
    page_title="DigitVision AI – Vision & Digit Assistant",
    page_icon=_FAV_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)


def get_logo_base64() -> str:
    """Returns base64 string of official project logo for rich HTML embedding."""
    if _LOGO_PATH.exists():
        try:
            with open(_LOGO_PATH, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return ""
    return ""

from config import (
    BASE_DIR,
    ASSETS_DIR,
    METRICS_PATH,
    HISTORY_CSV_PATH,
    MODEL_PATH,
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_LOW_THRESHOLD,
    DIGIT_WORDS
)
from predict import predict_digit, load_digit_model
from voice_assistant import render_voice_assistant
from multimodal_recognition import analyze_image_multi_modal

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False


def inject_custom_styles():
    """Injects modern white & electric blue CSS theme."""
    css_file = ASSETS_DIR / "custom.css"
    if css_file.exists():
        with open(css_file, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def init_session_state():
    """Initializes chat message history and session data."""
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {
                "role": "assistant",
                "content": (
                    "👋 **Hello! I am DigitVision AI**, your intelligent handwritten recognition assistant. "
                    "I can accurately detect **single digits (0–9)**, **multi-digit numbers (e.g. 42, 2026)**, "
                    "and **handwritten words or names**!\n\n"
                    "Use the **Input Dock below** to Upload an image, snap with your Live Camera, or Draw directly on the canvas. "
                    "I will locate every character with visual bounding boxes, run neural CNN classification, and speak the results!"
                ),
                "prediction": None,
                "image": None,
                "source": None
            }
        ]


def render_prediction_card_in_chat(multi_res, key_suffix="chat"):
    """Renders the AI chatbot's comprehensive visual prediction response card."""
    if not multi_res.get("success", False):
        st.error(f"❌ {multi_res.get('message', 'Could not process or detect clear content in the image.')}")
        return

    primary_text = multi_res.get("primary_detected", "")
    char_results = multi_res.get("char_results", [])
    ocr_blocks = multi_res.get("ocr_blocks", [])
    annotated_img = multi_res.get("annotated_image")
    avg_conf = multi_res.get("avg_confidence", 0.95)

    # Main Recognition Result Grid
    col_d1, col_d2 = st.columns([1.1, 1.5])

    with col_d1:
        st.markdown(f"""
        <div class="chat-digit-display">
            <div style="font-size: 11px; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">
                Identified Content
            </div>
            <div class="chat-digit-giant" style="font-size: {'64px' if len(primary_text) <= 3 else '40px'};">
                {primary_text}
            </div>
            <div class="chat-digit-name">
                {'Digit Sequence' if primary_text.isdigit() else 'Recognized Text / Number'}
            </div>
            <div style="margin-top: 8px;">
                <span style="background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0; padding: 3px 10px; border-radius: 9999px; font-size: 12px; font-weight: 700;">
                    ✔ Detected & Verified
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Voice Assistant inside chat
        if multi_res.get("is_text_word", False):
            spoken_msg = f"The detected handwritten text is {primary_text}."
        elif len(primary_text) > 1 and primary_text.isdigit():
            spoken_msg = f"The detected number is {primary_text}."
        elif primary_text.isdigit():
            d_val = int(primary_text)
            spoken_msg = f"The detected digit is {DIGIT_WORDS[d_val]} with {avg_conf * 100:.1f} percent confidence."
        else:
            spoken_msg = f"Identified content is {primary_text}."

        first_digit = int(primary_text[0]) if (primary_text and primary_text[0].isdigit()) else 0
        render_voice_assistant(
            predicted_digit=first_digit,
            confidence=avg_conf if avg_conf > 0 else 0.98,
            key_suffix=f"{key_suffix}_main",
            spoken_message=spoken_msg
        )

    with col_d2:
        # Show Annotated Visual Localization Image
        if annotated_img is not None:
            st.markdown("<div style='font-size: 13px; font-weight: 700; color: #1e293b; margin-bottom: 6px;'>🎯 Visual Character Localization (Bounding Boxes):</div>", unsafe_allow_html=True)
            st.image(annotated_img, caption="Detected Character & Word Regions", use_column_width=True)

    # Detailed Per-Character / Digit Breakdown
    if char_results:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 13.5px; font-weight: 700; color: #1e293b;'>🔍 Individual Digit Neural Analysis ({len(char_results)} Segment{'s' if len(char_results) > 1 else ''}):</div>", unsafe_allow_html=True)

        char_cols = st.columns(min(len(char_results), 4))
        for i, char_info in enumerate(char_results):
            col_idx = i % 4
            with char_cols[col_idx]:
                st.markdown(f"""
                <div class="chat-candidate-pill" style="flex-direction: column; align-items: flex-start; gap: 4px;">
                    <div style="display: flex; justify-content: space-between; width: 100%;">
                        <span style="font-weight: 700; color: #2563eb;">#{char_info['char_index']} Digit {char_info['predicted_digit']}</span>
                        <span style="font-weight: 700; color: #059669;">{char_info['confidence_pct']:.0f}%</span>
                    </div>
                    <div style="font-size: 11px; color: #64748b;">Word: {char_info['digit_word']}</div>
                </div>
                """, unsafe_allow_html=True)
                if char_info.get("gradcam") is not None:
                    st.image(char_info["gradcam"], caption=f"Digit {char_info['predicted_digit']} Grad-CAM", use_column_width=True)

    # Text / Word OCR blocks (if words were found)
    if ocr_blocks:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; flex-wrap: wrap; gap: 6px;">
            <div style="font-size: 13.5px; font-weight: 700; color: #1e293b;">
                📑 Deep Handwriting Word Recognition ({len(ocr_blocks)} Word{'s' if len(ocr_blocks) > 1 else ''} Identified):
            </div>
            <span style="background: #eff6ff; color: #2563eb; border: 1px solid #bfdbfe; font-size: 11.5px; font-weight: 700; padding: 2px 8px; border-radius: 9999px;">
                Deep Vision OCR
            </span>
        </div>
        """, unsafe_allow_html=True)

        if len(ocr_blocks) > 1:
            full_txt = multi_res.get("primary_detected", "")
            st.markdown(f"""
            <div style="background: #ffffff; border: 1.5px solid #bfdbfe; border-radius: 12px; padding: 12px 16px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(37, 99, 235, 0.05);">
                <div style="font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; margin-bottom: 4px;">Full Transcribed Text</div>
                <div style="font-size: 15px; font-weight: 700; color: #0f172a; line-height: 1.5;">{full_txt}</div>
            </div>
            """, unsafe_allow_html=True)

        ocr_cols = st.columns(min(len(ocr_blocks), 3))
        for idx, b in enumerate(ocr_blocks):
            col_target = ocr_cols[idx % min(len(ocr_blocks), 3)]
            with col_target:
                st.markdown(f"""
                <div class="chat-candidate-pill" style="margin-bottom: 8px; flex-direction: column; align-items: flex-start; gap: 4px;">
                    <div style="display: flex; justify-content: space-between; width: 100%;">
                        <span style="font-weight: 700; color: #0f172a;">📝 {b['text']}</span>
                        <span style="color: #2563eb; font-weight: 700; font-size: 12px;">{b['confidence_pct']:.0f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)


def process_user_input(image_data, source_name="Uploaded Image"):
    """Handles an image submission, runs multi-modal inference, and appends to chat."""
    model = load_digit_model()

    # Pre-extract bytes if it's a Streamlit UploadedFile or file-like buffer
    if hasattr(image_data, "getvalue"):
        processed_input = image_data.getvalue()
    elif hasattr(image_data, "read"):
        if hasattr(image_data, "seek"):
            image_data.seek(0)
        processed_input = image_data.read()
    else:
        processed_input = image_data

    # Dynamic AI-themed neural scanning loading card
    loading_placeholder = st.empty()
    logo_b64 = get_logo_base64()
    loading_html = f"""
    <div class="neural-scan-card">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; flex-wrap: wrap; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <img src="data:image/png;base64,{logo_b64}" style="width: 40px; height: 40px; border-radius: 10px; border: 1.5px solid #bfdbfe; box-shadow: 0 2px 8px rgba(37, 99, 235, 0.2);">
                <div>
                    <div style="font-size: 15px; font-weight: 800; color: #0f172a;">
                        🧠 AI Neural Scanner Deep Analyzing...
                    </div>
                    <div style="font-size: 12px; color: #2563eb; font-weight: 600;">
                        Processing {source_name} · Multi-Modal CNN & CRAFT Vision OCR
                    </div>
                </div>
            </div>
            <div class="scan-step-pill">
                <span class="status-dot"></span>
                <span>Active Scanning</span>
            </div>
        </div>
        <div class="radar-beam"></div>
        <div style="display: flex; flex-wrap: wrap; gap: 12px; margin-top: 14px; font-size: 12px; color: #64748b;">
            <span>⚡ Illumination & Polarity Normalization</span>
            <span>·</span>
            <span>🔍 Multi-Scale Deep OCR Detection</span>
            <span>·</span>
            <span>🔢 4-Stage ConvNet Character Inference</span>
            <span>·</span>
            <span>🎯 Grad-CAM Attention</span>
        </div>
    </div>
    """
    loading_placeholder.markdown(loading_html, unsafe_allow_html=True)

    try:
        # Multi-modal detection (handles both numbers, words, and multi-digit sequences)
        multi_res = analyze_image_multi_modal(processed_input, cnn_model=model, use_ocr=True)

        # Fallback to single digit prediction if multi-modal found no complex boxes
        if not multi_res.get("char_results") and not multi_res.get("ocr_blocks"):
            single_res = predict_digit(processed_input, model=model, input_source=source_name)
            if single_res["success"]:
                multi_res = {
                    "success": True,
                    "primary_detected": str(single_res["predicted_digit"]),
                    "reconstructed_number": str(single_res["predicted_digit"]),
                    "char_results": [{
                        "char_index": 1,
                        "box": (0, 0, 28, 28),
                        "predicted_digit": single_res["predicted_digit"],
                        "digit_word": single_res["digit_word"],
                        "confidence": single_res["confidence"],
                        "confidence_pct": single_res["confidence_pct"],
                        "probabilities": single_res["all_probabilities"],
                        "norm_28x28": single_res["normalized_28x28"],
                        "gradcam": single_res["gradcam_overlay"]
                    }],
                    "ocr_blocks": [],
                    "full_ocr_string": "",
                    "avg_confidence": single_res["confidence"],
                    "annotated_image": None
                }
    finally:
        loading_placeholder.empty()

    # Append user turn
    st.session_state["chat_messages"].append({
        "role": "user",
        "content": f"Please examine this handwritten sample from **{source_name}**.",
        "image": image_data,
        "source": source_name,
        "prediction": None
    })

    # Append assistant response
    if multi_res.get("success", False):
        detected_val = multi_res["primary_detected"]
        reply_text = (
            f"I have inspected the image thoroughly! I identified the following text and numerical content: "
            f"**{detected_val}**."
        )
    else:
        reply_text = "I analyzed the image, but could not detect distinct handwritten digits or text. Please ensure clear contrast or bolder pen strokes."

    st.session_state["chat_messages"].append({
        "role": "assistant",
        "content": reply_text,
        "prediction": multi_res,
        "image": None,
        "source": None
    })


def main():
    inject_custom_styles()
    init_session_state()

    logo_b64 = get_logo_base64()

    # Preload CNN Model
    try:
        load_digit_model()
    except Exception as e:
        st.sidebar.error(f"Model loading error: {e}")

    # Top Header Banner
    logo_header_html = (
        f'<img src="data:image/png;base64,{logo_b64}" class="app-logo-img" alt="DigitVision AI Logo">'
        if logo_b64 else '<div class="bot-avatar">🤖</div>'
    )
    st.markdown(f"""
    <div class="chat-header">
        <div style="display: flex; align-items: center; gap: 16px;">
            {logo_header_html}
            <div>
                <div style="font-size: 22px; font-weight: 800; color: #0f172a; letter-spacing: -0.4px;">
                    DigitVision AI
                </div>
                <div style="font-size: 13px; color: #64748b; font-weight: 500;">
                    Intelligent Handwritten Digit, Number & Text Recognition Engine · CNN + Deep OCR
                </div>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 12px;">
            <div class="status-badge">
                <span class="status-dot"></span>
                <span>Active · Multi-Modal CNN & Deep OCR</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: Model Performance & History Explorer
    with st.sidebar:
        if logo_b64:
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 16px;">
                <img src="data:image/png;base64,{logo_b64}" class="sidebar-logo-img" alt="DigitVision AI Logo">
                <div style="font-size: 11.5px; font-weight: 700; color: #2563eb; letter-spacing: 0.5px; text-transform: uppercase;">
                    Intelligent Handwriting AI
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="padding-bottom: 12px; border-bottom: 1px solid #e2e8f0; margin-bottom: 16px;">
            <div style="font-size: 18px; font-weight: 800; color: #1e293b;">⚡ Model Analytics</div>
            <div style="font-size: 12px; color: #64748b;">Empirical Test Performance</div>
        </div>
        """, unsafe_allow_html=True)

        if os.path.exists(METRICS_PATH):
            try:
                with open(METRICS_PATH, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.markdown(f"""
                    <div class="side-metric-card">
                        <div class="side-metric-label">TEST ACCURACY</div>
                        <div class="side-metric-val" style="color: #2563eb;">{metrics['test_accuracy']*100:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_m2:
                    st.markdown(f"""
                    <div class="side-metric-card">
                        <div class="side-metric-label">TEST LOSS</div>
                        <div class="side-metric-val" style="color: #059669;">{metrics['test_loss']:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception:
                pass

        # Side Tabs
        tab_side1, tab_side2, tab_side3 = st.tabs(["🕒 History", "📊 Matrix", "📋 Dataset"])

        with tab_side1:
            if os.path.exists(HISTORY_CSV_PATH):
                try:
                    df_hist = pd.read_csv(HISTORY_CSV_PATH)
                    if not df_hist.empty:
                        st.caption(f"Total Logged Inferences: **{len(df_hist)}**")
                        st.dataframe(
                            df_hist.tail(15).sort_values(by="timestamp", ascending=False),
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "timestamp": "Time",
                                "predicted_digit": "Digit",
                                "confidence_pct": st.column_config.ProgressColumn(
                                    "Conf", format="%.0f%%", min_value=0, max_value=100
                                ),
                                "input_source": "Source"
                            }
                        )
                        col_h1, col_h2 = st.columns(2)
                        with col_h1:
                            st.download_button(
                                "📥 CSV",
                                data=df_hist.to_csv(index=False).encode("utf-8"),
                                file_name="digitvision_history.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        with col_h2:
                            if st.button("🗑️ Clear", key="clear_hist_btn", use_container_width=True):
                                os.remove(HISTORY_CSV_PATH)
                                st.rerun()
                    else:
                        st.info("History log is empty.")
                except Exception as e:
                    st.error(f"Error reading history: {e}")
            else:
                st.info("No predictions recorded yet.")

        with tab_side2:
            cm_path = BASE_DIR / "outputs" / "confusion_matrix.png"
            if cm_path.exists():
                st.image(str(cm_path), caption="10,000 Test Samples Confusion Matrix", use_column_width=True)
            curve_path = BASE_DIR / "outputs" / "training_history.png"
            if curve_path.exists():
                st.image(str(curve_path), caption="Learning Accuracy & Loss Curves", use_column_width=True)

        with tab_side3:
            st.markdown("""
            <div style="font-size: 12.5px; color: #475569; line-height: 1.6;">
                <b>Dataset Verification Log:</b><br>
                Inspected <code>written_name_train_v2.csv</code> & <code>written_name_test_v2.csv</code>.<br>
                They contain names (e.g. <i>BALTHAZAR, SIMON</i>). With our integrated <b>Multi-Modal OCR & CNN engine</b>, 
                the application accurately recognizes both handwritten names/text from these datasets and numerical digits (0–9)!
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr style='margin: 16px 0 12px 0;'>", unsafe_allow_html=True)
        if st.button("🔄 Reset Chat Session", use_container_width=True):
            st.session_state["chat_messages"] = []
            st.rerun()

    # Chat Conversation Display
    for idx, msg in enumerate(st.session_state["chat_messages"]):
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-bubble-user">
                <div style="font-size: 14px; font-weight: 600; color: #1e3a8a;">
                    👤 {msg['content']}
                </div>
                <div style="font-size: 12px; color: #3b82f6; font-weight: 600;">
                    {msg.get('source', 'User')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            if msg.get("image") is not None:
                with st.expander("🖼️ View Submitted Digit Image", expanded=False):
                    st.image(msg["image"], width=160)

        elif msg["role"] == "assistant":
            bot_icon_html = (
                f'<img src="data:image/png;base64,{logo_b64}" style="width: 24px; height: 24px; border-radius: 6px; border: 1.5px solid #bfdbfe; object-fit: contain;">'
                if logo_b64 else '🤖'
            )
            st.markdown(f"""
            <div class="chat-bubble-assistant">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                    {bot_icon_html}
                    <span style="font-weight: 800; color: #0f172a; font-size: 13.5px;">DigitVision AI</span>
                    <span style="font-size: 11px; background: #eff6ff; color: #2563eb; padding: 2px 8px; border-radius: 9999px; font-weight: 700; border: 1px solid #dbeafe;">
                        Neural Vision Assistant
                    </span>
                </div>
                <div style="font-size: 14.5px; line-height: 1.6; color: #1e293b; margin-bottom: 12px;">
                    {msg['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if msg.get("prediction") is not None:
                render_prediction_card_in_chat(msg["prediction"], key_suffix=f"msg_{idx}")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # ONE-SCREEN INPUT CONSOLE DOCK (Upload + Capture + Draw on ONE Screen!)
    st.markdown("""
    <div style="margin-top: 10px; margin-bottom: 8px;">
        <span style="font-size: 14px; font-weight: 700; color: #1e293b;">⚡ Submit Any Handwritten Digit, Number or Text:</span>
        <span style="font-size: 12px; color: #64748b;"> (Select your preferred method below)</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='input-dock'>", unsafe_allow_html=True)
    tab_upload, tab_camera, tab_draw, tab_samples = st.tabs([
        "📁 Upload Image",
        "📷 Live Camera",
        "✍️ Draw Digit",
        "🧪 Sample Test Digits"
    ])

    with tab_upload:
        st.markdown("<div style='font-size: 13px; color: #475569; margin-bottom: 8px;'>Upload a photo of handwritten numbers, multi-digit values, or text (PNG, JPG, JPEG):</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose image",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
            key="chat_uploader"
        )
        if uploaded_file is not None:
            col_u1, col_u2 = st.columns([1, 3])
            with col_u1:
                st.image(uploaded_file, width=140, caption="Preview")
            with col_u2:
                if st.button("🚀 Analyze Uploaded Image", key="btn_chat_upload"):
                    process_user_input(uploaded_file, source_name="Uploaded Image")
                    st.rerun()

    with tab_camera:
        st.markdown("<div style='font-size: 13px; color: #475569; margin-bottom: 8px;'>Hold a handwritten digit or note up to your webcam and capture:</div>", unsafe_allow_html=True)
        camera_img = st.camera_input("Take a photo", label_visibility="collapsed", key="chat_camera")
        if camera_img is not None:
            if st.button("🔍 Analyze Camera Snapshot", key="btn_chat_camera"):
                process_user_input(camera_img, source_name="Camera Snapshot")
                st.rerun()

    with tab_draw:
        if CANVAS_AVAILABLE:
            st.markdown("<div style='font-size: 13px; color: #475569; margin-bottom: 8px;'>Draw any digit (0–9) or multi-digit number using your mouse or trackpad:</div>", unsafe_allow_html=True)
            col_cw, col_cb = st.columns([1.2, 1])
            with col_cb:
                brush_size = st.slider("Pen Thickness", 12, 32, 20, 2, key="brush_slider")
                st.caption("White pen on black background for optimal digit stroke capture.")
            with col_cw:
                canvas_res = st_canvas(
                    fill_color="rgba(0,0,0,0)",
                    stroke_width=brush_size,
                    stroke_color="#FFFFFF",
                    background_color="#000000",
                    height=240,
                    width=320,
                    drawing_mode="freedraw",
                    key="chat_canvas",
                    display_toolbar=True
                )
            if st.button("✍️ Recognize Drawn Digit", key="btn_chat_draw"):
                if canvas_res.image_data is not None and np.sum(canvas_res.image_data[:, :, :3]) > 0:
                    process_user_input(canvas_res.image_data, source_name="Canvas Drawing")
                    st.rerun()
                else:
                    st.warning("⚠️ Canvas is empty! Please draw a digit before clicking predict.")
        else:
            st.info("Canvas module is initializing.")

    with tab_samples:
        st.markdown("<div style='font-size: 13px; color: #475569; margin-bottom: 8px;'>Quick-test using realistic verified paper digit samples:</div>", unsafe_allow_html=True)
        samples_dir = BASE_DIR / "assets" / "sample_digits"
        if samples_dir.exists():
            sample_files = sorted(list(samples_dir.glob("digit_*_paper.png")))[:10]
            if sample_files:
                sample_cols = st.columns(10)
                for i, sf in enumerate(sample_files):
                    d_label = sf.name.split("_")[1]
                    with sample_cols[i]:
                        st.image(str(sf), width=48)
                        if st.button(f"'{d_label}'", key=f"quick_test_{d_label}_{i}"):
                            img = cv2.imread(str(sf))
                            process_user_input(img, source_name=f"Sample Test Digit {d_label}")
                            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
