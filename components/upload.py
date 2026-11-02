"""
DigitVision AI - Upload Recognition Component
Handles single and batch handwritten digit image uploads, preprocessing previews,
CNN inference, confidence analysis, probability distributions, Grad-CAM, and voice readout.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

from config import CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_LOW_THRESHOLD, DIGIT_LABELS
from predict import predict_digit
from voice_assistant import render_voice_assistant


def plot_probability_distribution(probabilities, predicted_idx):
    """Generates an attractive bar chart for the 10-class probability distribution."""
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=200)
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    colors = [
        "#10b981" if i == predicted_idx else "#6366f1" if prob > 0.1 else "#374151"
        for i, prob in enumerate(probabilities)
    ]

    bars = ax.bar(range(10), [p * 100 for p in probabilities], color=colors, width=0.65, edgecolor="none", zorder=3)
    ax.set_xticks(range(10))
    ax.set_xticklabels(DIGIT_LABELS, color="#f8fafc", fontsize=11, fontweight="bold")
    ax.set_ylabel("Probability (%)", color="#94a3b8", fontsize=10)
    ax.set_ylim(0, 105)
    ax.tick_params(axis="y", colors="#94a3b8")
    ax.grid(axis="y", linestyle="--", alpha=0.2, zorder=0)

    # Highlight top bar with text
    for bar in bars:
        h = bar.get_height()
        if h >= 5.0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 2.0,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                color="#ffffff",
                fontsize=8.5,
                fontweight="bold"
            )

    for spine in ax.spines.values():
        spine.set_color("#1f2937")

    plt.tight_layout()
    return fig


def render_prediction_results(res, key_prefix="upload"):
    """Renders the comprehensive prediction result card, top-3, distribution, and Grad-CAM."""
    if not res["success"]:
        st.error(f"❌ {res['message']}")
        return

    pred_digit = res["predicted_digit"]
    confidence = res["confidence"]
    confidence_pct = res["confidence_pct"]
    is_uncertain = res["is_uncertain"]

    # Main Result Grid
    col_res1, col_res2 = st.columns([1.2, 1.8])

    with col_res1:
        st.markdown(f"""
        <div class="digit-result-box">
            <div style="font-size: 13px; color: #a5b4fc; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">
                Detected Digit
            </div>
            <div class="digit-giant">{pred_digit}</div>
            <div class="digit-name">{res['digit_word']}</div>
            <div style="margin-top: 14px;">
                {'<span class="badge-high">✔ HIGH CONFIDENCE</span>' if not is_uncertain and confidence >= 0.85 else ('<span class="badge-warning">⚠ MODERATE CONFIDENCE</span>' if not is_uncertain else '<span class="badge-danger">✖ UNCERTAIN PREDICTION</span>')}
            </div>
            <div style="font-size: 15px; color: #ffffff; font-weight: 700; margin-top: 10px;">
                Confidence: {confidence_pct:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Voice Assistant Integration
        render_voice_assistant(
            predicted_digit=pred_digit,
            confidence=confidence,
            is_uncertain=is_uncertain,
            key_suffix=f"{key_prefix}_{pred_digit}"
        )

    with col_res2:
        if is_uncertain:
            st.warning("⚠️ **Prediction is uncertain.** Please upload a clearer image or capture again with better lighting.")

        st.markdown("<h4 style='color: #f8fafc; margin-bottom: 12px;'>Top Candidates</h4>", unsafe_allow_html=True)
        for item in res["top3"]:
            rank = item["rank"]
            d = item["digit"]
            pct = item["percentage"]
            word = item["word"]
            pill_color = "#10b981" if rank == 1 else ("#6366f1" if rank == 2 else "#94a3b8")
            st.markdown(f"""
            <div class="candidate-pill">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="background: {pill_color}; color: #ffffff; border-radius: 50%; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700;">
                        #{rank}
                    </span>
                    <span style="font-size: 15px; font-weight: 700; color: #ffffff;">Digit {d}</span>
                    <span style="font-size: 13px; color: #94a3b8;">({word})</span>
                </div>
                <div style="font-size: 15px; font-weight: 700; color: {pill_color};">
                    {pct:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Probability distribution chart
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        fig_prob = plot_probability_distribution(res["all_probabilities"], pred_digit)
        st.pyplot(fig_prob)
        plt.close(fig_prob)

    # Preprocessing Stages & Explainability Section
    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 24px 0 16px 0;'>", unsafe_allow_html=True)
    exp_col1, exp_col2 = st.columns([1.5, 1])

    with exp_col1:
        with st.expander("🔬 View Multi-Stage Preprocessing Pipeline", expanded=False):
            stages = res.get("stages", {})
            stage_keys = [k for k in sorted(stages.keys())]
            if stage_keys:
                cols = st.columns(len(stage_keys))
                stage_labels = {
                    "1_original": "1. Original",
                    "2_grayscale": "2. Grayscale",
                    "3_noise_filtered": "3. Filtered",
                    "4_polarity_corrected": "4. Polarity Match",
                    "5_thresholded": "5. Threshold",
                    "6_centered_28x28": "6. MNIST 28x28"
                }
                for i, key in enumerate(stage_keys):
                    with cols[i]:
                        st.caption(stage_labels.get(key, key))
                        img = stages[key]
                        st.image(img, use_column_width=True)

    with exp_col2:
        with st.expander("🧠 Explainability: Grad-CAM Heatmap", expanded=True):
            if res.get("gradcam_overlay") is not None:
                st.image(
                    res["gradcam_overlay"],
                    caption=f"Grad-CAM Feature Activation (Digit {pred_digit})",
                    use_column_width=True
                )
                st.caption("Bright regions highlight the stroke contours that most strongly triggered the CNN prediction.")
            else:
                st.info("Grad-CAM visualization not available for this sample.")


def render_upload_page():
    """Renders the image upload module."""
    st.markdown("""
    <div class="glass-card">
        <h2 style="color: #f8fafc; font-weight: 800; margin-bottom: 6px;">🖼️ Handwritten Digit Image Recognition</h2>
        <p style="color: #94a3b8; font-size: 14.5px; margin-bottom: 0;">
            Upload scanned documents, photo crops, or synthetic digit images. The automated preprocessing pipeline 
            corrects lighting, isolates the foreground digit, removes noise, and passes a standardized 28×28 tensor to the CNN.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_single, tab_batch = st.tabs(["Single Image Recognition", "Batch Prediction (Multiple Images)"])

    with tab_single:
        uploaded_file = st.file_uploader(
            "Choose a handwritten digit image (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"],
            key="single_digit_uploader"
        )

        if uploaded_file is not None:
            col_img, col_btn = st.columns([1, 2])
            with col_img:
                st.image(uploaded_file, caption="Uploaded Input Preview", use_column_width=True)
            with col_btn:
                st.write("")
                st.write("")
                predict_clicked = st.button("🚀 Predict Digit", key="btn_predict_upload", use_container_width=True)

            if predict_clicked or "last_upload_res" in st.session_state:
                if predict_clicked:
                    with st.spinner("Analyzing handwritten digit through CNN pipeline..."):
                        res = predict_digit(uploaded_file, input_source="Upload")
                        st.session_state["last_upload_res"] = res
                else:
                    res = st.session_state["last_upload_res"]

                st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
                render_prediction_results(res, key_prefix="single_up")

    with tab_batch:
        st.markdown("Upload multiple images at once to run high-throughput batch inference.")
        uploaded_files = st.file_uploader(
            "Upload multiple digit images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="batch_digit_uploader"
        )

        if uploaded_files:
            st.success(f"{len(uploaded_files)} image(s) queued for recognition.")
            if st.button("⚡ Run Batch Prediction", key="btn_run_batch"):
                batch_records = []
                progress_bar = st.progress(0)

                for i, file in enumerate(uploaded_files):
                    res = predict_digit(file, input_source="Batch Upload", generate_explanation=False)
                    if res["success"]:
                        batch_records.append({
                            "Filename": file.name,
                            "Predicted Digit": res["predicted_digit"],
                            "Word": res["digit_word"],
                            "Confidence (%)": res["confidence_pct"],
                            "Status": "Uncertain" if res["is_uncertain"] else "High Confidence"
                        })
                    else:
                        batch_records.append({
                            "Filename": file.name,
                            "Predicted Digit": "N/A",
                            "Word": "Invalid",
                            "Confidence (%)": 0.0,
                            "Status": res.get("message", "Error")
                        })
                    progress_bar.progress((i + 1) / len(uploaded_files))

                df_batch = pd.DataFrame(batch_records)
                st.markdown("<h4 style='color: #f8fafc; margin-top: 15px;'>Batch Recognition Results</h4>", unsafe_allow_html=True)
                st.dataframe(df_batch, use_container_width=True)

                # Export CSV button
                csv_data = df_batch.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Batch Results (CSV)",
                    data=csv_data,
                    file_name="digitvision_batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

