"""
DigitVision AI - Model Inference & Prediction Service
Handles single and batch digit inference, confidence scoring,
uncertainty estimation, Grad-CAM explanation generation, and history recording.
"""

from typing import Dict, Any, List, Optional
import os
import csv
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from datetime import datetime
import numpy as np
import tensorflow as tf

from config import (
    MODEL_PATH,
    HISTORY_CSV_PATH,
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_LOW_THRESHOLD,
    DIGIT_LABELS,
    DIGIT_WORDS
)
from preprocessing import preprocess_image
from model import make_gradcam_heatmap, overlay_gradcam

# Global in-memory cache for model
_LOADED_MODEL = None


def load_digit_model(model_path=MODEL_PATH) -> tf.keras.Model:
    """
    Loads and caches the trained Keras CNN model.
    """
    global _LOADED_MODEL
    if _LOADED_MODEL is not None:
        return _LOADED_MODEL

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            f"Please run 'python train_model.py' first to train and save the model."
        )

    model = tf.keras.models.load_model(model_path)
    _LOADED_MODEL = model
    return model


def log_prediction_to_history(
    input_source: str,
    predicted_digit: int,
    confidence: float,
    status: str = "Success",
    csv_path=HISTORY_CSV_PATH
):
    """
    Appends a prediction event into persistent history storage.
    """
    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        file_exists = os.path.exists(csv_path)

        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "prediction_id",
                    "timestamp",
                    "input_source",
                    "predicted_digit",
                    "digit_word",
                    "confidence_pct",
                    "status"
                ])
            pred_id = f"PRED_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:21]}"
            writer.writerow([
                pred_id,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                input_source,
                predicted_digit,
                DIGIT_WORDS[predicted_digit] if 0 <= predicted_digit <= 9 else "Unknown",
                round(confidence * 100, 2),
                status
            ])
    except Exception as e:
        print(f"Failed to record prediction history: {e}")


def predict_digit(
    image_input,
    model: Optional[tf.keras.Model] = None,
    input_source: str = "Upload",
    generate_explanation: bool = True,
    record_history: bool = True
) -> Dict[str, Any]:
    """
    Performs complete end-to-end digit prediction:
      1. Preprocesses image
      2. Validates image quality
      3. Passes tensor to CNN
      4. Computes probability distribution, top-3 candidates
      5. Analyzes confidence and uncertainty
      6. Generates Grad-CAM visual heatmap
      7. Records into persistent history log
    """
    if model is None:
        model = load_digit_model()

    # Step 1: Preprocess
    prep_res = preprocess_image(image_input)
    if not prep_res["is_valid"]:
        return {
            "success": False,
            "message": prep_res["message"],
            "stages": prep_res.get("stages", {}),
            "diagnostics": prep_res.get("diagnostics", {})
        }

    tensor = prep_res["tensor"]
    norm_28x28 = prep_res["normalized_28x28"]

    # Step 2: CNN Inference
    probabilities = model.predict(tensor, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])

    # Step 3: Top-3 Predictions
    top3_indices = np.argsort(probabilities)[::-1][:3]
    top3_list = [
        {
            "rank": rank + 1,
            "digit": int(idx),
            "word": DIGIT_WORDS[idx],
            "probability": float(probabilities[idx]),
            "percentage": round(float(probabilities[idx]) * 100, 2)
        }
        for rank, idx in enumerate(top3_indices)
    ]

    # Step 4: Confidence & Uncertainty Evaluation
    is_uncertain = confidence < CONFIDENCE_LOW_THRESHOLD
    is_moderate = CONFIDENCE_LOW_THRESHOLD <= confidence < CONFIDENCE_HIGH_THRESHOLD
    is_high = confidence >= CONFIDENCE_HIGH_THRESHOLD

    confidence_level = "High" if is_high else ("Moderate" if is_moderate else "Low (Uncertain)")

    if is_uncertain:
        status_msg = (
            f"Prediction is uncertain (confidence {confidence*100:.1f}%). "
            f"Please upload a clearer image or capture again."
        )
    elif is_moderate:
        status_msg = f"Digit detected with moderate confidence ({confidence*100:.1f}%)."
    else:
        status_msg = f"Digit recognized with high confidence ({confidence*100:.1f}%)."

    # Step 5: Explainability (Grad-CAM)
    gradcam_overlay = None
    if generate_explanation:
        try:
            heatmap = make_gradcam_heatmap(tensor, model, pred_index=predicted_digit)
            gradcam_overlay = overlay_gradcam(norm_28x28, heatmap)
        except Exception as e:
            print(f"Grad-CAM generation error: {e}")

    # Step 6: Log history
    if record_history:
        log_prediction_to_history(
            input_source=input_source,
            predicted_digit=predicted_digit,
            confidence=confidence,
            status="Uncertain" if is_uncertain else "Success"
        )

    return {
        "success": True,
        "predicted_digit": predicted_digit,
        "digit_word": DIGIT_WORDS[predicted_digit],
        "confidence": confidence,
        "confidence_pct": round(confidence * 100, 2),
        "confidence_level": confidence_level,
        "is_uncertain": is_uncertain,
        "status_message": status_msg,
        "top3": top3_list,
        "all_probabilities": [float(p) for p in probabilities],
        "stages": prep_res["stages"],
        "diagnostics": prep_res["diagnostics"],
        "normalized_28x28": norm_28x28,
        "gradcam_overlay": gradcam_overlay
    }
