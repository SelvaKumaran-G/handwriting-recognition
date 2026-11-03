"""
DigitVision AI - Production Serverless API for Vercel
Uses Pure Pillow + NumPy + ONNX Runtime for ultra-lightweight, 100% reliable deployment.
Zero OpenCV dependencies: avoids all missing .so library crashes on AWS Lambda / Vercel Linux.
"""

import os
import sys
import json
import base64
import io
import traceback
from pathlib import Path
from http.server import BaseHTTPRequestHandler
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

DIGIT_WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
MIN_STROKE_PIXELS = 20

# Cached ONNX inference session
_ort_session = None

def get_session():
    """Lazily load the ONNX session from api/ or models/ directory."""
    global _ort_session
    if _ort_session is not None:
        return _ort_session

    import onnxruntime as ort
    here = Path(__file__).resolve().parent
    candidates = [
        here / "digit_model.onnx",
        here.parent / "models" / "digit_model.onnx",
        here.parent / "api" / "digit_model.onnx"
    ]
    model_path = None
    for c in candidates:
        if c.exists():
            model_path = c
            break

    if model_path is None:
        raise FileNotFoundError(f"ONNX model file not found in candidates: {[str(c) for c in candidates]}")

    _ort_session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return _ort_session


def otsu_threshold(gray: np.ndarray) -> np.ndarray:
    """Computes Otsu's optimal binarization threshold using pure NumPy."""
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = gray.size
    current_max, threshold = 0.0, 128
    sum_total = np.dot(np.arange(256), hist)
    sum_b, w_b = 0.0, 0.0

    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        between_var = w_b * w_f * (m_b - m_f) ** 2
        if between_var > current_max:
            current_max = between_var
            threshold = t

    return ((gray > threshold) * 255).astype(np.uint8)


def center_of_mass(img: np.ndarray):
    """Calculates Center of Mass using pure NumPy."""
    total = float(np.sum(img))
    if total == 0:
        return 14.0, 14.0
    y_idx, x_idx = np.indices(img.shape).astype(np.float64)
    cy = float(np.sum(y_idx * img) / total)
    cx = float(np.sum(x_idx * img) / total)
    return cy, cx


def encode_pil_to_b64(pil_img: Image.Image) -> str:
    """Converts a PIL Image into a base64 Data URL."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def preprocess_crop_to_28x28(binary_crop: np.ndarray):
    """Formats a binary character crop into a centered 28x28 MNIST-aligned tensor & image."""
    h, w = binary_crop.shape
    if h > w:
        new_h = 20
        new_w = max(1, int(round((w * 20.0) / h)))
    else:
        new_w = 20
        new_h = max(1, int(round((h * 20.0) / w)))

    crop_pil = Image.fromarray(binary_crop).resize((new_w, new_h), Image.Resampling.BILINEAR)
    resized_arr = np.array(crop_pil)

    # Frame inside 28x28 canvas
    framed = np.zeros((28, 28), dtype=np.uint8)
    oy = (28 - new_h) // 2
    ox = (28 - new_w) // 2
    framed[oy:oy + new_h, ox:ox + new_w] = resized_arr

    # Center of Mass alignment
    cy, cx = center_of_mass(framed)
    if not (np.isnan(cy) or np.isnan(cx)):
        sx = int(np.clip(np.round(14.0 - cx), -4, 4))
        sy = int(np.clip(np.round(14.0 - cy), -4, 4))
        framed = np.roll(np.roll(framed, sy, axis=0), sx, axis=1)
        if sy > 0:
            framed[:sy, :] = 0
        elif sy < 0:
            framed[sy:, :] = 0
        if sx > 0:
            framed[:, :sx] = 0
        elif sx < 0:
            framed[:, sx:] = 0

    norm = framed.astype(np.float32) / 255.0
    tensor = np.expand_dims(norm, axis=(0, -1))
    return tensor, framed


def segment_character_columns(binary: np.ndarray):
    """Finds individual character regions using horizontal projection profiles."""
    col_sums = np.sum(binary > 0, axis=0)
    in_char = False
    start_col = 0
    chars = []

    for col_idx, s in enumerate(col_sums):
        if s > 0 and not in_char:
            in_char = True
            start_col = col_idx
        elif s == 0 and in_char:
            in_char = False
            if col_idx - start_col > 3:  # At least 4 pixels wide
                chars.append((start_col, col_idx))

    if in_char and len(col_sums) - start_col > 3:
        chars.append((start_col, len(col_sums)))

    # For each column range, get the tight vertical bounding box
    boxes = []
    for x1, x2 in chars:
        slice_img = binary[:, x1:x2]
        y_indices = np.where(np.sum(slice_img > 0, axis=1) > 0)[0]
        if len(y_indices) > 5:
            y1 = int(y_indices[0])
            y2 = int(y_indices[-1]) + 1
            if (x2 - x1) * (y2 - y1) >= MIN_STROKE_PIXELS:
                boxes.append((x1, y1, x2 - x1, y2 - y1))

    return boxes


def run_prediction_pipeline(raw_bytes: bytes):
    """Executes end-to-end digit recognition using pure Pillow and ONNX."""
    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    w_orig, h_orig = pil_img.size

    # Grayscale with slight blur to remove paper texture noise
    gray_pil = pil_img.convert("L").filter(ImageFilter.GaussianBlur(radius=1.0))
    gray = np.array(gray_pil)

    # Polarity detection: invert if dark handwriting on light paper
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    if float(np.mean(border)) > 127:
        dark_bg = 255 - gray
    else:
        dark_bg = gray.copy()

    # Otsu thresholding
    binary = otsu_threshold(dark_bg)

    # Check for empty image
    if int(np.count_nonzero(binary)) < MIN_STROKE_PIXELS:
        return {"success": False, "message": "No handwritten digit detected. Image appears blank or too faint."}

    # Segment character boxes
    boxes = segment_character_columns(binary)

    # Fallback to single overall bounding box if segmentation found 0
    if not boxes:
        y_idx, x_idx = np.where(binary > 0)
        if len(y_idx) > 0:
            x_min, x_max = int(np.min(x_idx)), int(np.max(x_idx))
            y_min, y_max = int(np.min(y_idx)), int(np.max(y_idx))
            boxes = [(x_min, y_min, x_max - x_min, y_max - y_min)]

    session = get_session()
    input_name = session.get_inputs()[0].name

    annotated_pil = pil_img.copy()
    draw = ImageDraw.Draw(annotated_pil)

    char_results = []
    digits_list = []
    confs_list = []

    for idx, (x, y, bw, bh) in enumerate(boxes):
        pad = 2
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_orig, x + bw + pad)
        y2 = min(h_orig, y + bh + pad)

        crop = binary[y1:y2, x1:x2]
        if crop.size == 0 or np.count_nonzero(crop) < 10:
            continue

        tensor, norm_img = preprocess_crop_to_28x28(crop)
        probs = session.run(None, {input_name: tensor})[0][0]
        pred_digit = int(np.argmax(probs))
        conf = float(probs[pred_digit])

        digits_list.append(str(pred_digit))
        confs_list.append(conf)

        norm_pil = Image.fromarray(norm_img)
        norm_b64 = encode_pil_to_b64(norm_pil)

        char_results.append({
            "char_index": idx + 1,
            "predicted_digit": pred_digit,
            "digit_word": DIGIT_WORDS[pred_digit],
            "confidence": conf,
            "confidence_pct": round(conf * 100, 1),
            "probabilities": [float(p) for p in probs],
            "gradcam_b64": norm_b64,
            "norm_b64": norm_b64
        })

        # Draw green bounding box on annotated image
        draw.rectangle([x, y, x + bw, y + bh], outline="#059669", width=2)
        draw.text((x + 2, max(2, y - 14)), f"{pred_digit} ({conf*100:.0f}%)", fill="#059669")

    if not char_results:
        return {"success": False, "message": "Could not identify clear digits. Please try drawing or uploading again."}

    primary = "".join(digits_list)
    avg_conf = float(np.mean(confs_list))
    content_type = "Handwritten Digit" if len(primary) == 1 else "Detected Numerical Sequence"

    if len(primary) == 1:
        d_val = int(primary)
        word = DIGIT_WORDS[d_val]
        spoken = f"The detected digit is {word} with {avg_conf * 100:.1f} percent confidence."
    else:
        spoken = f"The detected number sequence is {primary}."

    annotated_b64 = encode_pil_to_b64(annotated_pil)

    return {
        "success": True,
        "primary_detected": primary,
        "content_type": content_type,
        "is_text_word": False,
        "reconstructed_number": primary,
        "avg_confidence": avg_conf,
        "avg_confidence_pct": round(avg_conf * 100, 1),
        "annotated_image_b64": annotated_b64,
        "char_results": char_results,
        "ocr_blocks": [],
        "full_ocr_string": "",
        "spoken_message": spoken
    }


# ============================================================
# HTTP HANDLER FOR VERCEL SERVERLESS RUNTIME
# ============================================================

class handler(BaseHTTPRequestHandler):
    """Vercel HTTP handler for /api/predict."""

    def log_message(self, format, *args):
        """Silence logging to prevent AttributeError on Vercel AWS Lambda bridge."""
        pass

    def log_request(self, code='-', size='-'):
        """Silence request logging."""
        pass

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self._send_json(400, {"success": False, "message": "No data received in request."})
                return

            body_bytes = self.rfile.read(content_length)
            body = json.loads(body_bytes.decode("utf-8"))

            b64_image = body.get("image", "")
            source = body.get("source", "Uploaded Image")

            if not b64_image:
                self._send_json(400, {"success": False, "message": "Image data missing."})
                return

            # Strip Data URL prefix if present
            if "," in b64_image:
                b64_image = b64_image.split(",", 1)[1]

            raw_bytes = base64.b64decode(b64_image)
            if len(raw_bytes) == 0:
                self._send_json(400, {"success": False, "message": "Decoded image buffer is empty."})
                return

            result = run_prediction_pipeline(raw_bytes)
            result["source"] = source
            self._send_json(200, result)

        except Exception as e:
            traceback.print_exc()
            self._send_json(200, {
                "success": False,
                "message": f"Server processing error: {str(e)}"
            })

    def do_OPTIONS(self):
        """Handle CORS preflight checks."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _send_json(self, status: int, data: dict):
        response_bytes = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)
