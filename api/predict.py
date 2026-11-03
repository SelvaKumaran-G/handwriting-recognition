"""
DigitVision AI - Vercel Serverless Prediction API
Lightweight ONNX Runtime inference for handwritten digit recognition.
Replaces TensorFlow (2.5GB) + EasyOCR/PyTorch (2.5GB) with ONNX Runtime (~25MB).
Total bundle: ~70MB (Vercel limit: 500MB).
"""

import os
import sys
import json
import base64
import traceback
import numpy as np
import cv2
from pathlib import Path
from http.server import BaseHTTPRequestHandler

# Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIGIT_WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
MIN_CONTOUR_AREA = 25

# Lazy-loaded ONNX session
_ort_session = None

def get_onnx_session():
    """Loads and caches the ONNX Runtime inference session."""
    global _ort_session
    if _ort_session is not None:
        return _ort_session
    import onnxruntime as ort
    model_path = str(PROJECT_ROOT / "models" / "digit_model.onnx")
    _ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return _ort_session


def encode_np_to_b64(np_img: np.ndarray, is_rgb: bool = True) -> str:
    """Converts a numpy image into a base64 Data URL."""
    if np_img is None or np_img.size == 0:
        return ""
    if is_rgb and np_img.ndim == 3 and np_img.shape[2] == 3:
        bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    elif np_img.ndim == 2:
        bgr = np_img
    else:
        bgr = np_img
    success, buffer = cv2.imencode(".png", bgr)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


# ============================================================
# IMAGE PREPROCESSING (self-contained, lightweight)
# ============================================================

def decode_image(raw_bytes):
    """Decode raw bytes into RGB numpy array."""
    nparr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image from bytes.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detect_polarity(gray):
    """Auto-detect background polarity. Returns (corrected_img, was_inverted)."""
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    if float(np.mean(border)) > 127:
        return cv2.bitwise_not(gray), True
    return gray.copy(), False


def normalize_strokes(binary):
    """Normalize pen stroke thickness to match MNIST distribution."""
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    max_r = float(np.max(dist)) if np.max(dist) > 0 else 1.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if max_r < 1.8:
        return cv2.dilate(binary, k, iterations=1)
    elif max_r > 6.0:
        return cv2.erode(binary, k, iterations=1)
    return binary


def center_of_mass_numpy(img):
    """Pure-numpy Center of Mass (replaces scipy.ndimage.center_of_mass)."""
    total = float(np.sum(img))
    if total == 0:
        return 14.0, 14.0
    y_idx, x_idx = np.indices(img.shape).astype(np.float64)
    cy = float(np.sum(y_idx * img) / total)
    cx = float(np.sum(x_idx * img) / total)
    return cy, cx


def binarize_image(gray):
    """Hybrid Otsu + adaptive thresholding with morphological cleanup."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    dark_bg, _ = detect_polarity(blurred)
    _, thresh_otsu = cv2.threshold(dark_bg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adapt = cv2.adaptiveThreshold(dark_bg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -4)
    combined = cv2.bitwise_or(thresh_otsu, adapt)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned


def crop_to_28x28(binary_region):
    """Resize a binary digit crop into centered 28x28 MNIST-format tensor & image."""
    h, w = binary_region.shape
    if h > w:
        new_h, new_w = 20, max(1, int(round(w * 20.0 / h)))
    else:
        new_w, new_h = 20, max(1, int(round(h * 20.0 / w)))

    resized = cv2.resize(binary_region, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized = normalize_strokes(resized)

    framed = np.zeros((28, 28), dtype=np.uint8)
    oy = (28 - new_h) // 2
    ox = (28 - new_w) // 2
    framed[oy:oy + new_h, ox:ox + new_w] = resized

    # Center of Mass alignment
    cy, cx = center_of_mass_numpy(framed)
    if not (np.isnan(cy) or np.isnan(cx)):
        sx = max(-4, min(4, int(np.round(14.0 - cx))))
        sy = max(-4, min(4, int(np.round(14.0 - cy))))
        M = np.float32([[1, 0, sx], [0, 1, sy]])
        framed = cv2.warpAffine(framed, M, (28, 28), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    normalized = framed.astype(np.float32) / 255.0
    tensor = np.expand_dims(normalized, axis=(0, -1))  # (1, 28, 28, 1)
    return tensor, framed


def preprocess_single_image(raw_bytes):
    """Full preprocessing pipeline for a single-digit image. Returns (tensor, norm_img, bbox, rgb, error_msg)."""
    rgb = decode_image(raw_bytes)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cleaned = binarize_image(gray)

    white_count = int(np.count_nonzero(cleaned))
    if white_count < MIN_CONTOUR_AREA:
        return None, None, None, None, "No handwritten digit detected. Image appears blank or too faint."
    if white_count / cleaned.size > 0.85:
        return None, None, None, None, "Image contains excessive noise or glare."

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, None, "No digit contours found."

    img_h, img_w = cleaned.shape
    valid = [c for c in contours
             if cv2.contourArea(c) >= MIN_CONTOUR_AREA
             and not (cv2.boundingRect(c)[2] > 0.95 * img_w and cv2.boundingRect(c)[3] > 0.95 * img_h)]
    if not valid:
        valid = [max(contours, key=cv2.contourArea)]

    boxes = [cv2.boundingRect(c) for c in valid]
    x_min = min(b[0] for b in boxes)
    y_min = min(b[1] for b in boxes)
    x_max = max(b[0] + b[2] for b in boxes)
    y_max = max(b[1] + b[3] for b in boxes)

    pad = 4
    x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
    x_max, y_max = min(img_w, x_max + pad), min(img_h, y_max + pad)

    crop = cleaned[y_min:y_max, x_min:x_max]
    tensor, norm_img = crop_to_28x28(crop)
    return tensor, norm_img, (x_min, y_min, x_max - x_min, y_max - y_min), rgb, None


def segment_characters(raw_bytes):
    """Segment multi-digit images into individual character bounding boxes."""
    rgb = decode_image(raw_bytes)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cleaned = binarize_image(gray)
    img_h, img_w = cleaned.shape

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area >= 40 and w < 0.96 * img_w and h < 0.96 * img_h and w > 3 and h > 6:
            boxes.append((x, y, w, h))

    if not boxes:
        return [], cleaned, rgb

    # Sort left-to-right, merge overlapping strokes
    boxes.sort(key=lambda b: b[0])
    merged = [boxes[0]]
    for box in boxes[1:]:
        prev = merged[-1]
        x_overlap = box[0] < prev[0] + prev[2] + 3
        y_overlap = max(prev[1], box[1]) < min(prev[1] + prev[3], box[1] + box[3]) + 6
        if x_overlap and y_overlap:
            nx = min(prev[0], box[0])
            ny = min(prev[1], box[1])
            nw = max(prev[0] + prev[2], box[0] + box[2]) - nx
            nh = max(prev[1] + prev[3], box[1] + box[3]) - ny
            merged[-1] = (nx, ny, nw, nh)
        else:
            merged.append(box)

    merged.sort(key=lambda b: (b[1] // 40, b[0]))
    return merged, cleaned, rgb


def run_onnx_inference(tensor):
    """Run ONNX inference on a preprocessed (1,28,28,1) tensor."""
    session = get_onnx_session()
    input_name = session.get_inputs()[0].name
    probs = session.run(None, {input_name: tensor})[0][0]
    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    return pred, conf, [float(p) for p in probs]


# ============================================================
# VERCEL HANDLER
# ============================================================

class handler(BaseHTTPRequestHandler):
    """Vercel serverless function handler for /api/predict."""

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length))

            b64_image = body.get("image", "")
            source = body.get("source", "Uploaded Image")

            if not b64_image:
                self._json_response(400, {"success": False, "message": "No image data provided."})
                return

            if "," in b64_image:
                b64_image = b64_image.split(",", 1)[1]
            raw_bytes = base64.b64decode(b64_image)
            if len(raw_bytes) == 0:
                self._json_response(400, {"success": False, "message": "Empty image buffer."})
                return

            # Try multi-digit segmentation
            char_boxes, cleaned, rgb_img = segment_characters(raw_bytes)
            annotated_img = rgb_img.copy()

            if len(char_boxes) <= 1:
                # ── Single Digit Path ──
                tensor, norm_img, bbox, rgb, err = preprocess_single_image(raw_bytes)
                if tensor is None:
                    self._json_response(200, {"success": False, "message": err})
                    return

                pred, conf, probs = run_onnx_inference(tensor)
                primary = str(pred)
                content_type = "Handwritten Digit"
                avg_conf = conf

                # Draw bounding box on annotated image
                if bbox:
                    bx, by, bw, bh = bbox
                    cv2.rectangle(annotated_img, (bx, by), (bx + bw, by + bh), (5, 150, 105), 2)
                    cv2.putText(
                        annotated_img,
                        f"Digit {pred} ({conf*100:.0f}%)",
                        (bx, max(16, by - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (5, 150, 105),
                        2,
                        cv2.LINE_AA
                    )

                norm_b64 = encode_np_to_b64(norm_img, is_rgb=False)
                char_results = [{
                    "char_index": 1,
                    "predicted_digit": pred,
                    "digit_word": DIGIT_WORDS[pred],
                    "confidence": conf,
                    "confidence_pct": round(conf * 100, 1),
                    "probabilities": probs,
                    "gradcam_b64": norm_b64,
                    "norm_b64": norm_b64
                }]
                word = DIGIT_WORDS[pred]
                spoken = f"The detected digit is {word} with {conf * 100:.1f} percent confidence."

            else:
                # ── Multi-Digit Path ──
                img_h, img_w = cleaned.shape
                char_results = []
                for idx, (x, y, w, h) in enumerate(char_boxes):
                    p = 2
                    x1, y1 = max(0, x - p), max(0, y - p)
                    x2, y2 = min(img_w, x + w + p), min(img_h, y + h + p)
                    crop = cleaned[y1:y2, x1:x2]
                    if crop.size == 0 or np.count_nonzero(crop) < 15:
                        continue

                    tensor, norm_img = crop_to_28x28(crop)
                    pred, conf, probs = run_onnx_inference(tensor)
                    norm_b64 = encode_np_to_b64(norm_img, is_rgb=False)

                    char_results.append({
                        "char_index": idx + 1,
                        "predicted_digit": pred,
                        "digit_word": DIGIT_WORDS[pred],
                        "confidence": conf,
                        "confidence_pct": round(conf * 100, 1),
                        "probabilities": probs,
                        "gradcam_b64": norm_b64,
                        "norm_b64": norm_b64
                    })

                    # Draw bounding box on annotated image
                    cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (5, 150, 105), 2)
                    cv2.putText(
                        annotated_img,
                        f"{pred} ({conf*100:.0f}%)",
                        (x, max(16, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (5, 150, 105),
                        2,
                        cv2.LINE_AA
                    )

                if not char_results:
                    self._json_response(200, {"success": False, "message": "Could not identify digits."})
                    return

                primary = "".join(str(c["predicted_digit"]) for c in char_results)
                content_type = "Detected Numerical Sequence"
                avg_conf = float(np.mean([c["confidence"] for c in char_results]))
                spoken = f"The detected number sequence is {primary}."

            annotated_b64 = encode_np_to_b64(annotated_img, is_rgb=True)

            response = {
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
                "spoken_message": spoken,
                "source": source
            }

            self._json_response(200, response)

        except Exception as e:
            traceback.print_exc()
            self._json_response(500, {"success": False, "message": f"Server error: {str(e)}"})

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _json_response(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))
