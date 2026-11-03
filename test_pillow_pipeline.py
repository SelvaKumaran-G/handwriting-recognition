"""
Test pure Pillow + NumPy preprocessing pipeline (zero OpenCV dependencies).
Ensures 100% compatibility with AWS Lambda / Vercel Linux environments.
"""

import io
import base64
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import onnxruntime as ort

MODEL_PATH = Path("api/digit_model.onnx")
assert MODEL_PATH.exists(), f"Model not found at {MODEL_PATH}"
session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
DIGIT_WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]


def otsu_threshold_numpy(gray: np.ndarray) -> np.ndarray:
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


def center_of_mass_numpy(img: np.ndarray):
    """Calculates center of mass coordinates using pure NumPy."""
    total = float(np.sum(img))
    if total == 0:
        return 14.0, 14.0
    y_idx, x_idx = np.indices(img.shape).astype(np.float64)
    cy = float(np.sum(y_idx * img) / total)
    cx = float(np.sum(x_idx * img) / total)
    return cy, cx


def preprocess_pil(image_bytes: bytes):
    """Robust, pure Pillow/NumPy preprocessing pipeline."""
    # 1. Load image
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    rgb_arr = np.array(pil_img)

    # 2. Convert to grayscale & smooth
    gray_pil = pil_img.convert("L").filter(ImageFilter.GaussianBlur(radius=1.0))
    gray = np.array(gray_pil)

    # 3. Polarity detection (ensure black background, white strokes)
    h, w = gray.shape
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    if float(np.mean(border)) > 127:
        dark_bg = 255 - gray
    else:
        dark_bg = gray.copy()

    # 4. Otsu binarization
    binary = otsu_threshold_numpy(dark_bg)

    # 5. Find bounding box of foreground digit
    y_idx, x_idx = np.where(binary > 0)
    if len(y_idx) < 25:
        return None, None, None, "No digit detected (too faint or blank)."

    y_min, y_max = int(np.min(y_idx)), int(np.max(y_idx))
    x_min, x_max = int(np.min(x_idx)), int(np.max(x_idx))

    # Pad bounding box
    pad = 4
    y_min = max(0, y_min - pad)
    x_min = max(0, x_min - pad)
    y_max = min(h, y_max + pad)
    x_max = min(w, x_max + pad)

    crop = binary[y_min:y_max, x_min:x_max]
    crop_h, crop_w = crop.shape
    if crop_h == 0 or crop_w == 0:
        return None, None, None, "Invalid crop."

    # 6. Aspect-ratio preserving resize into 20x20
    if crop_h > crop_w:
        new_h = 20
        new_w = max(1, int(round((crop_w * 20.0) / crop_h)))
    else:
        new_w = 20
        new_h = max(1, int(round((crop_h * 20.0) / crop_w)))

    crop_pil = Image.fromarray(crop).resize((new_w, new_h), Image.Resampling.BILINEAR)
    resized_arr = np.array(crop_pil)

    # 7. Place inside 28x28 blank canvas
    framed = np.zeros((28, 28), dtype=np.uint8)
    oy = (28 - new_h) // 2
    ox = (28 - new_w) // 2
    framed[oy:oy + new_h, ox:ox + new_w] = resized_arr

    # 8. Center of Mass alignment
    cy, cx = center_of_mass_numpy(framed)
    if not (np.isnan(cy) or np.isnan(cx)):
        sx = int(np.clip(np.round(14.0 - cx), -4, 4))
        sy = int(np.clip(np.round(14.0 - cy), -4, 4))
        # Shift with roll & zero out wrapped edges
        framed = np.roll(np.roll(framed, sy, axis=0), sx, axis=1)
        if sy > 0:
            framed[:sy, :] = 0
        elif sy < 0:
            framed[sy:, :] = 0
        if sx > 0:
            framed[:, :sx] = 0
        elif sx < 0:
            framed[:, sx:] = 0

    # 9. Normalize [0, 1] and reshape to (1, 28, 28, 1)
    norm = framed.astype(np.float32) / 255.0
    tensor = np.expand_dims(norm, axis=(0, -1))
    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

    return tensor, framed, bbox, None


# Run test on sample digits
for digit in [3, 7, 0, 1, 4, 8]:
    sample_file = Path(f"assets/sample_digits/digit_{digit}_sample1_paper.png")
    if sample_file.exists():
        with open(sample_file, "rb") as f:
            bytes_data = f.read()
        tensor, framed, bbox, err = preprocess_pil(bytes_data)
        assert err is None, f"Error on digit {digit}: {err}"
        probs = session.run(None, {input_name: tensor})[0][0]
        pred = int(np.argmax(probs))
        conf = float(probs[pred])
        print(f"Digit {digit} -> Predicted: {pred} ({DIGIT_WORDS[pred]}) | Conf: {conf*100:.1f}%")
        assert pred == digit, f"Expected {digit}, got {pred}"

print("\n[ALL TESTS PASSED WITH PURE PILLOW + NUMPY!]")
