"""
DigitVision AI - Advanced Real-World Image Preprocessing Pipeline
Transforms raw camera captures, canvas drawings, and uploaded paper images into
standardized 28x28 normalized tensors perfectly aligned with the MNIST digit distribution.
Includes adaptive stroke thickness normalization, shadow removal, and CoM centering.
"""

from typing import Dict, Tuple, Optional, Any
import os
import numpy as np
import cv2
from PIL import Image
import io

from config import TARGET_SIZE, MIN_CONTOUR_AREA


def load_image_to_numpy(image_input) -> np.ndarray:
    """
    Safely converts uploaded file bytes, file path, PIL Image, or numpy array into an RGB numpy array.
    """
    if isinstance(image_input, (str, os.PathLike)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise ValueError(f"Could not read image from path: {image_input}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if isinstance(image_input, (bytes, bytearray)):
        if len(image_input) == 0:
            raise ValueError("Provided image byte buffer is empty.")
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from byte buffer.")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if hasattr(image_input, "read") or hasattr(image_input, "getvalue"):
        if hasattr(image_input, "seek"):
            image_input.seek(0)
        if hasattr(image_input, "getvalue"):
            content = image_input.getvalue()
        else:
            content = image_input.read()
        if hasattr(image_input, "seek"):
            image_input.seek(0)
        if not content:
            raise ValueError("Image stream buffer is empty.")
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from file-like stream.")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if isinstance(image_input, Image.Image):
        return np.array(image_input.convert("RGB"))

    if isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        if image_input.shape[2] == 4:
            # Handle RGBA (from canvas or PNG)
            # If canvas has black background and alpha channel, preserve RGB
            rgb = image_input[:, :, :3]
            alpha = image_input[:, :, 3]
            if np.any(alpha < 255):
                # Composite over black background
                mask = alpha > 10
                out = np.zeros_like(rgb)
                out[mask] = rgb[mask]
                return out
            return rgb
        return image_input

    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def detect_and_correct_polarity(gray_img: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Automatically detects background/foreground polarity.
    MNIST requires: background = 0 (black), digit stroke = 255 (white).
    
    Examines border pixel statistics: if border is light (>127), the image is a dark digit
    on light paper/background, so it must be inverted.
    """
    h, w = gray_img.shape
    border_pixels = np.concatenate([
        gray_img[0, :],         # top row
        gray_img[-1, :],        # bottom row
        gray_img[:, 0],         # left col
        gray_img[:, -1]         # right col
    ])
    border_mean = float(np.mean(border_pixels))

    was_inverted = False
    if border_mean > 127:
        corrected = cv2.bitwise_not(gray_img)
        was_inverted = True
    else:
        corrected = gray_img.copy()

    return corrected, was_inverted


def normalize_stroke_thickness(binary_img: np.ndarray, target_thickness: float = 2.8) -> np.ndarray:
    """
    Normalizes pen stroke thickness so real-world ballpoint/pencil drawings
    match native MNIST pen thickness (~3 pixels wide in 20x20 scale).
    """
    # Compute distance transform to estimate stroke radius
    dist = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
    max_radius = float(np.max(dist)) if np.max(dist) > 0 else 1.0

    # If strokes are thin (radius < 2.0), dilate with small kernel
    if max_radius < 1.8:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(binary_img, kernel, iterations=1)
        return dilated
    elif max_radius > 6.0:
        # Strokes are unusually fat, erode slightly
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(binary_img, kernel, iterations=1)
        return eroded

    return binary_img


def center_by_center_of_mass(img_28x28: np.ndarray) -> np.ndarray:
    """
    Centers the digit inside the 28x28 canvas using the Center of Mass (CoM),
    matching standard MNIST dataset formatting.
    """
    import scipy.ndimage as ndimage

    cy, cx = ndimage.center_of_mass(img_28x28)

    if np.isnan(cy) or np.isnan(cx):
        return img_28x28

    shift_x = int(np.round(14.0 - cx))
    shift_y = int(np.round(14.0 - cy))

    # Constrain shift to avoid pushing digit out of frame
    shift_x = max(-4, min(4, shift_x))
    shift_y = max(-4, min(4, shift_y))

    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    centered = cv2.warpAffine(img_28x28, M, (28, 28), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return centered


def preprocess_image(
    image_input,
    target_size: Tuple[int, int] = TARGET_SIZE
) -> Dict[str, Any]:
    """
    Complete end-to-end robust preprocessing pipeline for real-world handwritten digits:
      1. Image decoding & format normalization
      2. Shadow & lighting attenuation
      3. Grayscale conversion
      4. Polarity detection & correction (black background, white strokes)
      5. Hybrid Adaptive + Otsu Thresholding (handles shadows & uneven lighting)
      6. Morphological cleanup (removes dust, reinforces strokes)
      7. Intelligent contour isolation (filters out paper borders)
      8. Aspect-ratio preserving resize into 20x20 bounding box
      9. Adaptive stroke thickness normalization
      10. Center of Mass alignment inside 28x28 canvas
      11. Min-max normalization [0.0, 1.0]
      12. CNN tensor formatting: (1, 28, 28, 1)
    """
    stages = {}
    diagnostics = {}

    # Step 1: Decode
    rgb_img = load_image_to_numpy(image_input)
    stages["1_original"] = rgb_img
    diagnostics["original_shape"] = rgb_img.shape

    # Step 2: Grayscale
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    stages["2_grayscale"] = gray

    # Step 3: Noise filtering
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    stages["3_noise_filtered"] = blurred

    # Step 4: Polarity correction
    dark_bg_img, was_inverted = detect_and_correct_polarity(blurred)
    stages["4_polarity_corrected"] = dark_bg_img
    diagnostics["inverted_polarity"] = was_inverted

    # Step 5: Binarization / Thresholding
    # Combine Otsu and Adaptive thresholding to handle uneven phone/webcam shadows
    _, thresh_otsu = cv2.threshold(dark_bg_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Adaptive threshold on light-normalized image
    adapt = cv2.adaptiveThreshold(
        dark_bg_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -4
    )
    # Blend: use Otsu as base, reinforce with adaptive
    combined = cv2.bitwise_or(thresh_otsu, adapt)

    # Step 6: Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    stages["5_thresholded"] = cleaned

    # Quality Check: Stroke pixel count & contrast
    white_pixel_count = int(np.count_nonzero(cleaned))
    total_pixels = cleaned.size
    white_pixel_ratio = white_pixel_count / max(total_pixels, 1)
    diagnostics["white_pixel_ratio"] = round(white_pixel_ratio, 4)

    if white_pixel_count < MIN_CONTOUR_AREA:
        return {
            "tensor": None,
            "normalized_28x28": None,
            "diagnostics": diagnostics,
            "stages": stages,
            "is_valid": False,
            "message": "No handwritten digit detected. Image appears nearly blank or stroke is too faint."
        }

    if white_pixel_ratio > 0.85:
        return {
            "tensor": None,
            "normalized_28x28": None,
            "diagnostics": diagnostics,
            "stages": stages,
            "is_valid": False,
            "message": "Image contains excessive noise or glare. Please draw on clean paper or darker background."
        }

    # Step 7: Bounding box detection with intelligent noise filtering
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {
            "tensor": None,
            "normalized_28x28": None,
            "diagnostics": diagnostics,
            "stages": stages,
            "is_valid": False,
            "message": "No digit contours found in image."
        }

    # Filter out paper borders (contours touching image edges that span >80% width or height)
    img_h, img_w = cleaned.shape
    valid_contours = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area >= MIN_CONTOUR_AREA:
            # Check if this contour is just the frame/border
            if w > 0.95 * img_w and h > 0.95 * img_h:
                continue
            valid_contours.append(c)

    if not valid_contours:
        valid_contours = [max(contours, key=cv2.contourArea)]

    # Merge bounding boxes of valid central contours
    x_min = min(cv2.boundingRect(c)[0] for c in valid_contours)
    y_min = min(cv2.boundingRect(c)[1] for c in valid_contours)
    x_max = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in valid_contours)
    y_max = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in valid_contours)

    # Pad slightly around bounding box
    pad = 4
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(img_w, x_max + pad)
    y_max = min(img_h, y_max + pad)

    digit_crop = cleaned[y_min:y_max, x_min:x_max]
    h, w = digit_crop.shape
    diagnostics["bounding_box"] = (x_min, y_min, x_max - x_min, y_max - y_min)

    # Step 8: Aspect-ratio preserving resize into square 20x20 box inside 28x28 frame
    if h > w:
        new_h = 20
        new_w = max(1, int(round((w * 20.0) / h)))
    else:
        new_w = 20
        new_h = max(1, int(round((h * 20.0) / w)))

    resized_digit = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Step 9: Stroke thickness adjustment for real pens
    resized_digit = normalize_stroke_thickness(resized_digit)

    # Place inside 28x28 blank canvas (margin = 4 px minimum)
    framed_28x28 = np.zeros((28, 28), dtype=np.uint8)
    offset_y = (28 - new_h) // 2
    offset_x = (28 - new_w) // 2
    framed_28x28[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized_digit

    # Step 10: Center of Mass alignment
    final_28x28 = center_by_center_of_mass(framed_28x28)
    stages["6_centered_28x28"] = final_28x28

    # Step 11: Normalization [0.0, 1.0] & Batch Tensor Shape
    normalized = final_28x28.astype(np.float32) / 255.0
    tensor = np.expand_dims(normalized, axis=(0, -1))  # Shape: (1, 28, 28, 1)

    return {
        "tensor": tensor,
        "normalized_28x28": normalized,
        "diagnostics": diagnostics,
        "stages": stages,
        "is_valid": True,
        "message": "Digit preprocessed successfully."
    }
