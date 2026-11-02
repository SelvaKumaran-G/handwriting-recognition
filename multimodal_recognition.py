"""
DigitVision AI - Multi-Modal Text, Word & Multi-Digit Recognition Engine
Performs intelligent character segmentation, multi-digit sequence reconstruction,
visual bounding-box localization, and deep OCR for handwritten words and text.
"""

from typing import Dict, Any, List, Tuple
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import cv2
from PIL import Image

from config import TARGET_SIZE, MIN_CONTOUR_AREA, DIGIT_WORDS
from preprocessing import load_image_to_numpy, detect_and_correct_polarity, normalize_stroke_thickness, center_by_center_of_mass
from model import make_gradcam_heatmap, overlay_gradcam

# Global EasyOCR reader cache
_EASYOCR_READER = None


def get_ocr_reader():
    """Lazily loads EasyOCR reader with self-contained workspace storage."""
    global _EASYOCR_READER
    if _EASYOCR_READER is not None:
        return _EASYOCR_READER
    try:
        import easyocr
        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "easyocr"))
        user_network_dir = os.path.join(model_dir, "user_network")
        os.makedirs(user_network_dir, exist_ok=True)
        os.environ["EASYOCR_MODULE_PATH"] = model_dir

        craft_model = os.path.join(model_dir, "craft_mlt_25k.pth")
        rec_model = os.path.join(model_dir, "english_g2.pth")

        has_models = os.path.exists(craft_model) and os.path.exists(rec_model)
        _EASYOCR_READER = easyocr.Reader(
            ["en"],
            gpu=False,
            model_storage_directory=model_dir,
            user_network_directory=user_network_dir,
            download_enabled=(not has_models),
            verbose=False
        )
        return _EASYOCR_READER
    except Exception as e:
        print(f"EasyOCR fallback note: {e}")
        return None


def enhance_image_contrast_for_ocr(rgb_img: np.ndarray) -> np.ndarray:
    """Applies CLAHE illumination and contrast equalization for deep handwriting OCR."""
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
    return enhanced


def calculate_box_iou(boxA, boxB) -> float:
    """Calculates Intersection over Union between two quadrilateral bounding boxes."""
    ptsA = np.array(boxA, dtype=np.float32)
    ptsB = np.array(boxB, dtype=np.float32)
    xA1, yA1 = np.min(ptsA[:, 0]), np.min(ptsA[:, 1])
    xA2, yA2 = np.max(ptsA[:, 0]), np.max(ptsA[:, 1])
    xB1, yB1 = np.min(ptsB[:, 0]), np.min(ptsB[:, 1])
    xB2, yB2 = np.max(ptsB[:, 0]), np.max(ptsB[:, 1])

    x_inter1 = max(xA1, xB1)
    y_inter1 = max(yA1, yB1)
    x_inter2 = min(xA2, xB2)
    y_inter2 = min(yA2, yB2)

    inter_w = max(0.0, x_inter2 - x_inter1)
    inter_h = max(0.0, y_inter2 - y_inter1)
    inter_area = inter_w * inter_h

    areaA = max(1.0, (xA2 - xA1) * (yA2 - yA1))
    areaB = max(1.0, (xB2 - xB1) * (yB2 - yB1))
    union_area = areaA + areaB - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def sort_blocks_reading_order(blocks: List[Dict[str, Any]], line_threshold: int = 24) -> List[Dict[str, Any]]:
    """Sorts text blocks in natural reading order (top-to-bottom lines, left-to-right words)."""
    if not blocks:
        return []

    def get_coords(b):
        pts = np.array(b["bbox"])
        return np.min(pts[:, 1]), np.min(pts[:, 0])

    sorted_y = sorted(blocks, key=lambda b: get_coords(b)[0])
    lines = []
    current_line = [sorted_y[0]]

    for b in sorted_y[1:]:
        prev_y = get_coords(current_line[-1])[0]
        curr_y = get_coords(b)[0]
        if abs(curr_y - prev_y) <= line_threshold:
            current_line.append(b)
        else:
            lines.append(sorted(current_line, key=lambda x: get_coords(x)[1]))
            current_line = [b]
    if current_line:
        lines.append(sorted(current_line, key=lambda x: get_coords(x)[1]))

    ordered = []
    for line in lines:
        ordered.extend(line)
    return ordered


def segment_character_contours(binary_img: np.ndarray, min_area: int = 40) -> List[Tuple[int, int, int, int]]:
    """
    Finds and isolates individual character bounding boxes.
    Merges overlapping strokes and sorts them left-to-right, top-to-bottom.
    """
    img_h, img_w = binary_img.shape
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)

        # Ignore tiny noise or giant border frames
        if area >= min_area and w < 0.96 * img_w and h < 0.96 * img_h:
            # Aspect ratio and size filter
            if w > 3 and h > 6:
                valid_boxes.append((x, y, w, h))

    if not valid_boxes:
        return []

    # Merge boxes that horizontally overlap or are very close (e.g. multi-stroke digits like '=' or '4')
    merged_boxes = []
    sorted_boxes = sorted(valid_boxes, key=lambda b: b[0])

    for box in sorted_boxes:
        if not merged_boxes:
            merged_boxes.append(box)
            continue

        prev_x, prev_y, prev_w, prev_h = merged_boxes[-1]
        cur_x, cur_y, cur_w, cur_h = box

        x_overlap = (cur_x < prev_x + prev_w + 3) and (cur_x + cur_w > prev_x - 3)
        y_overlap = (max(prev_y, cur_y) < min(prev_y + prev_h, cur_y + cur_h) + 6)

        if x_overlap and y_overlap:
            new_x = min(prev_x, cur_x)
            new_y = min(prev_y, cur_y)
            new_w = max(prev_x + prev_w, cur_x + cur_w) - new_x
            new_h = max(prev_y + prev_h, cur_y + cur_h) - new_y
            merged_boxes[-1] = (new_x, new_y, new_w, new_h)
        else:
            merged_boxes.append(box)

    merged_boxes = sorted(merged_boxes, key=lambda b: (b[1] // 40, b[0]))
    return merged_boxes


def preprocess_single_crop(crop_img: np.ndarray) -> np.ndarray:
    """
    Formats a cropped character/digit into a standardized 28x28 MNIST tensor.
    """
    h, w = crop_img.shape
    if h > w:
        new_h = 20
        new_w = max(1, int(round((w * 20.0) / h)))
    else:
        new_w = 20
        new_h = max(1, int(round((h * 20.0) / w)))

    resized = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized = normalize_stroke_thickness(resized)

    framed = np.zeros((28, 28), dtype=np.uint8)
    oy = (28 - new_h) // 2
    ox = (28 - new_w) // 2
    framed[oy:oy + new_h, ox:ox + new_w] = resized

    centered = center_by_center_of_mass(framed)
    norm = centered.astype(np.float32) / 255.0
    return norm


def analyze_image_multi_modal(
    image_input,
    cnn_model=None,
    use_ocr: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive multi-modal analyzer:
      1. Detects both handwritten text/words AND numerical digits
      2. Segments individual characters and passes digits to CNN
      3. Performs OCR for full words, names, and sentences
      4. Draws visual bounding boxes directly on the input image
      5. Correctly categorizes and returns recognized text and numbers
    """
    rgb_img = load_image_to_numpy(image_input)
    annotated_img = rgb_img.copy()
    img_h, img_w, _ = rgb_img.shape

    # Step 1: Deep OCR Analysis (Words, Text, Alphanumeric Sequences)
    ocr_text_blocks = []
    full_ocr_string = ""
    if use_ocr:
        try:
            reader = get_ocr_reader()
            if reader is not None:
                # Pass 1: Standard high-sensitivity deep read
                pass1_results = reader.readtext(
                    rgb_img,
                    text_threshold=0.20,
                    low_text=0.20,
                    link_threshold=0.25,
                    slope_ths=0.35,
                    contrast_ths=0.05,
                    adjust_contrast=0.7,
                    mag_ratio=1.5
                )

                raw_blocks = []
                for bbox, text, conf in pass1_results:
                    clean_text = text.strip()
                    if conf > 0.12 and len(clean_text) > 0:
                        raw_blocks.append({
                            "text": clean_text,
                            "confidence": float(conf),
                            "confidence_pct": round(float(conf) * 100, 1),
                            "bbox": bbox
                        })

                # Pass 2: Deep CLAHE contrast & illumination equalization (catches faint strokes/words)
                if len(raw_blocks) < 3:
                    enhanced_rgb = enhance_image_contrast_for_ocr(rgb_img)
                    pass2_results = reader.readtext(
                        enhanced_rgb,
                        text_threshold=0.18,
                        low_text=0.18,
                        link_threshold=0.25,
                        slope_ths=0.35,
                        contrast_ths=0.05,
                        adjust_contrast=0.7,
                        mag_ratio=1.8
                    )
                    for bbox, text, conf in pass2_results:
                        clean_text = text.strip()
                        if conf > 0.14 and len(clean_text) > 0:
                            is_dup = any(calculate_box_iou(bbox, b["bbox"]) > 0.35 for b in raw_blocks)
                            if not is_dup:
                                raw_blocks.append({
                                    "text": clean_text,
                                    "confidence": float(conf),
                                    "confidence_pct": round(float(conf) * 100, 1),
                                    "bbox": bbox
                                })

                # Sort blocks in natural reading order
                ocr_text_blocks = sort_blocks_reading_order(raw_blocks)
                full_ocr_string = " ".join([b["text"] for b in ocr_text_blocks])

                # Draw high-contrast bounding boxes with filled badges on annotated_img
                for b in ocr_text_blocks:
                    pts = np.array(b["bbox"], dtype=np.int32)
                    cv2.polylines(annotated_img, [pts], isClosed=True, color=(37, 99, 235), thickness=2)
                    lbl_x = int(pts[0][0])
                    lbl_y = max(18, int(pts[0][1]) - 6)
                    label_text = f"{b['text']} ({b['confidence_pct']:.0f}%)"
                    (w_txt, h_txt), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(annotated_img, (lbl_x, lbl_y - h_txt - 4), (lbl_x + w_txt + 4, lbl_y + 2), (37, 99, 235), -1)
                    cv2.putText(annotated_img, label_text, (lbl_x + 2, lbl_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception as e:
            print(f"Deep OCR execution note: {e}")

    # Step 2: Grayscale & Polarity for Contour Digit Segmentation
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    dark_bg, _ = detect_and_correct_polarity(blurred)

    # Thresholding
    _, thresh_otsu = cv2.threshold(dark_bg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adapt = cv2.adaptiveThreshold(dark_bg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -4)
    cleaned = cv2.bitwise_or(thresh_otsu, adapt)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    # Segment individual character contours
    char_boxes = segment_character_contours(cleaned)
    char_results = []

    # If OCR detected clear alphabetic text (non-digit words), don't force digit CNN on every letter
    has_text_words = any(not b["text"].replace(" ", "").isdigit() for b in ocr_text_blocks)

    if cnn_model is not None and char_boxes and not has_text_words:
        for idx, (x, y, w, h) in enumerate(char_boxes):
            pad = 2
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img_w, x + w + pad)
            y2 = min(img_h, y + h + pad)

            char_crop = cleaned[y1:y2, x1:x2]
            if char_crop.size == 0 or np.count_nonzero(char_crop) < 15:
                continue

            norm_28x28 = preprocess_single_crop(char_crop)
            tensor = np.expand_dims(norm_28x28, axis=(0, -1))

            probs = cnn_model.predict(tensor, verbose=0)[0]
            pred_digit = int(np.argmax(probs))
            conf = float(probs[pred_digit])

            # Grad-CAM heatmap
            gradcam_img = None
            try:
                heatmap = make_gradcam_heatmap(tensor, cnn_model, pred_index=pred_digit)
                gradcam_img = overlay_gradcam(norm_28x28, heatmap)
            except Exception:
                pass

            char_results.append({
                "char_index": idx + 1,
                "box": (x, y, w, h),
                "predicted_digit": pred_digit,
                "digit_word": DIGIT_WORDS[pred_digit],
                "confidence": conf,
                "confidence_pct": round(conf * 100, 1),
                "probabilities": [float(p) for p in probs],
                "norm_28x28": norm_28x28,
                "gradcam": gradcam_img
            })

            # Draw green bounding box & label on annotated image
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (5, 150, 105), 2)
            cv2.putText(
                annotated_img,
                f"Digit {pred_digit} ({conf*100:.0f}%)",
                (x, max(14, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (5, 150, 105),
                2,
                cv2.LINE_AA
            )

    # Determine reconstructed number & primary output
    if char_results:
        reconstructed_number = "".join([str(c["predicted_digit"]) for c in char_results])
        avg_char_conf = float(np.mean([c["confidence"] for c in char_results]))
    else:
        reconstructed_number = ""
        avg_char_conf = 0.0

    # Categorize final detected content
    if full_ocr_string:
        primary_detected = full_ocr_string
        content_type = "Handwritten / Printed Text Word" if has_text_words else "Detected Numerical Sequence"
    elif reconstructed_number:
        primary_detected = reconstructed_number
        content_type = "Handwritten Digit Sequence"
    else:
        primary_detected = "No clear text or numbers detected"
        content_type = "Uncertain"

    return {
        "success": bool(char_results or ocr_text_blocks),
        "primary_detected": primary_detected,
        "content_type": content_type,
        "is_text_word": has_text_words,
        "reconstructed_number": reconstructed_number,
        "char_results": char_results,
        "ocr_blocks": ocr_text_blocks,
        "full_ocr_string": full_ocr_string,
        "avg_confidence": avg_char_conf,
        "annotated_image": annotated_img,
        "total_characters_found": len(char_results),
        "total_text_blocks_found": len(ocr_text_blocks)
    }
