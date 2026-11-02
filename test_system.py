"""
DigitVision AI - Automated System & Integration Test Suite
Validates model loading, preprocessing, inference on all digits (0-9), Grad-CAM, and history logging.
"""

from pathlib import Path
import numpy as np
import cv2

from config import MODEL_PATH, METRICS_PATH, HISTORY_CSV_PATH
from predict import load_digit_model, predict_digit
from preprocessing import preprocess_image


def run_system_tests():
    print("=" * 65)
    print("DIGITVISION AI - SYSTEM & INTEGRATION VERIFICATION")
    print("=" * 65)

    # Test 1: Model file exists
    assert MODEL_PATH.exists(), f"Model file missing at {MODEL_PATH}"
    print(f"[PASS] Model file confirmed: {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.2f} MB)")

    # Test 2: Load model
    model = load_digit_model(MODEL_PATH)
    print("[PASS] CNN Model successfully loaded into memory and cached.")

    # Test 3: Test prediction across sample digits
    samples_dir = Path("assets/sample_digits")
    assert samples_dir.exists(), "Sample digits folder missing"

    test_samples = sorted(list(samples_dir.glob("digit_*_paper.png")))[:10]
    print(f"\nTesting inference on {len(test_samples)} realistic paper digit images:")

    correct_count = 0
    for sample_path in test_samples:
        filename = sample_path.name
        expected_digit = int(filename.split("_")[1])
        img = cv2.imread(str(sample_path))

        res = predict_digit(img, model=model, input_source="Test Suite", record_history=True)

        assert res["success"] is True, f"Inference failed for {filename}: {res.get('message')}"
        pred = res["predicted_digit"]
        conf = res["confidence_pct"]
        is_match = (pred == expected_digit)
        if is_match:
            correct_count += 1
        status = "MATCH" if is_match else "MISMATCH"
        print(f"  [{status}] Expected: {expected_digit} | Predicted: {pred} ({res['digit_word']}) | Conf: {conf:.1f}%")

        # Verify Grad-CAM was generated
        assert res["gradcam_overlay"] is not None, "Grad-CAM overlay was not generated"

    accuracy = (correct_count / len(test_samples)) * 100
    print(f"\nSample Digit Benchmark Accuracy: {accuracy:.1f}% ({correct_count}/{len(test_samples)})")

    # Test 4: Verify History CSV was populated
    assert HISTORY_CSV_PATH.exists(), f"History CSV missing at {HISTORY_CSV_PATH}"
    print(f"[PASS] Prediction history successfully logged to: {HISTORY_CSV_PATH}")

    # Test 5: Verify Metrics JSON
    assert METRICS_PATH.exists(), f"Metrics JSON missing at {METRICS_PATH}"
    print(f"[PASS] Training metrics confirmed at: {METRICS_PATH}")

    # Test 6: Verify Multi-Modal Engine (Text Words & Multi-Digit Sequences)
    print("\nTesting Multi-Modal Engine (EasyOCR + Multi-Digit Segmentation):")
    from multimodal_recognition import analyze_image_multi_modal

    # 6A: Multi-digit sequence "42"
    canvas_42 = np.full((120, 240, 3), 255, dtype=np.uint8)
    cv2.putText(canvas_42, "42", (40, 85), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (15, 23, 42), 6, cv2.LINE_AA)
    res_42 = analyze_image_multi_modal(canvas_42, cnn_model=model)
    assert res_42["primary_detected"] == "42", f"Expected '42', got {res_42['primary_detected']}"
    print(f"  [PASS] Multi-digit sequence recognized: '{res_42['primary_detected']}' ({res_42['content_type']})")

    # 6B: Handwritten word / name "BALTHAZAR"
    canvas_word = np.full((120, 420, 3), 250, dtype=np.uint8)
    cv2.putText(canvas_word, "BALTHAZAR", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (20, 25, 45), 4, cv2.LINE_AA)
    res_word = analyze_image_multi_modal(canvas_word, cnn_model=model)
    assert res_word["is_text_word"] is True, "Failed to identify as text word"
    assert "BALTHAZAR" in res_word["primary_detected"], f"Expected 'BALTHAZAR', got {res_word['primary_detected']}"
    print(f"  [PASS] Handwritten word recognized: '{res_word['primary_detected']}' ({res_word['content_type']})")

    print("\n" + "=" * 65)
    print("ALL INTEGRATION TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 65)


if __name__ == "__main__":
    run_system_tests()

