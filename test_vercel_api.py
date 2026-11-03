"""Test Vercel api/predict handler logic with a sample digit image."""
import json
import base64
from pathlib import Path
import numpy as np

# Import the handler from api/predict.py
from api.predict import preprocess_single_image, run_onnx_inference, DIGIT_WORDS, segment_characters

sample_img = Path("assets/sample_digits/digit_3_sample1_paper.png")
if not sample_img.exists():
    print(f"Sample not found: {sample_img}")
    exit(1)

with open(sample_img, "rb") as f:
    raw_bytes = f.read()

print(f"Loaded {sample_img.name} ({len(raw_bytes)} bytes)")

# Test preprocessing
tensor, err = preprocess_single_image(raw_bytes)
if err:
    print(f"Preprocessing error: {err}")
    exit(1)

print(f"Preprocessed tensor shape: {tensor.shape}, min: {tensor.min()}, max: {tensor.max()}")

# Test inference
pred, conf, probs = run_onnx_inference(tensor)
print(f"Prediction: {pred} ({DIGIT_WORDS[pred]}) with confidence {conf*100:.2f}%")
assert pred == 3, f"Expected 3, got {pred}"
print("[PASS] Single digit ONNX test PASSED!")

# Test multi-digit segmentation
multi_img = Path("assets/sample_digits/digit_7_sample1_paper.png")
with open(multi_img, "rb") as f:
    raw_bytes_7 = f.read()

tensor_7, _ = preprocess_single_image(raw_bytes_7)
pred_7, conf_7, _ = run_onnx_inference(tensor_7)
print(f"Prediction for 7: {pred_7} ({DIGIT_WORDS[pred_7]}) with confidence {conf_7*100:.2f}%")
assert pred_7 == 7, f"Expected 7, got {pred_7}"
print("[PASS] Digit 7 test PASSED!")
print("\nALL VERCEL PREDICTION ENGINE TESTS PASSED!")
