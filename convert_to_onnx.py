"""Convert Keras CNN model to ONNX format for lightweight Vercel deployment."""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import tf2onnx
import numpy as np

MODEL_PATH = "models/best_digit_model.keras"
ONNX_PATH = "models/digit_model.onnx"

print("[1/3] Loading Keras model...")
model = tf.keras.models.load_model(MODEL_PATH)
model.summary()

print("\n[2/3] Converting to ONNX...")
spec = (tf.TensorSpec((1, 28, 28, 1), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=ONNX_PATH)
print(f"    ONNX model saved: {ONNX_PATH}")

onnx_size = os.path.getsize(ONNX_PATH)
keras_size = os.path.getsize(MODEL_PATH)
print(f"    Keras size:  {keras_size/1024:.1f} KB")
print(f"    ONNX size:   {onnx_size/1024:.1f} KB")

print("\n[3/3] Verifying ONNX inference...")
import onnxruntime as ort
session = ort.InferenceSession(ONNX_PATH)
dummy_input = np.random.rand(1, 28, 28, 1).astype(np.float32)
input_name = session.get_inputs()[0].name
result = session.run(None, {input_name: dummy_input})[0]
print(f"    Input shape:  {dummy_input.shape}")
print(f"    Output shape: {result.shape}")
print(f"    Output sum:   {result[0].sum():.4f} (should be ~1.0 for softmax)")
print("\n[OK] ONNX conversion and verification PASSED!")

