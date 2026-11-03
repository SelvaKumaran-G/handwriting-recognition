import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/best_digit_model.keras")
weights = model.get_weights()
print(f"Total weight arrays: {len(weights)}")
for i, w in enumerate(weights):
    print(f"  [{i}] shape={w.shape}, size={w.size}")

np.savez_compressed("api/model_weights.npz", *weights)
np.savez_compressed("models/model_weights.npz", *weights)
size_kb = os.path.getsize("api/model_weights.npz") / 1024
print(f"\n[OK] Saved api/model_weights.npz: {size_kb:.1f} KB")
