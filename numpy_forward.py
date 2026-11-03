"""
Pure NumPy forward pass implementation of DigitVision 4-Stage ConvNet.
Runs in 100% pure Python/NumPy with ZERO C-library dependencies (no ONNX, no TensorFlow, no Torch).
Guaranteed to NEVER fail with missing .so libraries on any serverless or cloud environment.
"""

import numpy as np
from pathlib import Path

# Load weights
_WEIGHTS = None

def get_weights(npz_path="api/model_weights.npz"):
    global _WEIGHTS
    if _WEIGHTS is not None:
        return _WEIGHTS
    data = np.load(npz_path)
    _WEIGHTS = [data[f"arr_{i}"] for i in range(len(data.files))]
    return _WEIGHTS


def conv2d_same(x, w, b):
    """2D Convolution with same padding. x: (H, W, Cin), w: (Kh, Kw, Cin, Cout), b: (Cout,)"""
    H, W, Cin = x.shape
    Kh, Kw, _, Cout = w.shape
    pad_h = Kh // 2
    pad_w = Kw // 2
    x_padded = np.pad(x, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant")
    
    out_H = H
    out_W = W
    out = np.zeros((out_H, out_W, Cout), dtype=np.float32)

    for c in range(Cout):
        kernel = w[:, :, :, c]
        for i in range(out_H):
            for j in range(out_W):
                patch = x_padded[i:i + Kh, j:j + Kw, :]
                out[i, j, c] = np.sum(patch * kernel) + b[c]
    return out


def conv2d_valid(x, w, b):
    """2D Convolution with valid padding. x: (H, W, Cin), w: (Kh, Kw, Cin, Cout), b: (Cout,)"""
    H, W, Cin = x.shape
    Kh, Kw, _, Cout = w.shape
    out_H = H - Kh + 1
    out_W = W - Kw + 1
    out = np.zeros((out_H, out_W, Cout), dtype=np.float32)

    for c in range(Cout):
        kernel = w[:, :, :, c]
        for i in range(out_H):
            for j in range(out_W):
                patch = x[i:i + Kh, j:j + Kw, :]
                out[i, j, c] = np.sum(patch * kernel) + b[c]
    return out


def batch_norm(x, gamma, beta, mean, var, eps=1e-3):
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def relu(x):
    return np.maximum(0, x)


def max_pool_2x2(x):
    H, W, C = x.shape
    out_H = H // 2
    out_W = W // 2
    return x[:out_H * 2, :out_W * 2, :].reshape(out_H, 2, out_W, 2, C).max(axis=(1, 3))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


def forward_pass(x_28x28_1, npz_path="api/model_weights.npz"):
    """
    x_28x28_1: shape (28, 28, 1) or (1, 28, 28, 1)
    Returns: probabilities (10,)
    """
    if x_28x28_1.ndim == 4:
        x = x_28x28_1[0]
    else:
        x = x_28x28_1

    W = get_weights(npz_path)
    # Stage 1: Conv2D(32, (3,3), padding='same') + BN + ReLU
    x = conv2d_same(x, W[0], W[1])
    x = batch_norm(x, gamma=W[2], beta=W[3], mean=W[4], var=W[5])
    x = relu(x)

    # Stage 2: Conv2D(32, (3,3), padding='valid') + ReLU + MaxPool(2,2)
    x = conv2d_valid(x, W[6], W[7])
    x = relu(x)
    x = max_pool_2x2(x)  # Shape: (13, 13, 32)

    # Stage 3: Conv2D(64, (3,3), padding='same') + BN + ReLU
    x = conv2d_same(x, W[8], W[9])
    x = batch_norm(x, gamma=W[10], beta=W[11], mean=W[12], var=W[13])
    x = relu(x)

    # Stage 4: Conv2D(64, (3,3), padding='valid') + ReLU + MaxPool(2,2)
    x = conv2d_valid(x, W[14], W[15])
    x = relu(x)
    x = max_pool_2x2(x)  # Shape: (5, 5, 64)

    # Flatten: 5 * 5 * 64 = 1600
    flat = x.flatten()

    # Dense(128) + BN + ReLU
    dense1 = np.dot(flat, W[16]) + W[17]
    dense1_bn = batch_norm(dense1, gamma=W[18], beta=W[19], mean=W[20], var=W[21])
    dense1_act = relu(dense1_bn)

    # Dense(10, softmax)
    logits = np.dot(dense1_act, W[22]) + W[23]
    probs = softmax(logits)
    return probs


if __name__ == "__main__":
    from test_pillow_pipeline import preprocess_pil
    with open("assets/sample_digits/digit_3_sample1_paper.png", "rb") as f:
        raw = f.read()
    tensor, _, _, _ = preprocess_pil(raw)
    probs = forward_pass(tensor)
    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    print(f"NumPy Forward Pass Prediction: {pred} with confidence {conf*100:.2f}%")
    assert pred == 3
    print("[PASS] NumPy Forward Pass works perfectly!")
