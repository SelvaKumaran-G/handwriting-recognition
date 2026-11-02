"""
DigitVision AI - CNN Architecture & Explainability Module
Defines the deep convolutional neural network for handwritten digit classification
and implements Grad-CAM (Gradient-weighted Class Activation Mapping).
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

from config import INPUT_SHAPE, NUM_CLASSES, INITIAL_LR


def build_digit_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES) -> keras.Model:
    """
    Constructs an optimized deep CNN architecture for robust handwritten digit recognition.
    
    Architecture:
      1. Input: (28, 28, 1)
      2. Conv2D: 32 filters, 3x3, same padding, ReLU
      3. Batch Normalization
      4. Conv2D: 32 filters, 3x3, valid padding, ReLU
      5. MaxPooling2D: 2x2
      6. Dropout: 0.25
      7. Conv2D: 64 filters, 3x3, same padding, ReLU
      8. Batch Normalization
      9. Conv2D: 64 filters, 3x3, valid padding, ReLU
      10. MaxPooling2D: 2x2
      11. Dropout: 0.25
      12. Flatten
      13. Dense: 128 units, ReLU
      14. Batch Normalization
      15. Dropout: 0.50
      16. Dense: 10 units, Softmax
    """
    inputs = keras.Input(shape=input_shape, name="digit_input")
    
    # Block 1
    x = layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", name="conv2d_1")(inputs)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Conv2D(32, kernel_size=(3, 3), padding="valid", activation="relu", name="conv2d_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_1")(x)
    x = layers.Dropout(0.25, name="dropout_1")(x)
    
    # Block 2
    x = layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", name="conv2d_3")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), padding="valid", activation="relu", name="conv2d_4")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_2")(x)
    x = layers.Dropout(0.25, name="dropout_2")(x)
    
    # Dense Classifier Head
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.BatchNormalization(name="bn_3")(x)
    x = layers.Dropout(0.50, name="dropout_head")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="digit_probabilities")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="DigitVision_CNN")
    return model


def compile_model(model: keras.Model, learning_rate: float = INITIAL_LR) -> keras.Model:
    """
    Compiles the model with Adam optimizer and Categorical Crossentropy.
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def get_last_conv_layer_name(model: keras.Model) -> str:
    """
    Finds the name of the last Conv2D layer in the model.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            return layer.name
    return "conv2d_4"


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: keras.Model,
    last_conv_layer_name: str = None,
    pred_index: int = None
) -> np.ndarray:
    """
    Computes the Grad-CAM (Gradient-weighted Class Activation Mapping) heatmap
    for visual explainability of the digit prediction.
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer_name(model)

    grad_model = keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def overlay_gradcam(
    original_img_28x28: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45
) -> np.ndarray:
    """
    Overlays the Grad-CAM heatmap onto the 28x28 grayscale image.
    Returns RGB image suitable for visualization.
    """
    import cv2

    # Rescale heatmap to 0-255
    heatmap_resized = cv2.resize(heatmap, (28, 28))
    heatmap_scaled = np.uint8(255 * heatmap_resized)

    # Colorize with JET colormap
    colormap = cv2.applyColorMap(heatmap_scaled, cv2.COLORMAP_JET)

    # Convert grayscale input to RGB
    if original_img_28x28.max() <= 1.0:
        base_img = np.uint8(255 * original_img_28x28)
    else:
        base_img = original_img_28x28.astype(np.uint8)

    if len(base_img.shape) == 2:
        base_rgb = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    elif base_img.shape[-1] == 1:
        base_rgb = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    else:
        base_rgb = base_img

    superimposed = cv2.addWeighted(colormap, alpha, base_rgb, 1 - alpha, 0)
    return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
