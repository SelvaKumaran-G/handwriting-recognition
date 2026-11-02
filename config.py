"""
DigitVision AI - Configuration Module
Defines system paths, model hyperparameters, UI styling parameters, and constants.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
COMPONENTS_DIR = BASE_DIR / "components"

# Ensure directories exist
for directory in [MODELS_DIR, ASSETS_DIR, OUTPUTS_DIR, DATA_DIR, COMPONENTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model Artifacts
MODEL_PATH = MODELS_DIR / "best_digit_model.keras"
METRICS_PATH = OUTPUTS_DIR / "training_metrics.json"
TRAINING_PLOT_PATH = OUTPUTS_DIR / "training_history.png"
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "confusion_matrix.png"
HISTORY_CSV_PATH = DATA_DIR / "prediction_history.csv"

# Model Hyperparameters
INPUT_SHAPE = (28, 28, 1)
TARGET_SIZE = (28, 28)
NUM_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 18
INITIAL_LR = 1e-3
MIN_LR = 1e-5
RANDOM_SEED = 42

# Confidence & Quality Thresholds
CONFIDENCE_HIGH_THRESHOLD = 0.85
CONFIDENCE_LOW_THRESHOLD = 0.50
MIN_CONTOUR_AREA = 25  # Minimum pixel area to consider as a valid digit stroke

# UI Theme & Colors
THEME_COLORS = {
    "primary": "#6366f1",         # Indigo
    "secondary": "#a855f7",       # Purple
    "accent_cyan": "#06b6d4",     # Cyan
    "accent_pink": "#ec4899",     # Pink
    "background_dark": "#0a0f1d", # Deep Dark Navy
    "card_dark": "#111827",       # Glassmorphism dark
    "card_border": "#1f293d",
    "text_main": "#f8fafc",
    "text_muted": "#94a3b8",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444"
}

# Digit Labels
DIGIT_LABELS = [str(i) for i in range(10)]
DIGIT_WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
