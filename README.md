# DigitVision AI – Intelligent Handwritten Digit & Text Recognition System

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.40%25-brightgreen)
![Frontend](https://img.shields.io/badge/Frontend-HTML5%20%2F%20CSS3%20%2F%20JS-blue)
![Backend](https://img.shields.io/badge/Backend-Starlette%20%2F%20Uvicorn-purple)
![Vercel](https://img.shields.io/badge/Deploy-Vercel-black?logo=vercel)
![License](https://img.shields.io/badge/License-MIT-green)

A production-grade, college final-year engineering project for **Intelligent Handwritten Digit, Multi-Digit & Word Recognition** powered by a Convolutional Neural Network (CNN) and Multi-Scale CRAFT Deep Vision OCR.

---

## 🚀 Key Highlights & Capabilities

- **🧠 4-Stage Deep CNN Architecture**:
  - Input: 28 × 28 × 1 (grayscale)
  - Conv2D (32/32/64/64 filters) + Batch Normalization + Spatial Dropout + Dense 128 + Softmax
  - **Empirical Measured Test Accuracy: 99.40%** (Test Loss: `0.0170` across 10,000 unseen test samples).
- **📝 Multi-Modal Text & Word Recognition**:
  - Integrated with **CRAFT Vision OCR** with adaptive CLAHE illumination normalization.
  - Accurately recognizes handwritten words and names (such as entries from `written_name_train_v2.csv`).
- **🔢 Multi-Digit Sequence Reconstruction**:
  - Character contour segmentation and left-to-right sorting (e.g. `42`, `2026`).
- **🎯 Explainable AI (Grad-CAM)**:
  - Visual class activation heatmaps overlaying convolutional attention regions.
- **🗣️ Interactive Web Speech Voice Assistant**:
  - Natural spoken announcements using the browser's HTML5 Speech Synthesis API.
- **💻 Responsive Pure HTML5/CSS3/JavaScript Frontend**:
  - Fluid auto-alignment across Mobile Phones, Tablets, Laptops, and Desktops.
  - **Unified 4-in-1 Input Console Dock**:
    1. 📁 **Upload Image** (with drag-and-drop & direct clipboard paste `Ctrl + V`).
    2. 📷 **Live Camera** (HTML5 WebRTC video stream with rear/front camera switcher).
    3. ✍️ **Draw Digit** (smooth touch/mouse drawing canvas with stroke width slider).
    4. 🧪 **Sample Test Digits** (10 verified realistic paper digit test samples).
  - High-tech **AI Neural Scanner** animated loading indicator.

---

## 📂 Project Directory Structure

```text
├── assets/
│   ├── logo.png                     # Official DigitVision AI project logo
│   └── sample_digits/               # 40 realistic digit test samples
├── data/
│   ├── mnist.npz                    # Benchmark MNIST digit dataset
│   └── prediction_history.csv       # Persistent inference history log
├── frontend/
│   ├── index.html                   # Semantic HTML5 single-page dashboard
│   ├── style.css                    # White & Electric Blue responsive theme
│   └── app.js                       # Vanilla JS touch canvas, camera & API engine
├── models/
│   ├── best_digit_model.keras       # Trained CNN weights (3.34 MB)
│   └── easyocr/                     # EasyOCR weights cache (.gitkeep)
├── outputs/
│   ├── confusion_matrix.png         # 10x10 empirical confusion matrix
│   ├── training_history.png         # Accuracy & loss learning curves
│   └── training_metrics.json        # Serialized evaluation metrics
├── config.py                        # Centralized hyperparameters & paths
├── model.py                         # CNN definition & Grad-CAM generator
├── multimodal_recognition.py        # EasyOCR + character contour segmenter
├── predict.py                       # Single/batch inference & explainability
├── preprocessing.py                 # Illumination, CoM alignment & distance transform
├── server.py                        # Starlette + Uvicorn REST backend
├── train_model.py                   # Data augmentation & model training script
├── test_system.py                   # Automated end-to-end integration test suite
├── written_name_train_v2.csv        # Dataset verification inspection CSV
├── written_name_test_v2.csv         # Dataset verification inspection CSV
└── requirements.txt                 # Project dependency list
```

---

## ⚡ Instant Vercel Deployment

DigitVision AI is optimized for serverless cloud deployment:

- **Ultra-lightweight bundle**: Replaces heavy TensorFlow/PyTorch with **ONNX Runtime** (~70MB total vs Vercel's 500MB limit).
- **Edge Static Serving**: HTML5/CSS3/JS UI served instantly via Vercel CDN from `/public`.
- **Serverless API**: Python function in `/api/predict.py` with ONNX CPU inference (<50ms per digit).

Simply connect your GitHub repository to [vercel.com](https://vercel.com) and click **Deploy**!

---

## 🛠️ Local Installation & Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/SelvaKumaran-G/handwriting-recognition.git
cd handwriting-recognition
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# For production serverless runtime:
pip install -r requirements.txt

# For full local development (TensorFlow training + EasyOCR):
pip install -r requirements-full.txt
```

pip install -r requirements.txt
```

### 3. Launch the Application
```bash
python server.py
```
Open your browser and navigate to:
```text
http://localhost:8501
```

---

## 🧪 Running System Verification Tests

Verify all model components, Grad-CAM overlays, history logs, and OCR pipelines:
```bash
python test_system.py
```

---

## 📊 Empirical Model Performance

| Metric | Measured Value |
| :--- | :--- |
| **Test Accuracy** | **99.40%** |
| **Validation Accuracy** | **99.43%** |
| **Test Loss** | **0.0170** |
| **Total Test Samples** | 10,000 (MNIST Benchmark) |
| **Optimizer** | Adam (lr=1e-3, ReduceLROnPlateau) |
| **Regularization** | Batch Normalization + Spatial Dropout (0.25, 0.50) |

---

## 📄 License
This project is open-source under the MIT License.
