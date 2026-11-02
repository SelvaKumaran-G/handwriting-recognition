"""
DigitVision AI - Sample Handwritten Digit Generator
Exports representative sample test digit images (0-9) to assets/sample_digits/
for viva demonstration and instant testing using local data/mnist.npz.
"""

from pathlib import Path
import numpy as np
import cv2

SAMPLES_DIR = Path("assets/sample_digits")
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = Path("data/mnist.npz")


def export_samples():
    if not DATA_PATH.exists():
        print(f"Data file not found at {DATA_PATH}")
        return

    with np.load(str(DATA_PATH)) as data:
        x_test = data["x_test"]
        y_test = data["y_test"]

    saved_counts = {d: 0 for d in range(10)}
    for img, label in zip(x_test, y_test):
        if saved_counts[label] < 2:
            idx = saved_counts[label] + 1

            # 1. Standard dark background version
            path_dark = SAMPLES_DIR / f"digit_{label}_sample{idx}_dark.png"
            cv2.imwrite(str(path_dark), img)

            # 2. Realistic paper version (dark ink on white paper, realistic noise)
            paper = np.ones((200, 200), dtype=np.uint8) * 248
            noise = np.random.normal(0, 4, (200, 200)).astype(np.int16)
            paper = np.clip(paper + noise, 0, 255).astype(np.uint8)

            inv_digit = 255 - img
            resized = cv2.resize(inv_digit, (120, 120), interpolation=cv2.INTER_CUBIC)
            paper[40:160, 40:160] = np.minimum(paper[40:160, 40:160], resized)

            path_paper = SAMPLES_DIR / f"digit_{label}_sample{idx}_paper.png"
            cv2.imwrite(str(path_paper), paper)

            saved_counts[label] += 1

        if all(count >= 2 for count in saved_counts.values()):
            break

    total_exported = len(list(SAMPLES_DIR.glob('*.png')))
    print(f"Successfully exported {total_exported} sample test digits to {SAMPLES_DIR}")


if __name__ == "__main__":
    export_samples()
