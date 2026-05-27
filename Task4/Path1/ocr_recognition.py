import cv2
import pytesseract
import numpy as np
import os
import sys
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def load_image(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)
    image = cv2.imread(path)
    if image is None:
        print(f"Could not read image: {path}")
        sys.exit(1)
    print(f"Image loaded — Shape: {image.shape}")
    return image


def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    if len(coords) == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:
        return image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    print(f"Deskew angle corrected: {angle:.2f}°")
    return rotated


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    deskewed = deskew(thresh)
    return deskewed


def run_ocr(preprocessed, psm=6):
    config = f"--oem 3 --psm {psm}"
    pil_image = Image.fromarray(preprocessed)
    return pytesseract.image_to_string(pil_image, config=config).strip()


def get_confidence(preprocessed, psm=6):
    config = f"--oem 3 --psm {psm}"
    pil_image = Image.fromarray(preprocessed)
    data = pytesseract.image_to_data(pil_image, config=config, output_type=pytesseract.Output.DICT)
    confidences = [int(c) for c in data["conf"] if str(c).lstrip("-").isdigit() and int(c) != -1]
    if not confidences:
        return 0.0
    return round(sum(confidences) / len(confidences), 2)


def run_pipeline(image_path, psm=6):
    print("\n" + "=" * 50)
    print("  PROJECT 4 — OCR PIPELINE")
    print("=" * 50)

    image = load_image(image_path)
    preprocessed = preprocess_image(image)

    cv2.imwrite("ocr_preprocessed.jpg", preprocessed)
    print("Pre-processed image saved → ocr_preprocessed.jpg")

    text = run_ocr(preprocessed, psm=psm)
    confidence = get_confidence(preprocessed, psm=psm)

    print("\n--- Extracted Text ---")
    print(text if text else "No text detected.")
    print(f"\nConfidence Score: {confidence}%")

    if confidence >= 80:
        print("PASS — Meets the 80% threshold.")
    else:
        print("FAIL — Below 80%. Try a different PSM or a cleaner image.")

    print("=" * 50 + "\n")
    return text, confidence


if __name__ == "__main__":
    IMAGE_PATH = "sample_text.jpg"
    PSM_MODE = 6
    run_pipeline(IMAGE_PATH, psm=PSM_MODE)
