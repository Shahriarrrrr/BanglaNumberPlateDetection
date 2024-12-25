from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import easyocr
import re


def standardize_text(text):
    """Standardize the license plate text format."""
    try:
        # Split by space to separate city name from the rest
        parts = text.split(' ')

        if len(parts) >= 2:
            city_name = parts[0]  # First part (city name)
            number_part = parts[-1]  # Last part (numbers)

            # Get the letter after hyphen from the original text
            middle_part = parts[1] if len(parts) > 1 else ""
            letter_after_hyphen = ""
            if '-' in middle_part:
                letter_after_hyphen = middle_part.split('-')[-1]

            # Replace the third digit with a hyphen
            if len(number_part) >= 3:
                number_part = f"{number_part[:2]}-{number_part[3:]}"

            # Create standardized text with মেট্রো and original letter
            standardized = f"{city_name} মেট্রো-{letter_after_hyphen} {number_part}"
            return standardized
    except Exception as e:
        return f"Error processing text: {e}"

def process_plate_region(image, results):
    """Extract and read text from detected license plates"""
    reader = easyocr.Reader(['bn'])
    plates_text = []

    boxes = results[0].boxes

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_region = image[y1:y2, x1:x2]

        # Preprocess the plate region
        plate_region = cv2.resize(plate_region, None, fx=2, fy=2)
        gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced)

        # Read text using EasyOCR
        ocr_results = reader.readtext(denoised)

        if ocr_results:
            # Combine all detected text
            original_text = " ".join([detection[1] for detection in ocr_results])
            # Standardize the text format
            modified_text = standardize_text(original_text)

            # Draw text on the original image
            cv2.putText(image, modified_text,
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

            plates_text.append({
                'original_text': original_text,
                'modified_text': modified_text,
                'bbox': (x1, y1, x2, y2)
            })

            # Display the processed plate region (optional)
            plt.figure(figsize=(10, 5))
            plt.imshow(cv2.cvtColor(plate_region, cv2.COLOR_BGR2RGB))
            plt.title('Detected Plate')
            plt.axis('off')
            plt.show()

    return plates_text