from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from banglaProcessing import process_plate_region


# Load the YOLO model
license_plate_detector = YOLO('licensePlatemodel/best (1).pt')




def main():
    image_path = "assets/premio.jpeg"
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not read the image file.")
        return

    print("Processing image...")

    results = license_plate_detector(image)
    annotated_image = results[0].plot()
    plates_text = process_plate_region(annotated_image, results)

    # Display results
    for i, plate in enumerate(plates_text, 1):
        print(f"\nPlate {i}:")
        print(f"Original Text: {plate['original_text']}")
        print(f"Modified Text: {plate['modified_text']}")

    output_path = "output_image_with_text.jpg"
    cv2.imwrite(output_path, annotated_image)

    print(f"\nProcessing complete. Output saved to {output_path}")

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()