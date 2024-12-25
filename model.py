from ultralytics import YOLO
import cv2

# Load the YOLO model
license_plate_detector = YOLO('licensePlatemodel/best (1).pt')

# Path to your input image
image_path = "assets/deshiplate1.png"

# Read the input image
image = cv2.imread(image_path)

# Check if the image was successfully read
if image is None:
    print("Error: Could not read the image file.")
    exit()

print("Processing image...")

# Perform inference on the image
results = license_plate_detector(image)

# Annotate the image with detection results
annotated_image = results[0].plot()  # Annotated image with detections

# Path to save the output image
output_path = "output_image.jpg"

# Save the annotated image
cv2.imwrite(output_path, annotated_image)

print("Processing complete. Output saved to", output_path)
