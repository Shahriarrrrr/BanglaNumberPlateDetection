# from ultralytics import YOLO
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import imutils
#
# # Load the YOLO model
# license_plate_detector = YOLO('licensePlatemodel/best (1).pt')
#
# # Path to your input image
# image_path = "assets/chittagong.jpg"
#
# # Read the input image
# image = cv2.imread(image_path)
#
# # Check if the image was successfully read
# if image is None:
#     print("Error: Could not read the image file.")
#     exit()
#
# print("Processing image...")
#
# # Perform inference on the image
# results = license_plate_detector(image)
#
# # Annotate the image with detection results
# annotated_image = results[0].plot()  # Annotated image with detections
#
# # Path to save the output image
# output_path = "output_image.jpg"
#
# # Save the annotated image
# cv2.imwrite(output_path, annotated_image)
#
# # Plate Reading Implementation
#
# # Re-read the saved annotated image
# image = cv2.imread(output_path)
#
# if image is None:
#     raise FileNotFoundError("Error: Could not read the annotated image.")
#
# # Convert to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Apply bilateral filter for noise reduction
# filtered = cv2.bilateralFilter(gray, 11, 17, 17)
#
# # Edge detection
# edges = cv2.Canny(filtered, 30, 200)
#
# # Display edges (optional)
# plt.figure()
# plt.title("Edge Detection")
# plt.imshow(edges, cmap='gray')
# plt.axis("off")
# plt.show()
#
#
#
#
# print("Processing complete. Output saved to", output_path)
