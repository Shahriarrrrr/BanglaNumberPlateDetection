#Video Implementation
from ultralytics import YOLO
import cv2
from sort import *
from banglaProcessing import *

# Initialize the YOLO models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('licensePlatemodel/best (1).pt')

cap = cv2.VideoCapture('assets/premiovid.mp4')
mot_tracker = Sort()

frame_nmr = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_nmr += 1

    # Detect vehicles
    detections = coco_model(frame)[0]
    detections_ = []

    # Filter out vehicles
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in [2, 3, 4, 5, 6, 7]:  # Vehicle classes
            detections_.append([x1, y1, x2, y2, score])

    # Track vehicles
    track_ids = mot_tracker.update(np.asarray(detections_))

    # Detect license plates
    license_plate_results = license_plate_detector(frame)[0]

    # Annotate the frame with license plate bounding boxes
    annotated_frame = license_plate_results.plot()

    # Process detected license plates
    plates_text = process_plate_region(frame, license_plate_results)

    # # Display results for the current frame
    # cv2.imshow("Annotated Frame", annotated_frame)


cap.release()
cv2.destroyAllWindows()









#Image Implementation Only

# from ultralytics import YOLO
# import cv2
# import matplotlib.pyplot as plt
# from banglaProcessing import process_plate_region
#
#
# # Load the YOLO model
# license_plate_detector = YOLO('licensePlatemodel/best (1).pt')
# coco_model = YOLO('yolov8n.pt')
#
#
#
# def main():
#
#     image_path = "assets/premio.jpeg"
#     image = cv2.imread(image_path)
#
#     if image is None:
#         print("Error: Could not read the image file.")
#         return
#
#     print("Processing image...")
#
#     results = license_plate_detector(image)
#     annotated_image = results[0].plot()
#     plates_text = process_plate_region(annotated_image, results)
#
#     # Display results
#     for i, plate in enumerate(plates_text, 1):
#         print(f"\nPlate {i}:")
#         print(f"Original Text: {plate['original_text']}")
#         print(f"Modified Text: {plate['modified_text']}")
#
#     output_path = "output_image_with_text.jpg"
#     cv2.imwrite(output_path, annotated_image)
#
#     print(f"\nProcessing complete. Output saved to {output_path}")
#
#     plt.figure(figsize=(12, 8))
#     plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.show()
