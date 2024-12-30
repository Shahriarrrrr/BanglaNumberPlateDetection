from ultralytics import YOLO
import cv2
from sort import *
from secondBanglaProcess import *
import numpy as np


#Translator
import cv2

# Replace with the URL shown in the DroidCam app
DROIDCAM_URL = "http://192.168.0.107:4747/video"

# def process_droidcam_feed():
#     # Open the video stream from DroidCam
#     cap = cv2.VideoCapture(DROIDCAM_URL)
#
#     if not cap.isOpened():
#         print("Error: Unable to access the DroidCam video stream.")
#         return
#
#     print("Press 'q' to quit.")
#
#     while True:
#         # Read a frame from the video stream
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to retrieve frame.")
#             break
#
#         # Display the video feed
#         #cv2.imshow("DroidCam Feed", frame)
#
#         # Quit on 'q' key press
#         # if cv2.waitKey(1) & 0xFF == ord('q'):
#         #     break
#
#     # Release resources and close windows
#     cap.release()
#     cv2.destroyAllWindows()
#
# if __name__ == "__main__":
#     process_droidcam_feed()



# Load YOLO models
#coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('licensePlatemodel/best (1).pt')

# Initialize video capture
cap = cv2.VideoCapture(DROIDCAM_URL)

# Check if video file is opened successfully
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

# Get video properties for output
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video writer
output_video = cv2.VideoWriter(
    'output/video/nHighway4.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

# Initialize SORT tracker
#mot_tracker = Sort()

# Skip logic
#skip_frames = int(fps * 2)  # Skip 2 seconds' worth of frames
frame_nmr = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Finished processing video.")
        break
    frame_nmr += 1

    # Vehicle detection
    # detections = coco_model(frame)[0]
    # detections_ = []

    # Filter and draw vehicle detections
    # for detection in detections.boxes.data.tolist():
    #     x1, y1, x2, y2, score, class_id = detection
    #     if int(class_id) in [2, 3, 4, 5, 6, 7]:  # Vehicle classes
    #         detections_.append([x1, y1, x2, y2, score])
    #         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    # Update tracker
    # if len(detections_) > 0:
    #     track_ids = mot_tracker.update(np.asarray(detections_))
    # else:
    #     track_ids = np.empty((0, 5))
    # License plate detection and processing
    license_plates = license_plate_detector(frame)[0]
    if len(license_plates.boxes) > 0:
        plates_text = process_plate_region(frame, license_plates)
        for plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = plate
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Write processed frame to output video
    output_video.write(frame)

# Release resources
cap.release()
output_video.release()
#cv2.destroyAllWindows()