# Import required libraries
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the pre-trained YOLOv8 model
model = YOLO('path_to_model_with_obstacle_classes.pt')  # Add the correct path

# Start video capture from webcam
cap = cv2.VideoCapture(0)

# Loop for real-time detection
while cap.isOpened():
    ret, frame = cap.read()  # Read frame from video
    
    if not ret:
        print("Failed to grab frame")
        break

    # Run object detection on the frame
    results = model(frame)
    
    # Loop through results and draw bounding boxes
    for r in results:
        for box in r.boxes:
            # Get box coordinates and class label
            x1, y1, x2, y2 = map(int, box.xyxy)
            label = model.names[int(box.cls)]  # Get label from class index

            # Set colors: blue for potholes, green for obstacles
            color = (0, 255, 0) if label == 'obstacle' else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the frame with detections
    cv2.imshow('Pothole and Obstacle Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
