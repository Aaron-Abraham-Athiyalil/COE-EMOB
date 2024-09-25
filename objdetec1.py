# Import required libraries
from ultralytics import YOLO
import cv2
import numpy as np

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # Add the correct path to your model

# Start video capture from webcam
cap = cv2.VideoCapture(0)

# Define a function to calculate distance (simple method)
def calculate_distance(width_in_feet, focal_length, width_in_pixels):
    """Calculate the distance to an object based on its width in feet and width in pixels."""
    return (width_in_feet * focal_length) / width_in_pixels

# Assume a known focal length (you may need to calibrate this for your camera)
focal_length = 800  # Example focal length in pixels (adjust as necessary)
object_width = 2.0  # Average width of the object (in feet, for example)

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
            # Convert box.xyxy to a numpy array
            xyxy = box.xyxy.cpu().numpy()  # Ensure it's on CPU and convert to numpy
            
            # Check if xyxy is a 2D array and extract the coordinates
            if xyxy.ndim == 2:  # If it's a 2D array
                for bbox in xyxy:  # Iterate through each bounding box
                    x1, y1, x2, y2 = map(int, bbox)  # Convert to integers
                    label = model.names[int(box.cls)]  # Get label from class index

                    # Set colors: blue for potholes, green for obstacles
                    color = (0, 255, 0) if label == 'obstacle' else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    # Calculate the width of the detected object in pixels
                    width_in_pixels = x2 - x1

                    # Calculate the distance to the object
                    distance = calculate_distance(object_width, focal_length, width_in_pixels)

                    # Display distance on the video frame
                    cv2.putText(frame, f'Distance: {distance:.2f} ft', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            else:  # If it's a single bounding box (1D array)
                x1, y1, x2, y2 = map(int, xyxy)  # Convert to integers
                label = model.names[int(box.cls)]  # Get label from class index

                # Set colors: blue for potholes, green for obstacles
                color = (0, 255, 0) if label == 'obstacle' else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Calculate the width of the detected object in pixels
                width_in_pixels = x2 - x1

                # Calculate the distance to the object
                distance = calculate_distance(object_width, focal_length, width_in_pixels)

                # Display distance on the video frame
                cv2.putText(frame, f'Distance: {distance:.2f} ft', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display the frame with detections
    cv2.imshow('Pothole and Obstacle Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
