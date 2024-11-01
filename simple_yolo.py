import cv2
import torch

# Loads the YoloV5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

cap = cv2.VideoCapture(0)  # Access default camera

while True:
    
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5 predictions
    results = model(frame)
    detections = results.xyxy[0]

    people_count = 0
    
    # Get people and chair count
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)

        if cls == 0:  # Assuming class 0 is "person"
            people_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

    message = f"There are {people_count} people."
    
    # Display message
    cv2.putText(frame, message, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
