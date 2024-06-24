import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")

# Load classes
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getUnconnectedOutLayersNames()

# Read and Process Image
img = cv2.imread("Brother.PNG")

# Obtain image dimensions
height, width, _ = img.shape

# Object Detection
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(layer_names)

# Process Output
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # Set your confidence threshold
            # Object detected, process it
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Calculate coordinates for the bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Draw bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display class label and confidence
            label = f"{classes[class_id]}: {confidence:.2f}"
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display Results
cv2.imshow("Object Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
