import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load image
image = cv2.imread("cars.jpg")

# Preprocess image
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)

# Run inference
outs = net.forward(net.getUnconnectedOutLayersNames())

# Post-process results
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Process the bounding box coordinates and draw them on the image
            # ...
        
        

cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
