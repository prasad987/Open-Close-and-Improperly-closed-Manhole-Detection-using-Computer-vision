import cv2
import numpy as np

# Load YOLOv3 model and weights
net = cv2.dnn.readNet("yolov3_manhole3_final.weights", "yolov3_manhole3.cfg")

# Load COCO class labels
classes = []
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Function to perform object detection and return detected objects
def detect_objects(image):
    blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    detected_objects = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Adjust confidence threshold as needed
                box = detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                (center_x, center_y, width, height) = box.astype("int")
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))
                detected_objects.append((classes[class_id], confidence, (x, y, int(width), int(height))))
    return detected_objects

# Sample image for testing
image = cv2.imread("test7.jpeg")

# Perform object detection
detected_objects = detect_objects(image)

# Print detected objects
for obj in detected_objects:
    print("Class:", obj[0], "- Confidence:", obj[1], "- Bounding Box:", obj[2])

# Calculate mAP (replace this with your actual mAP calculation)
mAP = 0.75  # Sample mAP value

print("Mean Average Precision (mAP):", mAP)
