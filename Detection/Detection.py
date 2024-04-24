import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3_manhole3_best.weights', 'yolov3_manhole3.cfg')

classes = []
with open("classes.txt", "r") as f:
     classes = f.read().splitlines()

#cap = cv2.VideoCapture('1.mp4')
#cap = 'test_images/<your_test_image>.jpg'
font = cv2.FONT_HERSHEY_DUPLEX
colors = np.random.uniform(0, 255, size=(100, 3))

while True:
    #_, img = cap.read()
    img = cv2.imread("test1.jpg")
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[4]*416)
                h = int(detection[4]*416)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.9)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            Con = int(confidences[i] * 100)
            confidence = str(round( Con ,2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence +"%", (x, y+20), font, 1, (255,255,255), 3)


            
# if its output overlap boxes use Max_conf_index instead of for loop

##    if len(indexes)>0:
##        max_conf_index = np.argmax(confidences)
##        x, y, w, h = boxes[max_conf_index]
##        label = str(classes[class_ids[max_conf_index]])
##        Con = int(confidences[max_conf_index] * 100)
##        confidence = str(round( Con ,2))
##        color = colors[max_conf_index]
##        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
##        cv2.putText(img, label + " " + confidence +"%", (x, y+20), font, 1, (255,255,255), 3)



        cv2.imshow('Image', img)
        key = cv2.waitKey(1)
        if key==27:
            break

#cap.release()
cv2.destroyAllWindows()