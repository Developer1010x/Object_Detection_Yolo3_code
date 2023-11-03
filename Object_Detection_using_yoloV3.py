import cv2
import numpy as np

#this was done on MacOS  and Ubuntu it works fine without error !!!!! your have to enter the package location correctly in ''
#press Q for exiting after output comes 




yolo_cfg = '/home//darknet/cfg/yolov3.cfg'#location of Ubuntu
yolo_weights = '/home//darknet/yolov3.weights'#location of weights 
# Load the YOLO model and its configuration and weights files
net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)

# Load the COCO names file (contains class names)
with open('/home//darknet/data/coco.names', 'r') as f:#location of COCOnames
    classes = f.read().strip().split('\n')

# Create a VideoCapture object to access the default camera (usually index 0)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    layer_names = net.getUnconnectedOutLayersNames()

    # Get the height and width of the frame
    height, width, _ = frame.shape

    # Prepare the frame for YOLO object detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    
    outs = net.forward(layer_names)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame with object detection results
    cv2.imshow('Object Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
