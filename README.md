# Object Detection with YOLOv3

This Python script utilizes the YOLOv3 (You Only Look Once) model for real-time object detection using a webcam feed. The script uses the OpenCV library to capture video frames, process them through the YOLO model, and display the results with bounding boxes around detected objects.

Prerequisites

Make sure you have the following dependencies installed:

Python
OpenCV (cv2)
NumPy (numpy)
Getting Started

Clone the repository:

git clone https://github.com/Developer1010x/Object_Detection_Yolo3_code.git
cd Object_Detection_Yolo3_code
Set the correct file paths:
Update yolo_cfg with the path to your YOLOv3 configuration file (yolov3.cfg).
Update yolo_weights with the path to your YOLOv3 weights file (yolov3.weights).
Update the class names file path (coco.names) accordingly.
Run the script:

python object_detection_yolo3.py
Usage

The script captures video from the default camera (usually index 0).
Detected objects are surrounded by bounding boxes and labeled with their class names and confidence scores.
Press 'Q' to exit the application.
Notes

Ensure that your OpenCV version includes the DNN module to use cv2.dnn.readNetFromDarknet.
Make sure the YOLO configuration, weights, and class names files are correctly specified.
Acknowledgments




YOLOv3: YOLO: Real-Time Object Detection
COCO Dataset: COCO - Common Objects in Context
