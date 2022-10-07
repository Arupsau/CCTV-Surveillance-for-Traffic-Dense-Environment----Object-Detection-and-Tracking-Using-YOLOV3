# CCTV Surveillance for Traffic Dense Environment -- Object Detection and Tracking Using YOLOV3

**Introduction :**

Deep learning has gained a tremendous influence on how the world is adapting to Artificial Intelligence since past few years. Some of the popular object detection algorithms are Region-based Convolutional Neural Networks (RCNN), Faster-RCNN, Single Shot Detector (SSD) and You Only Look Once (YOLO). Amongst these, Faster-RCNN and SSD have better accuracy, while YOLO performs better when speed is given preference over accuracy. Deep learning combines SSD and Mobile Nets to perform efficient implementation of detection and tracking. This algorithm performs efficient object detection while not compromising on the performance.

**Our Vision and Mission :**

* Creating a Custom Data-set of desired classes and labeling the Images.
* Detecting moving objects of specified classes in a Traffic heavy Environment using Deep learning.
* Track down the detected Objects using Deep Learning based algorithm Deep-SORT.
* Implementing our solution for any specific surveillance related problems to those detected and tracked Objects.

E.g: 
    1. Velocity Estimation of a Vehicle, 
    2. Distance between Two or more Tracked Down Objects  etc.                                                         

**Dataset Creation and Labeling :**

* We used images from Google’s OpenImagesV6 dataset, publicly available online. It is a very big dataset with more than 600 different categories of an object. The dataset contains the bounding box, segmentation, or relationship annotations for these objects.

* We collected images of 8 classes.  **“Person”**, **“Motorbike”**, **“Traffic light”**, **“Car”**, **“Bus”**, **“Truck”**, **“Bicycle”** and **“Umbrella”**.

* We used LebelImg to label each and every Image for collecting dimensions of anchor boxes.


**What is Object Detection ? :**

* Object Detection is a common Computer Vision problem which deals with identifying and locating object of in the image.(Object Recognition recognise what kind of object they are : class labels)

* Interpreting the object localisation can be done in various ways, including creating a bounding box around the object or marking every pixel in the image which contains the object (called segmentation).
* With the need of real time object detection, many one-step object detection architectures have been proposed, like YOLO, YOLOv2, YOLOv3, SSD, RetinaNet etc. which try to combine the detection and classification step.
* In our project we will use YOLOv3 for training our object detection model.



## Technology and frameworks used: 
1. YOLOv3
1. Darknet 53
1. Jupyter Notebook
1. Deepsort
1. Python 
