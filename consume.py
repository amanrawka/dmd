import numpy as np
import argparse
import time
import cv2
import os
import redis
import numpy as np
# import cv2
import redis
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO




r = redis.Redis(host='localhost', port=6379, db=0)

labelsPath = os.path.sep.join(["/home/pi/dark_net/darknet/data", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

weightsPath = os.path.sep.join(["/home/pi/dark_net/darknet", "yolov3.weights"])
configPath = os.path.sep.join(["/home/pi/dark_net/darknet/cfg", "yolov3.cfg"])


print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

def redis_stream(stream):
    data_image = r.xread({"demo_":b"0-0"})
    data= data_image[0][1]
    data = data.pop()[1]
    for key in data.items():
        image = key[1]
    stream = BytesIO(image)
    image = Image.open(stream).convert("RGB")
    img_array = np.array(image)
    
    return image,img_array
    
    

def data_grab():
    image,img_array = redis_stream(stream = "demo_")

    (H, W) = img_array.shape[:2]

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(img_array, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    boxes = []
    confidences = []
    classIDs = []

    conf = output_pred(layerOutputs,H,W)
    return conf


def output_pred(layerOutputs,H,W):
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.5:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.4)
    return confidences





while True:
    shuru = time.time()
    
    conf = data_grab()
    
    end_n = time.time()
    diff = end_n - shuru
    print("Time required : {}".format(diff))
     	


