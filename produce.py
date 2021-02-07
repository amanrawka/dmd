import numpy as np
# import cv2
import redis
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
# import base64
import cv2

r = redis.Redis(host='localhost', port=6379, db=0)

def redis_transfer(frame):
    output = BytesIO()

    frame = Image.fromarray(frame)
    frame.save(output, format="JPEG")
    r.xadd(name = "demo_",fields={"frame":output.getvalue()})

cam = cv2.VideoCapture("outfile.mp4")
f = cam.read()
i = 0
while True:
    frame = f[1]
    frame[0][0][0] = frame[0][0][0] + i
    i = i+1
    redis_transfer(frame)
