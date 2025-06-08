import urllib.request
import cv2
import numpy as np
import imutils


url='http://192.0.0.4:8080/shot.jpg'

while True:
    imgPath=urllib.request.urlopen(url)
    imgNp=np.array(bytearray(imgPath.read()), dtype=np.uint8)
    frame=cv2.imdecode(imgNp, -1)

    frame=imutils.resize(frame,width=600)
    cv2.imshow("Frame",frame)
    if ord('q')==cv2.waitKey(1):
        exit(0)
