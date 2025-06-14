import imutils as imu
import cv2
import time
import numpy as np

prototxt = "MobileNetSSD_deploy.prototxt.txt"
modelfile = "MobileNetSSD_deploy.caffemodel"

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor","mobile"]

COLORS = np.random.uniform(0,255,size=(len(CLASSES),3))

print("\nModel is Loading...")
model = cv2.dnn.readNetFromCaffe(prototxt,modelfile)
print(" Model is Loaded successfully.\n")

print("Camera initializing...")
cam = cv2.VideoCapture(0)
# by 'footage'
# cam = cv2.VideoCapture("footage.mp4")
time.sleep(1.0)
print(" Object Recognition is started.\n  Press 'ESC' to close.\n")

threshConfidence = 0.2

while True:
    hasCam,frame = cam.read()
    
    if not hasCam:
        print("Error occured in camera!")
        break
    
    frame = imu.resize(frame,width=800)
    (h,w) = frame.shape[:2]
    
    resizedImg = cv2.resize(frame,(300,300))
    blogImg = cv2.dnn.blobFromImage(resizedImg,0.007843,(300,300),127.5)
    
    model.setInput(blogImg)
    detection = model.forward()
    
    length = detection.shape[2]
    
    for i in range(0,length):
        
        confidence = detection[0,0,i,2]
        
        if confidence > threshConfidence:
            
            id = int(detection[0,0,i,1])
            print(f"Class ID: {id}")
            
            box = detection[0,0,i,3:7] * np.array([w,h,w,h])
            (stx,sty,endx,endy) = box.astype("int")
            cv2.rectangle(frame,(stx,sty),(endx,endy),COLORS[id],4)
            
            label = "{}: {:.2f}%".format(CLASSES[id],confidence * 100)
            if sty - 15 > 15:
                sty = sty - 15
            else:
                sty = sty + 15
            cv2.putText(frame,label,(stx,sty),cv2.FONT_HERSHEY_SIMPLEX,0.5,COLORS[id],2)
    
    cv2.imshow("Object Recognition",frame)
    # ESC to close window
    if cv2.waitKey(1) == 27:
        print("\n Object Recognition was Stoped.")
        exit(0)
