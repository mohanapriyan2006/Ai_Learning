import urllib.request
from facial_emotion_recognition import EmotionRecognition
import cv2
import numpy as np

er = EmotionRecognition(device="cpu")

ip = "http://192.0.0.4:8080/shot.jpg"

print("  Press 'ESC' to close app.\nEmotion Recognition is Running...")

while True:
    
    Imgurl = urllib.request.urlopen(ip)
    Imgnp = np.array(bytearray(Imgurl.read()),dtype=np.uint8)
    frame = cv2.imdecode(Imgnp,-1)
    
    frame = cv2.resize(frame,(500,400))
    frame = er.recognise_emotion(frame,return_type="BGR")
    
    cv2.imshow("Emotion Recognition",frame)
    
    if cv2.waitKey(1) == 27: #ESC == 27
        print("Emotion Recognition was Stoped.")
        exit(0)