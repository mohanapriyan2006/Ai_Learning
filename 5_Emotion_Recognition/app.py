from facial_emotion_recognition import EmotionRecognition
import cv2

Emo_Rec = EmotionRecognition(device='cpu')

cam = cv2.VideoCapture(0)

print("  Press 'ESC' to close app.\nEmotion Recognition is Running...")

while True:
    hasFrame,frame = cam.read()
    
    if not hasFrame:
        print("Error occured in camera-read()")
        break
    frame = Emo_Rec.recognise_emotion(frame,return_type="BGR")
    
    cv2.imshow("Emotion Recognition",frame)
    
    # ESC == 27
    if cv2.waitKey(1) == 27:
        print("Emotion Recognition was Stoped.")
        exit(0)