import os , cv2,imutils

haar_file = "haarcascade_frontalface_default.xml"
dataset = "datasets"
sub_data = "Ratan Tata"

path = os.path.join(dataset,sub_data)
if not os.path.isdir(path):
    os.mkdir(path)

face_algrm = cv2.CascadeClassifier(haar_file)
cam = cv2.VideoCapture(0)

width , height = (130,100)

count = 1

while count < 51:
    _,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_algrm.detectMultiScale(gray,1.3,4)
    
    for (x,y,w,h) in faces:
        print(count)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        face = gray[y:y+h , x:x+w]
        resized_face = imutils.resize(face,width,height)
        cv2.imwrite("%s/%s.png" % (path,count),resized_face)
        count += 1
        
    cv2.imshow("Create Face Dataset",img)
    key = cv2.waitKey(1)
    # ESC == 27
    if key == 27:
        break
    
cam.release()
cv2.destroyAllWindows()
