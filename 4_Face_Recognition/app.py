import os, cv2, numpy

haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
print('Training...')

(width,height) = (130,100)

(images,labels,names,id) = ( [] , [] , {} , 0)

for (_ ,dirs , _) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subdirpath = os.path.join(datasets,subdir)
        for filename in os.listdir(subdirpath):
            path = subdirpath + '/' + filename
            label = id
            Img = cv2.imread(path,0)
            resize_Img = cv2.resize(Img,(width,height))
            images.append(resize_Img)
            labels.append(int(label))
        id += 1
        
(images,labels) = [numpy.array(lis) for lis in [images,labels]]

model = cv2.face.FisherFaceRecognizer_create()
model.train(images,labels)

print("Training completed.\n\nFace Recognition running...\n Press 'ESC' to close.\n\n")

face_algrm = cv2.CascadeClassifier(haar_file)
cam = cv2.VideoCapture(0)

cnt = 0
unknown_face = 1
while True:
    hasCam , img = cam.read()
    
    if not hasCam:
        print("Camera not detected !")
        break
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_algrm.detectMultiScale(gray,1.3,5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        face = gray[y:y+h , x:x+w]
        resize_face = cv2.resize(face,(width,height))
        
        label,confidence = model.predict(resize_face)
        
        if(confidence < 800):
            cv2.putText(img,"%s - %.0f" % (names[label] ,confidence),(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)
            print(names[label])
            cnt = 0
        else:
            cnt+=1
            cv2.putText(img,"Unknown",(x-10,y-10),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,255),2)
            if(cnt>50):
                cv2.imwrite("unknown-%d.png" % unknown_face , img)
                unknown_face += 1
                print("Unknown Face Caught")
                cnt = 0
        cv2.imshow("Face Recognition",img)
        
    # ESC == 27
    key = cv2.waitKey(10)
    if (key == 27):
        print("Face Recognition Stopped.")
        break
        
cam.release()
cv2.destroyAllWindows()