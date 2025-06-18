'''
    First RUN the train.py file using CMD 'py train.py'
    and then RUN it.
'''

from keras.models import load_model
import os
import numpy as np
from keras.preprocessing import image

print("\nModel is Loading ... \n")
model = load_model("model.keras")
print("\n Model is Loaded successfully. \n")


def classify(f):
    filePath = f
    img = image.load_img(filePath,target_size=(256,256),color_mode="grayscale")
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    
    result = model.predict(img)
    arr = np.array(result[0])
    print("ARRAY : ",arr)
    max_prop = arr.argmax(axis=0) + 1
    classes=["NONE", "ONE", "TWO", "THREE", "FOUR", "FIVE"]
    label = classes[max_prop - 1]
    print(f," ==> ",label)


print("Datasets are loading...\n")
path= "HandGestureDataset/check"
files=[]
for rt,dir,file in os.walk(path):
    for f in file:
        if ".png" in f:
            files.append(os.path.join(path,f))
print(" Datasets are Loaded.\n")

print("\nHand Gesture is started:")
for f in files:
    classify(f)
    print()
print("\n  Hand Gesture Ended.")
