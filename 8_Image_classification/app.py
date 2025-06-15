import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import os

print("\nModel is Loading...\n")
modelfile = open("model.json",'r')
modeljson = modelfile.read()
modelfile.close()
model = model_from_json(modeljson)
model.load_weights("model.weights.h5")
print("\n  Model is Loaded successfully.\n")

def classify(f):
    filePath = f
    img = image.load_img(filePath,target_size=(64,64))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    
    result = model.predict(img)
    if result[0][0] == 1:
        prediction = "(Thalapathy) Vijay"
    else:
        prediction = "(Thala) Ajith"
    print(f," ==> ",prediction)

print("Datasets is loading ...")
path = "Dataset/test"
files = []
for root,dir,file in os.walk(path):
    for f in file:
        if(f.endswith(".jpeg")):
            files.append(os.path.join(path,f))
            
print("  Datasets are loaded.\n\nThala/Thalapathy Image Classification started:\n")
for f in files:
    classify(f)
    print()

print("  Image Classification was ended.")