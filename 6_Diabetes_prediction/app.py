from numpy import loadtxt
from keras.models import model_from_json

print("\n Datasets are loading...")
dataset = loadtxt("pima-indians-diabetes.csv" , delimiter=",")
x = dataset[:,0:8]
y = dataset[:, 8]

print(" Datasets are loaded.\n\n Model is loading...\n")
loadFile = open("model.json","r")
json_file = loadFile.read()
model = model_from_json(json_file)
model.load_weights("model.weights.h5")
print("\nModel is loaded successfully.\n")

# Prediction
print("Prediction Started:\n")
prediction = model.predict(x)

for i in range(1,11):
    print("%s ==> output: %d (expected: %d)" % (x[i].tolist(),prediction[i],y[i]))
