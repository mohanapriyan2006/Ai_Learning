'''
For Dataset column headings:
   1. Number of times pregnant
   2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
   3. Diastolic blood pressure (mm Hg)
   4. Triceps skin fold thickness (mm)
   5. 2-Hour serum insulin (mu U/ml)
   6. Body mass index (weight in kg/(height in m)^2)
   7. Diabetes pedigree function
   8. Age (years)
   9. Class variable (0 or 1)
'''

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense


dataset = loadtxt("pima-indians-diabetes.csv" , delimiter=",")

x = dataset[:,0:8]
y = dataset[:,8]

model = Sequential()
model.add(Dense(12,input_dim=8,activation="relu"))
model.add(Dense(8,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])

# Training
print("\n\n  Model is training...")
model.fit(x,y,batch_size=10,epochs=40)

# Evaulate
print("\n\n Model is Evaulating...")
_,accuracy = model.evaluate(x,y)

print("\n Accuracy: %.2f percentage." % (accuracy*100))

# # save model
json_model = model.to_json()
with open("model.json","w") as modelfile:
    modelfile.write(json_model)
model.save_weights("model.weights.h5")
print("\n\n Model and Its weight was saved successfully.")