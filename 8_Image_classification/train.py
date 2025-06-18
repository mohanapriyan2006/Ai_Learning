'''
    RUN IT first and the RUN app.py or else
'''

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

train_genImg = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_genImg = ImageDataGenerator(rescale=1. / 255)

train_dataset = train_genImg.flow_from_directory(
    "Dataset/train",
    target_size=(64,64),
    batch_size=8,
    class_mode="binary"
)

val_dataset = val_genImg.flow_from_directory(
    "Dataset/val",
    target_size=(64,64),
    batch_size=8,
    class_mode="binary"
)

model.fit(
    train_dataset,
    steps_per_epoch=10,
    epochs=50,
    validation_data=val_dataset,
    validation_steps=2
)

# save model
model_json = model.to_json()
with open("model.json",'w') as file:
    file.write(model_json)
model.save_weights("model.weights.h5")
print("\n   Trained Model is saved in your disk.")