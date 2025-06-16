from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping,ModelCheckpoint

model = Sequential()

model.add(Conv2D(32,(3,3),input_shape=(256,256,1),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(150,activation="relu"))
model.add(Dropout(0.25))

model.add(Dense(6,activation="softmax"))

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 12.,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   zoom_range=0.15,
                                   horizontal_flip = True)

val_datagen = ImageDataGenerator(rescale = 1./255)

train_dataset = train_datagen.flow_from_directory('HandGestureDataset/train',
                                                 target_size = (256, 256),
                                                 color_mode = 'grayscale',
                                                 batch_size = 8,
                                                 classes = ['NONE','ONE','TWO','THREE','FOUR','FIVE'],
                                                 class_mode = 'categorical')

val_dataset = val_datagen.flow_from_directory('HandGestureDataset/test',
                                            target_size = (256, 256),
                                            color_mode='grayscale',
                                            batch_size = 8,
                                            classes=['NONE', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'],
                                            class_mode='categorical')


callback_list = [
    EarlyStopping(monitor="val_loss",patience=10),
    ModelCheckpoint(filepath="model2.weights.h5",monitor="val_loss",save_best_only=True,verbose=1)
]

model.fit(
    train_dataset,
    steps_per_epoch=10,
    epochs=30,
    validation_data=val_dataset,
    validation_steps=2,
    callbacks=callback_list
)

# save the model
modelJson = model.to_json()
with open("model2.json",'w') as file:
    file.write(modelJson)
model.save_weights("model2.weights.h5")
print("\n  Model is saved to your disk.")