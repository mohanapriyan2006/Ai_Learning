'''
    RUN IT first and the RUN app.py or else
'''


from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,BatchNormalization,Flatten,Dropout,Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.layers import Input
print("\nModel is initailizing...\n")

model = Sequential()

model.add(Input(shape=(128,128,1)))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(96,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(45,activation='softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = None,
                                shear_range = 0.2,
                                zoom_range = 0.2,
                                horizontal_flip = True)

val_datagen = ImageDataGenerator(rescale = 1./255)

train_datasets = train_datagen.flow_from_directory(
    "Dataset/train",
    target_size = (128, 128),
    batch_size = 8,
    color_mode='grayscale', 
    class_mode = 'categorical'
)

val_datasets = val_datagen.flow_from_directory(
    "Dataset/val",
    target_size = (128, 128),
    batch_size = 8,
    color_mode='grayscale', 
    class_mode = 'categorical'
)

print("\n Model was initailzied.\nModel is training...\n")

model.fit(
    train_datasets,
    steps_per_epoch=100,
    epochs=15,
    validation_data=val_datasets,
    validation_steps=125
)

# save model

# # for New version
model.save("model.keras") 

## (OR) for old version

# model.save("model.h5")

print("Model is saved on your disk.")
