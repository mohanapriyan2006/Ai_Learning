'''
    First RUN the train.py file using CMD 'py train.py'
    and then RUN it.
'''


from PyQt5 import QtCore, QtGui, QtWidgets
from keras.models import Sequential,load_model
from keras.layers import Conv2D,MaxPooling2D,BatchNormalization,Flatten,Dropout,Dense
from keras.preprocessing import image
import numpy


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 593)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.BrowseImage = QtWidgets.QPushButton(self.centralwidget)
        self.BrowseImage.setGeometry(QtCore.QRect(70, 380, 151, 51))
        self.BrowseImage.setObjectName("BrowseImage")
        self.imageLbl = QtWidgets.QLabel(self.centralwidget)
        self.imageLbl.setGeometry(QtCore.QRect(110, 90, 361, 261))
        self.imageLbl.setFrameShape(QtWidgets.QFrame.Box)
        self.imageLbl.setText("")
        self.imageLbl.setObjectName("imageLbl")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(110, 20, 621, 20))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.Classify = QtWidgets.QPushButton(self.centralwidget)
        self.Classify.setGeometry(QtCore.QRect(70, 460, 151, 51))
        self.Classify.setObjectName("Classify")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(340, 380, 111, 16))
        self.label.setObjectName("label")
        self.Training = QtWidgets.QPushButton(self.centralwidget)
        self.Training.setGeometry(QtCore.QRect(310, 460, 151, 51))
        self.Training.setObjectName("Training")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(310, 400, 211, 51))
        self.textEdit.setObjectName("textEdit")
        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget.setGeometry(QtCore.QRect(510, 120, 256, 231))
        self.listWidget.setObjectName("listWidget")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(510, 90, 81, 16))
        self.label_3.setObjectName("label_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        self.BrowseImage.clicked.connect(self.loadImage)

        self.Classify.clicked.connect(self.classifyFunction)

        self.Training.clicked.connect(self.trainingFunction) 

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.BrowseImage.setText(_translate("MainWindow", "Browse Image"))
        self.label_2.setText(_translate("MainWindow", "GUJARATHI CHARACTER RECOGNITION USING CNN"))
        self.Classify.setText(_translate("MainWindow", "Classify"))
        self.label.setText(_translate("MainWindow", "Recognized Class"))
        self.Training.setText(_translate("MainWindow", "Training"))
        self.label_3.setText(_translate("MainWindow", "RESULT LOGS"))
        
    def loadImage(self):
        
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp);;All Files (*)")
        if fileName:
            print(fileName)
            self.fileName = fileName
            pixmap = QtGui.QPixmap(fileName)
            pixmap = pixmap.scaled(self.imageLbl.width(), self.imageLbl.height(), QtCore.Qt.KeepAspectRatio)
            self.imageLbl.setPixmap(pixmap)
            self.imageLbl.setAlignment(QtCore.Qt.AlignCenter)

    def classifyFunction(self):
        
        filePath = self.fileName
        
        self.textEdit.setText("Model is loading...")
        
        model = load_model("model.keras") ## (OR)
        # model = load_model("model.h5")
        
        self.textEdit.setText("Model is predicting...")
        
        CLASSES = ["sunna","ek","das","be","tran","char","panc","cha","sat","at","nav","ALA","ANA","B","BHA","CH","CHH","D","DA","DH","DHA","F","G","GH","GNA","H","J","JH","K","KH","KSH","L","M","N","P","R","S","SH","SHH","T","TA","TH","THA","V","Y"]
        
        img = image.load_img(filePath,color_mode="grayscale",target_size=(128,128))
        img = image.img_to_array(img)
        img = numpy.expand_dims(img,axis=0)
        
        results = model.predict(img)
        
        label = CLASSES[numpy.argmax(results)]
        
        print(filePath , " =====> ",label)
        
        self.textEdit.setText("RESULT => ",label)
        
            
    def trainingFunction(self):
        self.textEdit.setText("Training under process...")
        print("\nModel is initailizing...\n")

        model = Sequential()

        model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(128,128,1)))
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
        model.save("model.keras")
        print("Model is saved on your disk.")
        self.textEdit.setText("Model is saved on your disk.")
        



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

