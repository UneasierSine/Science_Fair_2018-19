import cv2
import os 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory

from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Concatenate, Conv3D, MaxPooling2D, Lambda

baseKernels = []
baseKernels.append((1,4,6,4,1))
baseKernels.append((1,-2,0,-2,1))
baseKernels.append((-1,0,2,0,-1))
baseKernels.append((1,-4,6,-4,1))
baseKernels = np.array(baseKernels)

def image_transform(image):
    image = np.asarray(image)
    imageForms = np.array(image)
    for x in range(4):
        for y in range(4):
            imageForms = np.append(imageForms, cv2.filter2D(src=image,ddepth=-1,kernel=np.multiply(baseKernels[x],np.reshape(baseKernels[y],[5,1]))), axis=2)
    return imageForms

#get directory of input images and create array of images and store images in the directory to the array
train_dir = "C:/Users/panka/OneDrive/Desktop/Aditya/image data 2018-19/Train_Resized"
#get labels pickle and convert to dataframe then sort by the filename to go along with the images
train_labels_file = "C:/Users/panka/OneDrive/Desktop/Aditya/image data 2018-19/Training_Input_Resized.pkl"

train_labels = pd.read_pickle(train_labels_file)

train_datagen = ImageDataGenerator(rescale=1./255,preprocessing_function=image_transform)
train_generator = train_datagen.flow_from_dataframe(dataframe=train_labels,directory=train_dir,target_size=(108,192),x_col='Filename',y_col=['Right Ankle x','Right Knee x','Right Hip x','Left Hip x','Left Knee x','Left Ankle x','Pelvis x','Thorax x','Upper Neck x','Head Top x','Right Wrist x','Right Elbow x','Right Shoulder x','Left Shoulder x','Left Elbow x','Left Wrist x','Right Ankle y','Right Knee y','Right Hip y','Left Hip y','Left Knee y','Left Ankle y','Pelvis y','Thorax y','Upper Neck y','Head Top y','Right Wrist y','Right Elbow y','Right Shoulder y','Left Shoulder y','Left Elbow y','Left Wrist y'],class_mode='other',batch_size=8)
    
#get directory of input images and create array of images and store images in the directory to the array
test_dir = "C:/Users/panka/OneDrive/Desktop/Aditya/image data 2018-19/Test_Resized"
#get labels pickle and convert to dataframe then sort by the filename to go along with the images
test_labels_file = "C:/Users/panka/OneDrive/Desktop/Aditya/image data 2018-19/Testing_Input_Resized.pkl"

test_labels = pd.read_pickle(test_labels_file)

test_datagen = ImageDataGenerator(rescale=1./255,preprocessing_function=image_transform)
test_generator = test_datagen.flow_from_dataframe(dataframe=test_labels,directory=test_dir,target_size=(108,192),x_col='Filename',y_col=['Right Ankle x','Right Knee x','Right Hip x','Left Hip x','Left Knee x','Left Ankle x','Pelvis x','Thorax x','Upper Neck x','Head Top x','Right Wrist x','Right Elbow x','Right Shoulder x','Left Shoulder x','Left Elbow x','Left Wrist x','Right Ankle y','Right Knee y','Right Hip y','Left Hip y','Left Knee y','Left Ankle y','Pelvis y','Thorax y','Upper Neck y','Head Top y','Right Wrist y','Right Elbow y','Right Shoulder y','Left Shoulder y','Left Elbow y','Left Wrist y'],class_mode='other',batch_size=8)

#create model
model = Sequential()

#add model layers
model.add(Conv2D(1, kernel_size=1, input_shape=(108,192,3), activation='relu'))
model.add(Lambda(image_transform))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(256, kernel_size=3, activation='relu'))
model.add(Conv2D(256, kernel_size=3, activation='relu'))
model.add(Conv2D(256, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

#train the model
STEP_SIZE_TRAIN = train_generator.n/8
STEP_SIZE_TEST = test_generator.n/8
hist = model.fit_generator(train_generator,epochs=100,validation_data=test_generator, steps_per_epoch=STEP_SIZE_TRAIN, validation_steps=STEP_SIZE_TEST)

#Show graph of validation and training loss
loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, color='red', label='Training loss')
plt.plot(epochs, val_loss, color='green', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig("loss.png")

#Show graph of validation and training accuracy
acc = hist.history['acc']
val_acc = hist.history['val_acc']
plt.plot(epochs, acc, color='red', label='Training acc')
plt.plot(epochs, val_acc, color='green', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig("/accuracy.png")

with open('file.json','w') as f:
    json.dump(hist.history,f)
