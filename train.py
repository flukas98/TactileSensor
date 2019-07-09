import tensorflow as tf
import keras
import PIL
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import sys
import glob
import numpy as np

image_list = []
image_list_reshaped = []
for filename in glob.glob('C:/Users/Filip/Documents/002 FER/6. semestar/zavrsni_rad/dataset/data2/nista_bwr2/*.jpg'): 
    im = Image.open(filename)
    image_list.append(im)
    image = np.array(im)
    image = np.reshape(image, 4096)
    image_list_reshaped.append(image)
for filename in glob.glob('C:/Users/Filip/Documents/002 FER/6. semestar/zavrsni_rad/dataset/data2/ostro_bwr2/*.jpg'): 
    im = Image.open(filename)
    image_list.append(im)
    image = np.array(im)
    image = np.reshape(image, 4096)
    image_list_reshaped.append(image)
for filename in glob.glob('C:/Users/Filip/Documents/002 FER/6. semestar/zavrsni_rad/dataset/data2/tupo_bwr2/*.jpg'): 
    im = Image.open(filename)
    image_list.append(im)
    image = np.array(im)
    image = np.reshape(image, 4096)
    image_list_reshaped.append(image)
	
x_trainf = np.zeros(shape=(240, 4096))
i = 0
for image in image_list_reshaped:
    x_trainf[i] = image
    i = i + 1
x_traini = x_trainf.astype(int)
max = 0
min = 255
for i in range (240):
    for j in range (4096):
        if x_trainf[i][j] > max:
            max = x_trainf[i][j]
        if x_trainf[i][j] < min:
            min = x_trainf[i][j]

x_trainfn = np.zeros(shape=(240, 4096))
for i in range (240):
    for j in range (4096):
        b = (x_trainf[i][j]-min)/(max-min)
        x_trainfn[i][j] = b
y_traini = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
                     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
                     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
y_trainf = y_traini.astype(float)

from keras.utils import to_categorical

y_traini = to_categorical(y_traini, num_classes = 3) 

from keras.layers import Input, Dense, Activation
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

img_input = Input(shape=(4096,))
x = Dense(units = 256, activation = "relu")(img_input)
x = Dense(units = 128, activation = "relu")(x)
x = Dense(units = 64, activation = "relu")(x)
x = Dense(units = 32, activation = "relu")(x)
x = Dense(units = 3, activation = "sigmoid")(x)

model = Model(inputs = img_input, outputs = x)

model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
model.fit(x_trainfn, y_traini, batch_size = 25, epochs = 10, validation_split = 0.2)

model.save('model.h5')
