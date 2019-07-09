import tensorflow as tf
import keras
import PIL
from PIL import Image, ImageOps
import sys
import glob
import numpy as np
from keras.layers import Input, Dense, Activation
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import os

model = load_model('/home/pi/zavrsni/model.h5')                 # Loading trained model

count = 0                                                       

while(count < 5):
    
    os.system("raspistill -o slika.jpg")                        # Capturing image
    
    color = Image.open("/home/pi/slika.jpg")                    # Uploading image
    grayscale = color.convert('L')                              # Converting to grayscale
    bw = np.asarray(grayscale).copy()                           # Grayscale as array
    bw[bw < 35] = 0                                             # Black pixel
    bw[bw >= 35] = 255                                          # White pixel
    imfile = Image.fromarray(bw)                                # Returning to image format
    imfile.save("/home/pi/zavrsni/slika_bw.jpg")                # Saving image

    size = 64, 64                                               # Declaring size of new image
    im = Image.open("/home/pi/zavrsni/slika_bw.jpg")            # Uploading black/white image
    im_resized = im.resize(size, Image.ANTIALIAS)               # Resizing image to 64x64 
    im_resized.save("/home/pi/zavrsni/slika_bwr.jpg")           # Saving resized image image
     
    im = Image.open("/home/pi/zavrsni/slika_bwr.jpg")           # Uploading resized image

    image = np.array(im)                                        # Image as array
    image = np.reshape(image, 4096)                             # Reshaping array to 4096x1

    imagen = np.zeros(shape=(1, 4096))                          # Declaring array of new image

    for i in range (4096):
        b = image[i]/255.0                                      # Normalizing pixels
        imagen[0][i] = b                                        # Writing normalized pixels in new array

    preds = model.predict(imagen)                               # Predicting class of the image 
    preds = preds.argmax(axis = 1)                              # Selecting class with highest probability

    print(preds)                                                # Printing predicted class
    count = count + 1                                           # Counter is increasing to 5 
