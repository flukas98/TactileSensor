import tensorflow as tf
import keras
import PIL
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import sys
import glob
import numpy as np
from keras.layers import Input, Dense, Activation
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

model = load_model('C:/Users/Filip/Documents/002 FER/6. semestar/zavrsni_rad/model/model.h5')

image_list2 = []
image_list_reshaped2 = []

for filename in glob.glob('C:/Users/Filip/Documents/002 FER/6. semestar/zavrsni_rad/dataset/test3/bwr/tupo/*.jpg'): 
    im = Image.open(filename)
    image_list2.append(im)
    image = np.array(im)
    image = np.reshape(image, 4096)
    image_list_reshaped2.append(image)
	
x_testf = np.zeros(shape = (13, 4096))
i = 0
for image in image_list_reshaped2:
    x_testf[i] = image
    i = i + 1
	
preds = model.predict(x_testfn)
preds = preds.argmax(axis = 1)

preds, y_testi