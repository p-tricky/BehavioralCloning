import csv
import cv2
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from data_gen import DataGen
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save_to',
        type=str,
        default='model.h5',
        nargs='?'
)
args = parser.parse_args()

model = Sequential()
model.add(Lambda(lambda x: x/255 - .5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 50), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(MaxPooling2D(padding='same'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(MaxPooling2D(padding='same'))
model.add(Convolution2D(48, 5, 5, subsample=(1,1), activation='relu'))
model.add(MaxPooling2D(padding='same'))
model.add(Convolution2D(64, 3, 3, subsample=(1,1), activation='relu'))
model.add(MaxPooling2D(padding='same'))
model.add(Convolution2D(64, 3, 3, subsample=(1,1), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

gen = DataGen()

model.compile(optimizer=Adam(lr=.0001), loss='mse')
model.fit_generator(gen.next_train(), samples_per_epoch=gen._train.shape[0], nb_epoch=2,
    validation_data=gen.next_valid(), nb_val_samples=gen._validation.shape[0])

model.save(args.save_to)
