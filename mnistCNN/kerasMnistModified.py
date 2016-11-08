'''
Modified from:
https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
'''

from __future__ import print_function
import numpy as np, os, sys
np.random.seed(9713)  # for reproducibility

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.noise import GaussianNoise
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
from keras.datasets import mnist

batch_size = 128
nb_classes = 10
nb_epoch = 15

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
# Noise sigma
sigma = 0.2

# the data, shuffled and split between train and test sets
print("Loading data")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

##############################################
print("Building Model")

model = Sequential()
model.add(GaussianNoise(sigma,input_shape=input_shape))

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

print("Built Model")
############################################

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
print("Compiled Model")

# this will do preprocessing and realtime data augmentation
datagen = ImageDataGenerator(
	featurewise_center=False,  # set input mean to 0 over the dataset
	samplewise_center=False,  # set each sample mean to 0
	featurewise_std_normalization=False,  # divide inputs by std of the dataset
	samplewise_std_normalization=False,  # divide each input by its std
	zca_whitening=False,  # apply ZCA whitening
	rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
	width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
	height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
	horizontal_flip=False,  # randomly flip images
	vertical_flip=False)  # randomly flip images

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(X_train)

# fit the model on the batches generated by datagen.flow()
model.fit_generator(datagen.flow(X_train, Y_train,
				 batch_size=batch_size),
		samples_per_epoch=X_train.shape[0],
		nb_epoch=nb_epoch,
		verbose=1,
		validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Save Model
model.save('mnist_predictor_shifting_noise.h5')



