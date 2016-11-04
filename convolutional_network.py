import gzip
import sys
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import theano
import theano.tensor as T

import scipy
import scipy.ndimage as ndimage

import numpy as np

from urllib.request import urlretrieve
from random import randint
from sklearn.metrics import accuracy_score

import lasagne

class Convolutional_Network(object):

    def __init__(self,x_train_data,y_train_data,x_test_data):
        self.x_train,self.y_train,self.x_val,self.y_val,self.x_test = self.load_dataset(x_train_data,y_train_data,x_test_data)
        self.input_var = T.tensor4('inputs')
        self.target_var = T.ivector('targets')
        self.network = self.build_main_network(input_var=self.input_var)

        prediction = lasagne.layers.get_output(self.network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, self.target_var)
        loss = loss.mean()
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
        test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,self.target_var)
        test_loss = test_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), self.target_var),dtype=theano.config.floatX)
                        
        self.train_fn = theano.function([self.input_var, self.target_var], loss, updates=updates)
        self.val_fn = theano.function([self.input_var, self.target_var], [test_loss, test_acc])
    
    def load_dataset(self,x_train_data,y_train_data,x_test_data):

        def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
            print("Downloading %s" % filename)
            urlretrieve(source + filename, filename)
        def move_random(imgs):
            new_images = []
            for i in imgs:
                rndX = randint(0,8)
                rndY = randint(0,8)
                new_image = np.zeros(shape=(28,28),dtype='f')
                digit = i[0][3:23,3:23]
                new_image[rndX:rndX+20,rndY:rndY+20] = digit
                new_images.append([new_image])
            return new_images

        def load_mnist_images(filename):
            if not os.path.exists(filename):
                download(filename)
            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
            data = data.reshape(-1, 1, 28, 28)
            data = move_random(data)
            data = np.array(data, dtype=np.float32) 
            return data / np.float32(256)

        def load_mnist_labels(filename):
            if not os.path.exists(filename):
                download(filename)
            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)
            return data

        def shuffle_dataset(x,y):
            x_shuff = []
            y_shuff = []
            index_shuf = list(range(len(x)))
            np.random.shuffle(index_shuf)
            for i in index_shuf:
                x_shuff.append(x[i])
                y_shuff.append(y[i])
            x_shuff = np.asarray(x_shuff)
            y_shuff = np.asarray(y_shuff)
            return x_shuff,y_shuff
        
        def load_custom_data(x_train_data,y_train_data,x_test_data):
            x_train = np.fromfile(x_train_data, dtype='uint8')
            x_train = x_train.reshape((-1,1,60,60))
            x_train = x_train/np.float32(256)
            x_test = np.fromfile(x_test_data, dtype='uint8')
            x_test = x_test.reshape((-1,1,60,60))
            x_test = x_test/ np.float32(256)

            y_train = np.loadtxt(open(y_train_data, "rb"), delimiter=",", skiprows=1)
            y_train = scipy.delete(y_train, 0, 1)
            y_train = np.asarray(y_train.reshape(-1))
            y_train =  y_train.astype(int)

            x_shuff,y_shuff = shuffle_dataset(x_train,  y_train)

            test_percent = 0.8
            test_size = len(x_shuff) * test_percent

            x_train = x_shuff[:2000]
            y_train = y_shuff[:2000]
            x_val = x_shuff[2000:3000]
            y_val = y_shuff[2000:3000]
            return x_train,y_train

        x_train = load_mnist_images('train-images-idx3-ubyte.gz')
        y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')

        x_test,y_test = load_custom_data(x_train_data,y_train_data,x_test_data)
        x_train, x_val = x_train[:-10000], x_train[-10000:]
        y_train, y_val = y_train[:-10000], y_train[-10000:]
        return x_train,y_train,x_val,y_val,x_test        
    
    def build_main_network(self,input_var=None):
        '''
        Architecture inspired by lasagne tutorial
        '''
        network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),input_var=input_var)
        network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(5, 5),nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
        network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(5, 5),nonlinearity=lasagne.nonlinearities.rectify)
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
        network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5),num_units=256,nonlinearity=lasagne.nonlinearities.rectify)
        network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5),num_units=10,nonlinearity=lasagne.nonlinearities.softmax)

        return network
    
    def iterate_minibatches(self,inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]


    def train(self,num_epochs=500,epoch_size=500,debug=False):
        for epoch in range(num_epochs):
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in self.iterate_minibatches(self.x_train, self.y_train, epoch_size, shuffle=True):
                inputs, targets = batch
                train_err += self.train_fn(inputs, targets)
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in self.iterate_minibatches(self.x_val, self.y_val, epoch_size, shuffle=False):
                inputs, targets = batch
                err, acc = self.val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch:
            if debug:
                print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
                print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
                print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
                print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))
        return self

    def fit(self,epoch_size=500,debug=False):
        fit_data = lasagne.layers.get_output(self.network,inputs=self.x_test,deterministic=True)
        fit_predictions = fit_data.eval()
        final_predictions = []
        for i in fit_predictions:
            final_predictions.append(np.argmax(i))
        if debug:
            print(final_predictions)
        return final_predictions


