import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy.misc

import theano
import theano.tensor as T

import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def shared_dataset(data_x, data_y):
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    return shared_x, T.cast(shared_y, 'int32')

def shuffle_dataset(x,y):
    x_shuff = []
    y_shuff = []
    index_shuf = list(range(len(x)))
    np.random.shuffle(index_shuf)
    for i in index_shuf:
        x_shuff.append(x[i])
        y_shuff.append(int(y[i]))
    x_shuff = np.asarray(x_shuff)
    y_shuff = np.asarray(y_shuff)
    return x_shuff,y_shuff

def load_dataset():
    x = np.fromfile('data/train_x.bin', dtype='uint8')
    x = x.reshape((-1,1,60,60))
    x_test = np.fromfile('data/test_x.bin', dtype='uint8')
    x_test = x_test.reshape((-1,1,60,60))
    y_test = np.zeros((1,len(x_test)))

    y = np.loadtxt(open("data/train_y.csv", "rb"), delimiter=",", skiprows=1)
    y = scipy.delete(y, 0, 1)
    y = np.asarray(y).reshape(-1)

    x,y = shuffle_dataset(x,y)
    x_train = x[:200]
    y_train = y[:200]
    x_val = x[200:300]
    y_val = y[200:300]

    return x_train,y_train,x_val,y_val,x_test,y_test

def build_cnn(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 1, 60,60),input_var=input_var)
    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(5, 5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(5, 5), nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=256, nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5),num_units=19,nonlinearity=lasagne.nonlinearities.softmax)
    return network

def main():
    x_train, y_train,x_val, y_val,x_test, y_test = load_dataset()

    x_train, y_train = shared_dataset(x_train, y_train)
    x_val, y_val = shared_dataset(x_val, y_val)
    x_test, y_test = shared_dataset(x_test, y_test)

    train_data = (x_train,y_train)
    val_data = (x_val,y_val)
    test_data = (x_test,y_test)

    mini_batch_size = 10
    net = Network([FullyConnectedLayer(n_in=3600, n_out=100),SoftmaxLayer(n_in=100, n_out=19)], mini_batch_size)
    net.SGD(train_data, 60, mini_batch_size, 0.1, val_data, test_data)

if __name__ == '__main__':main() 