import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy.misc

import theano
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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

    y = np.loadtxt(open("data/train_y.csv", "rb"), delimiter=",", skiprows=1)
    y = scipy.delete(y, 0, 1)
    y = np.asarray(y).reshape(-1)

    x,y = shuffle_dataset(x,y)

    print(y)

    x_train = x[:200]
    y_train = y[:200]
    x_val = x[200:300]
    y_val = y[200:300]

    return x_train,y_train,x_val,y_val,x_test

if __name__ == '__main__':
    x_train,y_train,x_val,y_val,x_test = load_dataset()
    # plt.imshow(x_train[0], cmap=cm.binary)
    # plt.show()

    net1 = NeuralNet( 
        layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('dense', layers.DenseLayer),
            ('dropout2', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
        # input layer
        input_shape=(None, 1, 60, 60),
        # layer conv2d1
        conv2d1_num_filters=32,
        conv2d1_filter_size=(5, 5),
        conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d1_W=lasagne.init.GlorotUniform(),  
        # layer maxpool1
        maxpool1_pool_size=(2, 2),    
        # layer conv2d2
        conv2d2_num_filters=32,
        conv2d2_filter_size=(5, 5),
        conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
        # layer maxpool2
        maxpool2_pool_size=(2, 2),
        # dropout1
        dropout1_p=0.5,    
        # dense
        dense_num_units=256,
        dense_nonlinearity=lasagne.nonlinearities.rectify,    
        # dropout2
        dropout2_p=0.5,    
        # output
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=19,
        # optimization method params
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,
        max_epochs=10,
        verbose=1
        )
    # Train the network
    nn = net1.fit(x_train, y_train)

    preds = net1.predict(x_val)

    cm = confusion_matrix(y_val, preds)
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
