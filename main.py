#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.
This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html
More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

from sklearn.metrics import confusion_matrix


import csv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import scipy
import lasagne

def apply_threshold(img,blur):
    for i in range(len(blur)):
        for j in range(len(blur[0])):
            if blur[i][j] < 0.9:
                img[i][j] = 0.0
    return img

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

def k_norm(y_val):
    arr = np.zeros(19)
    arr[y_val] = 1
    arr = arr.astype(int)
    return arr


def load_dataset():
    x = np.fromfile('data/trainBW.bin', dtype='uint8')
    x = x.reshape((-1,1,60,60))
    x = x/ np.float32(256)
    x_test = np.fromfile('data/testBW.bin', dtype='uint8')
    x_test = x_test.reshape((-1,1,60,60))
    x_test = x_test/ np.float32(256)
    y_test = np.zeros(len(x_test))

    y_vals = np.loadtxt(open("data/train_y.csv", "rb"), delimiter=",", skiprows=1)
    y_vals = scipy.delete(y_vals, 0, 1)
    y_vals = np.asarray(y_vals).reshape(-1)
    y_vals = y_vals.astype(int)
    # y = []
    # for val in y_vals:
    #     y.append(k_norm(val))
    x,y = shuffle_dataset(x,y_vals)
    x_train = x[:]
    y_train = y[:]
    x_val = x[80000:]
    y_val = y[80000:]

    return x_train,y_train,x_val,y_val,x_test,y_test

def build_cnn(input_var=None):
 
    network = lasagne.layers.InputLayer(shape=(None, 1, 60, 60),
                                        input_var=input_var)
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=500,
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=19,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
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


def shared_dataset(data_x, data_y):
    # print(np.asarray(data_x, dtype=theano.config.floatX))
    shared_x = theano.shared(data_x)
    shared_y = theano.shared(data_y)
    return shared_x, T.cast(shared_y, 'int32')

def main(num_epochs=40):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    # X_test,y_train = shared_dataset(X_train,y_train)
    # X_val,y_val = shared_dataset(X_val,y_val)
    # X_test,y_test = shared_dataset(X_test,y_test)


    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_cnn(input_var)
    # with np.load('model7.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    predict_function = theano.function([input_var], prediction)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    test_pred = T.argmax(test_prediction,axis=1)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    test_fn = theano.function([input_var],[test_pred])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    # for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        # train_err = 0
        # train_batches = 0
        # start_time = time.time()
        # for batch in iterate_minibatches(X_train, y_train, 500,shuffle=True):
        #     inputs, targets = batch
        #     train_err += train_fn(inputs, targets)
        #     train_batches += 1

        # # And a full pass over the validation data:
        # val_err = 0
        # val_acc = 0
        # val_batches = 0
        # for batch in iterate_minibatches(X_val, y_val, 500):
        #     inputs, targets = batch
        #     # test = lasagne.layers.get_output(network,inputs=inputs)
        #     # print(test.eval()[0])
        #     # plt.imshow(inputs[0][0], cmap=cm.binary)
        #     # plt.show()
        #     err, acc = val_fn(inputs, targets)
        #     val_err += err
        #     val_acc += acc
        #     val_batches += 1

        # if epoch % 20 == 0 :
        #     np.savez('serialized_models/model8_' + str(epoch) + '.npz', *lasagne.layers.get_all_param_values(network))
        # # Then we print the results for this epoch:
        # print("Epoch {} of {} took {:.3f}s".format(
        #     epoch + 1, num_epochs, time.time() - start_time))
        # print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        # print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        # print("  validation accuracy:\t\t{:.2f} %".format(
        #     val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    # test_err = 0
    # test_acc = 0
    # test_batches = 0
    # for batch in iterate_minibatches(X_test, y_test, 500):
    #     inputs, targets = batch
    #     err, acc = val_fn(inputs, targets)
    #     test_err += err
    #     test_acc += acc
    #     test_batches += 1
    # print("Final results:")
    # print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    # print("  test accuracy:\t\t{:.2f} %".format(
    #     test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model8.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    with np.load('model8.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)
    all_preds = []
    for batch in iterate_minibatches(X_val, y_val, 500):
        inputs, targets = batch
        pred = test_fn(inputs)
        all_preds = all_preds + pred
    all_preds = np.asarray(all_preds).flatten()
    cm = confusion_matrix(y_val, all_preds)
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("plots/confussion_matrix.png")
    plt.show()
    # for_csv = []
    # for i,x in enumerate(all_preds):
    #     for_csv.append([i,x])
    # with open("pred_try6.csv", "w",newline='') as f:
    #     writer = csv.writer(f,delimiter=',')
    #     writer.writerows(for_csv)



if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        main(**kwargs)