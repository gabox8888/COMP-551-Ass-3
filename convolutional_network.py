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
import csv

import lasagne

class Convolutional_Network(object):

    def __init__(self,x_train_data,y_train_data,x_test_data):
        self.x_train,self.y_train,self.x_val,self.y_val,self.x_test,self.y_test = self.load_dataset(x_train_data,y_train_data,x_test_data)
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
        test_pred = T.argmax(test_prediction,axis=1)
                        
        self.train_fn = theano.function([self.input_var, self.target_var], loss, updates=updates)
        self.val_fn = theano.function([self.input_var, self.target_var], [test_loss, test_acc])
        self.test_fn = theano.function([self.input_var],[test_pred])

    
    def load_dataset(self,x_train_data,y_train_data,x_test_data):

        x = np.fromfile(x_train_data, dtype='uint8')
        x = x.reshape((-1,1,60,60))
        x = x/ np.float32(256)
        x_test = np.fromfile(x_test_data, dtype='uint8')
        x_test = x_test.reshape((-1,1,60,60))
        x_test = x_test/ np.float32(256)
        y_test = np.zeros(len(x_test))

        y = np.loadtxt(open(y_train_data, "rb"), delimiter=",", skiprows=1)
        y = scipy.delete(y, 0, 1)
        y = np.asarray(y).reshape(-1)
        y = y.astype(int)
        
        x_train = x[:80000]
        y_train = y[:80000]
        x_val = x[80000:]
        y_val = y[80000:]

        return x_train,y_train,x_val,y_val,x_test,y_test

    def apply_filters(self,batch,centers_file=None,):
        def find_centers(img, original):
            current_avg = 0
            subJ = 0
            subK = 0
            image = img[0:28,0:28]
            for j in range(40):
                for k in range(40):
                    new_img = img[j:j+20,k:k+20]
                    if new_img.mean() > current_avg :
                        subJ = j
                        subK = k
                        stuff = np.zeros(shape=(28,28),dtype='f')
                        stuff[4:24,4:24] = original[j:j+20,k:k+20]
                        current_avg = new_img.mean()
                        test_img =stuff
                        image = test_img
            temp_img = np.copy(img)
            temp_img[subJ:subJ+20,subK:subK+20] = np.zeros((20,20),dtype=int)
            return image, temp_img
        def centers_from_file(img,coords):
            x1,y1 = (int(coords[0]),int(coords[1]))
            x2,y2 = (int(coords[2]),int(coords[3]))
            img1 = np.zeros(shape=(28,28),dtype='f')
            img2 = np.zeros(shape=(28,28),dtype='f')
            img1[4:24,4:24] = img[y1-10:y1+10,x1-10:x1+10]
            img2[4:24,4:24] = img[y2-10:y2+10,x2-10:x2+10]
            return img1,img2

        new_batch = []
        if centers_file == None:
            for i in batch:
                img = i[0]
                image1,i = find_centers(img,img)
                image2,j = find_centers(i,img)
                new_batch.append((image1,image2))
        else:
            centers = np.loadtxt(open(centers_file, "rb"), delimiter=",", skiprows=0)
            for i,x in enumerate(batch):
                new_batch.append(centers_from_file(x[0],centers[i]))

        return new_batch   

    
    def build_main_network(self,input_var=None):
        network = lasagne.layers.InputLayer(shape=(None, 1, 60, 60),input_var=input_var)
        network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(5, 5),nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
        network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(5, 5),nonlinearity=lasagne.nonlinearities.rectify)
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
        network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),nonlinearity=lasagne.nonlinearities.rectify)
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
        network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5),num_units=500,nonlinearity=lasagne.nonlinearities.rectify)
        network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5),num_units=256,nonlinearity=lasagne.nonlinearities.rectify)
        network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5),num_units=19,nonlinearity=lasagne.nonlinearities.softmax)

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


    def train(self,num_epochs=200,epoch_size=500,debug=False,serialize=None):
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
        if serialize != None:
            np.savez(serialize, *lasagne.layers.get_all_param_values(self.network))
        return self

    def deserialize(self,filename):
        with np.load(filename) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.network, param_values)
        return

    def fit(self,filename=None,debug=False):
        all_preds = self.test_fn(self.x_test[:100])
        all_preds = np.asarray(all_preds).flatten()
        if filename != None:
            for_csv = []
            for i,x in enumerate(all_preds):
                for_csv.append([i,x])
            with open(filename, "w",newline='') as f:
                writer = csv.writer(f,delimiter=',')
                writer.writerows(for_csv)

        return all_preds



