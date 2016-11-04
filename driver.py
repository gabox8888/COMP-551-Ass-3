import convolutional_network

def main():
    x_train = 'data/trainBW.bin'
    y_train = 'data/train_y.csv'
    x_test = 'data/testBW.bin'
    cnn = convolutional_network.Convolutional_Network(x_train,y_train,x_test)
    cnn.train(debug=True,num_epochs=4)
    
if __name__ == '__main__':main()