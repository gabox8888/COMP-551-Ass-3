COMP 551 Assignment 3

###
To run logistic regression (by hand implementation):
	- Set the file locations of the extracted feature representations (SURF, etc...) in the script featuresTest.py
	- run each part of the the python script LogisticRegression.ipynb it will create a graph of the accuracy for each different feature vector on a validation and test set. 

###
To run logistic regression (using sklearn):
	- Set the file locations of the extracted feature representations (SURF, etc...) in the script featuresTest.py
 	- Run the python script featuresTest.py
 	  Set the following variables to true in the script
			runLogReg = True
			transform = True
	  This will	run logistic regression across the given hard-coded parameters and data sets.

###
To run the implemented neural network
	Running the network on mock data
		- Run the script ffann.py 
		- Parameters can be set in the main() method of that script
	Running the network on the new image data 
		- Set the following parameters in featuresTest.py
			runLogReg = False
			transform = False
		- For hyper-parameter selection via cross-validation, also set
			crossvalidateAnn = True
		  Else, set it to false to run a specific model parameterization
		  Further, to get the results on the test set, let
		  	runAnnOnTestSet = True
		- Run featuresTest.py

###
To run the CNN using the estimated digit centers (i.e. MNIST-CNN)
	- Ensure the following required files are present
		+ Estimated centers file
		+ Binary image forms of input two-digit images
	- Train a CNN on the original MNIST data set by running kerasMnistModified.py.
	  Parameters are hard-coded. It saves an hdf5 model file used below.
	- Run kerasCenterPred.py to perform the subimage extraction and prediction with the above model.

###
	To run Theano CNN
			Ensure training.bin and test.bin files are in data folder
			Run driver.py

