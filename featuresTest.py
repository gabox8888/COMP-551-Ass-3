import os, sys, numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.decomposition import PCA
from ffann import FeedForwardArtificialNeuralNetwork

### Input files ###
# Features
surfFile    = "surfTrain.csv"
minEvalFile = "minEigenValueTrain.csv"
hogFile     = "hogTrain.csv" 
# Data
dx     		= "train_x.bin"
dy			= "train_y.csv"
test_x		= "test_x.bin"

def main():
	
	# Read data
	surfData     = np.loadtxt(surfFile,    delimiter=",") # 100000 x 50
	minEvalData  = np.loadtxt(minEvalFile, delimiter=",") # 100000 x 50
	hogData      = np.loadtxt(hogFile, 	   delimiter=",") # 100000 x 50
	combinedData = [ np.concatenate((a,b,c)) for a,b,c in zip(surfData,minEvalData,hogData) ]
	train_x = (np.fromfile(dx, dtype='uint8')).reshape((100000,60,60))
	train_y = np.array([ q.split(",")[1].strip() for q in open(dy,"r").readlines()[1:] ])

	# Raw data
	rawOnly = False
	if rawOnly:
		train_x = train_x.reshape((100000,60*60))
		tx = train_x[0:1000]
		ty = train_y[0:1000]
		clf = LogisticRegression()
		scores = cross_validation.cross_val_score(clf, train_x, train_y, cv=2)
		print(scores)
		sys.exit(0)

	# Transform
	print("Transforming")
	dim = 50
	pca = PCA(n_components=dim)
	pca.fit(combinedData) #[0:80000])
	combinedTransformedData = pca.transform( combinedData )

	# Run Learning (sklearn)
	data = minEvalData
	clf = LogisticRegression()
	runLogReg = False
	clf2 =  SVC(C=1.0, kernel='poly', degree=3, gamma=0.0, 
				coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, 
				class_weight=None, verbose=False, max_iter=-1, random_state=None) #LinearSVC() #(kernel='linear', C=1)
	if runLogReg:
		print("Starting CV (dim = " + str(dim) + ")")
		scores = cross_validation.cross_val_score(clf, combinedTransformedData, train_y, cv=2)
		print("SKLearn scores")
		print(scores)
		sys.exit(0)

	# Run Learning (custom)
	print("FFANN section")
	ss = 15000
	x, y = combinedTransformedData[0:ss], train_y[0:ss]
	FeedForwardArtificialNeuralNetwork.crossValidate(x,y,maxIters=3)
#	ffann = FeedForwardArtificialNeuralNetwork(dim, 
#	    numHiddenLayers = 3, 
#	    alpha = 0.08, 
#	    sizeOfHiddenLayers = [25, 20, 19],
#	    maxIters = 1)
#	ffann.display()
#	ss, ts = 10000, 15000
#	xtrain, ytrain, xtest, ytest = combinedTransformedData[0:ss], train_y[0:ss], combinedTransformedData[ss:ts], train_y[ss:ts]
#	ffann.train(xtrain,ytrain) 
#	y_ann = ffann.predict(xtest)
#	ffann.display()
#	ffann.checkPerformance(ytest,y_ann)


if __name__ == '__main__': main()
