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
waveletFile = "waveletTrain.csv"
# Data
dx     		= "train_x.bin"
dy			= "train_y.csv"
test_x		= "test_x.bin"

### Learning parameters ###
runLogReg = False 			# Run logisitc regression
transform = False			# Transform concatenated data via PCA
crossvalidateAnn = True		# Run cross-validation via the implemented ANN

def main():
	
	# Read data
	surfData     = np.loadtxt(surfFile,    delimiter=",") # 100000 x 50
	minEvalData  = np.loadtxt(minEvalFile, delimiter=",") # 100000 x 50
	hogData      = np.loadtxt(hogFile, 	   delimiter=",") # 100000 x 50
	waveData     = np.loadtxt(waveletFile, delimiter=",") # 100000 x 33
	combinedData = [ np.concatenate((a,b,c,d)) for a,b,c,d in zip(surfData,minEvalData,hogData,waveData) ]	
	train_x = (np.fromfile(dx, dtype='uint8')).reshape((100000,60,60))
	train_y = np.array([ q.split(",")[1].strip() for q in open(dy,"r").readlines()[1:] ])

	# Transform via PCA
	dims = [20,50,100]
	if transform:
		print("Transforming")
		combinedTransformedData = []
		for dim in dims:
			pca = PCA(n_components=dim)
			pca.fit(combinedData) 
			combinedTransformedData.append( pca.transform( combinedData ) )

	##### Run logistic regression learning (sklearn) #####
	if runLogReg:
		print("Logistic Regression")
		alldata = {"surf" : surfData, "me" : minEvalData, "hog" : hogData, "wave" : waveData}
		for ii,d in enumerate(dims):
			alldata['ctd' + str(d)] = combinedTransformedData[ii]
		for regc in [0.1, 1.0, 10.0]:
			print("\nC = " + str(regc))
			for curr in alldata:
				clf = LogisticRegression(C=regc)
				print("On " + curr)
				data = np.array(     alldata[curr]     )
				print("\tStarting CV (dims = " + str( data.shape ) + ")")
				scores = cross_validation.cross_val_score(clf, data, train_y, cv=2)
				print("\tSKLearn scores")
				print("\t"+str(scores) + " -> " + str(np.mean(scores)))
		sys.exit(0)

	##### Run Learning (custom ffann) #####
	print("FFANN section")
	ss, tes = 20000, 1000
	numclasses = 19
	data = np.array(     waveData     ) ## <--- Change to alter input data type
	x, y = data[0:ss], train_y[0:ss]
	print("Starting CV (dims = " + str( x.shape ) + ")")
	if crossvalidateAnn:
		FeedForwardArtificialNeuralNetwork.crossValidate(x,y,maxIters=3)
	else:
		d = data.shape[1]
		ffann = FeedForwardArtificialNeuralNetwork(
			d, 
			numHiddenLayers = 2, 
			alpha = 0.05, 
			sizeOfHiddenLayers = [50, numclasses], #[35, 35, numclasses],
			maxIters = 3)
		ffann.display()
		ffann.train(x, y) 
		y_ann = ffann.predict( data[ss:ss+tes] )
		ffann.display()
		outvals = [str(y) for y in y_ann]
		outY_val = train_y[ss:ss+tes]
		anny = ffann.checkPerformance(outY_val, outvals)
		print(y_ann[0:20])
		print(anny[0:20])
		print(outY_val[0:20])


if __name__ == '__main__': main()





