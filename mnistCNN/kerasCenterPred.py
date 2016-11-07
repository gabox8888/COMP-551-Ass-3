from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.models import load_model
from sklearn.neighbors import KNeighborsClassifier
import numpy as np, os, sys
from keras.preprocessing.image import ImageDataGenerator
import warnings

# Load in MNIST predictor
print("Loading mnist model")
recognizer = load_model('mnist_predictor_shifting_noise.h5')	
print("Reading mnist sum data")
trainingSet, testSet = "trainBW.bin", "testBW.bin"  # "train_x.bin", "test_x.bin"  #"trainBW.bin"  #"testBW.bin"
train_x = (np.fromfile(trainingSet, dtype='uint8')).reshape((100000,60,60))
train_y = np.array([ q.split(",")[1].strip() for q in open("train_y.csv","r").readlines()[1:] ])
test_x = (np.fromfile(testSet, dtype='uint8')).reshape((20000,60,60))

# Preprocess new mnist sum images
print("Norming and thresholding")
threshold = False
if threshold:
	targInds = train_x < (255 * 0.9)
	train_x[ targInds ] = 0.0
	train_x = train_x.astype('float32')
	train_x /= 255
# Extract centers determined by k-means
print("Extracting centers")
cenfile = "trainCenter.csv"
cenfile_test = "testCenter.csv"
def extractCenters(c):
	with open(c,"r") as cens:
		cenlines = cens.readlines()
		extractNums = lambda lin: [int(float(r.strip())) for r in lin.split(",")]
		cens = [ extractNums(lin) for lin in cenlines ]	
	return cens
trainCens = extractCenters(cenfile)
testCens = extractCenters(cenfile_test)
print("Extracting subimages")
dx, dy = 28 / 2, 28 / 2
brm = 2
def getPredictions(tx,cs):
	anss = []
	totalN = len(tx)
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore",category=DeprecationWarning)
		assert dx==dy, "Currently requires dx=dy"
		for i,img in enumerate(tx):
			if   i == totalN / 4: print("\t25%")
			elif i == totalN / 2: print("\t50%")
			elif i == 3*totalN/4: print("\t75%")
			currCens = cs[i]
			# Get estimated centers
			x1,y1 = currCens[0]-1, currCens[1]-1
			x2,y2 = currCens[2]-1, currCens[3]-1
			# Retrieve subimage
			tempimg = np.lib.pad(img, (dx,dy), 'constant') # zeros 
			digit1a = tempimg[y1:y1+2*dy:1, x1:x1+2*dx:1]
			digit1w = digit1a[brm:2*dy-brm, brm:2*dx-brm]
			digit1  = np.lib.pad(digit1w, (brm,brm), 'constant').reshape((1,28,28,1))
			digit2a = tempimg[y2:y2+2*dy:1, x2:x2+2*dx:1]
			digit2w = digit2a[brm:2*dy-brm, brm:2*dx-brm]
			digit2  = np.lib.pad(digit2w, (brm,brm), 'constant').reshape((1,28,28,1))
			# Run learner on subimages
			preds1 = recognizer.predict( digit1 )[0] 
			preds2 = recognizer.predict( digit2 )[0] 
			val1 = np.argmax(preds1)
			val2 = np.argmax(preds2)
			s = val1 + val2
			anss.append( str(s) ) 
	return anss
computeTrainingSetAcc = False
if computeTrainingSetAcc:
	anss = getPredictions(train_x,trainCens)
	# Examine score
	print("Validation Accuracy")
	from sklearn import metrics as skm
	print( skm.accuracy_score(train_y,anss) )

print("Test set output")
outputFile = "cnn-mnist.csv"
with open(outputFile,"w") as testOutFile:
	anss = getPredictions(test_x,testCens)
	testOutFile.write("Id,Prediction\n")
	for w,ans in enumerate( anss ):
		testOutFile.write( str(w) + "," + str(ans) + "\n" )




