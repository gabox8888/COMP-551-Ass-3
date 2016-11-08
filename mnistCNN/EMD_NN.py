from sklearn.neighbors import KNeighborsClassifier
import numpy as np, os, sys, warnings
from sklearn import metrics as skm
from keras.datasets import mnist
from pyemd import emd

print("Read MNIST Data")
(x_train, y1), (x_test, y2) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), 28, 28, 1)) #np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), 28, 28, 1)) #np.prod(x_test.shape[1:])))
x = np.concatenate((x_train,x_test))
y = np.concatenate((y1,y2))

print("Reading two-digit sum data")
train_x = (np.fromfile("trainBW.bin", dtype='uint8')).reshape((100000,60,60))
train_y = np.array([ q.split(",")[1].strip() for q in open("train_y.csv","r").readlines()[1:] ])
train_x = train_x[0:20]
train_y = train_y[0:20]

#print("Norming and thresholding")
#targInds = train_x < (255 * 0.8)
#train_x[ targInds ] = 0.0
#train_x = train_x.astype('float32')
#x_train /= 255

print("Extracting centers")
cenfile = "trainCenter.csv"
with open(cenfile,"r") as cens:
	cenlines = cens.readlines()
	extractNums = lambda lin: [int(float(r.strip())) for r in lin.split(",")]
	cens = [ extractNums(lin) for lin in cenlines ]

print("Generating EMD distance matrix")
n = 28
arrLen = n * n
def distanceWeight(ind1,ind2):
	x1, y1, x2, y2 = ind1 % n, ind1 / n, ind2 % n, ind2 / n
	return np.sqrt( (x1 - x2)**2 + (y1 - y2)**2 )
distanceMatrix = np.zeros((arrLen,arrLen)).astype('float64')
for i in range(0,arrLen):
	for j in range(0,arrLen):
		distanceMatrix[i][j] = distanceWeight(i,j)
print("\tShape = " + str(distanceMatrix.shape))

print("Extract MNIST representatives")
numExamples = 10
labelToExamples = {}
fxq = lambda q: x[q].reshape((n*n)).astype('float64')
for i in range(0,10):
	examples = []
	for q,labelCurr in enumerate(y):
		if int(labelCurr) == i: examples.append( fxq(q) )
		if len(examples) == numExamples: break
	labelToExamples[i] = examples

print("Nearest Neighbours via EMD")
dx, dy, p = 28 / 2, 28 / 2, -1
totalN, anss = len(train_x), []
def computeEMDs(digit):
	emds = np.zeros(10) #np.array([float('inf') for i in range(0,10)])
	for k in range(0,10):
		currExamples = labelToExamples[k]
		for example in currExamples:
			emdCurr = emd(digit, example, distanceMatrix, extra_mass_penalty=p)
			emds[k] += emdCurr
		emds[k] /= float(numExamples)
	return emds
with warnings.catch_warnings():
	warnings.filterwarnings("ignore",category=DeprecationWarning)
	assert dx==dy, "Currently requires dx=dy"
	print("\tTotal n = " + str(totalN))
	for i,img in enumerate(train_x):
		# Print status
		if   i == totalN / 4: print("\t25%")
		elif i == totalN / 2: print("\t50%")
		elif i == 3*totalN/4: print("\t75%")
		# Extract current centers
		currCens = cens[i]
		x1,y1 = currCens[0]-1, currCens[1]-1
		x2,y2 = currCens[2]-1, currCens[3]-1
		tempimg = np.lib.pad(img, (dx,dy), 'constant') # zeros
		# Subimages as arrays
		digit1 = tempimg[y1:y1+2*dy:1, x1:x1+2*dx:1].reshape( (n*n) )
		digit2 = tempimg[y2:y2+2*dy:1, x2:x2+2*dx:1].reshape( (n*n) )
		# Compute EMDs
		print("\tlabsStart")
		closestLabel1 = np.argmin( computeEMDs(digit1.astype('float64')) )
		print("\tlabelMid")
		closestLabel2 = np.argmin( computeEMDs(digit2.astype('float64')) )
		print("\tlabsEnd")
		# Predicted classes
		s = closestLabel1 + closestLabel2
		print(s)
		print(train_y[i])
		# Save answer
		anss.append( str(s) )

print("Compute accuracies")
print(anss[0:10])
print(train_y[0:10])
print( skm.accuracy_score(train_y,anss) )
