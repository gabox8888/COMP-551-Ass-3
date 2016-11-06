#from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
#from keras.models import Model, load_model
from sklearn.neighbors import KNeighborsClassifier
import numpy as np, os, sys, warnings
from sklearn import metrics as skm
#from keras.preprocessing.image import ImageDataGenerator
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
#			print('--')
#			print(example)
#			print(digit)
			emdCurr = emd(digit, example, distanceMatrix, extra_mass_penalty=p)
#			print(emdCurr)
#			if emds[k] > emdCurr: emds[k] = emdCurr
			emds[k] += emdCurr
		emds[k] /= float(numExamples)
#	print(emds)
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


#
#
# startInd, endInd = dx + 1, 60 - dy - 1
# stepSize = 5
# tempimg = np.lib.pad(img, (1,1), 'constant')
# dmap = {}
# for xind in range(startInd, endInd, stepSize):
# 	for yind in range(startInd, endInd, stepSize):
# 		currDigit = tempimg[yind-dy:yind+dy:1, xind-dx:xind+dx:1]
# 		currDigit = currDigit.reshape((1,28,28,1))
# 		# Encoded form
# 		v = encoder.predict(currDigit)
# 		# Reshape encoded vectors
# 		v = v.reshape((128,))
# 		# Predicted classes
# 		p = knn.kneighbors(v,n_neighbors=1,return_distance=True)
# 		dist = p[0][0][0]
# 		predLabel = y[ p[1][0][0] ] # MNIST Label
# 		if dmap.get(predLabel) is None or dmap[ predLabel ] > dist:
# 			dmap[ predLabel ] = dist
# # Save answer
# print(dmap)
# ans1 = min(dmap, key=dmap.get)
# del dmap[ans1]
# ans2 = min(dmap, key=dmap.get)
# ans = int(ans1) + int(ans2)
# anss.append( str( ans  ) )
#
#
# ###
#
# # Whether to generate a new model for autoencoding mnist or test knn with it on the new data
# makeModel = False
# # Prediction mode
# predMode = 1
# ###
#
# # Test nearest neighbours on patches
# if predMode == 1:
# 	# Plotting imports
# 	from matplotlib import pyplot as plt
#
# 	# Load MNIST data
# 	print("Reading mnist data")
#         from keras.datasets import mnist
#         (x_train, y1), (x_test, y2) = mnist.load_data()
# 	x_train = x_train.astype('float32') / 255.
#         x_test = x_test.astype('float32') / 255.
#         x_train = x_train.reshape((len(x_train), 28, 28, 1)) #np.prod(x_train.shape[1:])))
#         x_test = x_test.reshape((len(x_test), 28, 28, 1)) #np.prod(x_test.shape[1:])))
# 	x = np.concatenate((x_train,x_test))
# 	y = np.concatenate((y1,y2))
# 	print(x.shape)
# 	print(y.shape)
#
# 	# Load encoder
# 	print("Loading encoder")
# 	encoder = load_model('mnist_encoder_wshifting.h5')
#
# #	test = np.array([ a for a in range(0,784) ]).reshape((1,784))
# #	encoder.predict(test)
# #	sys.exit(0)
#
# 	# Encode mnist in low dim
# 	print("Encoding mnist data")
# 	encodedVecs = encoder.predict(x)
# 	encodedVecs = encodedVecs.reshape((len(encodedVecs),128))
#
# 	# Build nearest neighbour predictor for mnist
# 	print("Building KNN classifier for mnist")
# 	knn = KNeighborsClassifier(n_neighbors=50)
# 	print(encodedVecs.shape)
# 	knn.fit(encodedVecs, y)
#
# 	# Read in new images
# 	print("Reading sum data")
# 	train_x = (np.fromfile("train_x.bin", dtype='uint8')).reshape((100000,60,60))
# 	train_y = np.array([ q.split(",")[1].strip() for q in open("train_y.csv","r").readlines()[1:] ])
# 	train_x = train_x[0:100]
# #	test_x = train_x[80000:]
# 	train_y = train_y[0:100]
# #	test_y = test_y[80000:]
#
# 	# Preprocess new images
# 	print("Norming and thresholding")
# 	targInds = train_x < (255 * 0.8)
# 	train_x[ targInds ] = 0.0
# 	train_x = train_x.astype('float32')
# #	X_test = X_test.astype('float32')
# 	x_train /= 255
# #	X_test /= 255
#
# 	# Extract centers
# 	print("Extracting centers")
# 	cenfile = "trainCenter.csv"
# 	with open(cenfile,"r") as cens:
# 		cenlines = cens.readlines()
# 		extractNums = lambda lin: [int(float(r.strip())) for r in lin.split(",")]
# 		cens = [ extractNums(lin) for lin in cenlines ]
#
# 	# Extract subimages
# 	print("Extracting subimages")
# 	useCenters = False
# 	dx, dy = 28 / 2, 28 / 2
# 	totalN = len(train_x)
# 	anss = []
# 	nonsim = False
# 	plt.gray()
# 	with warnings.catch_warnings():
#    		warnings.filterwarnings("ignore",category=DeprecationWarning)
# 		assert dx==dy, "Currently requires dx=dy"
# 		for i,img in enumerate(train_x):
# 			if   i == totalN / 4: print("\t25%")
# 			elif i == totalN / 2: print("\t50%")
# 			elif i == 3*totalN/4: print("\t75%")
#
# 			if useCenters:
# 				print("Using centers")
# 				currCens = cens[i]
# 				x1,y1 = currCens[0]-1, currCens[1]-1
# 				x2,y2 = currCens[2]-1, currCens[3]-1
# 				tempimg = np.lib.pad(img, (dx,dy), 'constant') # zeros
# 				# Subimage
# 				digit1 = tempimg[y1:y1+2*dy:1, x1:x1+2*dx:1].reshape((1,28,28,1))
# 		#		digit1 = tempimg[y1-dy:y1+dy, x1-dx:x1+dx].reshape((1,28,28,1))
# 		#		digit1 = digit1.reshape(1,-1)
# 				#.reshape((1,-1)) #.reshape((1,784))
# 				digit2 = tempimg[y2:y2+2*dy:1, x2:x2+2*dx:1].reshape((1,28,28,1))
# 		#		digit2 = tempimg[y2-dy:y2+dy, x2-dx:x2+dx].reshape((1,28,28,1))
# 				print("Digits")
# 				plt.imshow(digit1.reshape((28,28)))
# 				plt.title("Digit 1")
# 				plt.show()
# 				plt.imshow(digit2.reshape((28,28)))
# 				plt.title("Digit 2")
# 				plt.show()
# 		#		digit2 = digit2.reshape(1,-1)
# 				#.reshape((1,-1)) #.reshape((1,784))
# 				# Encoded form
# 				vec1, vec2 = encoder.predict(digit1), encoder.predict(digit2)
# 				# Reshape encoded vectors
# 				vec1 = vec1.reshape((128,))
# 				vec2 = vec2.reshape((128,))
# 				# Predicted classes
# 				p1 = knn.kneighbors(vec1,n_neighbors=3,return_distance=True)  #predict(vec1)
# 				p2 = knn.kneighbors(vec2,n_neighbors=3,return_distance=True)
# 				print(p1)
# 				print(p2)
# 				print("x")
# 				plt.imshow( x[ p1[1][0][0] ].reshape((28,28)) )
# 				plt.title( "Image Matched (X) 1" )
# 				plt.show()
# 				plt.imshow( x[ p2[1][0][0] ].reshape((28,28)) )
# 				plt.title( "Image Matched (X) 2" )
# 				plt.show()
# 				val1 = y[ p1[1][0][0] ]
# 				val2 = y[ p2[1][0][0] ]
# 				s = val1 + val2
# 		#		print("--")
# 				print(str(val1)+" + "+str(val2)+" = "+ str(s))
# 				print(train_y[i])
# 				# Save answer
# 				anss.append( str(s) )
# 			elif nonsim:
# 				startInd, endInd = dx + 1, 60 - dy - 1
# 				stepSize = 5
# 				tempimg = np.lib.pad(img, (1,1), 'constant')
# 				dmap = {}
# 				for xind in range(startInd, endInd, stepSize):
# 					for yind in range(startInd, endInd, stepSize):
# 						currDigit = tempimg[yind-dy:yind+dy:1, xind-dx:xind+dx:1]
# 						#xind-dx:xind+dx:1, yind-dy:yind+dy:1]
# 						currDigit = currDigit.reshape((1,28,28,1))
# 						# Encoded form
# 						v = encoder.predict(currDigit)
# 						# Reshape encoded vectors
# 						v = v.reshape((128,))
# 						# Predicted classes
# 						p = knn.kneighbors(v,n_neighbors=1,return_distance=True)
# 						dist = p[0][0][0]
# 						predLabel = y[ p[1][0][0] ] # MNIST Label
# 						if dmap.get(predLabel) is None or dmap[ predLabel ] > dist:
# 							dmap[ predLabel ] = dist
# 				# Save answer
# 				print(dmap)
# 				ans1 = min(dmap, key=dmap.get)
# 				del dmap[ans1]
# 				ans2 = min(dmap, key=dmap.get)
# 				ans = int(ans1) + int(ans2)
# 				anss.append( str( ans  ) )
# 			else:
# 				startInd, endInd = dx + 1, 60 - dy - 1
# 				stepSize = 3
# 				tempimg = np.lib.pad(img, (1,1), 'constant')
# 				dmap = {}
# 				K, thresh = 50, 0.81
# 				for xind in range(startInd, endInd, stepSize):
# 					for yind in range(startInd, endInd, stepSize):
# 						currDigit = tempimg[yind-dy:yind+dy:1, xind-dx:xind+dx:1]
# 						 # [xind-dx:xind+dx:1, yind-dy:yind+dy:1]
# 						currDigitrs = currDigit.reshape((1,28,28,1))
# 						currDigit = currDigit.reshape((28,28))
# 						# Encoded form
# 						v = encoder.predict(currDigitrs)
# 						# Reshape encoded vectors
# 						v = v.reshape((128,))
# 						# Predicted classes
# 						p = knn.kneighbors(v,n_neighbors=K,return_distance=True)
# 						inds = p[1][0]
# 				#		dist = p[0][0][0]
# 				#		predLabel = y[ p[1][0][0] ] # MNIST Label
# 						bestInd = 0; lowestDist = float("inf")
# 						for ind in inds:
# 							sim = 0
# 							mnistImg = x[ind].reshape((28,28))
# 							for k in range(0,28):
# 								for j in range(0,28):
# 									if mnistImg[k][j] > thresh:
# 										if currDigit[k][j] > thresh:
# 											sim += 1
# 							dist = 1.0 / (sim + 1)
# 							if dist < lowestDist:
# 								lowestDist = dist
# 								bestInd = ind
# 						predLabel = y[ bestInd ]
# 						if dmap.get(predLabel) is None or dmap[ predLabel ] > lowestDist:
# 							dmap[ predLabel ] = lowestDist
# 				# Save answer
# 				ans1 = min(dmap, key=dmap.get)
# 				del dmap[ans1]
# 				ans2 = min(dmap, key=dmap.get)
# 				ans = int(ans1) + int(ans2)
# 				print(str(ans1)+" + "+str(ans2)+" = "+str(ans))
# 				print( train_y[i] )
# 				anss.append( str( ans  ) )
#
#
# 	# Examine score
# 	print("Accuracy")
# 	print(anss[0:20])
# 	print(train_y[0:20])
# 	from sklearn import metrics as skm
# 	print( skm.accuracy_score(train_y,anss) )
#
#
#
# else:
# 	# Load in MNIST predictor
# 	print("Loading mnist model")
# 	recognizer = load_model('mnist_predictor_shifting_noise.h5')
#
# 	#
# 	print("Reading mnist sum data")
#         train_x = (np.fromfile("train_x.bin", dtype='uint8')).reshape((100000,60,60))
#         train_y = np.array([ q.split(",")[1].strip() for q in open("train_y.csv","r").readlines()[1:] ])
#         train_x = train_x[0:100000]
# #       test_x = train_x[80000:]
#         train_y = train_y[0:100000]
# #       test_y = test_y[80000:]
#
#         # Preprocess new images
#         print("Norming and thresholding")
#         targInds = train_x < (255 * 0.8)
#         train_x[ targInds ] = 0.0
#         train_x = train_x.astype('float32')
# #       X_test = X_test.astype('float32')
#         train_x /= 255
# #       X_test /= 255
#
#         # Extract centers
#         print("Extracting centers")
#         cenfile = "trainCenter.csv"
#         with open(cenfile,"r") as cens:
#                 cenlines = cens.readlines()
#                 extractNums = lambda lin: [int(float(r.strip())) for r in lin.split(",")]
#                 cens = [ extractNums(lin) for lin in cenlines ]
#
# 	print("Extracting subimages")
# #	useCenters = False
# 	dx, dy = 28 / 2, 28 / 2
# 	totalN = len(train_x)
# 	anss = []
# #	nonsim = False
# #	plt.gray()
# 	with warnings.catch_warnings():
#    		warnings.filterwarnings("ignore",category=DeprecationWarning)
# 		assert dx==dy, "Currently requires dx=dy"
# 		for i,img in enumerate(train_x):
# 			if   i == totalN / 4: print("\t25%")
# 			elif i == totalN / 2: print("\t50%")
# 			elif i == 3*totalN/4: print("\t75%")
#
# 			currCens = cens[i]
# 			x1,y1 = currCens[0]-1, currCens[1]-1
# 			x2,y2 = currCens[2]-1, currCens[3]-1
# 			tempimg = np.lib.pad(img, (dx,dy), 'constant') # zeros
# 			# Subimage
# 			digit1 = tempimg[y1:y1+2*dy:1, x1:x1+2*dx:1].reshape((1,28,28,1))
# 	#		digit1 = tempimg[y1-dy:y1+dy, x1-dx:x1+dx].reshape((1,28,28,1))
# 	#		digit1 = digit1.reshape(1,-1)
# 			#.reshape((1,-1)) #.reshape((1,784))
# 			digit2 = tempimg[y2:y2+2*dy:1, x2:x2+2*dx:1].reshape((1,28,28,1))
# 	#		digit2 = tempimg[y2-dy:y2+dy, x2-dx:x2+dx].reshape((1,28,28,1))
# 	#		print("Digits")
# 	#		plt.imshow(digit1.reshape((28,28)))
# 	#		plt.title("Digit 1")
# 	#		plt.show()
# 	#		plt.imshow(digit2.reshape((28,28)))
# 	#		plt.title("Digit 2")
# 	#		plt.show()
# 			preds1 =  recognizer.predict( digit1 )[0]
# 			preds2 =  recognizer.predict( digit2 )[0]
# 			val1 = np.argmax(preds1)
# 			val2 = np.argmax(preds2)
# 	#		digit2 = digit2.reshape(1,-1)
# 			#.reshape((1,-1)) #.reshape((1,784))
# 			# Encoded form
# 	#		vec1, vec2 = encoder.predict(digit1), encoder.predict(digit2)
# 			# Reshape encoded vectors
# 	#		vec1 = vec1.reshape((128,))
# 	#		vec2 = vec2.reshape((128,))
# 			# Predicted classes
# 	#		p1 = knn.kneighbors(vec1,n_neighbors=3,return_distance=True)  #predict(vec1)
# 	#		p2 = knn.kneighbors(vec2,n_neighbors=3,return_distance=True)
# 	#		print(p1)
# 	#		print(p2)
# 	#		print("x")
# 	#		plt.imshow( x[ p1[1][0][0] ].reshape((28,28)) )
# 	#		plt.title( "Image Matched (X) 1" )
# 	#		plt.show()
# 	#		plt.imshow( x[ p2[1][0][0] ].reshape((28,28)) )
# 	#		plt.title( "Image Matched (X) 2" )
# 	#		plt.show()
# 	#		val1 = y[ p1[1][0][0] ]
# 	#		val2 = y[ p2[1][0][0] ]
# 			s = val1 + val2
# 	#		print("--")
# #			print(str(val1)+" + "+str(val2)+" = "+ str(s))
# #			print(train_y[i])
# 			# Save answer
# 			anss.append( str(s) )
#
# 	 # Examine score
#         print("Accuracy")
#         print(anss[0:20])
#         print(train_y[0:20])
#         from sklearn import metrics as skm
#         print( skm.accuracy_score(train_y,anss) )
