# Implementation of a fully connected, feed forward ANN
# 10/30/16

import os, sys, numpy as np, random as r, math

# Used for performance measurement
from sklearn import metrics as skm

# Used for preprocessing the given image data
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# For testing the ANN
def main():
  # Generate some practice data
  d = 3
  [xtrain,ytrain,xtest,ytest] = practiceMultiData(d, nc = 9, ntrain=450, ntest=150) #practiceData(d,2500,50)
  # Create an ANN
  ffann = FeedForwardArtificialNeuralNetwork(d, 
    numHiddenLayers = 3, 
    alpha = 0.1, 
    sizeOfHiddenLayers = [7, 7, 9])

  ffann.display()
  ffann.train(xtrain,ytrain) 
  y_ann = ffann.predict(xtest)
  ffann.display()
  ffann.checkPerformance(ytest,y_ann)

### Implementation of Feed-Forward Fully Connected Artificla Neural Network ###
# Uses back-propagation with gradient descent, based on squared error loss 
# Note: multiclass input should be given as an integer; it will be converted
# to a 1-hot encoding. 
class FeedForwardArtificialNeuralNetwork(object):

  ### Layer Class ###
  # ANN layer object
  class AnnLayer(object):
    def __init__(self, numNodes, numConnectingNodes, layerNum):
      self.nodes = [ FeedForwardArtificialNeuralNetwork.AnnNode(numConnectingNodes,layerNum) for _ in range(0,numNodes) ] 
      self.numNodes, self.nWeightsPerNode = len(self.nodes), numConnectingNodes + 1
    # Update the layer in backprop
    def update(self,alpha,delta,x_layerIn):
      for i,node in enumerate(self.nodes):
        node.update(alpha, delta[i], x_layerIn)
    # Layer-level integration of artificial neural input
    def integrate(self,stim):
      return np.array([n.integrate(stim) for n in self.nodes])

  ### Node Class ###
  # ANN node object
  class AnnNode(object): 
    # Constructor
    def __init__(self, numBackwardsNodes, layerNum):
      self._sigmoid, self._weightSigma = lambda z: 1.0 / (1.0 + math.exp(-z)), 0.01
      # Initialize weights to a small number (note bias)
      self.weights = np.array([ np.random.normal(0.0, self._weightSigma) 
                                for _ in range(0,numBackwardsNodes+1) ])
      self.delta, self.layerIndex, self.nWeights = None, layerNum, len(self.weights)
    # Sigmoid filter linear combination (implicit 1 for bias weight term)
    def integrate(self,stim):
      return self._sigmoid(np.dot(self.weights[:-1], stim) + self.weights[-1])
    # Update single node during backprop
    def update(self,alpha,delta,x):
      self.delta = delta
      for i in range(0, len(x)):
        self.weights[i] = self.weights[i] + alpha * x[i] * delta 
    def weightString(self): return ",".join(map(lambda x: str(round(x,3)), self.weights))

  # Initializer
  # Note: the last "hidden" layer is the output layer
  def __init__(self,
               inputSize,               # Dimensionality of inputs
               numHiddenLayers = 2,     # Number of hidden layers
               alpha = 0.05,            # Learning rate
               sizeOfHiddenLayers = [], # Size of the hidden layers
               defaultLayerSize = 5,    # Defaults for layer sizes (except for the output) 
               maxIters = 100
               ):  

    # Set defaults for hidden layer size (including single output node default)
    self.isMulticlass, self.numClasses = False, 1
    if sizeOfHiddenLayers == []:
      sizeOfHiddenLayers = [defaultLayerSize for i in range(0,numHiddenLayers-1)] + [1]
    if type(sizeOfHiddenLayers) is int:
      sizeOfHiddenLayers = [sizeOfHiddenLayers for i in range(0,numHiddenLayers-1)] + [1]
    if len(sizeOfHiddenLayers) > 1:
      self.isMulticlass, self.numClasses = True, sizeOfHiddenLayers[-1]
    # Set to true if the labels should be treated as regression values, not discrete unrelated classes
    self.treatLabelsRegressively = False
    self.maxIters = maxIters
    # Error check
    assert len(sizeOfHiddenLayers) == numHiddenLayers, "Untenable hidden layer size mismatch"
    # Insert input nodes implicitly
    sizeOfHiddenLayers.insert(0, inputSize)
    # Save network properties
    self.numInputs, self.alpha = inputSize, alpha
    self.layerSizes = sizeOfHiddenLayers[1:]
    self.nLayers = len(self.layerSizes)
    # Generate layers (fully connected; no need to track graph connections)
    self.layers = [FeedForwardArtificialNeuralNetwork.AnnLayer(sizeOfHiddenLayers[i+1],sizeOfHiddenLayers[i],i) 
                   for i in range(0,numHiddenLayers)]

  def checkPerformance(self, y_true, y_comp, short=False):
    if self.isMulticlass:
      # PredY (y_comp) will be 1-hot encoded, while trueY will be in the origin encoding
      # Assumes the encoding is for labels in 0 -> NumClasses
      normDiff = lambda x,y: np.sqrt(sum([ (x[i] - y[i])**2 for i in range(0,len(x)) ]))
      maxedYs = [ str(np.argmax(y)) for y in y_comp ]
      if short:
        print("Acc = " + str(skm.accuracy_score(y_true,maxedYs)))
        return
      print("\nPerformance")
      print("max"); print(maxedYs); print("true"); print(y_true)
      print('Accuracy: ' + str(skm.accuracy_score(y_true,maxedYs)))
      # Also look at average 1-hot encoded vector metric difference
      # Little non-sensical though since a reasonable predictor should take the max
      preprocForEncoder = [[q] for q in y_true]
      encodedTrue = self.labEncoder.transform(preprocForEncoder).toarray()
      mean2NormDistance = np.mean([ normDiff(u,v) for u,v in zip(encodedTrue,y_comp) ])
      print("Mean 2-norm 1-hot vec dist: " + str(mean2NormDistance))
    else: # For binary classification, do some extra things
      y_c = [ 1 if q[0] > 0.5 else 0 for q in y_comp ]
      print('Accuracy: ' + str(skm.accuracy_score(y_true,y_c)))
      print('F1: ' + str(skm.f1_score(y_true,y_c)))
      se = sum( [ (yt - yc[0])**2 for yt,yc in zip(y_true,y_comp) ] ) / float(len(y_c))
      print('Average Squared Error: ' + str( se ))

  # 
  @staticmethod
  def crossValidate(x,y,maxIters=3):
    ### Parameters ###
    # Number of layers (1 + output vs 2 + output)
    layerNumbers = [2, 3]
    # Architectures: for 2 layer and 3 layer
    sizes = [10,20,30,40]
    layerArch2 = [ [s] for s in sizes ]
    layerArch3 = [ [i,j] for i in sizes for j in sizes ]  
    # Alpha values
    alphas = [0.05, 0.1]

    ### Preprocessing ###
    classSize = 19
    layerArch2 = [u + [classSize] for u in layerArch2]
    layerArch3 = [u + [classSize] for u in layerArch3]

    ### Single CV Run ###
    def annCv(alpha,layerNum,hiddenLayerSizes):
      dim = len(x[0])
      ffann = FeedForwardArtificialNeuralNetwork(
          dim, 
          numHiddenLayers = layerNum, 
          alpha = alpha, 
          sizeOfHiddenLayers = hiddenLayerSizes,
          maxIters = maxIters)
      print(str(alpha) + ", " + str(hiddenLayerSize))
#      ffann.display()
#      ss, ts = trainSize, trainSize + testSize
      n = len(x) / 2
#      xtrain, ytrain, xtest, ytest = x[0:n], y[0:n], x[n:], y[n:]  # x[0:ss], y[0:ss], x[ss:ts], y[ss:ts]
      def cv2(xtrain,ytrain,xtest,ytest):
        ffann.train(xtrain,ytrain,silent=True) 
        y_ann = ffann.predict(xtest,silent=True)
        ffann.checkPerformance(ytest, y_ann, short=True)
      cv2(x[0:n], y[0:n], x[n:], y[n:])
      cv2(x[n:], y[n:], x[0:n], y[0:n])
      print("----")      

    ### Cross-validate for hyper-parameter selection ###
    totalNum = len(layerArch2 + layerArch3) * len(alphas)
    print("Total = " + str(totalNum) + "\n")
    for alpha in alphas:
      for layerNum in layerNumbers:
        if layerNum == 2:
          for hiddenLayerSize in layerArch2:
            annCv(alpha, layerNum, hiddenLayerSize)
        elif layerNum == 3:
          for hiddenLayerSize in layerArch3:
            annCv(alpha, layerNum, hiddenLayerSize)

  # Training method
  def train(self,X,Y,method=1, silent=False):
    # Scale the data (the same scaler will be applied to the test data, but not fit to it of course)
    self.scaler = StandardScaler()
    self.scaler.fit( X )
    X = self.scaler.transform( X )
    # Preprocess multiclass data
    if self.isMulticlass:
      self.labEncoder = OneHotEncoder()
      Y = self.labEncoder.fit_transform([ [lab] for lab in Y ]).toarray()
    # Checks
    islist = lambda x: isinstance(x, list)
    assert method in [1], "Unknown training method"
    assert not islist(Y[0]) or (islist(Y[0]) and len(Y[0])==self.numClasses), "Label size mismatch"
    # Method 1: run backprop once per x \in X
    if method == 1:
      epsilon, maxIters, minIters, verbose = 0.0000001, self.maxIters, 250, True ####################################
      if silent: verbose = False
      p, ws = lambda x: sys.stdout.write(x), self.weightVec()
      for gen in range(0,maxIters):
        if verbose: p("Iter " + str(gen) + ": ")
        elif not silent: 
          if gen % 50 == 0: p("Iter "+str(gen)+"\n") 
        for i,(v,y) in enumerate(zip(X,Y)): 
          if verbose:
            if i == len(X)   / 4: p("25%.")
            elif i == len(X)   / 2: p("..50%.")
            elif i == 3*len(X) / 4: p("..75%\n")
          self._backpropagateTrainingSingle(v, y)
        currWs = self.weightVec() 
        avgdiff = np.mean( [ abs(a - b) for a,b in zip(ws,currWs) ] )
        if avgdiff < epsilon and gen > minIters: return
        else: ws = currWs

  # Testing method
  def predict(self,X): 
    X = self.scaler.transform( X )
    return [ self.computeOutput(t) for t in X ]

  def weightVec(self):
    ws = []
    for layer in self.layers:
      for neuron in layer.nodes:
        for weight in neuron.weights:
          ws.append( weight )
    return ws

  # Print out the ANN
  def display(self):
    print("Current ANN Status\n" + 'Input dimensions ' + str(self.numInputs))
    print('Params: alpha = ' + str(self.alpha) + ", numLayers = " + str(self.nLayers))
    for i,s in enumerate(self.layerSizes):
      sys.stdout.write( str(s) + ( '' if i==self.nLayers-1 else ' -> ' ) )
    print('\nLayer Weights')
    for i,layer in enumerate(self.layers):
      print('\tLayer ' + str(i))
      for j,neuron in enumerate(layer.nodes):
        print('\t\t' + str(j) + ') ' + neuron.weightString())
    print('-')

  # Compute y = h_w(x) via the ANN
  def computeOutput(self, x, returnStorage=False):
    assert len(x) == self.numInputs, "Unexpected input size"
    # Cascade values through the network
    currOutput, storage = np.array(x), [np.array(x)]
    for i in range(0, self.nLayers):
      currOutput = self.layers[i].integrate(currOutput)
      storage.append(currOutput) # [x, layer1_out, ..., finalOut]
    return currOutput if not returnStorage else storage

  # Correct the weights based on the error of a single example (as in Hastie et al)
  def _backpropagateTrainingSingle(self, x, y_true):
    o, alpha = self.computeOutput(x,True), self.alpha
    # Update final output nodes
    if self.isMulticlass:
      assert len(y_true) == self.layers[-1].numNodes, "Labeler length mismatch"
      delta_final = [ o[-1][i] * (1 - o[-1][i]) * (y_true[i] - o[-1][i])  
                    for i in range(0, self.layers[-1].numNodes) ]
    else:
      delta_final = [ o[-1][i] * (1 - o[-1][i]) * (y_true - o[-1][i])  
                    for i in range(0, self.layers[-1].numNodes) ]
    self.layers[-1].update(alpha, delta_final, o[-2])
    # Update hidden layers
    for i in range(self.nLayers - 2, -1, -1):
      # CurrLayer = i, delta_i = x_i(1-x_i)sum_k w_ik delta_k
      for j,neuron in enumerate(self.layers[i].nodes):
        downstreamLayer, deltaProj = self.layers[i+1], 0.0
        for forwardNeuron in downstreamLayer.nodes: 
          deltaProj += forwardNeuron.weights[j] * forwardNeuron.delta
        x_out = o[i+1][j] # Output for this layer, this neuron
        deltaCurr = deltaProj * x_out * (1.0 - x_out) # Delta for this layer
        neuron.update(alpha, deltaCurr, o[i])

# Simple method for generating some practice data for binary classification
def practiceData(d, ntrain=100, ntest=50):
  n = int( (ntrain + ntest) / 2 )
  mu1, mu2, sigma1, sigma2 = 1, 9, 4, 5
  x_u, y_u = list(np.random.normal(mu1,sigma1,(n,d))) + list(np.random.normal(mu2,sigma2,(n,d))), [0]*n + [1]*n
  index_shuffled = list(range( n*2 ))
  r.shuffle(index_shuffled)
  x, y = [ x_u[i] for i in index_shuffled ], [ y_u[i] for i in index_shuffled]
  return [ x[0:ntrain], y[0:ntrain], x[ntrain:], y[ntrain:] ]

# Generates simple data set for testing multi-class classification
def practiceMultiData(d, nc=10, ntrain=500, ntest=50):
  n = int( (ntrain + ntest) / nc )
  means, variance = list(range(0,nc)), 0.2
  dataNest = ( np.random.normal(mean, variance,(n,d)) for mean in means )
  labNest = [ [k]*n for k in means ]
  flat = lambda xs: [w for u in xs for w in u]
  dataFlat, labFlat = flat(dataNest), flat(labNest)
  index_shuffled = list(range( n*nc ))
  r.shuffle(index_shuffled)
  x, y = [ dataFlat[i] for i in index_shuffled ], [ labFlat[i] for i in index_shuffled]
  return [ x[0:ntrain], y[0:ntrain], x[ntrain:], y[ntrain:] ]

### Main Method Invocation ###
if __name__ == '__main__': main()
