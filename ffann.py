# Implementation of a fully connected, feed forward ANN
# 10/30/16

import os, sys, numpy as np, random as r, math
from sklearn import metrics as skm


def main():
  # Generate some practice data
  d = 4
  [xtrain,ytrain,xtest,ytest] = practiceData(d,2500,50)
  # Create an ANN
  ffann = FeedForwardArtificialNeuralNetwork(d,numHiddenLayers = 3)
  ffann.display()
  ffann.train(xtrain,ytrain) 
  y_ann = ffann.predict(xtest)
  ffann.display()
  performance(ytest,y_ann)

# Note: two options for output nodes
#  - have 19 output nodes (1 for each possible sum value; take the max)
#  - have a non-sigmoid (e.g. linear) single output node [artificial relation]

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
    # Private static variables
    _weightSigma, _sigmoid = 0.01, lambda z: 1.0 / (1.0 + math.exp(-z))
    # Constructor
    def __init__(self, numBackwardsNodes, layerNum):
      # Initialize weights to a small number (note bias)
      self.weights = np.array([ np.random.normal(0.0, FeedForwardArtificialNeuralNetwork.AnnNode._weightSigma) 
                                for _ in range(0,numBackwardsNodes+1) ])
      self.nWeights = len(self.weights)
      self.delta, self.layerIndex = None, layerNum
    # Sigmoid filter linear combination (implicit 1 for bias weight term)
    def integrate(self,stim):
      return FeedForwardArtificialNeuralNetwork.AnnNode._sigmoid(np.dot(self.weights[:-1], stim) + self.weights[-1])
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
               defaultLayerSize = 5 ):  # Defaults for layer sizes (except for the output)

    # Set defaults for hidden layer size (including single output node default)
    if sizeOfHiddenLayers == []:
      sizeOfHiddenLayers = [defaultLayerSize for i in range(0,numHiddenLayers-1)] + [1]
    if type(sizeOfHiddenLayers) is int:
      sizeOfHiddenLayers = [sizeOfHiddenLayers for i in range(0,numHiddenLayers-1)] + [1]
    # Error check
    assert len(sizeOfHiddenLayers) == numHiddenLayers, "Untenable hidden layer size mismatch"
    # Insert input nodes implicitly
    sizeOfHiddenLayers.insert(0,inputSize)
    # Save network properties
    self.numInputs, self.alpha = inputSize, alpha
    self.layerSizes = sizeOfHiddenLayers[1:]
    self.nLayers = len(self.layerSizes)
    # Generate layers (fully connected; no need to track graph connections)
    self.layers = [FeedForwardArtificialNeuralNetwork.AnnLayer(sizeOfHiddenLayers[i+1],sizeOfHiddenLayers[i],i) 
                   for i in range(0,numHiddenLayers)]

  # Training method
  def train(self,X,Y,method=1):
    assert method in [1], "Unknown training method"
    # Method 1: run backprop once per x \in X
    if method == 1:
      epsilon = 0.001
      maxIters = 50
      ws = self.weightVec()
      for _ in range(0,maxIters):
        for v,y in zip(X,Y): 
          self._backpropagateTrainingSingle(v, y)
        currWs = self.weightVec()
        avgdiff = np.mean( [ abs(a - b) for a,b in zip(ws,currWs) ] )
        if avgdiff < epsilon: return
        else: ws = currWs


  # Testing method
  def predict(self,X): return [ self.computeOutput(t) for t in X ]

  def weightVec(self):
    ws = []
    for layer in self.layers:
      for neuron in layer.nodes:
        for weight in neuron.weights:
          ws.append( weight )
    return ws

  # Print out the ANN
  def display(self):
    print('Input dimensions ' + str(self.numInputs))
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
    delta_final = [ o[-1][i] * (1 - o[-1][i]) * (y_true - o[-1][i])  
                    for i in range(0, self.layers[-1].numNodes) ]
    self.layers[-1].update(alpha, delta_final, o[-2])
    # Update hidden layers
    for i in range(self.nLayers - 2, -1, -1):
      # CurrLayer = i, delta_i = x_i(1-x_i)sum_k w_ik delta_k
      for j,neuron in enumerate(self.layers[i].nodes):
        downstreamLayer = self.layers[i+1]
        deltaProj = 0.0
        for forwardNeuron in downstreamLayer.nodes: 
          deltaProj += forwardNeuron.weights[j] * forwardNeuron.delta
        x_out = o[i+1][j] # Output for this layer, this neuron
        deltaCurr = deltaProj * x_out * (1.0 - x_out) # 
        neuron.update(alpha, deltaCurr, o[i])

# Simple method for generating some practice data
def practiceData(d, ntrain=100, ntest=50):
  n = int( (ntrain + ntest) / 2 )
  mu1, mu2, sigma1, sigma2 = 1, 9, 4, 5
  x_u, y_u = list(np.random.normal(mu1,sigma1,(n,d))) + list(np.random.normal(mu2,sigma2,(n,d))), [0]*n + [1]*n
  index_shuffled = list(range( n*2 ))
  r.shuffle(index_shuffled)
  x, y = [ x_u[i] for i in index_shuffled ], [ y_u[i] for i in index_shuffled]
  return [ x[0:ntrain], y[0:ntrain], x[ntrain:], y[ntrain:] ]

# TODO issues if output node is not scalar
def performance(y_true, y_comp):
  y_c = [ 1 if q[0] > 0.5 else 0 for q in y_comp ]
  print("\nPerformance")
#  print(y_c)
#  print(y_comp)
#  print(y_true)
  print('Accuracy: ' + str(skm.accuracy_score(y_true,y_c)))
  print('F1: ' + str(skm.f1_score(y_true,y_c)))
  se = sum( [ (yt - yc[0])**2 for yt,yc in zip(y_true,y_comp) ] ) / float(len(y_c))
  print('Average Squared Error: ' + str( se ))


### Main Method Invocation ###
if __name__ == '__main__': main()
