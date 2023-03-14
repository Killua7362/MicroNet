import numpy as np

class Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons)
        self.bias = np.zeros((1,n_neurons))

    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.dot(inputs,self.weights) + self.bias
    
    def backward(self,dvalues):
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dbias = np.sum(dvalues,axis = 0,keepdims=True)
        self.dinputs = np.dot(dvalues,self.weights.T)

class Layer_input:
    def forward(self,inputs):
        self.output = inputs
