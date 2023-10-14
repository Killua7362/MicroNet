import numpy as np
import sys

class BaseLayer:
    instances = []
    def __init__(self,n_neurons=0,n_inputs=0,weights=None,bias=None,gamma=None,beta=None):
        self.weights = weights
        self.bias = bias
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.gamma = gamma
        self.beta = beta
        self.build(n_neurons,n_inputs)
        
        self.trainable_params = {}
        BaseLayer.instances.append(self)
    
    def __call__(self,inputs):
        self.forward(inputs)
        return self.output
         
    def build(self,n_neurons,n_inputs):
        if self.bias is None:
            self.bias = np.zeros((1,self.n_neurons))
        if self.weights is None:
            self.weights = 0.01 * np.random.randn(self.n_inputs,self.n_neurons)
        if self.gamma is None:
            self.gamma = np.ones(n_inputs)
        if self.beta is None:
            self.beta = np.zeros(n_inputs)

class Dense(BaseLayer):
    def __init__(self,/,n_inputs=0,n_neurons=0,**kwargs):
        self.n_neurons =n_neurons
        self.n_inputs = n_inputs
        super().__init__(n_inputs=n_inputs,n_neurons=n_neurons,**kwargs)
        
        self.trainable_params = {'bias':None,'weights':None}
    
    def load_params(**kwargs):
            pass
        
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

class LayerNorm(BaseLayer):
    def __init__(self,gamma=None,beta=None,epsilon=1e-5):
        self.epsilon = epsilon
        super().__init__(gamma=gamma,beta=beta)
    
    def forward(self,inputs):
        self.inputs = inputs
        mean = np.mean(inputs,axis=-1,keepdims=True)
        self.variance = np.var(inputs,axis=-1,keepdims=True)
        self.inputs_normalized =(inputs-mean) /np.sqrt(self.variance+self.epsilon)
        self.output = self.gamma * self.inputs_normalized +self.beta
    
    def backward(self,dvalues):
        self.dgamma = np.sum(dvalues * self.inputs_normalized, axis=1, keepdims=True)
        self.dbeta = np.sum(dvalues,axis=1,keepdims=True)
        dx_normalized = dvalues * self.gamma
        dvariance = np.sum(dx_normalized * (self.inputs - np.mean(self.inputs, axis=1, keepdims=True)), axis=1, keepdims=True) * -0.5 * (self.variance + self.epsilon) ** (-1.5)
        dmean = np.sum(dx_normalized, axis=1, keepdims=True) * -1 / np.sqrt(self.variance + self.epsilon) + dvariance * np.mean(self.inputs, axis=1, keepdims=True)
        N = self.inputs.shape[1]
        self.dinputs = dx_normalized / np.sqrt(self.variance + self.epsilon) + dvariance * 2 * (self.inputs - np.mean(self.inputs, axis=1, keepdims=True)) / N + dmean / N
 