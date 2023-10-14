import numpy as np
import sys

class BaseLayer:
    instances = []
    def __init__(self):
        BaseLayer.instances.append(self)
    
    def __call__(self,inputs):
        self.forward(inputs)
        return self.output

    def forward(self,inputs):
        self.build(inputs)
         
    def load_params(self,params):
        from lib.utils import update_dict
        self.trainable_params = update_dict(self.trainable_params,params)

class Dense(BaseLayer):
    def __init__(self,/,n_inputs=0,n_neurons=0):
        self.n_neurons =n_neurons
        self.n_inputs = n_inputs
        
        self.trainable_params = {'b':None,'w':None}
        super().__init__()    
    
    def build(self,inputs):
        if self.n_inputs == 0:
            self.n_inputs = inputs.shape[-1]
        if self.trainable_params['b'] is None:
            self.trainable_params['b'] = np.zeros((1,self.n_neurons))
        self.bias = self.trainable_params['b']
        if self.trainable_params['w'] is None:
            self.trainable_params['w'] = 0.01 * np.random.randn(self.n_inputs,self.n_neurons)
        self.weights = self.trainable_params['w']
        if self.n_neurons == 0:
            if self.weights is None:
                self.n_neurons = self.n_inputs
            else:
                self.n_neurons = self.weights.shape[0]
           
    def forward(self,inputs):
        super().forward(inputs)
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
        self.trainable_params = {'g':gamma,'b':beta}
        super().__init__()
    
    def build(self,inputs):
        if self.trainable_params ['g'] is None:
            self.trainable_params['g'] = np.ones(inputs.shape[-1])
        self.gamma = self.trainable_params['g']
        if self.trainable_params['b'] is None:
            self.trainable_params['b'] = np.zeros(inputs.shape[-1])
        self.beta= self.trainable_params['b']
        
    def forward(self,inputs):
        super().forward(inputs)
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
 