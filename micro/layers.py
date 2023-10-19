import numpy as np
from micro.tensor import Tensor,Hooks
from typing import Iterator
import inspect
from micro.autograd.parameters import Parameters

class BaseLayer:
    instances = []
    static_instances = []
    
    def __init__(self):
        if hasattr(self,'trainable_params'):
            BaseLayer.instances.append(self)
        else:
            BaseLayer.static_instances.append(self)
    
    def __call__(self,inputs):
        hooks = []
        self.forward(inputs)
        data = self.output
        if not isinstance(inputs,Tensor):
            return data
        
        if inputs.requires_grad:
            hooks.append(Hooks(inputs,self.backward))
        return Tensor(data,nodes=hooks,requires_grad=inputs.requires_grad)

    def forward(self,inputs):
        if isinstance(inputs,Tensor):
            self.inputs = inputs.data
        else:
            self.inputs = inputs
        self.build(self.inputs)
        
class Embeddings(BaseLayer):
    def __init__(self,n_inputs=0,n_neurons=0):
        self.n_neurons =n_neurons
        self.n_inputs = n_inputs
        self.trainable_params = {'w':None}
        super().__init__()    
        
    def build(self,inputs):
        if self.n_inputs == 0:
            self.n_inputs = inputs.shape[-1]
            
        if self.n_neurons == 0:
            self.n_neurons = self.n_inputs
                
        if self.trainable_params['w'] is None:
            self.trainable_params['w'] = 0.01 * np.random.randn(self.n_inputs,self.n_neurons)
        self.w = self.trainable_params['w']
           
    def forward(self,inputs):
        super().forward(inputs)
        inputs = self.inputs
        self.output = self.w[inputs]
        
    def backward(self,dvalues):
        self.dw = np.zeros_like(self.w)
        self.dw[self.inputs] = dvalues
        self.dinputs = np.dot(dvalues , self.w.T)
        return self.dinputs

class Dense(BaseLayer):
    def __init__(self,n_inputs=0,n_neurons=0):
        self.n_neurons =n_neurons
        self.n_inputs = n_inputs
        self.trainable_params = {'b':None,'w':None}
        super().__init__()    
    
    def build(self,inputs):
        if self.n_inputs == 0:
            self.n_inputs = inputs.shape[-1]
            
        if self.n_neurons == 0:
            self.n_neurons = self.n_inputs
                
        if self.trainable_params['w'] is None:
            self.trainable_params['w'] = 0.01 * np.random.randn(self.n_inputs,self.n_neurons)
        self.w = self.trainable_params['w']
        if self.trainable_params['b'] is None:
            self.trainable_params['b'] = np.zeros(shape=(self.n_neurons))
        self.b = self.trainable_params['b']
           
    def forward(self,inputs):
        super().forward(inputs)
        inputs = self.inputs
        self.output = np.dot(inputs,self.w) + self.b
        
    def backward(self,dvalues):
        self.dw = np.dot(self.inputs.T,dvalues)
        self.db = np.sum(dvalues,axis = 0,keepdims=True)
        self.dinputs = np.dot(dvalues,self.w.T)
        return self.dinputs

class Layer_input:
    def forward(self,inputs):
        self.output = inputs

class LayerNorm(BaseLayer):
    def __init__(self,epsilon=1e-5):
        self.epsilon = epsilon
        self.trainable_params = {'g':None,'b':None}
        super().__init__()
    
    def build(self,inputs):
        if self.trainable_params ['g'] is None:
            self.trainable_params['g'] = np.ones(inputs.shape[-1])
        self.g = self.trainable_params['g']
        
        if self.trainable_params['b'] is None:
            self.trainable_params['b'] = np.zeros(inputs.shape[-1])
        self.b = self.trainable_params['b']
        
    def forward(self,inputs):
        super().forward(inputs)
        inputs = self.inputs
        mean = np.mean(inputs,axis=-1,keepdims=True)
        variance = np.var(inputs,axis=-1,keepdims=True)
        inputs_normalized =(inputs-mean) /np.sqrt(variance+self.epsilon)
        self.output = self.g * inputs_normalized + self.b
        self.cache = (inputs_normalized, mean, variance)
        
    def backward(self,dvalues):
        inputs_normalized, mean, variance = self.cache
        N,D = self.inputs.shape
        self.db = np.sum(dvalues, axis=0)
        self.dg = np.sum(dvalues* inputs_normalized, axis=0)
        dx_normalized = dvalues * self.g
        dvar = np.sum(dx_normalized * (self.inputs - mean) * -0.5 * (variance+ self.epsilon)**(-1.5), axis=0)
        dmean = np.sum(dx_normalized * -1.0 / np.sqrt(variance+ self.epsilon), axis=0)
        self.dinputs = dx_normalized / np.sqrt(variance+ self.epsilon) + dvar * 2.0 * (self.inputs - mean) / D + dmean / D
 