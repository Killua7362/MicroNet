import numpy as np
from lib.layers import BaseLayer
from lib.tensor import Tensor

##these are not trainable

class Relu(BaseLayer):
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
        super().__init__()
        
    def backward(self,dvlaues):
        self.dinputs = dvlaues.copy()
        self.dinputs[self.inputs<=0] = 0
        
    def predictions(self, outputs):
        return outputs

class Gelu(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self,inputs):
        if isinstance(inputs,Tensor):
            self.inputs = inputs.data
        else:
            self.inputs = inputs

        self.inputs = inputs
        self.cumm_dist =  0.5  * (1 + np.tanh(np.sqrt(2 / np.pi) * (inputs + 0.044715 * inputs**3)))
        self.output = self.cumm_dist * inputs
    
    def backward(self,dvalues):
        prob_dist = (np.exp(-self.inputs**2 / 2.0) / np.sqrt(2.0 * np.pi))
        self.dinputs= dvalues * (self.cumm_dist+ self.inputs * prob_dist)
        return self.dinputs

    def predictions(self,outputs):
        return outputs

class SoftMax(BaseLayer):
    def forward(self,inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs-np.max(inputs,axis=-1,keepdims=True))
        probabilities = exp_values / np.sum(exp_values,axis=-1,keepdims=True)
        self.output = probabilities
        super().__init__()
        
    def backward(self,dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index,(single_output,single_dvalues) in enumerate(zip(self.output,dvalues)):
            single_output = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output,single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)
            
    def predictions(self,outputs):
        return np.argmax(outputs,axis=1)
