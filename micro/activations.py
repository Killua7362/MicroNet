import numpy as np
from micro.layers import BaseLayer
from micro.tensor import Tensor
import micro
##these are not trainable

class Relu(BaseLayer):
    def forward(self,inputs):
        self.inputs = inputs
        self.output = micro.maximum(0,inputs)
        super().__init__()
        
    def backward(self,dvlaues):
        self.dinputs = dvlaues.copy()
        self.dinputs[self.inputs<=0] = 0
        
    def predictions(self, outputs):
        return outputs

class Gelu(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
    
    def build(self,inputs):
        pass
    
    def forward(self,inputs):
        super().forward(inputs)
        inputs = self.inputs
        self.cumm_dist =  0.5  * (1 + micro.tanh(micro.sqrt(2 / np.pi) * (inputs + 0.044715 * inputs**3)))
        self.output = self.cumm_dist * inputs
    
    def backward(self,dvalues):
        prob_dist = (micro.exp(-self.inputs**2 / 2.0) / micro.sqrt(2.0 * np.pi))
        self.dinputs= dvalues * (self.cumm_dist+ self.inputs * prob_dist)
        return self.dinputs

    def predictions(self,outputs):
        return outputs
class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()    
        
    def build(self,inputs):
        pass
           
    def forward(self,inputs):
        super().forward(inputs)
        inputs = self.inputs
        exp_values = micro.exp(inputs-micro.max(inputs,axis=-1,keepdims=True))
        probabilities = exp_values / micro.sum(exp_values,axis=-1,keepdims=True)
        self.output = probabilities
        
    def backward(self,dvalues):
        self.dinputs = micro.empty_like(dvalues)
        for index,(single_output,single_dvalues) in enumerate(zip(self.output,dvalues)):
            single_output = single_output.reshape(-1,1)
            jacobian_matrix = micro.diagflat(single_output) - micro.dot(single_output,single_output.T)
            self.dinputs[index] = micro.dot(jacobian_matrix,single_dvalues)
        return self.dinputs
    
    def predictions(self,outputs):
        return micro.argmax(outputs,axis=1)