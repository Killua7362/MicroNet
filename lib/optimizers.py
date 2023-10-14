import numpy as np

class SGD:
    def __init__(self,lr=0.01):
        self.learning_rate  = lr

    def update_params(self,layer):
        if hasattr(layer,'weights'):
            layer.weights += (-self.learning_rate * layer.dweights)
        if hasattr(layer,'bias'):
            layer.bias += (-self.learning_rate * layer.dbias)
        if hasattr(layer,'dbeta'):
            layer.beta += (-self.learning_rate * layer.dbeta)
        if hasattr(layer,'dgamma'):
            layer.gamma +=(-self.learning_rate * layer.dgamma)

