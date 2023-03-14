import numpy as np

class SGD:
    def __init__(self,lr=0.01):
        self.learning_rate  = lr

    def update_params(self,layer):
        layer.weights += (-self.learning_rate * layer.dweights)
        layer.bias += (-self.learning_rate * layer.dbias)

