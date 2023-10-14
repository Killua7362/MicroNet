import numpy as np

class SGD:
    def __init__(self,lr=0.01):
        self.learning_rate  = lr

    def update_params(self,layer):
        for k,v in layer.trainable_params.items():
            layer.trainable_params[k] =  layer.trainable_params[k] + (-self.learning_rate * getattr(layer,f'd{k}'))