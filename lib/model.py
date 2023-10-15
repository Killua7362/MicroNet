import numpy as np
from lib.layers import Layer_input,BaseLayer
from lib.accuracy import Accuracy_Categorical
from lib.losses import CategoricalCrossEntropy
from lib.optimizers import SGD

class BaseModule:
    baseModule_instances = []
    def __init__(self):
        BaseModule.baseModule_instances.append(self)
        self.instances = BaseLayer.instances.copy()
        self.static_instances = BaseLayer.static_instances.copy()
        
class Model:
    def __init__(self,trainable=False):
        self.layers = []
        self.trainable_layers = []
        self.trainable = trainable
        
        self.instances = BaseLayer.instances.copy()
        BaseLayer.instances = []
        self.static_instances = BaseLayer.static_instances.copy()
        BaseLayer.static_instances = []
        
    def add(self,Layer):
        self.layers.append(Layer)

    def set(self,*,loss=CategoricalCrossEntropy(),optimizer=SGD(lr=0.001),accuracy=Accuracy_Categorical()):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
        
    def train(self,X,y,*,epochs=1,batch_size = None,print_every = 1,validation_data=None):
        self.accuracy.init(y)
        train_steps = 1
        if validation_data is not None:
            validation_steps = 1
            X_val,y_val = validation_data
        if batch_size is not None:
            train_steps = len(X)//batch_size
            if train_steps*batch_size<len(X):
                train_steps+=1
            if validation_data is not None:
                validation_steps = len(X_val)//batch_size
                if validation_steps * batch_size < len(X_val):
                    validation_steps+=1
                
        for epoch in range(1,epochs+1):
            print(f'epoch: {epoch}')
            self.loss.new_pass()
            self.accuracy.new_pass()
            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]
                output = self._forward(batch_X)
                loss = self.loss.calculate(output,batch_y)

                predictions = self.output_layer_activation.predictions(output)

                accuracy = self.accuracy.calculate(predictions,batch_y)
                self._backward(output,batch_y)
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                if not step % print_every or step == train_steps - 1:
                    print(f'step is {step}')
        epoch_loss = self.loss.calculate_accum()
        epoch_accuracy = self.accuracy.calculate_accum()
        print(f'epoch accuracy{epoch_accuracy} and loss is {epoch_loss}')
        
        if validation_data is not None:
            self.loss.new_pass()
            self.accuracy.new_pass()
            for step in range(validation_steps):
                if batch_size is None:
                    batch_X,batch_y=X_val,y_val 
                else:
                    batch_X = X_val[step*batch_size:(step+1)*batch_size]
                    batch_y = y_val[step*batch_size:(step+1)*batch_size]
                
                output = self._forward(batch_X)
                self.loss.calculate(output,batch_y)
                predictions = self.output_layer_activation.predictions(output)
                self.accuracy.calculate(predictions,batch_y)
            accuracy =self.accuracy.calculate_accum() 
            loss = self.loss.calculate_accum()
            print(f'validation, ' + f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}')

    def _forward(self,X):
        if hasattr(self,'forward'):
            self.input_layer.forward(self.forward(X))
        else:
            self.input_layer.forward(X)
            
        for layer in self.layers:
            layer.forward(layer.prev.output)
        return layer.output
    
    def _backward(self,outputs,y):
        self.loss.backward(outputs,y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def finalize(self):
        self.input_layer = Layer_input()
        layer_count = len(self.layers)
        
        if self.trainable:
            self.trainable_layers.extend(self.instances)
            
        for i in range(layer_count):
            if i == 0 :
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i<layer_count-1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
            if hasattr(self.layers[i],'trainable_params'):
                self.trainable_layers.append(self.layers[i])
            self.loss.remember_trainable_layers(self.trainable_layers)