import numpy as np

class Loss:
    def calculate(self,output,y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        return data_loss
    def calculate_accum(self):
        data_loss = self.accumulated_sum / self.accumulated_count
        return data_loss
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0 
    def remember_trainable_layers(self,trainable_layers):
        self.trainable_layers = trainable_layers

class CategoricalCrossEntropy(Loss):
    def forward(self,pred_val,true_val):
        samples = len(pred_val)
        pred_val_clipped = np.clip(pred_val,1e-7,1 - 1e-7)
        if len(true_val.shape) == 1:
            correct_confidence = pred_val_clipped[range(samples),true_val]
        elif len(true_val.shape) == 2:
            correct_confidence = np.sum(
                pred_val_clipped * true_val,axis = 1
            )
        return  -np.log(correct_confidence)

    def backward(self,dvalues,true_val):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(true_val.shape) == 1:
            true_val = np.eye(labels)[true_val]
        self.dinputs = -true_val / dvalues
        self.dinputs = self.dinputs/samples
