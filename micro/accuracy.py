import numpy as np

class Accuracy:
    def calculate(self,pred_val,true_val):
        comparisons = self.compare(pred_val,true_val)
        accuracy = np.mean(comparisons)
        self.accum_sum += np.sum(comparisons)
        self.accum_count += len(comparisons)
        return accuracy
    
    def calculate_accum(self):
        accuracy= self.accum_sum/self.accum_count
        return accuracy
    
    def new_pass(self):
        self.accum_sum = 0
        self.accum_count = 0

class Accuracy_Categorical(Accuracy):
    def init(self, y):
        pass
    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y
