import numpy as np

class Hooks:
    def __init__(self,tensor,backward):
        self.tensor = tensor
        self.grad_fn = backward

class Tensor:
    def __init__(self,data,node=None, op='',label='',backward = lambda:None):
        self.data = data
        self._prev = node or []
        self._op = op
        self._label = label
        self.grad = 0
        self._backward = backward
        self._for_test= lambda:None
        self.shape = self.getshape()  
        self.ndim = 1 if self.shape == 1 or self.shape[0] == 1 else len(self.shape)
        
    def getshape(self):    
        if type(self.data) == np.ndarray:
            return self.data.shape
        elif type(self.data) == list:
            return len(self.data)
        else:
            return (1,)
    
    def __repr__(self):
        return f'Tensor(data={self.data},label={self._label})'
    
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self): # Python 2: def next(self)
        if self.index< len(self.data):
            res = self.data[self.index]
            self.index+= 1
            return res 
        raise StopIteration
    
    def __add__(self,other):
        other = other if isinstance(other,Tensor) else Tensor(other)
        data = self.data + other.data
        hooks = []
        op = '+'
        def backward_1(gradient):
            ndims_added = gradient.ndim -  self.ndim
            for _ in range(ndims_added):
                gradient = gradient.sum(axis=0)
            for i,dim in enumerate(self.shape):
                if dim == 1:
                    gradient = gradient.sum(axis=i,keepdims=True)
            return gradient
            
        def backward_2(gradient):
            ndims_added = gradient.ndim - other.ndim 
            for _ in range(ndims_added):
                gradient = gradient.sum(axis=0)
            for i,dim in enumerate(other.shape):
                if dim == 1:
                    gradient = gradient.sum(axis=i,keepdims=True)
            return gradient
                        
        hooks.append(Hooks(self,backward_1))    
        hooks.append(Hooks(other,backward_2))    
        return Tensor(data,node=hooks,op=op)
    
    def __mul__(self,other):
        other = other if isinstance(other,Tensor) else Tensor(other)
        data = self.data * other.data
        hooks = []
        op ='*'
        
        def backward_1(gradient):
            self.grad+= other.data * gradient
            return self.grad
            
        def backward_2(gradient):
            other.grad += self.data * gradient
            return other.grad
            
        hooks.append(Hooks(self,backward_1))    
        hooks.append(Hooks(other,backward_2))    
        return Tensor(data=data,node=hooks,op=op)
      
    def __pow__(self,other):
        data = self.data ** other
        op = '**'
        hooks = []
        def backward(gradient):
            self.grad += gradient * (other * (self.data ** (other-1)))
            return self.grad
        hooks.append(Hooks(self,backward))    
        return Tensor(data=data,node=hooks,op=op)
    
    def sum(self,axis=None,keepdims=False):
        data = self.data.sum(axis=axis,keepdims=keepdims)
        hooks = []
        op = 'sum'
        def backward(gradient):
            keep = self.data.sum(axis=axis,keepdims=True)
            return gradient.reshape(keep.shape) + np.zeros_like(self.data)
        hooks.append(Hooks(self,backward=backward))
        return Tensor(data=data,node=hooks,op=op)
    
    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1
    
    def __neg__(self): # -self
        return self * Tensor(-1)
    
    def __rmul__(self,other):
        return self*other
    
    def __radd__(self,other):
        return self + other
  
    
    def __radd__(self, other): # other + self
        return self + other
    
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other): 
        return other + (-self)
    
    def __getitem__(self,index):
        return self.data[index]
    
    def __setitem__(self,index,item):
        self.data[index] = item

    def backprop(self,gradient=None):
        if gradient is None:
            if self.shape == ():
                grad = Tensor(np.array([1.0]))
            else:
                raise RuntimeError('Mention grads')
        else:
            grad = gradient
        prev = self._prev
        if prev is not None:
            for node in self._prev:
                backward_grad = node.grad_fn(grad)
                node.tensor.backprop(backward_grad)
            

def dummy_loss(x):
    x = x.data
    x = x.mean()
    return Tensor(x)