import numpy as np

class Hooks:
    def __init__(self,tensor,backward):
        self.tensor = tensor
        self.grad_fn = backward

class Tensor:
    def __init__(self,data,node=None, op='',label='',backward = lambda:None):
        self.data = to_array(data)
        self._prev = node or []
        self._op = op
        self._label = label
        self._backward = backward
        self._for_test= lambda:None
        self.shape = self.getshape()  
        self.grad = Tensor(np.zeros_like(self.data,dtype=np.float64))
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
        return _add(self,other)
    
    def __mul__(self,other):
        return _mul(self,other)
      
    def __pow__(self,other):
        return _pow(self,other)
    
    def sum(self,axis=None,keepdims=False):
        return _sum(self,axis=axis,keepdims=keepdims)
    
    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1
    
    def __neg__(self): # -self
        return self * Tensor(-1)
    
    def __rmul__(self,other):
        return _mul(self,other)
    
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
                gradient = Tensor(1.0)
            else:
                raise RuntimeError('Mention grads')
            
        self.grad = self.grad + gradient
        
        prev = self._prev
        if prev is not None:
            for node in self._prev:
                backward_grad = node.grad_fn(gradient)
                node.tensor.backprop(backward_grad)
            

def dummy_loss(x):
    x = x.data
    x = x.mean()
    return Tensor(x)

def to_tensor(x):
    if not isinstance(x,Tensor):
        return Tensor(x)
    else:
        return x

def to_array(x):
    if isinstance(x,np.ndarray):
        return x
    else:
        return np.array([x])


def _add(t1,t2):
    t1 = to_tensor(t1)   
    t2 = to_tensor(t2)   
    
    data = t1.data + t2.data
    hooks = []
    op = '+'
    def backward_1(gradient):
        ndims_added = gradient.ndim -  t1.ndim
        for _ in range(ndims_added):
            gradient = gradient.sum(axis=0)
        for i,dim in enumerate(t1.shape):
            if dim == 1:
                gradient = gradient.sum(axis=i,keepdims=True)
        return gradient
        
    def backward_2(gradient):
        ndims_added = gradient.ndim - t2.ndim 
        for _ in range(ndims_added):
            gradient = gradient.sum(axis=0)
        for i,dim in enumerate(t2.shape):
            if dim == 1:
                gradient = gradient.sum(axis=i,keepdims=True)
        return gradient
                    
    hooks.append(Hooks(t1,backward_1))    
    hooks.append(Hooks(t2,backward_2))    
    return Tensor(data,node=hooks,op=op)

def _mul(t1,t2):
        t1 = to_tensor(t1)
        t2 = to_tensor(t2)
        data = t1.data * t2.data
        hooks = []
        op ='*'
        
        def backward_1(gradient):
            t1.grad+= t2.data * gradient
            return t1.grad
            
        def backward_2(gradient):
            t2.grad += t1.data * gradient
            return t2.grad
            
        hooks.append(Hooks(t1,backward_1))    
        hooks.append(Hooks(t2,backward_2))    
        return Tensor(data=data,node=hooks,op=op)

def _pow(t1,power):
        data = t1.data ** power 
        op = '**'
        hooks = []
        def backward(gradient):
            t1.grad += gradient * (power* (t1.data ** (power-1)))
            return t1.grad
        hooks.append(Hooks(t1,backward))    
        return Tensor(data=data,node=hooks,op=op)

def _sum(t,axis=None,keepdims=False):
        data = t.data.sum(axis=axis,keepdims=keepdims)
        hooks = []
        op = 'sum'
        def backward(gradient):
            keep = t.data.sum(axis=axis,keepdims=True)
            return gradient.reshape(keep.shape) + np.zeros_like(t.data)
        hooks.append(Hooks(t,backward=backward))
        return Tensor(data=data,node=hooks,op=op)
