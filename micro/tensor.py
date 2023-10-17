from typing import List, NamedTuple,Callable,Optional,Union
import numpy as np

class Hooks(NamedTuple):
    tensor:'Tensor'
    grad_fn: Callable[[np.ndarray],np.ndarray]

class Tensor:
    def __init__(self,data,requires_grad=False,nodes=[]):
        self._data = to_array(data) 
        self.requires_grad = requires_grad
        self.nodes = nodes
        self.shape = self._data.shape
        self.grad:Optional['Tensor'] = None
        self.dtype = self._data.dtype
        if self.requires_grad:
            self.zero_grad()
    
    def __repr__(self) ->str:
        return f'Tensor(data={self.data}, requires_grad={self.requires_grad})'
    
    ##getter and setter
    @property
    def data(self) -> np.ndarray:
        return self._data   
    
    @data.setter
    def data(self,value) -> None:
        self._data = value
        self.grad = None
    
    def zero_grad(self):
        self.grad = Tensor(np.zeros_like(self.data,dtype=np.float64))
    
    def backward(self,grad:'Tensor'=None):
        if not self.requires_grad:
           raise AssertionError('non-requires-grad') 
       
        if grad is None:
            if self.shape == ():
                grad = Tensor(1.0)
            else:
                raise RuntimeError('specify grad for non-0-tensor')
            
        self.grad.data = self.grad.data + grad.data
         
        for node in self.nodes:
            back_grad = node.grad_fn(grad.data)      
            node.tensor.backward(Tensor(back_grad))
    
    def sum(self) -> 'Tensor':
        return _sum(self)
    
    def __add__(self,other) -> 'Tensor':
        return _add(self,other)
    
    def __radd__(self,other) -> 'Tensor':
        return _add(other,self)
    
    def __iadd__(self,other) -> 'Tensor':
        self.data = self.data + to_tensor(other).data
        return self
        
    def __isub__(self,other) -> 'Tensor':
        self.data = self.data - to_tensor(other).data
        return self
    
    def __imul__(self,other) -> 'Tensor':
        self.data = self.data * to_tensor(other).data
        return self
        
    def __mul__(self,other) -> 'Tensor':
        return _mul(self,other)

    def __rmul__(self,other) -> 'Tensor':
        return _mul(other,self)
    
    def __neg__(self) ->'Tensor':
        return minus(self)
    
    def __sub__(self,other) -> 'Tensor':
        return self + (-other)
    
    def __rsub__(self,other) -> 'Tensor':
        return other + (-self)
    
    def __pow__(self,power) -> 'Tensor':
        return _pow(self,power)
    
    def __truediv__(self, other) -> 'Tensor': 
        return self * other **-1

    def __rtruediv__(self, other) -> 'Tensor': # other / self
        return other * self **-1
    
    def __matmul__(self,other) -> 'Tensor':
        return _matmul(self,other)
    
    def __getitem__(self,idxs) -> 'Tensor':
        return _slice(self,idxs)
    
    def __len__(self) -> 'int':
        return len(self.data)
    
    @property
    def T(self,indices=None) -> 'Tensor':
        return _transpose(self,indices=indices)
    
def _transpose(t:Tensor,indices=None) -> 'Tensor':
    if indices is None:
        indices = tuple(range(t.data.ndim -1,-1,-1))
    data = t.data.transpose(*indices)
    hooks = []
    if t.requires_grad:
        def backward(gradient):
            inverse = [0] * len(indices)
            for i,p in enumerate(indices):
                inverse[p] = i
            indices_back = tuple(inverse)
            gradient = gradient.transpose(*indices_back)
            return gradient
    return Tensor(data,requires_grad=t.requires_grad,nodes=hooks)

def _sum(t:Tensor) -> Tensor:
    data = t.data.sum()
    hooks = []
    if t.requires_grad:
        def backward(gradient:np.ndarray) -> np.ndarray:
            return gradient * np.ones_like(t.data)
        hooks.append(Hooks(t,backward))
        
    return Tensor(data,requires_grad=t.requires_grad,nodes=hooks)

def _add(t1:Tensor,t2:Tensor) -> Tensor:
    t1 = to_tensor(t1)
    t2 = to_tensor(t2)
    data = t1.data + t2.data
    hooks = []
    if t1.requires_grad:
        def backward(gradient:np.ndarray) -> np.ndarray:
            ndims_added = gradient.ndim - t1.data.ndim
            for _ in range(ndims_added):
                gradient = gradient.sum(axis=0)
            for i,dims in enumerate(t1.shape):
                if dims == 1:
                    gradient = gradient.sum(axis=i,keepdims=True)
            return gradient
        hooks.append(Hooks(t1,backward))
            
    if t2.requires_grad:
        def backward(gradient:np.ndarray) -> np.ndarray:
            ndims_added = gradient.ndim - t2.data.ndim
            for _ in range(ndims_added):
                gradient = gradient.sum(axis=0)
            for i,dims in enumerate(t2.shape):
                if dims == 1:
                    gradient = gradient.sum(axis=i,keepdims=True)
            return gradient
        hooks.append(Hooks(t2,backward))
    return Tensor(data=data,requires_grad=t1.requires_grad or t2.requires_grad,nodes=hooks)

def _pow(t,power):
        t = to_tensor(t)
        data = t.data ** power 
        hooks = []
        if t.requires_grad:
            def backward(gradient:np.ndarray) -> np.ndarray:
                return  gradient * power * t.data ** (power-1)
            hooks.append(Hooks(t,backward))    
        return Tensor(data=data,requires_grad=t.requires_grad,nodes=hooks)

def _matmul(t1:Tensor,t2:Tensor)->'Tensor':
    t1 = to_tensor(t1)
    t2 = to_tensor(t2)
    data = t1.data @ t2.data
    hooks = []
    if t1.requires_grad:
        def backward(gradient:np.ndarray) -> np.ndarray:
            return gradient @ t2.data.T
        hooks.append(Hooks(t1,backward))
            
    if t2.requires_grad:
        def backward(gradient:np.ndarray) -> np.ndarray:
            return t1.data.T @ gradient
        hooks.append(Hooks(t2,backward))
    return Tensor(data=data,requires_grad=t1.requires_grad or t2.requires_grad,nodes=hooks)

def _mul(t1:Tensor,t2:Tensor) -> Tensor:
    t1 = to_tensor(t1)
    t2 = to_tensor(t2)
    data = t1.data * t2.data
    hooks = []
    if t1.requires_grad:
        def backward(gradient:np.ndarray) -> np.ndarray:
            gradient = gradient * t2.data
            ndims_added = gradient.ndim - t1.data.ndim
            for _ in range(ndims_added):
                gradient = gradient.sum(axis=0)
            for i,dims in enumerate(t1.shape):
                if dims == 1:
                    gradient = gradient.sum(axis=i,keepdims=True)
            return gradient
        hooks.append(Hooks(t1,backward))
            
    if t2.requires_grad:
        def backward(gradient:np.ndarray) -> np.ndarray:
            gradient = gradient * t1.data
            ndims_added = gradient.ndim - t2.data.ndim
            for _ in range(ndims_added):
                gradient = gradient.sum(axis=0)
            for i,dims in enumerate(t2.shape):
                if dims == 1:
                    gradient = gradient.sum(axis=i,keepdims=True)
            return gradient
        hooks.append(Hooks(t2,backward))
    return Tensor(data=data,requires_grad=t1.requires_grad or t2.requires_grad,nodes=hooks)

Arrayable = Union[float,int,np.ndarray]
Tensorable = Union[Tensor,float,np.ndarray]

def to_array(x:Arrayable)->np.ndarray:
    if isinstance(x,np.ndarray):
        return x
    else: 
        return np.array(x)
    
def to_tensor(x:Tensorable) ->Tensor:
    if isinstance(x,Tensor):
        return x
    else:
        return Tensor(x)

def minus(t:Tensor)->Tensor:
    t = to_tensor(t)
    data = -t.data
    hooks = []
    if t.requires_grad:
        def backward(gradient):
            return -gradient
        hooks.append(Hooks(t,backward))
    return Tensor(data=data,requires_grad=t.requires_grad,nodes=hooks) 

def dummy_loss(x:Tensor) -> 'Tensor':
    data = x.data.mean()
    hooks = []
    if x.requires_grad:
        def backward(gradients):
            num = gradients.shape[0]
            grad = 1.0/ num
            return grad * gradients
        hooks.append(Hooks(x,backward))
    return Tensor(data,requires_grad=x.requires_grad,nodes=hooks)

def _slice(t:Tensor,idxs) -> Tensor:
    if isinstance(idxs,Tensor):
        idxs = idxs.data
    data = t.data[idxs]
    hooks = []
    if t.requires_grad:
        def backward(gradient):
            grad = np.zeros_like(data)
            grad[idxs] = gradient
            return grad
        hooks.append(Hooks(t,backward))
    return Tensor(data,nodes=hooks,requires_grad=t.requires_grad)
        

def split(t:Tensor,parts,axis=0):
    a = t.shape[axis]
    a = a / parts
    if (a % 1 != 0.0):
        raise RuntimeError('Tensor is not equally splittable')
    arr = []
    for i in range(parts):
        arr.append(t[ : ,int(a*i):int(a*(i+1))])
    return arr

def hstack(arrays:List[Tensor]) -> Tensor:
    requires_grad = False
    hooks = []
    data = arrays[0].data
    for t in range(1,len(arrays)):
        requires_grad = arrays[t-1].requires_grad or requires_grad
        data = np.hstack([data,arrays[t].data])
        if arrays[t-1].requires_grad:
            def backward(gradient):
                return gradient[t-1:t-1+arrays[t-1].shape[0]]
            hooks.append(Hooks(arrays[t-1],backward))
    requires_grad = arrays[-1].requires_grad or requires_grad
    if arrays[-1].requires_grad:
        def backward(gradient):
            return gradient[len(arrays)-1:len(arrays)-1+arrays[-1].shape[0]]
        hooks.append(Hooks(arrays[-1],backward))
    return Tensor(data,requires_grad=requires_grad,nodes=hooks)


def tri(shape,dtype=np.float64,requires_grad=False):
    data = np.tri(shape,dtype=dtype)
    return Tensor(data,requires_grad=requires_grad)

def sqrt(t:Tensor) -> 'Tensor':
    t = to_tensor(t)
    data = np.sqrt(t.data)
    hooks = []
    
    if t.requires_grad:
        hooks.append(Hooks(t,lambda gradient: -1/(2*np.sqrt(t.data)) * gradient))
    return Tensor(data,requires_grad=t.requires_grad,nodes=hooks)

def argmax(t:Tensor,axis=None)->'Tensor':
    t = to_tensor(t)
    data = t.data
    if axis is None:
        data = np.unravel_index(np.argmax(data),shape=t.shape)
    else:
        data = np.argmax(data,axis=axis)
    return Tensor(data)