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
        self.graph = None
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
    
    #for debugging
    def build_graph(self):
        topo = []
        visited = set()
        def _build_graph(root):
            if root not in visited:
                visited.add(root)
                for child in root.tensor.nodes:
                    _build_graph(child)
                topo.append(root)
        _build_graph(Hooks(self,lambda: None))
        self.graph = topo
        del self.graph[-1]
    
    def backward(self,grad:'Tensor'=None):
        if not self.requires_grad:
           raise AssertionError('non-requires-grad') 
       
        if grad is None:
            if self.shape == () or self.data == []:
                grad = Tensor(1.0)
            else:
                raise RuntimeError('specify grad for non-0-tensor')
        self.grad.data = grad.data + self.grad.data
        
        hooks = self.nodes
        if hooks is not None:
            for node in self.nodes:
                backward_grad = node.grad_fn(grad.data)
                node.tensor.backward(Tensor(backward_grad))
    
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

    def __ge__(self,other):
        return self.data >= other.data

    def register_hook(self,hook):
        self.nodes.append(hook)
    
    
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
            num = gradients[()]
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
            if gradient.shape != grad.shape:
                grad[idxs] = gradient
            else:
                grad = gradient
            return grad
        hooks.append(Hooks(t,backward))
    return Tensor(data,nodes=hooks,requires_grad=t.requires_grad)
        
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
    return Tensor(data,requires_grad=t.requires_grad)

def concatenate(tensor,axis=0) -> 'Tensor':
    requires_grad = False
    hooks = []
    data = np.array([])
    
    for idx,t in enumerate(tensor):
        t = to_tensor(t)
        requires_grad = t.requires_grad or requires_grad
        if data.size == 0:
            data = t.data
        else:
            data = np.concatenate((t.data,data),axis=axis)
        if t.requires_grad:
            def backward(gradient):
                if axis == 0:
                    return gradient[idx:idx+t.shape[0],:]
                elif axis == 1:
                    return gradient[:,idx:idx+t.shape[1]]
            hooks.append(Hooks(t,backward))
    return Tensor(data,requires_grad=requires_grad,nodes=hooks)


def hstack(arrays:List[Tensor]) -> Tensor:
    requires_grad = any(t.requires_grad for t in arrays)
    data_list = [t.data for t in arrays]
    stacked = np.hstack(data_list)
    hooks = []
    if requires_grad:
        current_idx = 0
        for t in arrays:
            if t.requires_grad:
                num_col = t.shape[1]
                def backward(gradient):
                    return gradient[:,current_idx:current_idx+num_col]
                hooks.append(Hooks(t,backward))
            current_idx += num_col
    return Tensor(stacked,requires_grad=requires_grad,nodes=hooks)

def append(t,value):
    t = to_tensor(t)
    value = to_tensor(value)
    requires_grad = t.requires_grad or value.requires_grad
    if t.data.size == 0:
        data = [value.data]
    elif value.data.size == 0:
        data = [t.data]
    else:
        data = t.data.tolist()
        data.append(value.data)
    hooks = []
    if t.requires_grad:
        def backward(grad):
            return grad[:-1]
        hooks.append(Hooks(t, backward))

    if value.requires_grad:
        def backward(grad):
            return grad[-1]
        hooks.append(Hooks(value, backward))
    return Tensor(data, requires_grad,nodes=hooks)


def split(input_tensor, num_splits, axis=-1):
    if axis<0:
        axis = len(input_tensor.shape) + axis
        
    input_tensor = to_tensor(input_tensor)
    input_shape = input_tensor.shape
    assert input_shape[axis] % num_splits == 0, "Invalid split size"

    split_size = input_shape[axis] // num_splits
    split_tensors = []

    for i in range(num_splits):
        start = i * split_size
        end = (i + 1) * split_size

        def backward(gradient):
            # Create a tensor with zeros of the same shape as the input
            grad = np.zeros_like(input_tensor.data)

            # Assign the received gradient to the appropriate split range
            if axis == 0:
                grad[start:end, :] = gradient
            elif axis == 1:
                grad[:, start:end] = gradient

            return grad

        split_tensors.append(Tensor(input_tensor.data[:, start:end], requires_grad=input_tensor.requires_grad, nodes=[Hooks(input_tensor, backward)]))

    return split_tensors