import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError as err:
    import numpy as cp
    
def ones(shape,dtype=np.float64,device='cpu'):
    if device == 'cpu':
        return np.ones(shape,dtype=dtype)
    elif device == 'cuda':
        return cp.ones(shape,dtype=dtype)
    
#np.zeros
def zeros(shape,dtype=np.float64,device='cpu'):
    if device == 'cpu':
        return np.zeros(shape,dtype=dtype)
    elif device == 'cuda':
        return cp.zeros(shape,dtype=dtype)
    
#np.ndarray

#np.tri
def tri(N,*args,dtype=np.float64,**kwargs):
    if 'numpy' in str(dtype):
        return np.tri(N,*args,dtype=dtype,**kwargs)
    return cp.tri(N,*args,dtype=dtype,**kwargs)
 
#np.array
def array(a,dtype=None,copy=True,device='cpu'):
    if device == 'cpu':
        return np.array(a,dtype,copy)
    elif device == 'cuda':
        return cp.array(a,dtype,copy)
    