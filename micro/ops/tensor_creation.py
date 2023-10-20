import numpy as np
import cupy as cp

#np.ones
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
def tri(N,M=None,k=0,dtype=cp.float64):
     return np.tri(N,M,k,dtype)
 
#np.array
def array(a,dtype=None,copy=True,device='cpu'):
    if device == 'cpu':
        return np.array(a,dtype,copy)
    elif device == 'cuda':
        return cp.array(a,dtype,copy)
    