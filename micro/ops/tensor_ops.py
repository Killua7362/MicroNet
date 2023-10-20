import numpy as np
import cupy as cp

#np.empty_liek
def empty_like(a,dtype=None,order='K',shape=None):
    if 'numpy' in str(type(a)):
        return np.empty_like(a,dtype=dtype,order=order,shape=shape)
    if 'cupy' in str(type(a)):
        return cp.empty_like(a,dtype=dtype,order=order,shape=shape)
    
#np.diagflat
def diagflat(a,k):
    if 'numpy' in str(type(a)):
        return np.diagflat(a,k)
    if 'cupy' in str(type(a)):
        return cp.diagflat(a,k)
    
#np.dot
def dot(a,b,out=None):
    if 'numpy' in str(type(a)):
        return np.dot(a,b)
    if 'cupy' in str(type(a)):
        return cp.dot(a,b)
    
#np.eye
def eye(N,M=None,k=0,dtype=cp.float64):
     return cp.eye(N,M,k,dtype)
    
#np.zeros_like
def zeros_like(a,dtype=None,order='K',shape=None):
    if 'numpy' in str(type(a)):
        return np.zeros_like(a,dtype,order,shape)
    if 'cupy' in str(type(a)):
        return cp.zeros_like(a,dtype,order,shape)
    
#np.ones_like
def ones_like(a,dtype=None,order='K',shape=None):
    if 'numpy' in str(type(a)):
        return np.ones_like(a,dtype=dtype,order=order,shape=shape)
    if 'cupy' in str(type(a)):
        return cp.ones_like(a,dtype=dtype,order=order,shape=shape)
    
#np.unravel_index
def unravel_index(a,dims,order='C'):
    if 'numpy' in str(type(a)):
        return np.unravel_index(a,dims,order)
    if 'cupy' in str(type(a)):
        return cp.unravel_index(a,dims,order)
    
#np.concatenate
def concatenate(a,axis=0,out=None,*,dtype=None):
    if 'numpy' in str(type(a)):
        return np.concatenate(a,axis,out,dtype)
    if 'cupy' in str(type(a)):
        return cp.concatenate(a,axis,out,dtype)
    
#np.clip
def clip(a,a_min,a_max,out=None):
    if 'numpy' in str(type(a)):
        return np.clip(a,a_min,a_max,out=out)
    if 'cupy' in str(type(a)):
        return cp.clip(a,a_min,a_max,out=out)
