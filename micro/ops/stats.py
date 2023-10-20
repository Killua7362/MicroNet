import numpy as np
import cupy as cp

#np.mean
def mean(a,axis=None,dtype=None,out=None,keepdims=False):
    if 'numpy' in str(type(a)):
        return np.mean(a,axis,dtype,out,keepdims)
    elif 'cupy' in str(type(a)):
        return cp.mean(a,axis,dtype,out,keepdims)
    
#np.sum
def sum(a,axis=None,dtype=None,out=None,keepdims=False):
    if 'numpy' in str(type(a)):
        return np.sum(a,axis,dtype,out,keepdims)
    elif 'cupy' in str(type(a)):
        return cp.sum(a,axis,dtype,out,keepdims)
    
#np.argmax
def argmax(a,axis=None,dtype=None,out=None,keepdims=False):
    if 'numpy' in str(type(a)):
        return np.argmax(a,axis,dtype,out,keepdims)
    elif 'cupy' in str(type(a)):
        return cp.argmax(a,axis,dtype,out,keepdims)
    
#np.maximum
def maximum(x1,x2):
    if 'numpy' in str(type(x1)):
        return np.maximum(x1,x2)
    elif 'cupy' in str(type(x1)):
        return cp.maximum(x1,x2)
    
#np.tanh
def tanh(x):
    if 'numpy' in str(type(x)):
        return np.tanh(x)
    if 'cupy' in str(type(x)):
        return cp.tanh(x)
    
#np.sqrt
def sqrt(x):
    if 'cupy' in str(type(x)):
        return cp.sqrt(x)
    return np.sqrt(x)
    
#np.pi

#np.exp
def exp(x):
    if 'numpy' in str(type(x)):
        return np.exp(x)
    if 'cupy' in str(type(x)):
        return cp.exp(x)

#np.max
def max(x,axis,keepdims,out=None):
    if 'numpy' in str(type(x)):
        return np.max(x,axis,keepdims=keepdims,out=out)
    if 'cupy' in str(type(x)):
        return cp.max(x,axis,keepdims)
    
#np.var
def var(x,axis=None,dtype=None,out=None,keepdims=False):
    if 'numpy' in str(type(x)):
        return np.var(x,axis=None,dtype=None,out=None,keepdims=False)
    if 'cupy' in str(type(x)):
        return cp.var(x,axis=None,dtype=None,out=None,keepdims=False)
