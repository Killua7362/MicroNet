import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError as err:
    import numpy as cp
   
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
def argmax(a,*args,**kwargs):
    if 'numpy' in str(type(a)):
        return np.argmax(a,*args,**kwargs)
    elif 'cupy' in str(type(a)):
        return cp.argmax(a,*args,**kwargs)
    
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
        return cp.max(x,axis,keepdims=keepdims,out=None)
    
#np.var
def var(x,axis=None,dtype=None,out=None,keepdims=False):
    if 'numpy' in str(type(x)):
        return np.var(x,axis=axis,dtype=dtype,out=out,keepdims=keepdims)
    if 'cupy' in str(type(x)):
        return cp.var(x,axis=axis,dtype=dtype,out=out,keepdims=keepdims)
