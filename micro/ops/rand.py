import numpy as np 
import cupy as cp

#np.random.randn
def randn(*size,device='cpu',**kwargs):
    if device == 'cpu':
        return np.random.randn(*size,**kwargs)
    elif device == 'cuda':
        return cp.random.randn(*size,**kwargs)

    