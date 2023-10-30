import numpy as np 
try:
    import cupy as cp
except ModuleNotFoundError as err:
    import numpy as cp
   
#np.random.randn
def randn(*size,device='cpu',**kwargs):
    if device == 'cpu':
        return np.random.randn(*size,**kwargs)
    elif device == 'cuda':
        return cp.random.randn(*size,**kwargs)

    