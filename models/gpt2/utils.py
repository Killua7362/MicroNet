import tensorflow as tf
import numpy as np
import re

try:
    import cupy as cp
except ModuleNotFoundError as err:
    pass
    
def nested_params(dict,keys,val):
    if not keys:
        return val
    if keys[0] not in dict:
        dict[keys[0]] = {}
    dict[keys[0]] = nested_params(dict[keys[0]],keys[1:],val)
    return dict
    
def get_param_dict(pnt,hparams,device='cpu'):
    params = {'blocks':[{} for _ in range(hparams['n_layer'])]}
    for name,_ in tf.train.list_variables(pnt):
        array = np.squeeze(tf.train.load_variable(pnt,name))
        
        if device == 'cuda':
            array = cp.asarray(array)
            
        name = name[len('model/'):]
        if name.startswith('h'):
            m = re.match(r'h([0-9]+)/(.*)',name)
            n = int(m[1])
            sub_name = m[2]
            nested_params(params['blocks'][n],sub_name.split('/'),array)
        else: 
            nested_params(params,name.split('/'),array)
    return params