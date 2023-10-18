import sys
#this will get leaf params from the dictionary
#currently tested with llama that is it
def get_flattended_weights(d):
    leaf = []
    def flatten_dict(nested_dict):
        for key, value in nested_dict.items():
            if isinstance(value, dict):
                if all(not isinstance(v, dict) for v in value.values()):
                    leaf.append(value)
                else:
                    flatten_dict(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        if all(not isinstance(v, dict) for v in item.values()):
                            leaf.append(item.values())
                        else:
                            flatten_dict(item)
            else:
                leaf.append({key:value})
    flatten_dict(d)
    return leaf


def update_dict(a,b):
    for key,value in b.items():
        if key in a:
            a[key] = value
        else:
            raise KeyError(f'The key {key} does not exist')
    return a 

def load_params(model,params,emb=True):
    from micro.layers import BaseLayer
    if model.instances == []:
        raise AttributeError(f'BaseLayer instance is empty. do not call this function without initializing the model')
    params = get_flattended_weights(params)    
    temp = 0
    if emb:
        params[-2]['w'] = params[-2].pop('wpe')
        params[-1]['w'] = params[-1].pop('wte')
    else:
        temp = -2
    for i,val in enumerate(params[:temp]):
        model.instances[i].trainable_params = update_dict(model.instances[i].trainable_params,val)