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

def load_params(model,params):
    from lib.layers import BaseLayer
    if model.instances == []:
        raise AttributeError(f'BaseLayer instance is empty. do not call this function without initializing the model')
    params = get_flattended_weights(params)    
    for i,val in enumerate(model.instances):
        val.trainable_params = update_dict(val.trainable_params,params[i])
