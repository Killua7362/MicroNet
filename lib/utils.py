#this will get leaf params from the dictionary
#currently tested with llama that is it
def get_flattended_weights(d):
    leaf = []
    def flatten_dict(nested_dict):
        for key, value in nested_dict.items():
            if isinstance(value, dict):
                if all(not isinstance(v, dict) for v in value.values()):
                    leaf.append({key:value})
                else:
                    flatten_dict(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        if all(not isinstance(v, dict) for v in item.values()):
                            leaf.append(item)
                        else:
                            flatten_dict(item)
            else:
                leaf.append({key:value})
    flatten_dict(d)
    return leaf