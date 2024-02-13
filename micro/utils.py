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
    if emb:
        params.insert(0,params.pop())
        params.insert(0,params.pop())
        params[0]['w'] = params[0].pop('wpe')
        params[1]['w'] = params[1].pop('wte')
        for i,val in enumerate(params):
            model.instances[i].trainable_params = update_dict(model.instances[i].trainable_params,val)
    else:
        for i,val in enumerate(params[:-2]):
            model.instances[i].trainable_params = update_dict(model.instances[i].trainable_params,val)
        
def args_expander(f,transpiled_graph,args,kwargs):
    from torch import tensor
    #traced_graph is passed so that we can get it from locals() after python exec
    ##gets the source code and removes the indentation error
    if args == None:
        args = []
    if kwargs == None:
        kwargs = {}
    args = list(args)
    source = textwrap.dedent(inspect.getsource(f))
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
            node.args.args=[]
            node.args.defaults = []
            if args != ():
                ##goes through the args and creates t1,t2,...,len(args) temporary variables inside function definition call
                for i,val in enumerate(args):
                    args_arg = ast.arg()
                    args_arg.arg = 't'+str(i)
                    node.args.args.append(args_arg)

            if kwargs != {}:
                ##goes through kwargs and and creates kwargs1=None,kwarg2=None,...len(kwargs) assignments inside function definition call
                for key,val in kwargs.items():
                    arg = ast.arg()
                    arg.arg = key
                    node.args.args.append(arg)
                    const = ast.Constant()
                    const.value = None
                    node.args.defaults.append(const)
        elif isinstance(node,ast.Call):
            node.args = []
            #goes through args and adds all the constants
            if args!= ():
                for i,val in enumerate(args):
                    s = StringIO()
                    print(val, file=s) #does not print anything
                    result = s.getvalue()
                    node.args.append(ast.parse(result).body[0].value)
            ##goes through kwargs and creates kwarg1=kwarg1,kwarg2=kwarg2,....len(kwargs) inside model call
            node.keywords = []
            if kwargs != {} :
                for key in kwargs.keys():
                    keyword = ast.keyword()
                    keyword.arg = key
                    value = ast.Name()
                    value.id = key
                    value.ctx = ast.Load()
                    keyword.value = value
                    node.keywords.append(keyword)

    tree = ast.fix_missing_locations(tree)
    name = f.__code__.co_name
    ##runs a function once so that it is store in python local/global stack
    code = compile(tree, name, 'exec')
    temp_locals = dict(locals())
    exec(code, temp_locals)
    ##returns a modified function
    return temp_locals[name]
