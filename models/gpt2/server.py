import sys
import os
import base64
script_dir = os.path.dirname(sys.path[0])
micro_net_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(micro_net_dir)

import tensorflow as tf
import json
from torch.nn import functional as F

try:
    import cupy as cp
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
except ModuleNotFoundError as err:
    pass
    
from encoder import get_encoder
from utils import get_param_dict
from get_weights import get_weights

from micro.utils import load_params
from micro.tensor import Tensor,argmax
from model import GPT

from flask import Flask,jsonify,request
from flask_cors import CORS

def regress(model,inputs,n_tokens_gen,wte,wpe):
    from tqdm import tqdm
    for _ in tqdm(range(n_tokens_gen),'generating'):
        embds = wte[inputs] + wpe[range(len(inputs))]
        logits = model(embds)
        logits = logits @ wte.T
        next_id = argmax(logits[-1]).sum()
        next_id = next_id.data
        inputs.append(int(next_id))
        
        del embds
        del logits
        try:
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
        except:
            pass
    return inputs[len(inputs)-n_tokens_gen:]

device = 'cpu'
model_name = '124M' 
models_dir = 'Weights'
get_weights(models_dir,model_name)
path = os.path.join(micro_net_dir,models_dir,model_name)
check_point = tf.train.latest_checkpoint(path)
hparams = json.load(open(os.path.join(path,'hparams.json')))
params = get_param_dict(check_point,hparams,device=device)
encoder = get_encoder(model_name,models_dir)

gpt = GPT(hparams)
wte = Tensor(params['wte'],requires_grad=False,device=device)
wpe = Tensor(params['wpe'],requires_grad=False,device=device)
load_params(gpt,params,emb=False)

app = Flask(__name__)
CORS(app)

@app.route("/home",methods=['POST'])
def return_home():
    input = request.json
    user = input['user']
    bot = input['bot']
    context = input['context']
    
    convo = []
    for i,val in enumerate(user):
        convo.append(f'user:{val} bot:{bot[i]}')
    convo = ' '.join(convo)
    question = input['question']
    if len(user) == 0:
        prompt = question
    else:
        prompt = f'This is the conversation between you and me, here bot is you and user is me\
        {convo}\
        This is my current question\
        {question}\
        '
    if context != '':
        base64_encoded = context.split(',')[1]
        base64_bytes = base64.b64decode(base64_encoded)
        context = base64_bytes.decode('utf-8')
        prompt = 'This is the context: ' + context + ' ' + prompt
        
    prompt = " ".join(prompt.split())
    input = encoder.encode(prompt)
    out = regress(gpt,input,10,wte,wpe)
    return jsonify({
        'result':encoder.decode(out)
    })
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000)