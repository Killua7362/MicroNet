import sys
import os

script_dir = os.path.dirname(sys.path[0])
micro_net_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(micro_net_dir)

import tensorflow as tf
import json
from torch.nn import functional as F
import cupy as cp
from encoder import get_encoder
from utils import get_param_dict

from micro.utils import load_params
from micro.tensor import Tensor,argmax
from model import GPT

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

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
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        
    return inputs[len(inputs)-n_tokens_gen:]

device = 'cuda'
text = "Quantum physics is"
model_name = '1558M' 
models_dir = 'Weights'
path = os.path.join(micro_net_dir,models_dir,model_name)
check_point = tf.train.latest_checkpoint(path)
hparams = json.load(open(os.path.join(path,'hparams.json')))
params = get_param_dict(check_point,hparams,device=device)
encoder = get_encoder(model_name,models_dir)

inputs = encoder.encode(text)

# inputs = Tensor(inputs,requires_grad = True)
# targets = Tensor(targets,requires_grad = True)

gpt = GPT(hparams)

wte = Tensor(params['wte'],requires_grad=False,device=device)
wpe = Tensor(params['wpe'],requires_grad=False,device=device)

import time
load_params(gpt,params,emb=False)
start_time = time.perf_counter()
out = regress(gpt,inputs,10,wte,wpe)
end_time = time.perf_counter()
print(encoder.decode(out))
print(end_time-start_time)
