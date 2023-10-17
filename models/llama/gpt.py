import sys
import os
import numpy as np
import tensorflow as tf
import json

sys.path.insert(0,os.path.abspath(""))

from micro.activations import Gelu,SoftMax
from micro.layers import Dense,LayerNorm
from micro.utils import load_params
from micro.model import Model
from encoder import get_encoder
from utils import get_param_dict
from micro.tensor import Tensor,split,hstack,tri,sqrt,argmax

class FeedForward:
    def __init__(self):
        self.gelu = Gelu()
        self.dense_1  = Dense() 
        self.dense_2  = Dense() 
    
    def __call__(self,inputs):
        inputs = self.gelu(self.dense_1(inputs))
        inputs = self.dense_2(inputs) 
        return inputs

class Attention:
    def __init__(self):
        self.softmax = SoftMax()
    
    def __call__(self,q,k,v,mask=None):
        return self.softmax(q @ k.T / sqrt(q.shape[-1])+mask) @ v
    
class MultiHeadAttention:
    def __init__(self,n_heads=4):
        self.dense_1 = Dense() 
        self.dense_2 = Dense()
        self.attention = Attention()
        self.n_heads = n_heads
    
    def __call__(self,inputs):
        inputs = self.dense_1(inputs)
        qkv = split(inputs,3,axis=-1)
        qkv_heads = list(map(lambda inputs:split(inputs,self.n_heads,axis=-1),qkv))
        mask = (1-tri(inputs.shape[0],dtype=inputs.dtype)) * -1e10
        out_head = [self.attention(q,k,v,mask) for q,k,v in zip(*qkv_heads)]
        inputs = hstack(out_head)
        inputs = self.dense_2(inputs)
        return inputs
    
class TransformerBlock:
    def __init__(self,n_heads=4):
        self.mha = MultiHeadAttention(n_heads=n_heads)
        self.layer_norm_1 = LayerNorm()
        self.layer_norm_2 = LayerNorm()
        self.ffn = FeedForward()
    
    def __call__(self,inputs):
        inputs = inputs + self.mha(self.layer_norm_1(inputs))
        inputs = inputs + self.ffn(self.layer_norm_2(inputs))
        return inputs


class GPT(Model):
    def __init__(self,n_blocks=4,n_head=4):
        self.transformer_blocks = []
        for _ in range(n_blocks):
            self.transformer_blocks.append(TransformerBlock(n_heads=n_head))
        self.layer_norm = LayerNorm()
        super().__init__()
    
    def forward(self,inputs):
        for block in self.transformer_blocks:
            inputs = block(inputs)
        return self.layer_norm(inputs)
    
    def __call__(self,inputs):
        return self.forward(inputs)
    
def regress(inputs,n_tokens_gen,n_head,params):
    from tqdm import tqdm
    wte = Tensor(params['wte'],requires_grad=True)
    wpe = Tensor(params['wpe'],requires_grad=True)
    gpt = GPT(n_blocks=len(params['blocks']),n_head=n_head)
    load_params(gpt,params)
    for _ in tqdm(range(n_tokens_gen),'generating'):
        emb = wte[inputs] + wpe[range(len(inputs))]
        out = gpt(emb)
        logits = out @ wte.T
        next_id = argmax(logits[-1]).sum()
        inputs = inputs.append(next_id)
    return inputs[len(inputs)-n_tokens_gen:]

text = "Quantum physics is"
model_name = '1558M' 
models_dir = 'Weights'
path = os.path.join(models_dir,model_name)
check_point = tf.train.latest_checkpoint(path)
hparams = json.load(open(os.path.join(path,'hparams.json')))
params = get_param_dict(check_point,hparams)
encoder = get_encoder(model_name,models_dir)

ids = encoder.encode(text)
ids = Tensor(ids,requires_grad = True)
out = regress(ids,4,hparams['n_head'],params)
out = out.data
print(encoder.decode(out))
# 24915   388 11887   318