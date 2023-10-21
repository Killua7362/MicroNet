import sys
import os
import numpy as np
import tensorflow as tf
import json
from torch.nn import functional as F
import cupy as cp
sys.path.insert(0,os.path.abspath(""))

from micro.activations import Gelu,SoftMax
from micro.layers import Dense,LayerNorm
from micro.utils import load_params
from micro.model import Model
from encoder import get_encoder
from utils import get_param_dict
from micro.tensor import Tensor,split,concatenate,tri,sqrt,argmax,append,dummy_loss
from micro.losses import CategoricalCrossEntropy

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
        res = self.softmax(q @ k.T / sqrt(q.shape[-1])+mask) @ v
        return res
    
class MultiHeadAttention:
    def __init__(self,n_heads=25):
        self.dense_1 = Dense() 
        self.dense_2 = Dense()
        self.attention = Attention()
        self.n_heads = n_heads
    
    def __call__(self,inputs):
        inputs = inputs
        x = self.dense_1(inputs)
        mask = (1-tri(inputs.shape[0],dtype=cp.float32)) * -1e10
        qkv = split(x,3,axis=-1)
        q = split(qkv[0],self.n_heads,axis=-1)
        k = split(qkv[1],self.n_heads,axis=-1)
        v = split(qkv[2],self.n_heads,axis=-1)
            
        out_head = []
        for q, k, v in zip(q,k,v):
            out_head.append(self.attention(q,k,v,mask))
        x = concatenate(out_head,axis=1)
        y = self.dense_2(x)
        return y
    

class TransformerBlock:
    def __init__(self,n_heads=25):
        self.mha = MultiHeadAttention(n_heads=n_heads)
        self.layer_norm_1 = LayerNorm()
        self.layer_norm_2 = LayerNorm()
        self.ffn = FeedForward()
    
    def __call__(self,inputs):
        inputs = inputs + self.mha(self.layer_norm_1(inputs))
        inputs = inputs + self.ffn(self.layer_norm_2(inputs))
        return inputs

class GPT(Model):
    def __init__(self,hparams):
        #pos emb and time emb
        self.n_ctx = hparams['n_ctx']
        self.n_embd = hparams['n_embd']
        self.n_vocab = hparams['n_vocab']
        self.n_blocks = hparams['n_layer']
        self.n_head = hparams['n_head']
        self.transformer_blocks = []
        for _ in range(self.n_blocks):
            self.transformer_blocks.append(TransformerBlock(n_heads=self.n_head))
        self.layer_norm = LayerNorm()

        super().__init__()
    
    def forward(self,inputs):
        for block in self.transformer_blocks:
            inputs = block(inputs)
        logits = self.layer_norm(inputs)
        return logits
    
    def __call__(self,inputs):
        return self.forward(inputs)
    
def regress(model,inputs,targets,n_tokens_gen,wte,wpe):
    from tqdm import tqdm
    for _ in tqdm(range(n_tokens_gen),'generating'):
        embds = wte[inputs] + wpe[range(len(inputs))]
        logits = model(embds)
        logits = logits @ wte.T
        next_id = argmax(logits[-1]).sum()
        next_id = next_id.data
        inputs.append(int(next_id))
    return inputs[len(inputs)-n_tokens_gen:]

loss_fn = CategoricalCrossEntropy()
text = "Quantum physics is"
target = "The telephone was invented by"
model_name = '1558M' 
models_dir = 'Weights'
path = os.path.join(models_dir,model_name)
check_point = tf.train.latest_checkpoint(path)
hparams = json.load(open(os.path.join(path,'hparams.json')))
params = get_param_dict(check_point,hparams,device='cuda')
encoder = get_encoder(model_name,models_dir)

inputs = encoder.encode(text)
targets = encoder.encode(target)

# inputs = Tensor(inputs,requires_grad = True)
# targets = Tensor(targets,requires_grad = True)

gpt = GPT(hparams)

wte = Tensor(params['wte'],requires_grad=True,device='cuda')
wpe = Tensor(params['wpe'],requires_grad=True,device='cuda')

load_params(gpt,params,emb=False)
out = regress(gpt,inputs,targets,10,wte,wpe)
print(encoder.decode(out))
# 24915   388 11887   318