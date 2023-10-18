import sys
import os
import numpy as np
import tensorflow as tf
import json
from torch.nn import functional as F
import torch

sys.path.insert(0,os.path.abspath(""))

from micro.activations import Gelu,SoftMax
from micro.layers import Dense,LayerNorm,Embeddings
from micro.utils import load_params
from micro.model import Model
from encoder import get_encoder
from utils import get_param_dict
from micro.tensor import Tensor,split,concatenate,tri,sqrt,argmax,hstack,append
from micro.losses import CategoricalCrossEntropy

class FeedForward:
    def __init__(self):
        self.gelu = Gelu()
        self.dense_1  = Dense() 
    
    def __call__(self,inputs):
        inputs = self.gelu(self.dense_1(inputs))
        return inputs

m = FeedForward()
inputs = Tensor(np.random.randn(3,),requires_grad=True)
out = m(inputs)
out.backward(Tensor(np.ones_like(out)))
print(inputs.grad)
# out.build_graph()
# print(out.graph)

sys.exit(0)
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
        mask = (1-tri(inputs.shape[0],dtype=inputs.dtype)) * -1e10
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
        self.wpe= Embeddings(self.n_ctx,self.n_embd)
        self.wte= Embeddings(self.n_vocab,self.n_embd)
        super().__init__()
    
    def forward(self,inputs,targets=None):
        inputs = self.wte(inputs) + self.wpe(range(len(inputs)))
        for block in self.transformer_blocks:
            inputs = block(inputs)
        logits= self.layer_norm(inputs)
        logits = logits @ self.wte.w.T
        # print(F.cross_entropy(torch.tensor( logits.data ),torch.tensor( targets.data )))
        return logits, None
    
    def __call__(self,inputs,targets):
        return self.forward(inputs,targets)
    
def regress(model,inputs,targets,n_tokens_gen):
    from tqdm import tqdm
    for _ in tqdm(range(n_tokens_gen),'generating'):
        logits,loss = model(inputs,targets)
        next_id = argmax(logits[-1]).sum()
        inputs = inputs.append(next_id)
    return inputs[len(inputs)-n_tokens_gen:]

loss_fn = CategoricalCrossEntropy()
text = "context: Akshay has neck pain. Q: Who has neck pain?"
target = "The telephone was invented by"
model_name = '1558M' 
models_dir = 'Weights'
path = os.path.join(models_dir,model_name)
check_point = tf.train.latest_checkpoint(path)
hparams = json.load(open(os.path.join(path,'hparams.json')))
params = get_param_dict(check_point,hparams)
encoder = get_encoder(model_name,models_dir)

inputs = encoder.encode(text)
targets = encoder.encode(target)
inputs = Tensor(inputs,requires_grad = True)
targets = Tensor(targets,requires_grad = True)
gpt = GPT(hparams)
load_params(gpt,params)
# print(len(ids))
out = regress(gpt,inputs,targets,1)
out = out.data
print(encoder.decode(out))
# 24915   388 11887   318