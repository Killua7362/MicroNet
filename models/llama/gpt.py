import sys
import os
import numpy as np
import tensorflow as tf
import json

sys.path.insert(0,os.path.abspath(""))

from micro.activations import Gelu,SoftMax
from micro.layers import Dense,LayerNorm,Embeddings
from micro.utils import load_params
from micro.model import Model
from encoder import get_encoder
from utils import get_param_dict
from micro.tensor import Tensor,split,hstack,tri,sqrt,argmax
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
    def __init__(self,hparams):
        #pos emb and time emb
        self.n_ctx = hparams['n_ctx']
        self.n_embd = hparams['n_embd']
        self.n_vocab = hparams['n_vocab']
        self.n_blocks = hparams['n_layer']
        self.n_head = hparams['n_head']
        self.wpe= Embeddings(self.n_ctx,self.n_embd)
        self.wte= Embeddings(self.n_vocab,self.n_embd)
        self.transformer_blocks = []
        for _ in range(self.n_blocks):
            self.transformer_blocks.append(TransformerBlock(n_heads=self.n_head))
        self.layer_norm = LayerNorm()
        super().__init__()
    
    def forward(self,inputs,targets=None):
        inputs = self.wte(inputs) + self.wpe(range(len(inputs)))
        for block in self.transformer_blocks:
            inputs = block(inputs)
        inputs = self.layer_norm(inputs)
        inputs = inputs @ self.wte.w.T
        if target is None:
            loss = None
        else:
            print(inputs.shape)
            loss = None
        return inputs, loss
    
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
text = "Quantum physics is"
target = "Quantum physics is study of quantum particles"
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
out = regress(gpt,inputs,targets,4)
out = out.data
print(encoder.decode(out))
# 24915   388 11887   318