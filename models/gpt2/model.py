
import sys
import os

script_dir = os.path.dirname(sys.path[0])
micro_net_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(micro_net_dir)

from micro.activations import Gelu,SoftMax
from micro.layers import Dense,LayerNorm
from micro.model import Model
from micro.tensor import Tensor,split,concatenate,tri,sqrt,argmax,append,dummy_loss
import numpy as np

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
        res = self.softmax(q @ k.T / sqrt(q.shape[-1],device=mask.device)+mask) @ v
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
        mask = (1-tri(inputs.shape[0],dtype=np.float32 if 'numpy' in str(type(x.data)) else None)) * -1e10
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
    def __init__(self,hparams,past=None,):
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
    