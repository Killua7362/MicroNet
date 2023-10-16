import unittest
import torch
from MicroNet.micro.tensor import Tensor
import numpy as np

class TestSanity(unittest.TestCase):
    def test_sanity(self):
        x = torch.Tensor([1.])
        x.requires_grad = True
        z = 2 * x + 2 + x
        q = z * z
        h = z * z
        h = h + q 
        h = h ** 2
        h = h / 2
        h.backward(torch.tensor([1.]))
        t1 = x.grad
        
        x = Tensor([1.],requires_grad=True)
        z = 2 * x + 2 + x
        q = z * z
        h = z * z
        h = h + q 
        h = h ** 2
        h = h / 2
        h.backward(Tensor(np.ones_like(h.data)))
        t2 = x.grad
        assert t1.tolist() == t2.data.tolist()
