import unittest
import torch
from micro.tensor import Tensor
import numpy as np

class TestTensorMatMul(unittest.TestCase):
    def test_simple_matmul(self):
        x = Tensor([1.],requires_grad=True)
        y = Tensor([1.],requires_grad=True)
        t1 = x @ y 
        
        x = np.array([1.])
        y = np.array([1.])
        t2 = x @ y 
        assert t1.data.tolist() == t2.tolist()
