import unittest
from MicroNet.micro.tensor import Tensor

class TestTensorNeg(unittest.TestCase):
    def test_simple_neg(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t1 = -t1
        assert t1.data.tolist() == [-1, -2, -3]

class TestTensorSub(unittest.TestCase):
    def test_simple_sub(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([1, 2, 3], requires_grad=True)
        
        t3 = t1 - t2
        assert t3.data.tolist() == [0,0,0]