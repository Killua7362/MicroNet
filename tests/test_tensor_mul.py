import unittest
from MicroNet.micro.tensor import Tensor

class TestTensorMul(unittest.TestCase):
    def test_simple_mul(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([1, 2, 3], requires_grad=True)
        t3 = t1 * t2       
        assert t3.data.tolist() == [1.,4.,9.]