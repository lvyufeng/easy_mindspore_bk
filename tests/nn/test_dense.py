import unittest
from easy_mindspore.nn import BiDense
import mindspore.numpy as mnp

class TestDense(unittest.TestCase):
    def test_bidense(self):
        m = BiDense(20, 30, 40)
        input1 = mnp.randn(128, 20)
        input2 = mnp.randn(128, 30)
        output = m(input1, input2)
        assert output.shape == (128, 40)
    
    def test_bidense_nd(self):
        m = BiDense(20, 30, 40)
        input1 = mnp.randn(128, 4, 20)
        input2 = mnp.randn(128, 4, 30)
        output = m(input1, input2)
        assert output.shape == (128, 4, 40)

    def test_bidense_1d(self):
        m = BiDense(20, 30, 40)
        input1 = mnp.randn(20)
        input2 = mnp.randn(30)
        output = m(input1, input2)
        assert output.shape == (40,)