import unittest
from easy_mindspore.nn import BiDense
from easy_mindspore import randn

class TestDense(unittest.TestCase):
    def test_bidense(self):
        m = BiDense(20, 30, 40)
        input1 = randn(128, 20)
        input2 = randn(128, 30)
        output = m(input1, input2)
        assert output.shape == (128, 40)